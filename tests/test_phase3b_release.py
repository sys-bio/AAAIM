import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from benchmark.scripts import phase3b_release as release


def _write_jsonl(path: Path, rows):
    path.write_text("".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8", newline="\n")


def test_selected_checkpoint_and_committed_metadata_match_expected_values():
    actual = release.verify_selected_checkpoint()
    assert actual["selected_epoch"] == 1
    assert actual["checkpoint_sha256"] == release.EXPECTED_CHECKPOINT_SHA256
    assert actual["revision"] == release.biencoder.MODEL_REVISION
    assert actual["recall_at_1"] == 0.840041
    assert actual["recall_at_10"] == 0.921569


def test_complete_rrf_is_deterministic_one_indexed_and_tie_broken():
    catalog = ["R00001", "R00002", "R00003", "R00004"]
    # R00001 and R00002 have symmetric scores; ascending ID resolves the tie.
    first = release.reciprocal_rank_fusion_complete(
        [["R00002", "R00003"], ["R00001", "R00003"]], catalog, k=60,
    )
    second = release.reciprocal_rank_fusion_complete(
        [["R00002", "R00003"], ["R00001", "R00003"]], catalog, k=60,
    )
    assert first == second == ["R00003", "R00001", "R00002", "R00004"]
    # The absent document contributes zero and follows every positive-score ID.
    assert first[-1] == "R00004"


def test_rrf_rejects_duplicate_or_non_catalog_component_ids():
    with pytest.raises(ValueError, match="duplicate"):
        release.reciprocal_rank_fusion_complete([["R00001", "R00001"]], ["R00001"])
    with pytest.raises(ValueError, match="non-catalog"):
        release.reciprocal_rank_fusion_complete([["R99999"]], ["R00001"])


def test_fusion_validation_rejects_test_rows_and_bad_catalog_membership(monkeypatch):
    monkeypatch.setattr(release, "EXPECTED_CATALOG_SIZE", 3)
    good = [{
        "schema": release.FUSION_SCHEMA,
        "split": "validation",
        "model_id": "m",
        "reaction_id": "r",
        "method": release.FUSION_METHOD,
        "ranked_ids": ["R00001", "R00002", "R00003"],
    }]
    release.validate_fusion_rows(good, ["R00001", "R00002", "R00003"], expected=1)
    with pytest.raises(ValueError, match="validation-only"):
        release.validate_fusion_rows([dict(good[0], split="test")], ["R00001", "R00002", "R00003"], expected=1)
    with pytest.raises(ValueError, match="permutation"):
        release.validate_fusion_rows([dict(good[0], ranked_ids=["R00001", "R00001", "R00003"])], ["R00001", "R00002", "R00003"], expected=1)
    with pytest.raises(ValueError, match="membership"):
        release.validate_fusion_rows([dict(good[0], ranked_ids=["R00001", "R00002", "R99999"])], ["R00001", "R00002", "R00003"], expected=1)


def test_label_free_fusion_construction_never_loads_truth(tmp_path, monkeypatch):
    bm25_path = tmp_path / "bm25.jsonl"
    trained_path = tmp_path / "trained.jsonl"
    catalog_path = tmp_path / "catalog.json"
    catalog_source = tmp_path / "catalog.lzma"
    catalog_source.write_bytes(b"catalog")
    catalog_path.write_text(json.dumps({"ids": ["R00001", "R00002", "R00003"], "n": 3, "source": "catalog.lzma"}), encoding="utf-8")
    _write_jsonl(bm25_path, [{"model_id": "m", "reaction_id": "r", "method": "bm25", "ranked_ids": ["R00002"]}])
    _write_jsonl(trained_path, [{"model_id": "m", "reaction_id": "r", "split": "validation", "ranked_ids": ["R00001"]}])
    monkeypatch.setattr(release, "BM25_RANKINGS", bm25_path)
    monkeypatch.setattr(release, "SELECTED_RANKINGS", trained_path)
    monkeypatch.setattr(release, "CATALOG_IDS", catalog_path)
    monkeypatch.setattr(release.retrieval, "CATALOG", catalog_source)
    monkeypatch.setattr(release, "EXPECTED_CATALOG_SIZE", 3)
    monkeypatch.setattr(release, "EXPECTED_VALIDATION_SIZE", 1)
    monkeypatch.setattr(release, "verify_selected_checkpoint", lambda: {})
    monkeypatch.setattr(release.retrieval, "load_query_population", lambda split: pd.DataFrame([{"model_id": "m", "reaction_id": "r", "split": "validation"}]))

    def forbidden():
        raise AssertionError("ground truth must not be touched during ranking")

    monkeypatch.setattr(release.retrieval, "_truth_and_metadata", forbidden)
    ranking = release.freeze_fusion_rankings(tmp_path / "out")
    rows = release._read_xz_jsonl(ranking)
    assert rows[0]["ranked_ids"] == ["R00001", "R00002", "R00003"]
    freeze = json.loads((tmp_path / "out" / "ranking_freeze.json").read_text())
    assert freeze["labels_loaded_during_ranking"] is False
    assert freeze["test_rows_read"] == 0
    assert freeze["sha256"] == release.sha256_file(ranking)


def _minimal_archive(path: Path):
    names = {
        "model.safetensors": b"weights",
        "config.json": b"{}\n",
        "tokenizer.json": b"{}\n",
        "tokenizer_config.json": b"{}\n",
        "vocab.txt": b"vocab\n",
        "special_tokens_map.json": b"{}\n",
        "inference_config.json": b"{}\n",
        "environment.json": b"{}\n",
        "requirements-inference.txt": b"transformers\n",
        "README.md": b"readme\n",
        "inference_fixture.json": b"{}\n",
    }
    manifest = {
        "schema": release.RELEASE_SCHEMA,
        "self_exclusion": "recursive self-digest",
        "n_payload_files": len(names),
        "files": [
            {"path": name, "bytes": len(blob), "sha256": hashlib.sha256(blob).hexdigest()}
            for name, blob in sorted(names.items())
        ],
    }
    names["archive_manifest.json"] = release._json_bytes(manifest)
    with zipfile.ZipFile(path, "w") as archive:
        for name, blob in names.items():
            archive.writestr(name, blob)


def test_archive_contents_and_payload_digests_are_verified(tmp_path):
    archive = tmp_path / "model.zip"
    _minimal_archive(archive)
    assert release.verify_archive_contents(archive) == []
    with zipfile.ZipFile(archive, "a") as handle:
        handle.writestr("checkpoint.pt", b"forbidden")
    problems = release.verify_archive_contents(archive)
    assert any("member mismatch" in problem for problem in problems)
    assert any("checkpoint state" in problem for problem in problems)


def test_restoration_path_uses_extracted_archive_and_reproduces_rankings(tmp_path, monkeypatch):
    archive = tmp_path / "model.zip"
    _minimal_archive(archive)
    expected_path = tmp_path / "expected.jsonl"
    _write_jsonl(expected_path, [{
        "schema": release.biencoder.FULL_RANKING_SCHEMA,
        "epoch": 1,
        "model_id": "m",
        "reaction_id": "r",
        "split": "validation",
        "ranked_ids": ["R00001", "R00002"],
    }])
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({"restoration": {}}), encoding="utf-8")
    monkeypatch.setattr(release, "DIST_DIR", tmp_path)
    monkeypatch.setattr(release, "ARCHIVE_REGISTRY", registry)
    monkeypatch.setattr(release, "SELECTED_RANKINGS", expected_path)
    monkeypatch.setattr(release, "EXPECTED_RANKING_SHA256", "expected")
    monkeypatch.setattr(release, "verify_archive_contents", lambda path: [])
    monkeypatch.setattr(release, "_verify_restored_fixture", lambda path: {"rankings_exact": True, "embedding_digests_exact": True})
    monkeypatch.setattr(release, "_load_archive_model", lambda path, device_name: (object(), object(), device_name))
    monkeypatch.setattr(release, "_encode_local", lambda model, tokenizer, texts, device, batch_size=64: np.array([[1.0, 0.0], [0.0, 1.0]]) if len(texts) == 2 else np.array([[1.0, 0.0]]))
    monkeypatch.setattr(release.retrieval, "load_query_population", lambda split: pd.DataFrame([{"model_id": "m", "reaction_id": "r"}]))
    monkeypatch.setattr(release.retrieval, "query_text", lambda row: "query")
    monkeypatch.setattr(release.retrieval, "load_catalog", lambda: [{"kegg_id": "R00001", "text": "a"}, {"kegg_id": "R00002", "text": "b"}])
    monkeypatch.setattr(release.biencoder, "validate_full_ranking_rows", lambda rows, epoch: None)
    result = release.restore_and_verify(archive, batch_size=2)
    assert result["loaded_from_extracted_archive_only"] is True
    assert result["source_checkpoint_accessed_for_restore"] is False
    assert result["source_model_cache_accessed_for_restore"] is False
    assert result["validation"]["ranked_kegg_ids_exact"] is True


def test_fusion_manifest_verification_is_read_only(tmp_path, monkeypatch):
    artifact = tmp_path / "x.json"
    artifact.write_text("{}\n", encoding="utf-8")
    release.write_artifact_manifest(tmp_path, [artifact], root=tmp_path)
    monkeypatch.setattr(release, "REPO_ROOT", tmp_path)
    before = (tmp_path / "artifact_manifest.json").read_bytes()
    assert release.verify_fusion_manifest(tmp_path) == []
    assert (tmp_path / "artifact_manifest.json").read_bytes() == before


def test_derived_fusion_rebuilds_are_byte_identical(tmp_path, monkeypatch):
    def fake_freeze(out):
        out.mkdir(parents=True, exist_ok=True)
        (out / release.FUSION_RANKING_NAME).write_bytes(b"ranking")
        (out / "ranking_freeze.json").write_text("{}\n", encoding="utf-8", newline="\n")
        return out / release.FUSION_RANKING_NAME

    def fake_evaluate(out):
        report = out / "REPORT.md"
        report.write_text("deterministic\n", encoding="utf-8", newline="\n")
        return [report]

    monkeypatch.setattr(release, "freeze_fusion_rankings", fake_freeze)
    monkeypatch.setattr(release, "evaluate_frozen_fusion", fake_evaluate)
    result = release.rebuild_fusion_twice(tmp_path / "fusion")
    assert result["byte_identical"] is True and result["passes"] == 2
    assert release.verify_fusion_manifest(tmp_path / "fusion") == []


def test_fusion_config_records_prespecified_label_free_validation_design():
    config = release.fusion_config()
    assert config["rrf"] == {
        "k": 60,
        "rank_origin": 1,
        "weights": "equal",
        "missing_contribution": 0.0,
        "tie_break": "KEGG identifier ascending",
    }
    assert "label-free" in config["construction"]
    assert config["test_rows_read"] == 0
