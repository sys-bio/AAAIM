import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from benchmark.scripts import phase3_biencoder as pb


def _example(**updates):
    value = {
        "schema": pb.DATASET_SCHEMA,
        "model_id": "m",
        "reaction_id": "r",
        "cluster_id": "c",
        "split": "train",
        "phase2_stratum": "empty_constrained",
        "query_template": pb.QUERY_TEMPLATE,
        "document_template": pb.DOCUMENT_TEMPLATE,
        "query": "Equation: A => B\nParticipants: A; B",
        "positive_ids": ["R00001", "R00002"],
        "negatives": [
            {"kegg_id": "R00003", "source": "bm25", "source_rank": 1,
             "uncertain_brite_orthology_sibling": False}
        ],
        "bm25_first_positive_rank": 2,
    }
    value.update(updates)
    return value


def test_training_population_is_frozen_train_only():
    rows = pb.load_training_rows()
    assert len(rows) == 3497
    assert set(rows.split) == {"train"}
    assert rows.positive_ids.map(len).min() >= 1
    assert set(rows.cluster_id).isdisjoint(
        set(pd.read_csv(pb.retrieval.SPLITS, dtype=str).query("split != 'train'").cluster_id)
    )


def test_no_catalog_positive_rows_are_explicitly_excluded(monkeypatch):
    frame = _mock_training_frame()
    frame.at[0, "ground_truth_kegg_all"] = "R99999"
    frame.at[0, "positive_ids"] = ["R99999"]
    monkeypatch.setattr(pb, "load_training_rows", lambda: frame)
    monkeypatch.setattr(pb.retrieval, "load_catalog", _mock_catalog)
    examples, summary, _ = pb.build_training_examples()
    assert examples == []
    assert summary["n_frozen_train_rows"] == 1
    assert summary["n_excluded_no_catalog_positive"] == 1
    assert summary["excluded_no_catalog_positive"][0]["positive_ids"] == ["R99999"]


@pytest.mark.parametrize("split", ["validation", "test"])
def test_training_rejects_validation_and_test_split(split):
    with pytest.raises(ValueError, match="only the frozen train"):
        pb.load_training_rows(split)
    with pytest.raises(ValueError, match="rejected"):
        pb.assert_train_only([_example(split=split)])


def test_query_and_documents_pass_digit_bounded_leakage_scanner():
    with pytest.raises(ValueError, match="leakage"):
        pb.assert_no_kegg_leakage("prefixR01234_suffix", where="unit test")
    row = pb.load_training_rows().iloc[0].to_dict()
    query = pb.retrieval.query_text({
        "reaction_equation": row["reaction_equation"],
        "participant_evidence": row["participant_evidence"],
    })
    pb.assert_no_kegg_leakage(query, where="unit test query")
    pb.assert_no_kegg_leakage(pb.retrieval.load_catalog()[0]["text"], where="unit test document")


def test_multi_positive_loss_uses_every_valid_positive():
    import torch

    query = torch.tensor([[1.0, 0.0]], requires_grad=True)
    docs = torch.tensor([[1.0, 0.0], [0.9, 0.1], [-1.0, 0.0]], requires_grad=True)
    multi = pb.multi_positive_contrastive_loss(query, docs, [2], [3], temperature=0.1)
    single = pb.multi_positive_contrastive_loss(query, docs, [1], [3], temperature=0.1)
    assert multi < single
    multi.backward()
    assert query.grad is not None and docs.grad is not None


def test_positive_ids_are_excluded_from_negatives():
    pb.validate_negative_list(_example())
    bad = _example(negatives=[
        {"kegg_id": "R00002", "source": "bm25", "source_rank": 1,
         "uncertain_brite_orthology_sibling": False}
    ])
    with pytest.raises(ValueError, match="positive IDs"):
        pb.validate_negative_list(bad)


def test_duplicate_negatives_are_rejected():
    negative = {"kegg_id": "R00003", "source": "bm25", "source_rank": 1,
                "uncertain_brite_orthology_sibling": False}
    with pytest.raises(ValueError, match="duplicate negatives"):
        pb.validate_negative_list(_example(negatives=[negative, dict(negative)]))


def _mock_training_frame():
    return pd.DataFrame([{
        "model_id": "m", "reaction_id": "r", "cluster_id": "c", "split": "train",
        "stratum": "empty_constrained", "status": "no_candidates", "candidate_set_size": "0",
        "reaction_equation": "A => B", "substrate_names": "A", "product_names": "B",
        "query_text": "A -> B", "participant_evidence": "A; B",
        "ground_truth_kegg_all": "R00001;R00002", "positive_ids": ["R00001", "R00002"],
    }])


def _mock_catalog():
    return [
        {"kegg_id": "R00001", "text": "alpha product"},
        {"kegg_id": "R00002", "text": "alpha product alternate"},
        {"kegg_id": "R00003", "text": "alpha reactant hard"},
        {"kegg_id": "R00004", "text": "alpha sibling"},
        {"kegg_id": "R00005", "text": "unrelated control"},
        {"kegg_id": "R00006", "text": "other control"},
    ]


def test_hard_negative_selection_is_deterministic_and_multilabel_safe(monkeypatch):
    monkeypatch.setattr(pb, "load_training_rows", _mock_training_frame)
    monkeypatch.setattr(pb.retrieval, "load_catalog", _mock_catalog)
    monkeypatch.setattr(pb, "is_equivalent", lambda candidate, truth, kind: False)
    first, summary1, negatives1 = pb.build_training_examples(bm25_hard_negatives=2, random_negatives=1, seed=7)
    second, summary2, negatives2 = pb.build_training_examples(bm25_hard_negatives=2, random_negatives=1, seed=7)
    assert first == second and summary1 == summary2 and negatives1 == negatives2
    assert first[0]["positive_ids"] == ["R00001", "R00002"]
    assert not set(first[0]["positive_ids"]) & {x["kegg_id"] for x in first[0]["negatives"]}


def test_brite_orthology_sibling_policy_excludes_uncertain_candidate(monkeypatch):
    monkeypatch.setattr(pb, "load_training_rows", _mock_training_frame)
    monkeypatch.setattr(pb.retrieval, "load_catalog", _mock_catalog)
    monkeypatch.setattr(pb, "is_equivalent", lambda candidate, truth, kind: candidate == "R00003")
    examples, _, summary = pb.build_training_examples(bm25_hard_negatives=1, random_negatives=0, seed=7)
    assert "R00003" not in {x["kegg_id"] for x in examples[0]["negatives"]}
    assert summary["n_uncertain_brite_orthology_siblings_excluded"] >= 1
    uncertain = _example(negatives=[
        {"kegg_id": "R00004", "source": "bm25", "source_rank": 2,
         "uncertain_brite_orthology_sibling": True}
    ])
    with pytest.raises(ValueError, match="uncertain"):
        pb.validate_negative_list(uncertain)


def test_training_configuration_hash_changes_for_every_field():
    base = pb.TrainingConfig()
    assert base.hash == pb.TrainingConfig().hash
    changed = pb.TrainingConfig(learning_rate=3e-5)
    assert changed.hash != base.hash
    assert pb.TrainingConfig(max_length=128).hash != base.hash


def _checkpoint_objects():
    import torch

    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    return model, optimizer, scheduler, scaler


def test_checkpoint_invalidation_and_resume_state(tmp_path):
    model, optimizer, scheduler, scaler = _checkpoint_objects()
    config = pb.TrainingConfig()
    path = tmp_path / "checkpoint.pt"
    expected_state = {"optimizer_step": 4, "epoch": 1, "cursor": 8, "examples_processed": 40}
    pb.save_checkpoint(
        path, model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
        state=expected_state, config=config, dataset_hash="dataset-a",
    )
    original = {key: value.detach().clone() for key, value in model.state_dict().items()}
    for parameter in model.parameters():
        parameter.data.zero_()
    resumed = pb.load_checkpoint(
        path, model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
        config=config, dataset_hash="dataset-a",
    )
    assert resumed == expected_state
    assert all(np.array_equal(model.state_dict()[k].numpy(), v.numpy()) for k, v in original.items())
    with pytest.raises(ValueError, match="dataset"):
        pb.load_checkpoint(
            path, model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
            config=config, dataset_hash="dataset-b",
        )
    with pytest.raises(ValueError, match="configuration"):
        pb.load_checkpoint(
            path, model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
            config=pb.TrainingConfig(max_length=128), dataset_hash="dataset-a",
        )


def test_tiny_shared_encoder_training_step_reduces_loss():
    import torch
    import torch.nn.functional as functional

    torch.manual_seed(5)
    encoder = torch.nn.Embedding(4, 3)
    optimizer = torch.optim.SGD(encoder.parameters(), lr=0.5)
    query_ids = torch.tensor([0])
    doc_ids = torch.tensor([1, 2, 3])
    losses = []
    for _ in range(10):
        optimizer.zero_grad()
        query = functional.normalize(encoder(query_ids), dim=1)
        docs = functional.normalize(encoder(doc_ids), dim=1)
        loss = pb.multi_positive_contrastive_loss(query, docs, [2], [3], temperature=0.2)
        losses.append(float(loss.detach()))
        loss.backward()
        optimizer.step()
    assert losses[-1] < losses[0]


def test_ranking_output_unique_and_consecutive():
    population = pd.DataFrame([{"model_id": "m", "reaction_id": "r"}])
    rows = pb.make_ranking_rows(population, [["R00002", "R00001"]])
    pb.validate_ranking_rows(rows, expected=1)
    rows[0]["ranked"][1]["rank"] = 3
    with pytest.raises(ValueError, match="consecutive"):
        pb.validate_ranking_rows(rows, expected=1)
    rows[0]["ranked"] = [{"rank": 1, "kegg_id": "R00001"}, {"rank": 2, "kegg_id": "R00001"}]
    with pytest.raises(ValueError, match="duplicate ranked"):
        pb.validate_ranking_rows(rows, expected=1)


def test_validation_only_and_frozen_before_label_join(tmp_path):
    bad = [{
        "model_id": "m", "reaction_id": "r", "split": "test",
        "ranked": [{"rank": 1, "kegg_id": "R00001"}],
    }]
    with pytest.raises(ValueError, match="validation-only"):
        pb.validate_ranking_rows(bad, expected=1)
    with pytest.raises(ValueError, match="frozen before"):
        pb._load_validation_truth_after_freeze(tmp_path / "missing.jsonl")


def test_manifest_verification_is_read_only_and_paths_are_posix(tmp_path, monkeypatch):
    artifact = tmp_path / "metric.json"
    artifact.write_text("{}\n", encoding="utf-8")
    pb.write_artifact_manifest(tmp_path, [artifact], root=tmp_path)
    monkeypatch.setattr(pb, "REPO_ROOT", tmp_path)
    manifest = tmp_path / "artifact_manifest.json"
    before = manifest.read_bytes()
    assert pb.verify_manifest(tmp_path) == []
    assert manifest.read_bytes() == before
    paths = [x["path"] for x in json.loads(before)["files"]]
    assert len(paths) == len(set(paths)) and all("\\" not in path for path in paths)


def test_loading_training_rows_does_not_mutate_frozen_artifacts():
    paths = [
        pb.REPO_ROOT / "benchmark" / "PHASE2_MANIFEST.json",
        pb.REPO_ROOT / "benchmark" / "phase3" / "validation" / "artifact_manifest.json",
        pb.REPO_ROOT / "benchmark" / "phase3" / "retrieval_baselines" / "artifact_manifest.json",
    ]
    before = {path: path.read_bytes() for path in paths}
    pb.load_training_rows()
    assert before == {path: path.read_bytes() for path in paths}
