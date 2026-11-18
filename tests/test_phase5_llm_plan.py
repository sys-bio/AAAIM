"""Synthetic cache tests plus the actual 163-case frozen clean construction."""
import copy
import json
import os
import socket
from pathlib import Path

import pytest

from benchmark.scripts import phase3c_grounded as grounded
from benchmark.scripts import phase5_llm_plan as plan
from benchmark.scripts.phase5_perturb import digest


def fixture():
    record = {"evidence_id": "E01", "kegg_id": "R99991", "fused_rank": 1, "fused_score": .03,
              "bm25_rank": 1, "trained_biencoder_rank": 1, "name": "Synthetic", "definition": "Synthetic candidate",
              "equation": "C00001 <=> C00002", "enzyme": "", "rclass": "", "brite": "", "provenance": {"synthetic": True}}
    query = "Synthetic substrate => synthetic product"
    payload = {"model": "synthetic-only", "input": grounded.build_user_prompt({"query": query}, [grounded.EvidenceRecord(**record)]), "tools": []}
    return query, [record], payload


def test_removing_visible_provenance_is_a_real_cache_miss():
    query, records, old = fixture()
    rebuilt, compatible = plan.payload_for(query, records, old, records)
    assert not compatible
    assert rebuilt != plan.provider_payload(old)
    assert all("provenance" not in r for r in plan.split_input(rebuilt["input"])["EVIDENCE"])
    again, compatible = plan.payload_for(query, records, rebuilt)
    assert compatible and again == rebuilt


def test_sidecar_changes_do_not_change_provider_input():
    query, records, old = fixture()
    before, _ = plan.payload_for(query, records, old)
    records[0]["provenance"].update(query_digest="new operational lineage", file="different location")
    after, _ = plan.payload_for(query, records, old)
    assert before == after and digest(before) == digest(after)


@pytest.mark.parametrize("field,new", [("kegg_id", "R99992"), ("fused_score", .031), ("trained_biencoder_rank", 2), ("definition", "changed biological evidence")])
def test_actual_evidence_changes_always_change_complete_cache_identity(field, new):
    query, records, template = fixture()
    before, _ = plan.payload_for(query, records, template)
    records[0][field] = new
    after, compatible = plan.payload_for(query, records, before)
    assert not compatible and digest(before) != digest(after)


@pytest.mark.parametrize("field,value", [("model", "another-model"), ("instructions", "other instructions"), ("max_output_tokens", 100), ("reasoning", {"effort": "high"}), ("tools", [{"type": "web_search_preview"}])])
def test_provider_settings_are_never_ignored_for_cache_identity(field, value):
    query, records, template = fixture()
    before, _ = plan.payload_for(query, records, template)
    changed = {**before, field: value}
    after, _ = plan.payload_for(query, records, changed)
    assert after[field] == value and digest(after) != digest(before)


def test_schema_and_literal_prompt_serialization_are_in_identity():
    query, records, template = fixture()
    before, _ = plan.payload_for(query, records, template)
    changed = copy.deepcopy(before)
    changed["text"]["format"]["schema"]["description"] = "changed response schema"
    after, _ = plan.payload_for(query, records, changed)
    assert digest(after) != digest(before)
    changed = copy.deepcopy(before)
    changed["input"] = changed["input"].replace("TARGET\n", "TARGET\n ")
    assert plan.split_input(changed["input"]) == plan.split_input(before["input"])
    report = plan.compare_payloads(before, changed)
    assert not report["complete_provider_payload_equal"]
    assert report["field_differences"][0]["category"] == "serialization"
    assert digest(changed) != digest(before)


def test_no_provider_or_network_needed_to_capture_real_sdk_body(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("network must not be constructed")
    monkeypatch.setattr(socket, "socket", forbidden)
    query, records, template = fixture()
    body = plan.provider_payload({**template, "prompt_version": "internal", "schema_version": "internal"})
    assert "prompt_version" not in body and "schema_version" not in body
    assert body["text"]["format"]["type"] == "json_schema"
    assert body["text"]["format"]["strict"] is True
    assert plan.provider_payload(body) == body


def test_json_numeric_type_differences_are_not_erased_by_python_equality():
    query, records, template = fixture()
    before, _ = plan.payload_for(query, records, template)
    records[0]["fused_rank"] = 1.0
    after, compatible = plan.payload_for(query, records, before)
    assert not compatible and digest(before) != digest(after)
    report = plan.compare_payloads(before, after)
    assert not report["complete_provider_payload_equal"]
    assert report["field_differences"][0]["category"] == "numerical"


def test_unknown_sdk_version_cannot_fake_a_historical_wire_comparison(monkeypatch):
    import openai
    monkeypatch.setattr(openai, "__version__", "unverified-future-version")
    _, _, template = fixture()
    with pytest.raises(RuntimeError, match="pinned"):
        plan.provider_payload(template)


def test_paid_execution_fails_before_file_access(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("input opened before execution denial")
    monkeypatch.setattr(Path, "read_text", forbidden)
    with pytest.raises(SystemExit, match="disabled"):
        plan.main(["--execute", "--source-root", "SYNTHETIC", "--out", "SYNTHETIC"])


@pytest.fixture(scope="module")
def real_context():
    root = plan.ROOT
    source = Path(os.environ.get("AAAIM_PHASE5_SOURCE_ROOT", plan.load(root / "benchmark/phase5/provenance.json")["source_workspace"]))
    if not (source / "data/kegg/kegg_reaction_features.lzma").exists():
        pytest.skip("real clean-control integration requires the locally frozen catalog; set AAAIM_PHASE5_SOURCE_ROOT")
    return plan.read_context(source, root / "benchmark/phase5/validation")


def test_real_original_clean_path_matches_all_163_captured_builder_digests(real_context):
    report = plan.clean_control(real_context)
    golden = plan.load(plan.ROOT / "benchmark/phase5/superseded/ea241ea/legacy_clean_request_spec_digests.json")
    assert len(report["cases"]) == 163
    assert all(r["legacy_request_spec_digest"] == golden[r["sample_id"]] for r in report["cases"])
    assert report["complete_equal_count"] == 0
    assert report["leaf_difference_counts"] == {"provenance": 4890, "numerical": 2, "biological_evidence": 0, "provider_settings": 0, "serialization": 0}
    assert report["cases_with_equal_candidate_fields"]["candidate_ids_and_order"] == 163
    assert report["cases_with_equal_candidate_fields"]["trained_biencoder_rank"] == 162


def test_real_clean_numerical_drift_is_reported_not_overwritten(real_context):
    report = plan.clean_control(real_context)
    assert report["cases_with_non_provenance_differences"] == ["P3P0096"]
    case = next(r for r in report["cases"] if r["sample_id"] == "P3P0096")
    fields = {d["path"]: d for d in case["field_differences"] if d["category"] == "numerical"}
    assert fields["input.EVIDENCE[6].trained_biencoder_rank"]["before"] == 70
    assert fields["input.EVIDENCE[6].trained_biencoder_rank"]["after"] == 69
    rows = plan.matched_requests(real_context)
    clean = next(r for r in rows if r["sample_id"] == "P3P0096" and r["arm"] == "clean")
    visible = plan.split_input(clean["payload"]["input"])["EVIDENCE"][6]
    assert visible["trained_biencoder_rank"] == 69
    assert visible["fused_score"] == .020909832721


def test_real_matched_clean_and_degraded_share_all_provider_settings(real_context):
    rows = plan.matched_requests(real_context)
    assert len(rows) == 326
    assert not any(r["compatible_historical_clean_body"] for r in rows)
    for sid in real_context["ids"]:
        clean, degraded = [r for r in rows if r["sample_id"] == sid]
        assert {k: v for k, v in clean["payload"].items() if k != "input"} == {k: v for k, v in degraded["payload"].items() if k != "input"}
        for request, condition in [(clean, "clean"), (degraded, plan.CONDITION)]:
            actual = plan.split_input(request["payload"]["input"])
            v, r = real_context["variants"][condition, sid], real_context["ranks"][condition, sid]
            assert actual["TARGET"] == {"query": v["query"]}
            assert actual["EVIDENCE"] == plan.scientific_records(real_context["catalog"], r)
            assert all(set(e) == plan.VISIBLE_FIELDS for e in actual["EVIDENCE"])
            assert request["sidecar"]["ranking_artifact_sha256_portable"] == real_context["prediction_digest"]
            assert request["sidecar"]["query_digest"] == digest(v["query"])
            rebuilt, compatible = plan.payload_for(v["query"], actual["EVIDENCE"], request["payload"])
            assert compatible and rebuilt == request["payload"]  # every complete field
        assert clean["canonical_provider_payload_digest"] != degraded["canonical_provider_payload_digest"]


def test_candidate_order_is_part_of_actual_provider_inputs():
    query, records, template = fixture()
    second = {**records[0], "kegg_id": "R99992", "evidence_id": "E02", "fused_rank": 2}
    before, _ = plan.payload_for(query, [records[0], second], template)
    after, compatible = plan.payload_for(query, [second, records[0]], before)
    assert not compatible and digest(after) != digest(before)
    diff = plan.compare_payloads(before, after)
    assert any(d["path"].endswith("kegg_id") and d["category"] == "biological_evidence" for d in diff["field_differences"])


def test_gpt61_model_override_changes_only_model_and_preserves_strict_schema(real_context):
    requests = plan.matched_requests(real_context, model_override="gpt-6.1-terra")
    original = real_context["original"]
    assert len(requests) == 326
    for row in requests:
        sid, body = row["sample_id"], row["payload"]
        old = plan.provider_payload(original[sid]["payload"])
        assert body["model"] == "gpt-6.1-terra" and row["sidecar"]["model_requested"] == "gpt-6.1-terra"
        assert body["text"] == old["text"] and body["reasoning"] == old["reasoning"]
        assert body["max_output_tokens"] == old["max_output_tokens"] and body["tools"] == old["tools"] == []
        assert body["instructions"] == old["instructions"]
        assert plan.provider_payload(body) == body
        changes = plan.field_differences(old, body)
        # Rebuilt scientific evidence is also a real provider-input change. The
        # new matched baseline must therefore use GPT-6.1; this is not a
        # historical-payload-compatible model-only substitution.
        changed_paths = {item["path"] for item in changes}
        assert changed_paths == {"model", "input"}
        assert next(item for item in changes if item["path"] == "model")["after"] == "gpt-6.1-terra"
        pair = [item for item in requests if item["sample_id"] == sid]
        clean = next(item for item in pair if item["arm"] == "clean")
        degraded = next(item for item in pair if item["arm"] == "degraded")
        assert {k: v for k, v in clean["payload"].items() if k != "input"} == {
            k: v for k, v in degraded["payload"].items() if k != "input"}


def test_real_generated_plans_keep_bodies_sidecars_and_manifests_consistent(real_context, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(plan, "read_context", lambda source, out: real_context)
    summary = plan.plan(tmp_path, tmp_path, "reproduced", publish_current=False)
    capsys.readouterr()
    out = tmp_path / "reproduced"
    full = plan.read_rows(out / "_requests/clean.jsonl") + plan.read_rows(out / "_requests/degraded.jsonl")
    public = plan.read_rows(out / "request_manifest.jsonl")
    sidecars = plan.read_rows(out / "provenance_sidecars.jsonl")
    assert len(full) == len(public) == len(sidecars) == summary["attempt_cap"] == 326
    assert summary["compatible_historical_clean_cache_hits"] == 0
    assert summary["unique_provider_bodies"] == 317
    assert not (tmp_path / "llm_dry_run.json").exists()
    corrected = plan.load(out / "corrected_clean_control_diagnostic.json")
    assert corrected["n"] == 163 and corrected["complete_equal_count"] == 0
    assert corrected["leaf_difference_counts"]["numerical"] == 2
    assert set(corrected["leaf_difference_counts"]) == {"provenance", "numerical"}
    assert all(not d["after_present"] for case in corrected["cases"] for d in case["field_differences"] if d["category"] == "provenance")
    for request, manifest, sidecar in zip(full, public, sidecars):
        assert manifest["canonical_provider_payload_digest"] == digest(request["payload"])
        assert "payload" not in manifest and "sidecar" not in manifest
        assert sidecar["response_schema_digest"] == digest(request["payload"]["text"]["format"])
        assert sidecar["sdk_version"] == plan.PINNED_SDK_VERSION
        assert sidecar["pydantic_version"] == plan.PINNED_PYDANTIC_VERSION
        assert "ranking_artifact" not in request["payload"]["input"]
