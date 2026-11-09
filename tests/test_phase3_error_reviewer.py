"""Focused tests for the offline blank human-review interface."""

from __future__ import annotations

import copy
import csv
import json
import re
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path

import pytest

from benchmark.scripts import phase3_error_reviewer as reviewer


class ResourceParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.resources: list[tuple[str, str, str]] = []
        self.ids: list[str] = []

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if "id" in values:
            self.ids.append(values["id"])
        for attribute in ("src", "href"):
            if attribute in values:
                self.resources.append((tag, attribute, values[attribute]))


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    out = tmp_path_factory.mktemp("error-reviewer")
    bundles = reviewer.build_bundle(out)
    return out, bundles


def test_exact_case_counts_alignment_and_blinded_order(bundle):
    out, bundles = bundle
    pass1, pass2 = bundles["pass1"], bundles["pass2"]
    assert len(pass1["cases"]) == len(pass2["cases"]) == 60
    assert [case["audit_id"] for case in pass1["cases"]] == [case["audit_id"] for case in pass2["cases"]]
    with (reviewer.ERROR_AUDIT / "formal_audit_blinded_order.csv").open(encoding="utf-8", newline="") as handle:
        frozen = sorted(csv.DictReader(handle), key=lambda row: int(row["blinded_order"]))
    assert [case["audit_id"] for case in pass1["cases"]] == [row["audit_id"] for row in frozen]
    assert _load(out / "formal_pass1_data.json")["dataset_digest"] == pass1["dataset_digest"]


def test_browse_all_partition_and_formal_membership(bundle):
    cases = bundle[1]["browse"]["cases"]
    assert len(cases) == 61
    assert Counter(case["facets"]["outcome_kind"] for case in cases) == {
        "incorrect_selection": 22, "abstention": 39,
    }
    assert all(case["facets"]["formal_sample_inclusion"] in {True, False} for case in cases)


def test_supplemental_e12_is_separate_training_case(bundle):
    bundles = bundle[1]
    supplemental = bundles["supplemental"]["cases"][0]
    assert (supplemental["model_id"], supplemental["reaction_id"]) == ("BIOMD0000000013", "E12")
    assert supplemental["split_status"] == "training"
    assert supplemental["formal_validation_audit_inclusion"] is False
    formal_keys = {(case["model_id"], case["reaction_id"]) for case in bundles["pass1"]["cases"]}
    assert (supplemental["model_id"], supplemental["reaction_id"]) not in formal_keys
    assert supplemental["manual_nomination"]["concern"] == "R01429 contains xylonolactone which is not reflected in reaction E12"
    r01429 = next(card for card in supplemental["existing_labels"] if card["kegg_id"] == "R01429")
    assert r01429["name"] and r01429["definition"] and r01429["equation"]
    assert supplemental["existing_method_results"] is None
    assert "no new inference" in supplemental["existing_method_results_status"].lower()


def test_no_held_out_rows_or_labels(bundle):
    config = _load(bundle[0] / "reviewer_build_config.json")
    assert config["test_rows_loaded"] == config["test_labels_loaded"] == 0
    assert all(case["model_id"] != "test" for data in bundle[1].values() for case in data["cases"])


def test_complete_reaction_side_display_data(bundle):
    cases = [*bundle[1]["pass1"]["cases"], *bundle[1]["browse"]["cases"], *bundle[1]["supplemental"]["cases"]]
    for case in cases:
        reaction = case["model_reaction"]
        assert reaction["model_name"] and reaction["model_id"] and reaction["reaction_id"]
        assert reaction["readable_equation"] and reaction["normalized_equation"]
        assert isinstance(reaction["reactants"], list) and isinstance(reaction["products"], list)
        assert isinstance(reaction["modifiers"], list) and reaction["direction"]
        assert reaction["source_annotation_provenance"]["sbml_path"]
        assert len(reaction["source_annotation_provenance"]["sbml_sha256"]) == 64
        for item in [*reaction["reactants"], *reaction["products"], *reaction["modifiers"]]:
            assert set(item) >= {
                "species_id", "display_name", "stoichiometry", "compartment_id", "compartment_name",
                "chebi_annotations", "kegg_compound_annotations", "annotation_resources",
            }


def test_catalog_backed_labels_are_readable_and_multilabels_are_separate(bundle):
    cases = bundle[1]["pass1"]["cases"]
    assert any(len(case["existing_labels"]) > 1 for case in cases)
    for case in cases:
        ids = [card["kegg_id"] for card in case["existing_labels"]]
        assert len(ids) == len(set(ids))
        for card in case["existing_labels"]:
            if card["catalog_status"] == "present_in_frozen_catalog":
                assert card["name"] and card["definition"] and card["equation"]
            else:
                assert card["catalog_status"] == "absent_from_frozen_catalog"


def test_missing_data_is_explicit_and_json_has_no_nan(bundle):
    out, bundles = bundle
    assert any(case["model_reaction"]["reaction_name"] is None for case in bundles["pass1"]["cases"])
    for name in reviewer.TRACKED_OUTPUTS:
        text = (out / name).read_text(encoding="utf-8")
        assert "NaN" not in text
    assert "Not provided in frozen local artifacts." in (out / "formal_pass1.html").read_text(encoding="utf-8")


def test_embedded_json_escaping_blocks_script_injection():
    dangerous = {"text": "</script><img src=x onerror=alert(1)>&\u2028"}
    encoded = reviewer.json_dumps_script(dangerous)
    assert "</script>" not in encoded
    assert "<img" not in encoded
    assert "\\u003c" in encoded and "\\u0026" in encoded


def test_offline_resources_csp_and_unique_static_ids(bundle):
    out = bundle[0]
    for name in ("index.html", "formal_pass1.html", "formal_pass2.html", "browse_phase3c_noncorrect.html", "supplemental_cases.html"):
        text = (out / name).read_text(encoding="utf-8")
        parser = ResourceParser(); parser.feed(text)
        assert len(parser.ids) == len(set(parser.ids)), f"duplicate static id in {name}"
        for tag, attribute, value in parser.resources:
            if tag == "a" and attribute == "href":
                assert not re.match(r"(?i)https?://", value)
            elif tag == "link":
                assert value == "reviewer.css"
            else:
                assert not re.match(r"(?i)(https?:)?//", value)
        assert "connect-src &#x27;none&#x27;" in text
        assert "fetch(" not in text and "XMLHttpRequest" not in text and "WebSocket(" not in text


def test_no_secrets_or_raw_provider_metadata(bundle):
    combined = "\n".join((bundle[0] / name).read_text(encoding="utf-8") for name in reviewer.TRACKED_OUTPUTS).lower()
    for token in ("api_key", "openai_api_key", "authorization: bearer", "langsmith", ".env", "provider_payload", "files_url", "download_url"):
        assert token not in combined


def test_pass1_physical_blinding_and_allowlist(bundle):
    out, bundles = bundle
    reviewer.assert_pass1_blinded(bundles["pass1"])
    assert all(set(case) == {
        "display_order", "audit_id", "sample_id", "model_id", "reaction_id", "model_reaction",
        "existing_labels", "existing_label_provenance",
    } for case in bundles["pass1"]["cases"])
    combined = ((out / "formal_pass1.html").read_text(encoding="utf-8") + (out / "formal_pass1_data.json").read_text(encoding="utf-8")).lower()
    for token in ("phase2", "bm25", "biencoder", "bi-encoder", "fusion", "phase3a", "phase3c", "primary_category", "suspicion_rules", "selection_priority", "retrieval_state"):
        assert token not in combined


def test_known_nonlabel_prediction_values_are_absent_from_pass1(bundle):
    pass1 = bundle[1]["pass1"]
    pass2 = bundle[1]["pass2"]
    source_ids = {card["kegg_id"] for case in pass1["cases"] for card in case["existing_labels"]}
    proposed = {
        system["prediction"]["kegg_id"]
        for case in pass2["cases"]
        for system in (case["systems"]["phase3a"], case["systems"]["phase3c"])
        if system["prediction"] and system["prediction"]["kegg_id"] not in source_ids
    }
    serialized = json.dumps(pass1, sort_keys=True)
    absent = [identifier for identifier in proposed if f'"{identifier}"' not in serialized]
    assert absent, "expected at least one system-only prediction identifier"
    assert all(f'"{identifier}"' not in serialized for identifier in absent)


def _valid_incomplete(pass1):
    reviews = [reviewer.blank_pass1_review(case["audit_id"]) for case in pass1["cases"]]
    payload = {
        "dataset_id": pass1["dataset_id"], "dataset_digest": pass1["dataset_digest"],
        "review_schema_version": reviewer.REVIEW_SCHEMA_VERSION, "review_kind": "formal_pass1",
        "reviewer_id": "synthetic-test", "reviews": reviews, "completion_count": 0, "complete": False,
    }
    payload["canonical_review_digest"] = reviewer._digest(payload)
    return payload


def test_valid_incomplete_checkpoint_and_completed_import_validation(bundle):
    pass1 = bundle[1]["pass1"]
    incomplete = _valid_incomplete(pass1)
    assert reviewer.validate_pass1_export(incomplete, pass1, require_complete=False) == []
    assert "all 60 Pass 1 reviews must be complete" in reviewer.validate_pass1_export(incomplete, pass1, require_complete=True)
    complete = copy.deepcopy(incomplete)
    for review in complete["reviews"]:
        review.update(verdict="unresolved", biological_rationale="Synthetic missing-evidence explanation.", complete=True)
    complete["completion_count"] = 60; complete["complete"] = True
    complete.pop("canonical_review_digest")
    complete["canonical_review_digest"] = reviewer._digest(complete)
    assert reviewer.validate_pass1_export(complete, pass1, require_complete=True) == []


@pytest.mark.parametrize("mutation", ["digest", "duplicate", "malformed", "verdict", "required"])
def test_pass1_validator_rejects_bad_imports(bundle, mutation):
    pass1 = bundle[1]["pass1"]
    payload = _valid_incomplete(pass1)
    if mutation == "digest": payload["dataset_digest"] = "0" * 64
    elif mutation == "duplicate": payload["reviews"][1]["audit_id"] = payload["reviews"][0]["audit_id"]
    elif mutation == "malformed": payload["reviews"] = "not-an-array"
    elif mutation == "verdict": payload["reviews"][0]["verdict"] = "not_allowed"
    else: payload["reviews"][0].update(verdict="label_correct", complete=True)
    assert reviewer.validate_pass1_export(payload, pass1)


def test_pass2_gate_import_lock_and_no_initial_case_display(bundle):
    text = (bundle[0] / "formal_pass2.html").read_text(encoding="utf-8")
    assert '<section id="gate"' in text
    assert '<div id="workspace" class="hidden">' in text
    assert "validatePass1Export" in text and "all 60" not in text.lower()
    assert "Pass 1 is locked" in text and "locked_pass1_export" in text


def test_export_schemas_and_local_storage_namespaces(bundle):
    out = bundle[0]
    pass1 = (out / "formal_pass1.html").read_text(encoding="utf-8")
    assert "canonical_review_digest" in pass1 and "completion_count" in pass1
    assert "pass1_review_" in pass1 and ".csv" in pass1 and ".json" in pass1
    assert "aaaim-review:formal-pass1:${DATA.review_schema_version}:${DATA.dataset_digest}" in pass1
    pass2 = (out / "formal_pass2.html").read_text(encoding="utf-8")
    assert "pass2_review" in pass2 and "locked_pass1_export" in pass2
    browse = (out / "browse_phase3c_noncorrect.html").read_text(encoding="utf-8")
    assert "exploratory_review" in browse


def test_browse_filters_are_complete(bundle):
    text = (bundle[0] / "browse_phase3c_noncorrect.html").read_text(encoding="utf-8")
    for identifier in ("f-outcome", "f-truth", "f-effect", "f-seen", "f-stratum", "f-model", "f-cluster", "f-compliance", "f-formal"):
        assert f'id="{identifier}"' in text
    assert "cannot estimate population label-error prevalence" in text
    assert "may unblind the formal audit" in text


def test_dataset_digests_are_stable_and_distinct(bundle):
    bundles = bundle[1]
    assert len({data["dataset_digest"] for data in bundles.values()}) == 4
    for data in bundles.values():
        core = {key: value for key, value in data.items() if key != "dataset_digest"}
        assert data["dataset_digest"] == reviewer._digest(core)


def test_generated_build_is_byte_deterministic(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    reviewer.build_bundle(first); reviewer.build_bundle(second)
    assert {path.name: path.read_bytes() for path in first.iterdir()} == {
        path.name: path.read_bytes() for path in second.iterdir()
    }


def test_reviewer_manifest_verification_is_read_only(tmp_path):
    out = tmp_path / "manifested"
    reviewer.build_twice(out)
    manifest = out / "artifact_manifest.json"
    before = manifest.read_bytes()
    assert reviewer.verify_manifest(out) == []
    assert manifest.read_bytes() == before


def test_frozen_prompt1_prompt2_and_prior_sources_unchanged():
    reviewer.verify_sources()


def test_no_human_answers_prepopulated(bundle):
    for data in bundle[1].values():
        assert "reviews" not in data
    config = _load(bundle[0] / "reviewer_build_config.json")
    assert config["human_judgments_prepopulated"] is False
    assert config["api_calls"] == 0 and config["new_inference"] is False
