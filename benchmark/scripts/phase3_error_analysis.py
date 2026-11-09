"""Describe explicitly selected human exports without adjudicating biological answers.

Run with --pass1 PATH --pass2 PATH. Outputs contain personal judgments and must
remain beneath ignored error_audit/review_work/. No inference or test data access.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

from benchmark.scripts import phase3_error_reviewer as reviewer

ROOT = reviewer.REPO_ROOT
AUDIT = reviewer.ERROR_AUDIT


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def csv_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare_reviews(previous, revised):
    """Compare answers by stable ID, ignoring row ordering and export metadata."""
    old = {r["audit_id"]: r for r in previous}
    new = {r["audit_id"]: r for r in revised}
    if len(old) != len(previous) or len(new) != len(revised) or old.keys() != new.keys():
        raise ValueError("revision comparison requires identical unique audit IDs")
    return [{"audit_id": aid, "field": key, "before": old[aid].get(key), "after": new[aid].get(key)}
            for aid in sorted(old) for key in sorted(old[aid].keys() | new[aid].keys())
            if old[aid].get(key) != new[aid].get(key)]


def validate(payload, bundle, phase):
    """Mirror browser completion rules; additionally check exported field types."""
    errors = []
    if not isinstance(payload, dict):
        return ["export must be an object"]
    if phase == 1:
        errors.extend(reviewer.validate_pass1_export(payload, bundle, require_complete=True))
    for field, expected in [("dataset_id", bundle["dataset_id"]),
                            ("dataset_digest", bundle["dataset_digest"]),
                            ("review_schema_version", reviewer.REVIEW_SCHEMA_VERSION),
                            ("review_kind", f"formal_pass{phase}")]:
        if payload.get(field) != expected:
            errors.append(f"{field} mismatch")
    core = {k: v for k, v in payload.items() if k not in {"canonical_review_digest", "export_timestamp"}}
    if reviewer._digest(core) != payload.get("canonical_review_digest"):
        errors.append("canonical review digest mismatch")
    rows = payload.get("reviews")
    if not isinstance(rows, list) or any(not isinstance(r, dict) for r in rows):
        return errors + ["reviews must be an array of objects"]
    expected = [c["audit_id"] for c in bundle["cases"]]
    ids = [r.get("audit_id") for r in rows]
    if len(rows) != 60 or len(set(ids)) != 60 or set(ids) != set(expected):
        errors.append("expected exactly 60 unique formal audit IDs")
    schema = reviewer.REVIEW_SCHEMA[f"pass{phase}"]
    bools = {"complete", "needs_second_review" if phase == 1 else "needs_adjudication"}
    for row in rows:
        aid = row.get("audit_id")
        # Optional schema fields may be blank, but exports include all fields.
        if set(row) != set(schema["fields"]):
            errors.append(f"{aid}: missing or unknown exported fields")
        for field in schema["fields"]:
            value = row.get(field)
            typ = bool if field in bools else list if field == "failure_taxonomy" else str
            if not isinstance(value, typ):
                errors.append(f"{aid}: invalid type for {field}")
        enums = {"confidence": schema["confidence"]}
        if phase == 1:
            enums.update(verdict=schema["verdicts"], kegg_mappable=schema["kegg_mappable"])
        else:
            enums.update(prediction_assessment=schema["prediction_assessments"],
                         exact_matching_too_strict=schema["yes_no_uncertain"])
            taxonomy = row.get("failure_taxonomy")
            if not isinstance(taxonomy, list) or any(v not in schema["failure_taxonomy"] for v in taxonomy):
                errors.append(f"{aid}: invalid failure taxonomy")
        for field, values in enums.items():
            if row.get(field) not in ["", *values]:
                errors.append(f"{aid}: invalid {field}")
        for field in schema["required_to_complete"]:
            if not isinstance(row.get(field), str) or not row[field].strip():
                errors.append(f"{aid}: missing required {field}")
        if row.get("complete") is not True:
            errors.append(f"{aid}: incomplete")
    if payload.get("completion_count") != 60 or payload.get("complete") is not True:
        errors.append("completion markers mismatch")
    return sorted(set(errors))


def aggregate(rows):
    """Exclusive verdict counts and separate reviewer multi-label selections."""
    result = {"n": len(rows)}
    for field in ("verdict", "prediction_assessment", "kegg_mappable", "exact_matching_too_strict"):
        result[field] = dict(sorted(Counter(r[field] or "[blank]" for r in rows).items()))
    result["failure_taxonomy_multilabel"] = dict(sorted(Counter(
        value for r in rows for value in set(r["failure_taxonomy"])).items()))
    result["taxonomy_blank_cases"] = sum(not r["failure_taxonomy"] for r in rows)
    result["needs_second_review"] = sum(r["needs_second_review"] for r in rows)
    result["needs_adjudication"] = sum(r["needs_adjudication"] for r in rows)
    result["frozen_abstention_by_assessment"] = dict(sorted(Counter(
        r.get("phase3c_abstain", "unknown") + " -> " + r["prediction_assessment"] for r in rows).items()))
    result["pass1_to_pass2_pairs"] = dict(sorted(Counter(
        r["verdict"] + " -> " + r["prediction_assessment"] for r in rows).items()))
    return result


def reasons(row):
    result = []
    if row.get("phase3c_abstain") == "true" and row["prediction_assessment"].startswith("prediction_"):
        result.append("clarify assessment target: frozen grounded output abstained but recorded assessment evaluates a prediction")
    if row["verdict"] == "unresolved" or row["prediction_assessment"] == "unresolved" or "unresolved" in row["failure_taxonomy"]:
        result.append("unresolved biological judgment")
    if row["verdict"] in {"label_incorrect", "label_correct_but_incomplete", "multiple_defensible_annotations", "not_kegg_mappable"} or set(row["failure_taxonomy"]) & {"source_label_error", "source_label_incomplete"}:
        result.append("potential source-label issue; documented finding, reference labels unchanged")
    if row["verdict"] in {"insufficient_model_information", "catalog_snapshot_issue"} or row["prediction_assessment"] == "insufficient_information" or set(row["failure_taxonomy"]) & {"catalog_coverage_failure", "query_information_failure"}:
        result.append("catalog or model evidence concern recorded")
    if row["needs_second_review"] or row["needs_adjudication"]:
        result.append("explicit independent-review/adjudication request")
    if row["prediction_assessment"] == "prediction_defensible_alternative" or row["exact_matching_too_strict"] == "yes":
        result.append("alternative/equivalence finding; independent confirmation deferred")
    return result


def write_csv(path, rows):
    if not rows:
        path.write_text("audit_id,follow_up_reason\n", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                         for k, v in row.items()} for row in rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pass1", type=Path, required=True)
    parser.add_argument("--pass2", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="new ignored output directory; existing outputs are never overwritten")
    parser.add_argument("--prior-analysis", type=Path, help="previous analysis directory for exact answer comparison")
    parser.add_argument("--scope-note", type=Path, required=True, help="explicit single-reviewer scope decision")
    args = parser.parse_args(argv)
    out = args.out.resolve()
    out.relative_to((AUDIT / "review_work").resolve())
    if subprocess.run(["git", "check-ignore", "--quiet", str(out)], cwd=ROOT).returncode:
        raise ValueError("output directory must be ignored")
    if out.exists():
        raise ValueError("output directory already exists; choose a new directory to preserve prior results")
    scope_note = args.scope_note.resolve()
    scope_text = scope_note.read_text(encoding="utf-8")
    out.mkdir(parents=True)
    paths = [args.pass1.resolve(), args.pass2.resolve()]
    bundles = [load(reviewer.OUT / f"formal_pass{i}_data.json") for i in (1, 2)]
    payloads = [load(p) for p in paths]
    inputs = paths + [scope_note, Path(__file__).resolve(), ROOT / "goals.md", ROOT / "benchmark/PHASE3.md",
                      AUDIT / "reviewer/README.md", AUDIT / "reviewer/review_schema.json",
                      ROOT / "benchmark/phase3/phase3c_validation/REPORT.md"] + [reviewer.OUT / f"formal_pass{i}_data.json" for i in (1, 2)] + [
        AUDIT / name for name in ("formal_audit_crosswalk.csv", "all_validation_outcomes.csv", "sampling_config.json")]
    if args.prior_analysis:
        inputs.extend(sorted(p.resolve() for p in args.prior_analysis.iterdir() if p.is_file()))
    before = {str(p.relative_to(ROOT)): checksum(p) for p in inputs}
    errors = {f"pass{i}": validate(payloads[i-1], bundles[i-1], i) for i in (1, 2)}
    locked = payloads[1].get("locked_pass1_export")
    if locked != payloads[0]:
        errors["pass2"].append("embedded Pass 1 differs from original export")
    crosswalk = csv_rows(AUDIT / "formal_audit_crosswalk.csv")
    outcomes = csv_rows(AUDIT / "all_validation_outcomes.csv")
    expected = {c["audit_id"] for c in bundles[0]["cases"]}
    if len(crosswalk) != 60 or {r["audit_id"] for r in crosswalk} != expected:
        errors["join"] = ["crosswalk formal sample mismatch"]
    for bundle in bundles:
        digest_core = {k: v for k, v in bundle.items() if k != "dataset_digest"}
        if reviewer._digest(digest_core) != bundle["dataset_digest"]:
            errors.setdefault("bundle", []).append("dataset content digest mismatch")
    manifests = {}
    from benchmark.scripts import phase3_error_inventory, phase3_error_sampling, phase3c_validation, phase3c_grounded
    for name, module in [("inventory", phase3_error_inventory), ("sampling", phase3_error_sampling),
                         ("reviewer", reviewer), ("phase3c_validation", phase3c_validation), ("phase3c_smoke", phase3c_grounded)]:
        manifests[name] = module.verify_artifacts() if name == "phase3c_smoke" else module.verify_manifest()
    report = {"input_sha256_bytes": before, "errors": errors, "manifest_errors": manifests,
              "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "git_branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
              "embedded_pass1_identical": locked == payloads[0],
              "location_exception": "Revised export found in reviewer/, none in review_work at discovery. Selected pair preserved as byte-identical ignored input snapshots. Prior export path was replaced before this task; prior answers compared from preserved joined_cases.csv, not reconstructed raw bytes.",
              "reviewer_identity": "Both reviewer_id fields blank; user identifies these as own two passes. No independent reviewer export located.",
              "synthetic_check": "Exact frozen formal IDs, dataset content digests, canonical response digests checked. Human origin cannot be cryptographically proven; no fixture exports merged.",
              "export_order_matches_blinded_bundle": [[r["audit_id"] for r in p["reviews"]] == [c["audit_id"] for c in b["cases"]] for p, b in zip(payloads, bundles)]}
    report["candidate_exports"] = []
    for candidate in sorted(AUDIT.glob("formal_pass1/raw/*/*.json")):
        checkpoint = load(candidate)
        report["candidate_exports"].append({"path": str(candidate.relative_to(ROOT)),
                                             "sha256_bytes": checksum(candidate),
                                             "completion_count": checkpoint.get("completion_count"),
                                             "identical_to_selected_pass1": candidate.read_bytes() == paths[0].read_bytes()})
    # CSV companion is informational; JSON remains authoritative.
    report["csv_companions"] = {}
    for path, payload in zip(paths, payloads):
        companion = path.with_suffix(".csv")
        if not companion.exists():
            report["csv_companions"][str(companion.relative_to(ROOT))] = "missing (optional)"
            continue
        before[str(companion.relative_to(ROOT))] = checksum(companion)
        actual = csv_rows(companion)
        by_id = {r["audit_id"]: r for r in payload["reviews"]}
        mismatches = []
        for row in actual:
            original = by_id.get(row.get("audit_id"))
            if original is None:
                mismatches.append(row.get("audit_id"))
                continue
            for k, value in row.items():
                expected_value = original.get(k, payload.get(k, ""))
                text = ";".join(expected_value) if isinstance(expected_value, list) else str(expected_value).lower() if isinstance(expected_value, bool) else str(expected_value)
                if value != text:
                    mismatches.append(f"{row['audit_id']}:{k}")
        if len(actual) != 60 or len({r['audit_id'] for r in actual}) != 60:
            mismatches.append("companion row count/uniqueness")
        report["csv_companions"][str(companion.relative_to(ROOT))] = mismatches or "matches JSON"
    (out / "validation_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if any(errors.values()) or any(manifests.values()):
        print(json.dumps(report, indent=2))
        return 1
    maps = [{r["audit_id"]: r for r in p["reviews"]} for p in payloads]
    if args.prior_analysis:
        prior_rows = csv_rows(args.prior_analysis / "joined_cases.csv")
        revisions = {f"pass{phase}": compare_reviews(
            [json.loads(r[f"pass{phase}_review"]) for r in prior_rows], payloads[phase-1]["reviews"])
            for phase in (1, 2)}
        (out / "revision_changes.json").write_text(json.dumps(revisions, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        write_csv(out / "revision_changes.csv", revisions["pass1"] + revisions["pass2"])
        (out / "REVISION_REPORT.md").write_text(
            "# Exact answer changes from previous analysis\n\n"
            "Comparison uses preserved prior joined_cases.csv: original prior raw Pass 2 bytes are no longer present at the replaced source path. "
            "The previous raw-file checksum remains in the preserved validation report. All 60 answers in both stages are compared by audit ID, including optional fields; metadata is not counted as an answer change.\n\n"
            + "```json\n" + json.dumps(revisions, indent=2, ensure_ascii=False) + "\n```\n\n"
            + "Revised export metadata:\n\n```json\n" + json.dumps({k: v for k, v in payloads[1].items()
                if k not in {"reviews", "locked_pass1_export"}}, indent=2) + "\n```\n", encoding="utf-8")
    outcome_map = {r["audit_id"]: r for r in outcomes}
    if len(outcome_map) != 163 or len(outcomes) != 163:
        raise ValueError("validation outcomes must contain 163 unique audit IDs")
    joined = []
    for cross in crosswalk:
        aid = cross["audit_id"]
        frozen = outcome_map[aid]
        for k in ("model_id", "reaction_id", "sample_id"):
            if cross[k] != frozen[k]:
                raise ValueError(f"stable identifier mismatch {aid}:{k}")
            for bundle in bundles:
                case = next(c for c in bundle["cases"] if c["audit_id"] == aid)
                if case[k] != cross[k]:
                    raise ValueError(f"reviewer bundle identity mismatch {aid}:{k}")
        row = {**cross, **maps[0][aid], **maps[1][aid]}
        # Keep both rationales and confidence values without overwriting them.
        row.update(pass1_review=maps[0][aid], pass2_review=maps[1][aid], frozen_outcome=frozen)
        joined.append(row)
    summary = {"all_formal": aggregate(joined)}
    for dimension in ("primary_category", "corrected_phase2_stratum", "retrieval_state"):
        groups = defaultdict(list)
        for row in joined:
            groups[row[dimension]].append(row)
        summary[dimension] = {k: aggregate(v) for k, v in sorted(groups.items())}
    summary["optional_blank_counts"] = {
        f"pass{phase}": {k: sum(r[k] in ("", []) for r in payload["reviews"])
                         for k in reviewer.REVIEW_SCHEMA[f"pass{phase}"]["fields"]}
        for phase, payload in enumerate(payloads, 1)}
    followups = []
    for row in joined:
        why = reasons(row)
        if why:
            followups.append({"audit_id": row["audit_id"], "model_id": row["model_id"],
                              "reaction_id": row["reaction_id"], "primary_category": row["primary_category"],
                              "pass1_verdict": row["verdict"], "pass2_assessment": row["prediction_assessment"],
                              "failure_taxonomy": row["failure_taxonomy"], "needs_second_review": row["needs_second_review"],
                              "needs_adjudication": row["needs_adjudication"], "follow_up_reason": "; ".join(why),
                              "completion_blocker": False,
                              "disposition": "documented finding/limitation; independent review deferred; reference labels unchanged",
                              "pass1_review": row["pass1_review"], "pass2_review": row["pass2_review"]})
    for name, value in [("aggregate_summary.json", summary), ("input_checksums.json", before)]:
        (out / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_csv(out / "joined_cases.csv", joined)
    write_csv(out / "follow_up_cases.csv", followups)
    text = ["# Recorded human audit judgments", "", "Both exports validate: 60/60 complete in each, same frozen formal sample; embedded Pass 1 identical.",
            "Blank optional fields and unresolved verdicts satisfy completion when required assessment and rationale are present.",
            "No missing/invalid formal export fields. CSV companion status is in validation_report.json.",
            "", "Counts describe this stratified 60-case sample only. No population percentages, weighting, corrected accuracy, new labels, or biological adjudication.",
            "Selection is cluster/model balanced and enriched; frozen design supplies no inclusion-probability weights. Even random-control counts remain descriptive here.",
            "", "Pass 1 assesses source labels; Pass 2 assesses predictions. The paired cross-tab describes their relationship, not a change in the same verdict or independent agreement.",
            "Embedded Pass 1 is unchanged. No revised source-verdict field exists; free text is preserved without inferring revised labels.",
            "Both reviewer IDs are blank; attribution rests on the user's statement. No independent review or adjudication has occurred in available exports.",
            "", "## Categories", "", json.dumps(reviewer.PASS1_VERDICT_DESCRIPTIONS, indent=2),
            "", "Pass 2 prediction_correct and prediction_defensible_alternative record acceptance; related_but_incorrect and incorrect record rejection; abstention_appropriate/inappropriate evaluate abstention; insufficient_information/unresolved retain uncertainty.",
            "failure_taxonomy is optional and multi-label: source label error/incompleteness, catalog coverage, query information, retrieval, ranking, LLM selection/overabstention, appropriate abstention, ontology equivalence, nonchemical event, no system failure, unresolved. Counts may overlap and do not sum to n; blank means no category recorded.",
            "", "## Counts by sampling stratum", ""]
    for name, group in [("All formal (descriptive)", summary["all_formal"]), *summary["primary_category"].items()]:
        text.extend([f"### {name} (n={group['n']})", "", "```json", json.dumps(group, indent=2), "```", ""])
    target_clarifications = [r["audit_id"] for r in joined if r["phase3c_abstain"] == "true" and r["prediction_assessment"].startswith("prediction_")]
    text.extend(["Additional counts by corrected Phase 2 stratum and retrieval state are in aggregate_summary.json.",
                 "Remaining assessment-target clarifications (frozen grounded output abstained but a prediction was assessed): " + (", ".join(target_clarifications) or "none") + ".",
                 "Unresolved categories and second-review/adjudication flags are tabulated above. Insufficient-information assessments remain evidence limitations even when complete. Blank optional fields do not establish that concerns are absent.",
                 "Recorded method assessments, proposed/defensible IDs, rationales, and all frozen outcomes remain in joined_cases.csv. Blank method fields cannot support method-specific biological conclusions.",
                 f"Follow-up triggers select {len(followups)}/60 cases: " + ", ".join(r["audit_id"] for r in followups) + ".",
                 "Flags document findings and limitations; they do not require another review to complete this scoped audit. Independent review and inter-reviewer adjudication are deferred by the explicit scope decision in AUDIT_SCOPE.md.", ""])
    (out / "AUDIT_SUMMARY.md").write_text("\n".join(text), encoding="utf-8")
    (out / "PHASE3C_CHECKLIST.md").write_text(CHECKLIST, encoding="utf-8")
    (out / "AUDIT_SCOPE.md").write_text(scope_text, encoding="utf-8")
    command = "python -m benchmark.scripts.phase3_error_analysis --pass1 " + str(paths[0].relative_to(ROOT)).replace("\\", "/") + " --pass2 " + str(paths[1].relative_to(ROOT)).replace("\\", "/")
    command += " --scope-note " + str(scope_note.relative_to(ROOT)).replace("\\", "/")
    if args.prior_analysis:
        command += " --prior-analysis " + args.prior_analysis.as_posix()
    command += " --out " + out.relative_to(ROOT).as_posix() + "_reproduced"
    (out / "REPRODUCE.md").write_text(
        "# Local reproduction\n\nRun from repository root (no network):\n\n```powershell\n" + command +
        "\n```\n\nScript: benchmark/scripts/phase3_error_analysis.py. Requires the existing benchmark environment (including python-libsbml). "
        "Explicit paths select one session pair; no discovery or merging. Original files are never written. "
        "Byte hashes differ intentionally from newline-normalized manifest hashes. "
        "Analysis date label is the user's 2026-11-05 local date; the Pass 2 export's UTC timestamp is preserved.\n", encoding="utf-8")
    for relative, digest in before.items():
        if checksum(ROOT / relative) != digest:
            raise ValueError(f"input changed during analysis: {relative}")
    print(json.dumps({"out": str(out), "validated": True, "followups": len(followups), "counts": summary["all_formal"]}, indent=2))
    return 0


CHECKLIST = """# Phase 3C completion checklist

Assessment: the implemented grounded-fusion paired validation pilot and scoped single-reviewer audit/analysis are complete. No remaining required technical work was identified within that scope. Independent review and inter-reviewer adjudication are explicitly deferred under AUDIT_SCOPE.md and do not block completion. No corrected accuracy is calculated. This does not certify fulfillment of every experiment in the older roadmap.

Paths below are repository-relative. Complete experimental items are supported by manifests checked read-only in validation_report.json. Older narrative documents have stale future-tense statements; available frozen artifacts establish later implementation.

| Requirement / scope | Status | Evidence and remaining action |
| --- | --- | --- |
| Current checkout preserved | Complete | benchmark/error-audit, HEAD recorded in validation_report.json; existing writing/ and exports preserved; no switch/commit/push |
| Applicable AGENTS.md | Unverified | No AGENTS.md found by repository/ancestor inspection; inaccessible .pytest_cache excluded from instruction search |
| Phase 2 heuristic and lexical baselines | Complete | goals.md Phase 2; benchmark/PHASE2_MANIFEST.json; Phase 3 retrieval_baselines/REPORT.md |
| Off-the-shelf embeddings, BM25, RRF | Complete | benchmark/phase3/retrieval_baselines/REPORT.md: 969 validation reactions; these are full-catalog retrieval comparisons, not all same-set rerankers |
| Trained bi-encoder and selected fusion | Complete | benchmark/phase3/phase3b_full/REPORT.md; phase3b_fusion/REPORT.md: frozen epoch-1 model and equal-weight RRF |
| Original roadmap's trained cross-encoder | Optional extension, not implemented | benchmark/PHASE3.md Optional cross-encoder makes it optional/secondary; no completed experiment located. goals.md still lists the broader comparison; roadmap reconciliation remains a separate scope decision, not a technical blocker for this pilot |
| All proposed rerankers compared on identical retrievable candidate sets | Not established; broader scope decision | goals.md Phase 3C proposes this; implemented artifacts compare full-catalog retrieval and grounded fusion Top-10 selection. The original full matrix is not declared fulfilled or silently superseded |
| Grounded smoke, prompt/evidence/cache/budget checks | Complete | benchmark/phase3/phase3c_smoke/REPORT.md and artifact_manifest.json |
| Paired 163-reaction grounded validation | Complete | benchmark/phase3/phase3c_validation/{REPORT.md,paired_pilot_plan.json,config.json,artifact_manifest.json}; 102/163 grounded exact vs 93/163 fusion and 49/163 direct target-only |
| Retrieval-state, harm/recovery, abstention, evidence compliance, cost, uncertainty | Complete | same directory: retrieval_state_analysis.json, paired_transitions.json, bootstrap_comparisons.json, cost_usage_report.json; 39 abstentions, 22 in-evidence incorrect selections, 2 compliance cases; confidence descriptive only |
| All-969 validation extension | Explicitly deferred | phase3c_validation/REPORT.md decision 6: not scientifically necessary without a separate prespecified question |
| Error inventories, frozen 60-case sample, blinded packets | Complete | error_audit/{artifact_manifest.json,SAMPLING_REPORT.md,sampling_config.json,formal_audit_crosswalk.csv}; reviewer/artifact_manifest.json |
| Formal Pass 1 and Pass 2 by current reviewer | Complete | explicit exports, byte hashes and schema checks in validation_report.json; 60 complete each; reviewer IDs blank, attribution supplied by user |
| Export preservation/location | Complete | explicit input pair preserved byte-for-byte under ignored review_work/inputs_revision_20261105/; original reviewer/ files untouched. Prior outputs and their original hashes retained |
| Descriptive audit analysis and case queue | Complete | AUDIT_SUMMARY.md, aggregate_summary.json, joined_cases.csv, follow_up_cases.csv, input_checksums.json; sample-stratum denominators included; no population estimates |
| Independent second human reviewer | Explicitly deferred; not required | user scope decision recorded in AUDIT_SCOPE.md supersedes reviewer/README.md step 8 for this report; historical protocol unchanged |
| Inter-reviewer adjudication/agreement | Explicitly deferred; unavailable | one reviewer, two dependent stages; no agreement calculation or independent adjudication; labels unchanged |
| Biological flags and uncertainty | Complete as documented findings/limitations | follow_up_cases.csv retains case-level concerns without making every flag a completion requirement; reference-label changes and sensitivity analyses require separately defined work |
| Optional browse-all 61 and supplemental training E12 | Pending (optional extension) | reviewer/README.md separates these from the 60 formal validation cases; neither is required to validate the two exports, neither included here; no completion claimed |
| Final method freeze and later held-out test | Pending | phase3c_validation/REPORT.md decision 7 recommends freeze before one held-out test; this task does not inspect test data or establish completion of that future evaluation |
| Routed hybrid system / roadmap Phase 4 | Explicitly deferred | goals.md marks stretch; benchmark/PHASE3.md says routed system not implemented; post-hoc routing unvalidated per Phase 3C report. Error-audit Prompt 4 is separate |

Audit scope decision: explicitly single-reviewer, per AUDIT_SCOPE.md. Experimental scope assessed here is the implemented grounded-fusion validation pilot. Separately reconcile goals.md's broader same-set reranker matrix with benchmark/PHASE3.md's optional cross-encoder; this report neither runs those experiments nor declares that broader matrix complete. Goals and manuscript files remain unchanged.

Recommended next project task: prepare and record the final method/evaluation freeze for the later one-time held-out test, explicitly specifying grounded_llm_on_every_reaction, retriever/checkpoint, prompts, evidence limits, abstention handling, metrics, and unchanged frozen reference labels. This follows phase3c_validation/REPORT.md decision 7 and goals.md's requirement that method choices be frozen before held-out evaluation. It is a planning/freeze milestone, not authorization to inspect or run the test now. Do not begin Phase 4 routing.
"""


if __name__ == "__main__":
    raise SystemExit(main())
