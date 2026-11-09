"""Read-only compatibility/parity checks for explicitly supplied, permitted artifacts.

Never reads benchmark split/label files, runs retrieval inference, or contacts a
provider. Frozen component rankings may only be reused after exact query checks.
Run from the repository root with python -m scripts.verify_rrf_grounded.
"""
from __future__ import annotations

import argparse
import json
import lzma
import pickle
from collections import defaultdict
from pathlib import Path

from core.reaction.grounded import canonical, digest, provider_payload, VISIBLE_FIELDS
from core.reaction.retrieval import CATALOG_SHA256, BM25, document_text, evidence_for, sha256
from core.reaction.species_evidence import build_queries


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def differences(before, after, path=""):
    """Every changed provider field, including numerical differences and ordering."""
    if type(before) is not type(after):
        return [{"field": path, "before": before, "after": after}]
    if isinstance(before, dict):
        rows = []
        for key in sorted(set(before) | set(after)):
            if key not in before or key not in after:
                rows.append({"field": path + "/" + key, "before": before.get(key), "after": after.get(key)})
            else:
                rows.extend(differences(before[key], after[key], path + "/" + key))
        return rows
    if isinstance(before, list):
        if len(before) != len(after):
            return [{"field": path, "before": before, "after": after}]
        return [item for i, (a, b) in enumerate(zip(before, after)) for item in differences(a, b, path + "/" + str(i))]
    return [] if before == after else [{"field": path, "before": before, "after": after}]


def compare_prepared_plan(model, species, source, plan, catalog_path, output):
    """Write a separate private report. Existing artifacts are checksum-guarded."""
    source, plan, output = Path(source), Path(plan), Path(output)
    if output.exists() or source.resolve() in output.resolve().parents or plan.resolve() in output.resolve().parents:
        raise ValueError("Use a new output directory outside preserved inputs")
    protected = {str(p.resolve()): sha256(p) for root in [source, plan] for p in root.rglob("*") if p.is_file()}
    protected.update({str(Path(p).resolve()): sha256(p) for p in [model, species, catalog_path]})
    if sha256(catalog_path) != CATALOG_SHA256:
        raise ValueError("catalog mismatch")
    with lzma.open(catalog_path, "rb") as handle:
        catalog = pickle.load(handle)
    old_queries = {q["reaction_id"]: q["query"] for q in jsonl(source / "queries.jsonl")}
    queries = build_queries(model, species)
    new_queries = {q["reaction_id"]: q["query"] for q in queries}
    if set(old_queries) != set(new_queries):
        raise ValueError("reaction coverage differs")
    changed = [rid for rid in new_queries if new_queries[rid] != old_queries[rid]]
    if changed:
        raise ValueError("Query semantics changed; frozen rankings cannot be reused. Retrieve anew into a separately versioned candidate plan: " + str(changed))
    components = source / "reused_source_artifacts"
    bm = {r["reaction_id"]: r["ranked_ids"] for r in jsonl(components / "phase3b_bm25_rankings_top100.jsonl")}
    dense = {r["reaction_id"]: r["ranked_ids"] for r in jsonl(components / "phase3b_trained_epoch1_rankings_top100.jsonl")}
    if any(set(rows) != set(new_queries) for rows in (bm, dense)):
        raise ValueError("component coverage mismatch")
    old_evidence = defaultdict(list)
    for row in jsonl(source / "top10_evidence.jsonl"):
        old_evidence[row["reaction_id"]].append({key: row[key] for key in VISIBLE_FIELDS})
    if set(old_evidence) != set(new_queries):
        raise ValueError("evidence coverage mismatch")
    requests = jsonl(plan / "requests.jsonl")
    if len(requests) != len(new_queries) or {r["unit"]["reaction_id"] for r in requests} != set(new_queries):
        raise ValueError("request coverage mismatch")
    report_rows, rebuilt = [], []
    index = BM25([{"kegg_id": kid, "text": document_text(kid, catalog[kid])} for kid in sorted(catalog)])
    bm_changed = []
    for row in requests:
        rid = row["unit"]["reaction_id"]
        if len(bm[rid]) != 100 or len(dense[rid]) != 100:
            raise ValueError("component depth mismatch")
        if index.rank(new_queries[rid], 100) != bm[rid]:
            bm_changed.append(rid)
        evidence = evidence_for(catalog, bm[rid], dense[rid])
        previous_evidence = sorted(old_evidence[rid], key=lambda e: e["fused_rank"])
        evidence_diff = differences(previous_evidence, evidence)
        before = row["provider_payload"]
        body = provider_payload(new_queries[rid], evidence, model=before["model"],
            reasoning_effort=before["reasoning"]["effort"], max_output_tokens=before["max_output_tokens"])
        body_diff = differences(before, body)
        report_rows.append({"reaction_id": rid, "evidence_differences": evidence_diff,
            "provider_differences": body_diff, "before_payload_sha256": digest(before), "after_payload_sha256": digest(body)})
        rebuilt.append({"unit": row["unit"], "provider_payload": body,
                        "request_sha256": digest(body), "cache_key": digest({"schema": "rrf-grounded-request-v1", "provider_payload": body})})
    after = {p: sha256(p) for p in protected}
    if after != protected:
        raise ValueError("protected artifacts changed during verification")
    summary = {"query_count": len(queries), "changed_queries": len(changed),
        "bm25_top100_exact": not bm_changed, "bm25_changed_reactions": bm_changed,
        "changed_evidence_records": sum(bool(r["evidence_differences"]) for r in report_rows),
        "changed_complete_provider_payloads": sum(bool(r["provider_differences"]) for r in report_rows),
        "provider_calls": 0, "encoder_inference_calls": 0, "protected_inputs_unchanged": after == protected,
        "serialization_plan_digest": digest(rebuilt), "old_plan_usable_unchanged": not bm_changed and not any(r["evidence_differences"] or r["provider_differences"] for r in report_rows),
        "limitations": ["Dense ranking parity uses frozen components only after exact query equality; full dense inference not rerun",
            "CPU/GPU numerical differences may reorder near ties; compare actual complete requests before response reuse",
            "Identical requests do not guarantee identical hosted-model behavior across time/provider versions",
            "No inference or spending authorization is implied"]}
    output.mkdir(parents=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "input_checksums.json").write_text(json.dumps(protected, indent=2) + "\n")
    (output / "field_differences.jsonl").write_bytes(b"".join(canonical(row) + b"\n" for row in report_rows))
    # Diagnostic reconstruction, not an executable purchase plan or cache migration.
    (output / "reconstructed_requests.jsonl").write_bytes(b"".join(canonical(row) + b"\n" for row in rebuilt))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["model", "species", "source", "plan", "catalog", "out"]:
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(compare_prepared_plan(args.model, args.species, args.source, args.plan, args.catalog, args.out), indent=2))


if __name__ == "__main__":
    main()
