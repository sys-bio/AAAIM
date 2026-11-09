"""Bounded local robustness experiment over frozen validation-only queries.

Stages: prepare (SBML roles, perturbations, strict rules, BM25), dense (offline
selected encoder and RRF), analyze (only then join validation references).
"""
from __future__ import annotations

import argparse
import copy
import csv
import gzip
import json
import lzma
import os
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from benchmark.scripts import phase3_retrieval as retrieval
from benchmark.scripts.phase3_common import redact_kegg_reaction_ids, sha256_portable
from benchmark.scripts.phase5_perturb import Ontology, annotation_links, conditions, digest, parse_query, perturb, render

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "benchmark/phase5/study_config.json"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def write_rows(path, values):
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in values:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n")


def csv_write(path, values):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)


def pickle_load(path):
    with lzma.open(path, "rb") as handle:
        return pickle.load(handle)


def assets(source):
    catalog = pickle_load(source / "data/kegg/kegg_reaction_features.lzma")
    if sha256_portable(source / "data/kegg/kegg_reaction_features.lzma") != load(CONFIG)["catalog_sha256"] or len(catalog) != 12312:
        raise ValueError("frozen catalog mismatch")
    return catalog, [{"kegg_id": key, "text": retrieval.document_text(key, catalog[key])} for key in sorted(catalog)]


def parent_map(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        raw = json.load(handle)
    result = {}
    for child, edges in raw.items():
        values = []
        for edge in edges:
            if isinstance(edge, str):
                values.append(edge)
            elif isinstance(edge, list) and len(edge) == 2 and edge[0] == "is_a":
                values.append(edge[1])
        result[child] = sorted(set(values))
    return result


def input_records(source):
    """Use only frozen validation query records and specific validation SBML models."""
    import libsbml
    samples = read_rows(source / "benchmark/phase3/phase3c_validation/sample_ids.jsonl")
    if len(samples) != 163 or len({r["sample_id"] for r in samples}) != 163 or any(r["split"] != "validation" for r in samples):
        raise ValueError("exact validation population required")
    queries = read_rows(source / "benchmark/phase3/retrieval_baselines/queries.jsonl")
    if any(q["split"] != "validation" for q in queries):
        raise ValueError("non-validation query source")
    lookup = {(q["model_id"], q["reaction_id"]): q["query"] for q in queries}
    registry = {r["model_id"]: r for r in load(source / "benchmark/manifest/model_registry.json")["models"]}
    records, sbml_hashes, models = [], {}, {}
    for sample in samples:
        mid, rid = sample["model_id"], sample["reaction_id"]
        if mid not in models:
            path = source / registry[mid]["local_path"]
            if sha256_portable(path) != registry[mid]["local_sha256"]:
                # SBML bytes are normally pinned raw; portable hash equals raw for .xml.
                raise ValueError("validation SBML digest mismatch")
            document = libsbml.readSBMLFromFile(str(path))
            models[mid] = document
            sbml_hashes[registry[mid]["local_path"]] = registry[mid]["local_sha256"]
        reaction = models[mid].getModel().getReaction(rid)
        if reaction is None:
            raise ValueError("validation reaction absent")
        record = parse_query(lookup[mid, rid])
        record.update(sample_id=sample["sample_id"], model_id=mid, reaction_id=rid, reversible=bool(reaction.getReversible()))
        for side, getter, count in [("reactants", reaction.getReactant, reaction.getNumReactants()), ("products", reaction.getProduct, reaction.getNumProducts())]:
            record[side] = {redact_kegg_reaction_ids(getter(i).getSpecies()): float(getter(i).getStoichiometry()) for i in range(count)}
        participant_ids = {p["species_id"] for p in record["participants"]}
        if not participant_ids.issubset(set(record["reactants"]) | set(record["products"])):
            raise ValueError("query participant/SBML side alignment mismatch")
        if render(record) != lookup[mid, rid]:
            raise ValueError("clean query round trip mismatch")
        records.append(record)
    return records, sbml_hashes


class StrictRules:
    """Existing strict filter/scorer; all participant mappings rebuilt from identifiers."""
    def __init__(self, source, mapping):
        from core import database_search as db
        from core.reaction.amendment_config import CofactorConfig
        self.db, self.mapping, self.cofactors = db, mapping, CofactorConfig().kegg_ids
        db.get_data_dir = lambda: source / "data"
        db._KEGG_REACTION_FEATURES_DICT = pickle_load(source / "data/kegg/kegg_reaction_features.lzma")
        db._KEGG_PARSED_REACTIONS_DICT = pickle_load(source / "data/kegg/parsed_kegg_reactions.lzma")
        self.cache = {}

    def rank(self, record):
        import pandas as pd
        from core.reaction.matching import map_reactions_to_kegg
        key = digest({k: record[k] for k in ("participants", "reactants", "products")})
        if key in self.cache:
            return self.cache[key], True
        compound_rows = []
        for p in record["participants"]:
            compounds = set(p["kegg"])
            for term in p["chebi"]:
                compounds.update(self.mapping.get(term, []))
            compound_rows.extend({"id": p["species_id"], "KEGG_ID": c, "match_score": 1.0} for c in sorted(compounds))
        if not compound_rows:
            self.cache[key] = []
            return [], False
        equation = " => ".join(" + ".join(f"{coeff:g} {sid}" for sid, coeff in record[side].items()) for side in ("reactants", "products"))
        normalized = map_reactions_to_kegg([equation], ["P5LOCAL"], pd.DataFrame(compound_rows), spectators=False)
        matched = self.db._get_kegg_recommendations_rulebased(normalized, cofactors_to_ignore=self.cofactors,
            spectators=False, evaluate_candidates=True, include_exchange_reactions=False, strict_errors=True)
        scores = {}
        for rec in matched:
            if not (rec.metadata or {}).get("filtered_species_count", 0):
                continue
            for candidate, score in zip(rec.candidates, rec.match_score):
                scores[candidate] = max(scores.get(candidate, -1), float(score))
        ranked = sorted(scores, key=lambda c: (-scores[c], c))
        self.cache[key] = ranked
        return ranked, False


def prepare(source, out):
    if out.exists():
        raise ValueError("choose a new output directory")
    config = load(CONFIG)
    catalog, docs = assets(source)
    mapping = pickle_load(source / "data/kegg/chebi_to_kegg_map.lzma")
    mapping = {k: [v] if isinstance(v, str) else v for k, v in mapping.items()}
    ontology = Ontology(parent_map(source / "data/chebi/chebi_parent_map.json.gz"), mapping)
    records, sbml_hashes = input_records(source)
    variants, accounting = [], []
    for condition, kind, level, seed in conditions(config):
        for record in records:
            changed, stats = perturb(record, kind, level, seed, ontology)
            variants.append({**changed, "condition": condition, "kind": kind, "level": level, "seed": seed,
                             "query": render(changed), "query_digest": digest(render(changed))})
            accounting.append({"condition": condition, "sample_id": record["sample_id"], **stats})
    if len(variants) != config["max_query_conditions"]:
        raise ValueError("prespecified query count mismatch")
    out.mkdir(parents=True)
    write_rows(out / "derived_inputs.jsonl", variants)
    write_rows(out / "perturbation_manifest.jsonl", accounting)
    write(out / "sbml_source_checksums.json", sbml_hashes)
    source_files = ["data/kegg/kegg_reaction_features.lzma", "data/kegg/parsed_kegg_reactions.lzma",
                    "data/kegg/chebi_to_kegg_map.lzma", "data/chebi/chebi_parent_map.json.gz",
                    "benchmark/phase3/retrieval_baselines/queries.jsonl", "benchmark/phase3/phase3c_validation/sample_ids.jsonl",
                    "benchmark/phase3/phase3b_full/_checkpoints/best.pt"]
    write(out / "source_checksums.json", {p: sha256_portable(source / p) for p in source_files})
    bm25, strict = retrieval.BM25(docs), StrictRules(source, mapping)
    start = time.monotonic()
    ranked_rows, query_cache = [], {}
    for i, variant in enumerate(variants):
        if time.monotonic()-start > config["cpu_time_limit_seconds"]:
            raise RuntimeError("CPU preprocessing retrieval resource limit reached")
        t = time.perf_counter()
        qkey = variant["query_digest"]
        hit = qkey in query_cache
        lexical = query_cache[qkey] if hit else bm25.rank(variant["query"], topn=100)
        query_cache[qkey] = lexical
        lexical_seconds = time.perf_counter()-t
        t = time.perf_counter()
        rules, rule_hit = strict.rank(variant)
        ranked_rows.append({"condition": variant["condition"], "sample_id": variant["sample_id"], "query_digest": qkey,
                            "bm25": lexical, "rule_strict": rules, "bm25_seconds": lexical_seconds,
                            "rule_seconds": time.perf_counter()-t, "bm25_exact_query_cache_hit": hit,
                            "rule_exact_mapping_cache_hit": rule_hit})
        if i % 163 == 0:
            print(f"CPU {i}/{len(variants)} {variant['condition']}", flush=True)
    write_rows(out / "local_rankings.jsonl", ranked_rows)
    write(out / "local_runtime.json", {"seconds": time.monotonic()-start, "unique_text_queries": len(query_cache),
                                      "unique_rule_mapping_inputs": len(strict.cache), "query_conditions": len(variants)})
    print("CPU retrieval complete", flush=True)


def dense(source, out):
    """Offline fixed-weight query inference only; no dataset loader or training."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import torch
    from transformers import AutoModel, AutoTokenizer
    from benchmark.scripts.phase3_biencoder import _encode_texts
    config = load(CONFIG)
    start = time.monotonic()
    checkpoint = source / "benchmark/phase3/phase3b_full/_checkpoints/best.pt"
    if sha256_portable(checkpoint) != config["trained_checkpoint_sha256"]:
        raise ValueError("selected checkpoint mismatch")
    snapshot = source / "benchmark/phase3/phase3b_smoke/_model_cache/models--BAAI--bge-small-en-v1.5/snapshots/5e62ea33e012fda8c02802b906664c915ebd1bb1"
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
    model = AutoModel.from_pretrained(str(snapshot), local_files_only=True)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if state["config_hash"] != "0b808b72948311a44ebfd5869168751297df47a6afcfd1484a3b6907c9cabe88":
        raise ValueError("checkpoint configuration mismatch")
    model.load_state_dict(state["model_state"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("bounded encoder run requires the available local GPU; do not silently launch large CPU inference")
    model.to(device).eval()
    _, docs = assets(source)
    t = time.monotonic()
    document_vectors = _encode_texts(model, tokenizer, [d["text"] for d in docs], device, max_length=256, batch_size=64)
    document_seconds = time.monotonic()-t
    variants = read_rows(out / "derived_inputs.jsonl")
    texts = list(dict.fromkeys(v["query"] for v in variants))
    t = time.monotonic()
    query_vectors = _encode_texts(model, tokenizer, texts, device, max_length=256, batch_size=64)
    query_seconds = time.monotonic()-t
    if time.monotonic()-start > config["dense_time_limit_seconds"]:
        raise RuntimeError("encoder time cap reached")
    catalog_ids = [d["kegg_id"] for d in docs]
    # Inner products on the same fixed normalized vectors; no labels involved.
    t = time.monotonic()
    ranks = {}
    gpu_docs = torch.tensor(document_vectors, device=device)
    for i in range(0, len(texts), 64):
        scores = (torch.tensor(query_vectors[i:i+64], device=device) @ gpu_docs.T).cpu().numpy()
        for text, score in zip(texts[i:i+64], scores):
            order = np.argsort(-score, kind="stable")[:100]
            ranks[digest(text)] = [catalog_ids[j] for j in order]
        if time.monotonic()-start > config["dense_time_limit_seconds"]:
            raise RuntimeError("dense ranking time cap reached")
    local = {(r["condition"], r["sample_id"]): r for r in read_rows(out / "local_rankings.jsonl")}
    predictions = []
    for v in variants:
        lexical = local[v["condition"], v["sample_id"]]
        trained = ranks[v["query_digest"]]
        predictions.append({**lexical, "trained": trained, "fusion": retrieval.rrf([lexical["bm25"], trained], k=60, topn=100)})
    write_rows(out / "frozen_rankings.jsonl", predictions)
    write(out / "prediction_freeze.json", {"sha256": sha256_portable(out / "frozen_rankings.jsonl"),
          "labels_loaded_during_ranking": False, "test_rows_read": 0, "catalog_size": len(docs),
          "unique_query_encodings": len(texts), "snapshot_files": {p.name: sha256_portable(p) for p in snapshot.iterdir() if p.is_file()},
          "checkpoint_sha256": config["trained_checkpoint_sha256"], "max_length": 256,
          "pooling": "cls", "normalization": "l2", "device": str(device), "torch": torch.__version__,
          "document_encoding_seconds": document_seconds, "query_encoding_seconds": query_seconds,
          "ranking_and_fusion_seconds": time.monotonic()-t, "total_seconds": time.monotonic()-start,
          "score_arithmetic": "float32 inner products on CUDA after mixed-precision encoding; small historical numerical drift is checked and reported"})
    print(f"Dense retrieval complete: {len(predictions)} aligned rows, {len(texts)} unique texts", flush=True)


def retrieval_metrics(ranking, truths, catalog_ids, full_catalog):
    truth = set(truths)
    first = next((i for i, value in enumerate(ranking, 1) if value in truth), None)
    result = {f"recall_at_{k}": int(first is not None and first <= k) for k in (1, 3, 5, 10, 100)}
    result.update(mrr_at_10=1/first if first and first <= 10 else 0.0,
                  mrr_at_100=1/first if first and first <= 100 else 0.0,
                  nonempty=int(bool(ranking)), empty=int(not ranking),
                  candidate_count=len(catalog_ids) if full_catalog else len(ranking),
                  reference_absent_top10=int(not truth.intersection(ranking[:10])),
                  recall_any_candidate=int(bool(truth.intersection(catalog_ids if full_catalog else ranking))))
    return result


def validate_rank_input_alignment(ranking, variant):
    if ranking["query_digest"] != variant["query_digest"] or digest(variant["query"]) != variant["query_digest"]:
        raise ValueError("ranking was not recomputed for the actual derived query")


def paired_interval(clean, perturbed, clusters, seed, replicates):
    """Cluster resampling plus replicate resampling, retaining paired denominators."""
    # perturbed shape: [perturbation replicate, query]; clean shape: [query].
    array = np.asarray(perturbed, dtype=float)-np.asarray(clean, dtype=float)[None, :]
    labels = sorted(set(clusters))
    indices = [np.flatnonzero(np.array(clusters) == c) for c in labels]
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(replicates):
        selected_clusters = rng.integers(0, len(labels), len(labels))
        selected_queries = np.concatenate([indices[c] for c in selected_clusters])
        selected_replicates = rng.integers(0, len(array), len(array))
        values.append(float(array[selected_replicates][:, selected_queries].mean()))
    return np.quantile(values, [.025, .975]).tolist()


def analyze(source, out):
    config = load(CONFIG)
    gate = load(out / "prediction_freeze.json")
    if sha256_portable(out / "frozen_rankings.jsonl") != gate["sha256"] or gate["labels_loaded_during_ranking"]:
        raise ValueError("ranking freeze invalid")
    rankings = read_rows(out / "frozen_rankings.jsonl")
    variants = read_rows(out / "derived_inputs.jsonl")
    manifest = read_rows(out / "perturbation_manifest.jsonl")
    truths = {r["sample_id"]: r for r in read_rows(source / "benchmark/phase3/phase3c_validation/answer_key_join.jsonl")}
    if len(truths) != 163 or len(rankings) != config["max_query_conditions"]:
        raise ValueError("paired validation count mismatch")
    _, docs = assets(source)
    catalog_ids = {d["kegg_id"] for d in docs}
    evaluated, by_method_condition = [], defaultdict(list)
    variant_lookup = {(v["condition"], v["sample_id"]): v for v in variants}
    for row in rankings:
        sid, condition = row["sample_id"], row["condition"]
        t, v = truths[sid], variant_lookup[condition, sid]
        validate_rank_input_alignment(row, v)
        if (v["model_id"], v["reaction_id"]) != (t["model_id"], t["reaction_id"]):
            raise ValueError("stable validation identity mismatch")
        for method in ("rule_strict", "bm25", "trained", "fusion"):
            value = {"condition": condition, "sample_id": sid, "method": method,
                     "cluster_id": t["cluster_id"], "stratum": t["corrected_stratum"],
                     **retrieval_metrics(row[method], t["ground_truth_ids"], catalog_ids, method != "rule_strict")}
            evaluated.append(value)
            by_method_condition[method, condition].append(value)
    numerical = list(retrieval_metrics([], ["SYN"], set(), False))
    summary, subgroups = [], []
    for (method, condition), values in by_method_condition.items():
        if len(values) != 163 or len({r["sample_id"] for r in values}) != 163:
            raise ValueError("every method/condition must align all 163 cases")
        summary.append({"method": method, "condition": condition, "n": len(values),
                        **{k: float(np.mean([r[k] for r in values])) for k in numerical}})
        for dimension in ("stratum", "cluster_id"):
            for group in sorted({r[dimension] for r in values}):
                subset = [r for r in values if r[dimension] == group]
                subgroups.append({"method": method, "condition": condition, "dimension": dimension,
                                  "group": group, "n": len(subset), **{k: float(np.mean([r[k] for r in subset])) for k in numerical}})
    summaries = {(r["method"], r["condition"]): r for r in summary}
    for row in summary:
        for metric in numerical:
            row["delta_" + metric] = row[metric]-summaries[row["method"], "clean"][metric]
    family_conditions = defaultdict(list)
    for name, kind, level, seed in conditions(config):
        if name != "clean":
            family_conditions[f"{kind}_{level:g}"].append(name)
    sid_order = sorted(truths)
    clusters = [truths[s]["cluster_id"] for s in sid_order]
    family = []
    family_maps = {}
    for method in ("rule_strict", "bm25", "trained", "fusion"):
        clean_by_id = {r["sample_id"]: r for r in by_method_condition[method, "clean"]}
        for name, names in family_conditions.items():
            maps = [{r["sample_id"]: r for r in by_method_condition[method, c]} for c in names]
            family_maps[method, name] = maps
            item = {"method": method, "family": name, "n_queries": 163, "perturbation_replicates": len(names)}
            for metric in numerical:
                item[metric] = float(np.mean([[m[s][metric] for s in sid_order] for m in maps]))
                item["delta_" + metric] = item[metric]-float(np.mean([clean_by_id[s][metric] for s in sid_order]))
            for metric in ("recall_at_1", "recall_at_10"):
                samples = [[m[s][metric] for s in sid_order] for m in maps]
                interval = paired_interval([clean_by_id[s][metric] for s in sid_order], samples, clusters,
                                           config["bootstrap_seed"], config["bootstrap_replicates"])
                item[metric + "_delta_ci_low"], item[metric + "_delta_ci_high"] = interval
                item[metric + "_seed_min"] = float(np.min(np.mean(samples, axis=1)))
                item[metric + "_seed_max"] = float(np.max(np.mean(samples, axis=1)))
            family.append(item)
    paired_methods = []
    for a, b in [("trained", "bm25"), ("fusion", "bm25"), ("trained", "fusion")]:
        ac = {r["sample_id"]: r for r in by_method_condition[a, "clean"]}
        bc = {r["sample_id"]: r for r in by_method_condition[b, "clean"]}
        for name in family_conditions:
            clean_difference = [ac[s]["recall_at_10"]-bc[s]["recall_at_10"] for s in sid_order]
            degraded_difference = [[am[s]["recall_at_10"]-bm[s]["recall_at_10"] for s in sid_order]
                                  for am, bm in zip(family_maps[a, name], family_maps[b, name])]
            interval = paired_interval(clean_difference, degraded_difference, clusters, config["bootstrap_seed"], config["bootstrap_replicates"])
            paired_methods.append({"method_a": a, "method_b": b, "family": name, "n_queries": 163,
                                   "recall10_difference_in_changes": float(np.mean(degraded_difference)-np.mean(clean_difference)),
                                   "ci_low": interval[0], "ci_high": interval[1],
                                   "interpretation": "positive means A has less recall loss (not necessarily higher absolute recall)"})
    realized = []
    for condition, _, _, _ in conditions(config):
        group = [m for m in manifest if m["condition"] == condition]
        counts = {k: sum(int(m[k]) for m in group) for k in group[0] if k not in {"condition", "sample_id", "events"}}
        counts["realized_changed_species_fraction"] = counts["changed_species"]/counts["eligible_species"] if counts["eligible_species"] else 0.0
        counts["realized_removed_identifier_link_fraction"] = counts["links_removed"]/counts["links_before"] if counts["links_before"] else 0.0
        realized.append({"condition": condition, "n": 163, **counts})
    csv_write(out / "comparison_by_condition.csv", summary)
    csv_write(out / "paired_family_summary.csv", family)
    csv_write(out / "paired_method_robustness.csv", paired_methods)
    csv_write(out / "subgroups.csv", subgroups)
    csv_write(out / "realized_corruptions.csv", realized)
    csv_write(out / "evaluated_rows.csv", evaluated)
    historical = {}
    for method, path in [("bm25", "benchmark/phase3/retrieval_baselines/rankings_bm25.jsonl"),
                         ("trained", "benchmark/phase3/phase3b_full/rankings_epoch_1.jsonl")]:
        old = {(r["model_id"], r["reaction_id"]): r["ranked_ids"] for r in read_rows(source / path)}
        clean = [r for r in rankings if r["condition"] == "clean"]
        historical[method] = {"n": 163, **{f"exact_top{k}_list_matches": sum(r[method][:k] == old[variant_lookup['clean', r['sample_id']]["model_id"], variant_lookup['clean', r['sample_id']]["reaction_id"]][:k] for r in clean) for k in (1, 10, 100)}}
    old_evidence = {r["sample_id"]: [e["kegg_id"] for e in r["records"]] for r in read_rows(source / "benchmark/phase3/phase3c_validation/frozen_evidence.jsonl")}
    historical["fusion"] = {"n": 163, "exact_top10_list_matches": sum(r["fusion"][:10] == old_evidence[r["sample_id"]] for r in rankings if r["condition"] == "clean")}
    write(out / "historical_clean_comparison.json", historical)
    write(out / "analysis_contract.json", {"reference_join_after_ranking_freeze": True, "prediction_sha256": gate["sha256"],
          "n_validation": 163, "n_full_validation": 969, "human_review_labels_used": False, "test_rows_read": 0,
          "primary_metrics": ["recall_at_1", "recall_at_10"], "bootstrap_replicates": config["bootstrap_replicates"],
          "bootstrap_seed": config["bootstrap_seed"], "bootstrap": "paired cluster and perturbation-replicate resampling",
          "full_catalog_candidate_recall_note": "query-independent catalog coverage; top-k metrics describe retrieval quality", "mrr_truncation": [10, 100]})
    print("Analysis complete; tables include every aligned condition", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "dense", "analyze"])
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    source, out = args.source_root.resolve(), args.out.resolve()
    out.relative_to((ROOT / "benchmark/phase5").resolve())
    {"prepare": prepare, "dense": dense, "analyze": analyze}[args.stage](source, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
