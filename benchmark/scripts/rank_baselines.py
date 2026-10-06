"""Phase 2 step 5: baseline rankers over the frozen candidate table.

Every ranker only *reorders* the candidate set produced in step 1. That is deliberate:
it bounds all rankers by the retrieval ceiling and splits errors cleanly into

retrieval failure
    the correct answer is absent from the candidate set, so no reranker can fix it
reranking failure
    the correct answer is present but not placed first

Denominators
------------
Only reactions with a nonempty candidate set can be ranked, so every per-ranker average
in ``baseline_rankings.csv`` is conditional on that. Two different quantities are easy to
confuse, and they differ by a factor of four here:

``conditional_retrieval_failure_rate_nonempty``
    nonempty candidate sets that omit the exact answer (~14%)
``overall_retrieval_failure_rate``
    all evaluable reactions with no exact answer reachable, counting the zero-candidate
    reactions that dominate the corpus (~65%)

``failure_decomposition`` in ``baseline_summary.json`` reports each rate with its own
numerator, denominator and population name so the two cannot be conflated.

Rankers
-------
``heuristic``
    The existing AAAIM rule-based score (the shipped behaviour).
``lexical``
    TF-IDF cosine between the reaction's text and each candidate's KEGG text.
``embedding``
    MiniLM sentence-embedding cosine over the same texts (via chromadb's bundled
    ONNX model, so it runs offline once the model is cached).
``llm``
    LLM reranking of the top candidates. Requires an API key; run explicitly with
    ``--rankers llm``, optionally on a stratified subsample via ``--llm-sample``.
``random`` / ``oracle``
    Reference points: a seeded random shuffle and a best-case reordering. They bracket
    what reranking can achieve on a given candidate set.

Usage::

    python benchmark/scripts/rank_baselines.py --rankers heuristic lexical embedding
    python benchmark/scripts/rank_baselines.py --rankers llm --llm-sample 400
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from benchmark.scripts.kegg_equivalence import EQUIVALENCE_KINDS, match_kinds

DATA_DIR = REPO_ROOT / "benchmark" / "data"
CACHE_DIR = DATA_DIR / "_embed_cache"

REACTIONS_CSV = DATA_DIR / "reactions.csv"
CANDIDATES_CSV = DATA_DIR / "candidates.csv"
TEXT_CSV = DATA_DIR / "reaction_text.csv"
RETRIEVAL_CSV = DATA_DIR / "reaction_retrieval.csv"

OUT_RANKINGS = DATA_DIR / "baseline_rankings.csv"
OUT_TABLE = DATA_DIR / "baseline_table.csv"
OUT_FAILURES = DATA_DIR / "failure_stratification.csv"
OUT_JSON = DATA_DIR / "baseline_summary.json"

CRITERIA = ("exact",) + EQUIVALENCE_KINDS
K_VALUES = (1, 3, 5, 10)
RANDOM_SEED = 20260901

# Candidates shown to the LLM per reaction. Reranking beyond this depth is pointless
# for recall@1..10 and the cost scales linearly.
LLM_TOP_N = 15

logger = logging.getLogger("rank_baselines")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csv(df: pd.DataFrame, path: Path) -> None:
    with open(path, "w", newline="\n", encoding="utf-8") as fh:
        df.to_csv(fh, index=False, lineterminator="\n")


def write_json(obj: Any, path: Path) -> None:
    with open(path, "w", newline="\n", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True)
        fh.write("\n")


def three_way(df: pd.DataFrame, column: str) -> Dict[str, Optional[float]]:
    if df.empty:
        return {"reaction_micro": None, "model_macro": None, "cluster_macro": None}
    return {
        "reaction_micro": round(float(df[column].astype(float).mean()), 4),
        "model_macro": round(float(df.groupby("model_id")[column].mean().mean()), 4),
        "cluster_macro": round(float(df.groupby("cluster_id")[column].mean().mean()), 4),
    }


# --------------------------------------------------------------------------------------
# Candidate documents


def candidate_documents(kegg_ids: Sequence[str]) -> Dict[str, str]:
    """Build one text document per KEGG reaction from its feature record."""
    from core.database_search import load_kegg_reaction_features_dict

    feats = load_kegg_reaction_features_dict()
    docs: Dict[str, str] = {}
    for kegg_id in kegg_ids:
        f = feats.get(kegg_id, {}) or {}
        name = str(f.get("NAME", "") or "")
        definition = str(f.get("DEFINITION", "") or "")
        orthology = str(f.get("ORTHOLOGY", "") or "")
        enzyme = str(f.get("ENZYME", "") or "")
        # Orthology text is long and repetitive; keep the enzyme names only.
        ko_names = " ".join(
            line.split("  ", 1)[1] if "  " in line else line
            for line in orthology.splitlines()[:4]
        )
        docs[kegg_id] = " | ".join(p for p in [name, definition, ko_names, enzyme] if p).strip()
    return docs


# --------------------------------------------------------------------------------------
# Rankers. Each returns {(model_id, reaction_id): {candidate: score}} (higher = better).

ScoreMap = Dict[Tuple[str, str], Dict[str, float]]


def rank_heuristic(ctx: "RankContext") -> ScoreMap:
    out: ScoreMap = {}
    for key, sub in ctx.candidates.groupby(["model_id", "reaction_id"], sort=False):
        scores = {}
        for row in sub.itertuples():
            score = row.heuristic_score
            # Preserve the shipped ordering: fall back to inverted rank when unscored.
            scores[str(row.candidate_kegg)] = (
                float(score) if pd.notna(score) else -float(row.raw_rank)
            )
        out[(str(key[0]), str(key[1]))] = scores
    return out


def rank_random(ctx: "RankContext") -> ScoreMap:
    rng = random.Random(RANDOM_SEED)
    out: ScoreMap = {}
    for key, sub in ctx.candidates.groupby(["model_id", "reaction_id"], sort=False):
        cands = sorted(str(c) for c in sub.candidate_kegg)
        rng.shuffle(cands)
        out[(str(key[0]), str(key[1]))] = {c: -float(i) for i, c in enumerate(cands)}
    return out


def rank_oracle(ctx: "RankContext") -> ScoreMap:
    """Best possible reordering: any correct candidate first. Upper bound on reranking."""
    out: ScoreMap = {}
    for key, sub in ctx.candidates.groupby(["model_id", "reaction_id"], sort=False):
        model_id, reaction_id = str(key[0]), str(key[1])
        truth = ctx.truth.get((model_id, reaction_id), set())
        scores = {}
        for row in sub.itertuples():
            cand = str(row.candidate_kegg)
            scores[cand] = 1.0 if cand in truth else 0.0
        out[(model_id, reaction_id)] = scores
    return out


def rank_lexical(ctx: "RankContext") -> ScoreMap:
    from sklearn.feature_extraction.text import TfidfVectorizer

    kegg_ids = sorted({str(c) for c in ctx.candidates.candidate_kegg})
    docs = candidate_documents(kegg_ids)
    doc_list = [docs.get(k, "") for k in kegg_ids]
    doc_index = {k: i for i, k in enumerate(kegg_ids)}

    queries = ctx.query_texts
    query_keys = sorted(queries)
    query_list = [queries[k] for k in query_keys]
    query_index = {k: i for i, k in enumerate(query_keys)}

    vectorizer = TfidfVectorizer(
        lowercase=True, analyzer="word", token_pattern=r"[A-Za-z0-9\-\+]{2,}",
        sublinear_tf=True, min_df=1,
    )
    vectorizer.fit(doc_list + query_list)
    doc_matrix = vectorizer.transform(doc_list)
    query_matrix = vectorizer.transform(query_list)

    return _score_by_similarity(ctx, doc_matrix, doc_index, query_matrix, query_index)


def rank_embedding(ctx: "RankContext") -> ScoreMap:
    kegg_ids = sorted({str(c) for c in ctx.candidates.candidate_kegg})
    docs = candidate_documents(kegg_ids)
    doc_vectors = _embed_cached([docs.get(k, "") for k in kegg_ids], "kegg_docs")
    doc_index = {k: i for i, k in enumerate(kegg_ids)}

    query_keys = sorted(ctx.query_texts)
    query_vectors = _embed_cached([ctx.query_texts[k] for k in query_keys], "queries")
    query_index = {k: i for i, k in enumerate(query_keys)}

    return _score_by_similarity(ctx, doc_vectors, doc_index, query_vectors, query_index)


def _score_by_similarity(ctx, doc_matrix, doc_index, query_matrix, query_index) -> ScoreMap:
    """Cosine similarity between each reaction's query vector and its candidates."""
    from sklearn.preprocessing import normalize

    doc_matrix = normalize(doc_matrix)
    query_matrix = normalize(query_matrix)

    out: ScoreMap = {}
    for key, sub in ctx.candidates.groupby(["model_id", "reaction_id"], sort=False):
        model_id, reaction_id = str(key[0]), str(key[1])
        qi = query_index.get((model_id, reaction_id))
        cands = [str(c) for c in sub.candidate_kegg]
        if qi is None:
            out[(model_id, reaction_id)] = {c: 0.0 for c in cands}
            continue
        rows = [doc_index[c] for c in cands if c in doc_index]
        known = [c for c in cands if c in doc_index]
        if not rows:
            out[(model_id, reaction_id)] = {c: 0.0 for c in cands}
            continue
        sims = (doc_matrix[rows] @ query_matrix[qi].T)
        sims = np.asarray(sims.todense()).ravel() if hasattr(sims, "todense") else np.asarray(sims).ravel()
        scores = {c: float(s) for c, s in zip(known, sims)}
        for c in cands:
            scores.setdefault(c, 0.0)
        out[(model_id, reaction_id)] = scores
    return out


def _embed_cached(texts: Sequence[str], tag: str) -> np.ndarray:
    """Embed texts with chromadb's bundled MiniLM, caching by content digest."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(("\u0000".join(texts)).encode("utf-8")).hexdigest()[:16]
    cache_file = CACHE_DIR / f"{tag}_{digest}.npy"
    if cache_file.exists():
        logger.info("embeddings cache hit: %s", cache_file.name)
        return np.load(cache_file)

    from chromadb.utils import embedding_functions

    fn = embedding_functions.DefaultEmbeddingFunction()
    logger.info("embedding %d texts (%s); this runs once and is cached", len(texts), tag)
    vectors: List[List[float]] = []
    batch = 256
    for start in range(0, len(texts), batch):
        chunk = [t if t.strip() else "unknown" for t in texts[start:start + batch]]
        vectors.extend(fn(chunk))
        if (start // batch) % 10 == 0:
            logger.info("  embedded %d/%d", min(start + batch, len(texts)), len(texts))
    arr = np.asarray(vectors, dtype=np.float32)
    np.save(cache_file, arr)
    return arr


def rank_llm(ctx: "RankContext") -> ScoreMap:
    """Rerank the top candidates per reaction with an LLM."""
    from core.llm_interface import query_llm

    kegg_ids = sorted({str(c) for c in ctx.candidates.candidate_kegg})
    docs = candidate_documents(kegg_ids)

    out: ScoreMap = {}
    keys = ctx.llm_keys if ctx.llm_keys is not None else sorted(ctx.reaction_keys)
    total = len(keys)
    logger.info("LLM reranking %d reactions (top %d candidates each)", total, LLM_TOP_N)

    for i, key in enumerate(keys, start=1):
        sub = ctx.candidates_by_key.get(key)
        if sub is None or sub.empty:
            continue
        top = sub.sort_values("raw_rank").head(LLM_TOP_N)
        cands = [str(c) for c in top.candidate_kegg]
        query = ctx.query_texts.get(key, "")

        listing = "\n".join(
            f"{j}. {c}: {docs.get(c, '')[:200]}" for j, c in enumerate(cands, start=1)
        )
        prompt = (
            "You are annotating a systems-biology model reaction with the correct KEGG "
            "reaction id.\n\n"
            f"Model: {ctx.model_names.get(key, '')}\n"
            f"Reaction: {query}\n"
            f"Equation (species ids): {ctx.equations.get(key, '')}\n\n"
            f"Candidate KEGG reactions:\n{listing}\n\n"
            "Rank the candidates from most to least likely. Reply with only a "
            "comma-separated list of KEGG ids, best first."
        )
        try:
            response, _ = query_llm(prompt, model="gpt-4o-mini", system_prompt=None)
        except Exception as exc:  # noqa: BLE001
            logger.error("LLM call failed for %s: %s", key, exc)
            ctx.llm_errors.append({"key": "|".join(key), "error": f"{type(exc).__name__}: {exc}"})
            continue

        ordered = [c for c in _parse_kegg_list(str(response)) if c in set(cands)]
        # Anything the model dropped keeps its original relative order behind the ranked
        # items, so a truncated reply degrades to the heuristic order rather than losing
        # candidates.
        remainder = [c for c in cands if c not in ordered]
        final = ordered + remainder
        out[key] = {c: -float(idx) for idx, c in enumerate(final)}
        if i % 25 == 0:
            logger.info("  LLM %d/%d", i, total)
    return out


def _parse_kegg_list(text: str) -> List[str]:
    import re

    return re.findall(r"\bR\d{5}\b", text)


RANKERS: Dict[str, Callable[["RankContext"], ScoreMap]] = {
    "heuristic": rank_heuristic,
    "lexical": rank_lexical,
    "embedding": rank_embedding,
    "llm": rank_llm,
    "random": rank_random,
    "oracle": rank_oracle,
}


# --------------------------------------------------------------------------------------


class RankContext:
    """Shared inputs for the rankers."""

    def __init__(self, llm_sample: Optional[int] = None):
        self.candidates = pd.read_csv(CANDIDATES_CSV)
        self.candidates["model_id"] = self.candidates.model_id.astype(str)
        self.candidates["reaction_id"] = self.candidates.reaction_id.astype(str)
        self.candidates["candidate_kegg"] = self.candidates.candidate_kegg.astype(str)

        reactions = pd.read_csv(REACTIONS_CSV)
        reactions = reactions[reactions.included_in_eval.astype(bool)]
        self.truth = {
            (str(r.model_id), str(r.reaction_id)):
                {t for t in str(r.ground_truth_kegg_all).split("|") if t}
            for r in reactions.itertuples()
        }

        text = pd.read_csv(TEXT_CSV).fillna("")
        self.query_texts = {
            (str(r.model_id), str(r.reaction_id)): str(r.query_text) for r in text.itertuples()
        }
        self.model_names = {
            (str(r.model_id), str(r.reaction_id)): str(r.model_name) for r in text.itertuples()
        }
        self.equations = {
            (str(r.model_id), str(r.reaction_id)): str(r.reaction_equation)
            for r in text.itertuples()
        }

        self.retrieval = pd.read_csv(RETRIEVAL_CSV)
        self.retrieval["model_id"] = self.retrieval.model_id.astype(str)
        self.retrieval["reaction_id"] = self.retrieval.reaction_id.astype(str)

        self.reaction_keys = sorted(
            {(str(m), str(r)) for m, r in
             zip(self.candidates.model_id, self.candidates.reaction_id)}
        )
        self.candidates_by_key = {
            (str(k[0]), str(k[1])): sub
            for k, sub in self.candidates.groupby(["model_id", "reaction_id"], sort=False)
        }
        self.llm_errors: List[Dict[str, str]] = []
        self.llm_keys: Optional[List[Tuple[str, str]]] = None
        if llm_sample:
            self.llm_keys = self._stratified_sample(llm_sample)

    def _stratified_sample(self, n: int) -> List[Tuple[str, str]]:
        """Sample reactions with candidates, spread across models and strata."""
        pool = self.retrieval[self.retrieval.has_candidates.astype(bool)].copy()
        pool["key"] = list(zip(pool.model_id, pool.reaction_id))
        pool = pool.sort_values(["model_id", "reaction_id"])
        strata = ["is_genome_scale", "species_annotation_source", "complexity_bucket"]
        pool["stratum"] = pool[strata].astype(str).agg("|".join, axis=1)
        rng = np.random.default_rng(RANDOM_SEED)
        chosen: List[Tuple[str, str]] = []
        groups = list(pool.groupby("stratum", sort=True))
        per_group = max(1, n // max(1, len(groups)))
        for _, sub in groups:
            take = min(per_group, len(sub))
            idx = rng.choice(len(sub), size=take, replace=False)
            chosen.extend([sub.iloc[i]["key"] for i in sorted(idx)])
        return sorted(set(chosen))[:n]


def evaluate_ranking(ctx: RankContext, ranker: str, scores: ScoreMap) -> pd.DataFrame:
    """Score one ranker: per-reaction hit positions under every criterion."""
    rows: List[Dict[str, Any]] = []
    strata = ctx.retrieval.set_index(["model_id", "reaction_id"])

    for key, cand_scores in scores.items():
        if not cand_scores:
            continue
        # Deterministic order: score descending, KEGG id ascending as tie-break.
        ordered = [c for c, _ in sorted(cand_scores.items(), key=lambda kv: (-kv[1], kv[0]))]
        truth = ctx.truth.get(key, set())

        first_hit: Dict[str, Optional[int]] = {c: None for c in CRITERIA}
        for rank, cand in enumerate(ordered, start=1):
            verdict = match_kinds(cand, truth)
            for crit in CRITERIA:
                if verdict[crit] and first_hit[crit] is None:
                    first_hit[crit] = rank
            if all(first_hit[c] is not None for c in CRITERIA):
                break

        try:
            s = strata.loc[key]
        except KeyError:
            continue

        row: Dict[str, Any] = {
            "ranker": ranker,
            "model_id": key[0],
            "reaction_id": key[1],
            "cluster_id": s["cluster_id"],
            "candidate_set_size": len(ordered),
            "is_genome_scale": bool(s["is_genome_scale"]),
            "species_annotation_source": s["species_annotation_source"],
            "any_missing_annotation": bool(s["any_missing_annotation"]),
            "relaxation_required": bool(s["relaxation_required"]),
            "complexity_bucket": s["complexity_bucket"],
        }
        for crit in CRITERIA:
            fh = first_hit[crit]
            row[f"first_hit_rank_{crit}"] = fh
            row[f"hit_any_{crit}"] = fh is not None
            for k in K_VALUES:
                row[f"hit_at_{k}_{crit}"] = fh is not None and fh <= k
        # The split the whole exercise is for.
        row["retrieval_failure"] = first_hit["exact"] is None
        row["reranking_failure"] = (
            first_hit["exact"] is not None and first_hit["exact"] > 1
        )
        rows.append(row)

    return pd.DataFrame(rows)


def baseline_table(all_rankings: pd.DataFrame, ctx: RankContext) -> pd.DataFrame:
    """Headline table: one row per (ranker, criterion, metric) with all three averages."""
    rows: List[Dict[str, Any]] = []
    n_evaluable = int(len(ctx.retrieval))

    for ranker, sub in all_rankings.groupby("ranker"):
        for crit in CRITERIA:
            for metric, column in (
                [("recall_any_rank", f"hit_any_{crit}")]
                + [(f"recall_at_{k}", f"hit_at_{k}_{crit}") for k in K_VALUES]
            ):
                agg = three_way(sub, column)
                # Denominator note: `scored` covers reactions this ranker ranked (i.e.
                # those with candidates); `all_evaluable` charges zero-candidate
                # reactions as failures.
                scored_mean = agg["reaction_micro"]
                rows.append({
                    "ranker": ranker,
                    "criterion": crit,
                    "metric": metric,
                    "reaction_micro_scored": scored_mean,
                    "model_macro_scored": agg["model_macro"],
                    "cluster_macro_scored": agg["cluster_macro"],
                    "reaction_micro_all_evaluable": (
                        round(float(sub[column].astype(float).sum()) / n_evaluable, 4)
                        if n_evaluable else None
                    ),
                    "n_reactions_scored": int(len(sub)),
                    "n_evaluable": n_evaluable,
                })
    return pd.DataFrame(rows)


def _rate(numerator: int, denominator: int, population: str) -> Dict[str, Any]:
    """A rate that carries its own denominator, so it cannot be misread.

    Every reported fraction in Phase 2 goes through this so the machine-readable output
    always states what population it is over. Reading `rate` alone is never enough to
    know whether it is over all evaluable reactions or only those with candidates.
    """
    return {
        "rate": round(numerator / denominator, 4) if denominator else None,
        "pct": round(100.0 * numerator / denominator, 2) if denominator else None,
        "numerator": int(numerator),
        "denominator": int(denominator),
        "population": population,
    }


def failure_decomposition(sub: pd.DataFrame, n_evaluable: int,
                          n_zero_candidate: int) -> Dict[str, Any]:
    """Split failure into retrieval vs reranking with unambiguous denominators.

    ``sub`` holds one row per reaction *that this ranker ranked*, i.e. only reactions
    with a nonempty candidate set. The overall quantities therefore have to charge the
    zero-candidate reactions explicitly rather than inheriting `sub`'s denominator --
    that conflation is what made a 14.4% conditional miss rate look like the benchmark's
    retrieval-failure rate when the true figure is about 65%.
    """
    n_scored = int(len(sub))
    retrievable = int(sub.hit_any_exact.astype(bool).sum())
    top1 = int(sub.hit_at_1_exact.astype(bool).sum())
    missed_in_nonempty = n_scored - retrievable
    rerank_failures = int(
        (sub.hit_any_exact.astype(bool) & ~sub.hit_at_1_exact.astype(bool)).sum())

    return {
        # Retrieval, independent of any ranker.
        "zero_candidate_rate": _rate(
            n_zero_candidate, n_evaluable, "all evaluable reactions"),
        "overall_retrieval_failure_rate": _rate(
            n_evaluable - retrievable, n_evaluable,
            "all evaluable reactions; failure = exact answer absent from candidate set "
            "(includes every zero-candidate reaction)"),
        "conditional_retrieval_failure_rate_nonempty": _rate(
            missed_in_nonempty, n_scored,
            "evaluable reactions with a nonempty candidate set"),
        # Reranking, conditional on the answer being reachable at all.
        "conditional_reranking_failure_rate_retrievable": _rate(
            rerank_failures, retrievable,
            "reactions whose candidate set contains the exact answer"),
        # End-to-end.
        "overall_top1_accuracy": _rate(
            top1, n_evaluable, "all evaluable reactions"),
        "overall_top1_failure_rate": _rate(
            n_evaluable - top1, n_evaluable, "all evaluable reactions"),
        "conditional_top1_accuracy_nonempty": _rate(
            top1, n_scored, "evaluable reactions with a nonempty candidate set"),
    }


def failure_stratification(all_rankings: pd.DataFrame) -> pd.DataFrame:
    """Split failures by cause and stratum for each ranker."""
    rows: List[Dict[str, Any]] = []
    strata_cols = [
        "is_genome_scale", "species_annotation_source", "any_missing_annotation",
        "relaxation_required", "complexity_bucket",
    ]
    for ranker, sub in all_rankings.groupby("ranker"):
        for column in strata_cols:
            for value, grp in sub.groupby(column, dropna=False):
                n = len(grp)
                retrievable = int(grp["hit_any_exact"].astype(bool).sum())
                rerank_failures = int(
                    (grp["hit_any_exact"].astype(bool)
                     & ~grp["hit_at_1_exact"].astype(bool)).sum())
                # Every rate here is conditional on a nonempty candidate set, because
                # only ranked reactions appear in `all_rankings`. The column names say
                # so; do not compare them against corpus-wide rates.
                rows.append({
                    "ranker": ranker,
                    "stratum": column,
                    "value": str(value),
                    "n_reactions_nonempty": int(n),
                    "conditional_top1_pct_nonempty": round(
                        100.0 * float(grp["hit_at_1_exact"].mean()), 2),
                    "conditional_retrieval_failure_pct_nonempty": round(
                        100.0 * float(grp["retrieval_failure"].mean()), 2),
                    "conditional_reranking_failure_pct_nonempty": round(
                        100.0 * float(grp["reranking_failure"].mean()), 2),
                    "n_retrievable": retrievable,
                    "conditional_reranking_failure_pct_retrievable": (
                        round(100.0 * rerank_failures / retrievable, 2)
                        if retrievable else None),
                    "equivalence_only_gain_pct_nonempty": round(100.0 * float(
                        (grp["hit_at_1_brite_orthology"] & ~grp["hit_at_1_exact"]).mean()
                    ), 2),
                    "mean_candidate_set_size": round(float(grp.candidate_set_size.mean()), 1),
                })
    return pd.DataFrame(rows).sort_values(["ranker", "stratum", "value"]).reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rankers", nargs="+", default=["heuristic", "lexical", "embedding",
                                                         "random", "oracle"])
    parser.add_argument("--llm-sample", type=int, default=None,
                        help="Rerank only a seeded stratified subsample with the LLM")
    parser.add_argument("--append", action="store_true",
                        help="Merge with existing baseline_rankings.csv instead of replacing")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    unknown = [r for r in args.rankers if r not in RANKERS]
    if unknown:
        logger.error("unknown rankers: %s (available: %s)", unknown, sorted(RANKERS))
        return 2

    if "llm" in args.rankers and not (
        os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    ):
        logger.error(
            "LLM reranking requested but no OPENAI_API_KEY / OPENROUTER_API_KEY is set. "
            "Set one and re-run with --rankers llm."
        )
        return 3

    ctx = RankContext(llm_sample=args.llm_sample)
    logger.info("reactions with candidates: %d", len(ctx.reaction_keys))

    frames: List[pd.DataFrame] = []
    if args.append and OUT_RANKINGS.exists():
        existing = pd.read_csv(OUT_RANKINGS)
        existing = existing[~existing.ranker.isin(args.rankers)]
        frames.append(existing)

    per_ranker_notes: Dict[str, Any] = {}
    for name in args.rankers:
        logger.info("running ranker: %s", name)
        scores = RANKERS[name](ctx)
        result = evaluate_ranking(ctx, name, scores)
        frames.append(result)
        per_ranker_notes[name] = {
            "reactions_ranked": int(len(result)),
            "recall_at_1_exact_micro": (
                round(float(result["hit_at_1_exact"].mean()), 4) if len(result) else None
            ),
        }
        logger.info("  %s: ranked %d reactions, recall@1(exact, micro)=%s",
                    name, len(result), per_ranker_notes[name]["recall_at_1_exact_micro"])

    all_rankings = pd.concat(frames, ignore_index=True)
    all_rankings = all_rankings.sort_values(["ranker", "model_id", "reaction_id"])
    write_csv(all_rankings.reset_index(drop=True), OUT_RANKINGS)

    table = baseline_table(all_rankings, ctx)
    write_csv(table, OUT_TABLE)
    write_csv(failure_stratification(all_rankings), OUT_FAILURES)

    n_evaluable = int(len(ctx.retrieval))
    n_zero_candidate = int((~ctx.retrieval.has_candidates.astype(bool)).sum())

    summary = {
        "rankers_run": args.rankers,
        "ranker_notes": per_ranker_notes,
        "llm_sample": args.llm_sample,
        "llm_errors": ctx.llm_errors,
        "criteria": list(CRITERIA),
        "k_values": list(K_VALUES),
        "random_seed": RANDOM_SEED,
        "llm_top_n": LLM_TOP_N,
        "populations": {
            "all_evaluable": {
                "n": n_evaluable,
                "description": "every reaction with ground truth in the frozen Phase 1 table",
            },
            "nonempty_candidate_set": {
                "n": n_evaluable - n_zero_candidate,
                "description": "evaluable reactions for which generation stored >=1 candidate; "
                               "the only reactions any ranker can score",
            },
            "zero_candidate": {
                "n": n_zero_candidate,
                "description": "evaluable reactions with no candidates; unreachable for every "
                               "ranker and charged as retrieval failures in overall metrics",
            },
        },
        "headline": {
            ranker: {
                # `_scored` suffixes mark averages over ranked (nonempty) reactions only.
                "recall_at_1_exact_scored": three_way(sub, "hit_at_1_exact"),
                "recall_at_1_brite_orthology_scored": three_way(
                    sub, "hit_at_1_brite_orthology"),
                "failure_decomposition": failure_decomposition(
                    sub, n_evaluable, n_zero_candidate),
            }
            for ranker, sub in all_rankings.groupby("ranker")
        },
        "inputs": {
            "candidates_csv_sha256": sha256_file(CANDIDATES_CSV),
            "reaction_retrieval_csv_sha256": sha256_file(RETRIEVAL_CSV),
            "reaction_text_csv_sha256": sha256_file(TEXT_CSV),
        },
        "outputs": {
            "baseline_rankings_csv_sha256": sha256_file(OUT_RANKINGS),
            "baseline_table_csv_sha256": sha256_file(OUT_TABLE),
            "failure_stratification_csv_sha256": sha256_file(OUT_FAILURES),
        },
    }
    write_json(summary, OUT_JSON)

    logger.info(
        "populations: %d evaluable reactions; %d with candidates; %d with zero candidates "
        "(zero_candidate_rate=%.1f%% of all evaluable)",
        n_evaluable, n_evaluable - n_zero_candidate, n_zero_candidate,
        100.0 * n_zero_candidate / n_evaluable if n_evaluable else 0.0,
    )
    for ranker, head in sorted(summary["headline"].items()):
        r1 = head["recall_at_1_exact_scored"]
        d = head["failure_decomposition"]
        # Spell out both denominators on every line. An unqualified "retrieval_fail"
        # reads as a corpus-wide rate and it is not one.
        logger.info(
            "%-10s | conditional (n=%d, nonempty sets): top1=%.1f%% "
            "retrieval_fail=%.1f%% | overall (n=%d, all evaluable): top1=%.1f%% "
            "retrieval_fail=%.1f%% | rerank_fail=%.1f%% of the %d retrievable",
            ranker,
            d["conditional_top1_accuracy_nonempty"]["denominator"],
            d["conditional_top1_accuracy_nonempty"]["pct"],
            d["conditional_retrieval_failure_rate_nonempty"]["pct"],
            d["overall_top1_accuracy"]["denominator"],
            d["overall_top1_accuracy"]["pct"],
            d["overall_retrieval_failure_rate"]["pct"],
            d["conditional_reranking_failure_rate_retrievable"]["pct"],
            d["conditional_reranking_failure_rate_retrievable"]["denominator"],
        )
        logger.info(
            "%-10s | recall@1 exact over ranked reactions only: micro=%.3f "
            "model_macro=%.3f cluster_macro=%.3f",
            ranker, r1["reaction_micro"], r1["model_macro"], r1["cluster_macro"],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
