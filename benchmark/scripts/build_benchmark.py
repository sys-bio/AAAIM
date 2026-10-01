#!/usr/bin/env python3
"""Build the frozen Phase-1 reaction-annotation benchmark table.

Design principles
-----------------

*Pipeline failures are not scientific exclusions.* A model that failed to
download, or that libSBML cannot parse, represents a defect in our tooling or
in the upstream snapshot. Such models are reported in ``pipeline_failures.csv``
and must be resolved or explicitly justified; they are never silently folded
into the benchmark's exclusion statistics.

*Scientific exclusions* are properties of the data itself and are recorded in
``exclusions.csv``:

- model level: the model genuinely carries no KEGG reaction ground truth
- reaction level: exchange/source-sink (SSX) stoichiometry, or a malformed
  ground-truth identifier

*SSX exclusions reduce reaction counts, not model counts.* A model whose
reactions are all SSX stays in the benchmark and contributes zero evaluable
reactions; that situation is reported explicitly rather than hidden.

*Parser problems are surfaced, not absorbed.* For every model we compare the
number of reactions whose raw annotation mentions ``kegg.reaction`` against the
number from which we actually extracted an identifier. Any shortfall indicates
a URI-form or parser defect and is flagged in ``parser_diagnostics.csv``.

Outputs (under ``benchmark/data/``)
-----------------------------------

``reactions.csv``            one row per (model, reaction) carrying ground truth
``model_context.csv``        model title and notes (join on ``model_id``)
``model_summary.csv``        per-model status, counts, and cluster assignment
``exclusions.csv``           scientific exclusions only
``pipeline_failures.csv``    tooling/provenance failures
``duplicate_groups.csv``     multi-member near-duplicate clusters
``model_clusters.csv``       every included model's stable cluster ID
``species_annotations.csv``  species-level annotation inventory
``parser_diagnostics.csv``   raw-vs-extracted KEGG mention comparison
``benchmark_summary.json``   observed counts and reconciliation
``invariants.json``          dataset invariant checks
``VERSION.json``             frozen version with artifact hashes

Run from the repo root::

    python benchmark/scripts/build_benchmark.py
    python benchmark/scripts/build_benchmark.py --with-candidates
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import libsbml
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.model_info import (  # noqa: E402
    exchange_constraint_skipped_reaction_ids,
    find_reactions_with_kegg_annotations,
    find_species_with_annotations_and_qualifiers,
    find_species_with_chebi_annotations,
)
from utils.constants import DatabaseID  # noqa: E402

DEFAULT_MANIFEST = REPO_ROOT / "benchmark" / "manifest" / "models.txt"
DEFAULT_MODELS_DIR = REPO_ROOT / "benchmark" / "models"
DEFAULT_REGISTRY = REPO_ROOT / "benchmark" / "manifest" / "model_registry.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "benchmark" / "data"

BENCHMARK_VERSION = "phase1-v1"

# KEGG reaction accessions are 'R' followed by exactly five digits.
VALID_KEGG_REACTION_RE = re.compile(r"^R\d{5}$")
RAW_KEGG_MENTION_RE = re.compile(r"kegg\.reaction", re.IGNORECASE)

# Overlap thresholds for treating two models as near-duplicate variants.
DEFAULT_DUPLICATE_THRESHOLD = 0.9
DEFAULT_CONTAINMENT_THRESHOLD = 0.9
# Containment alone is unsafe: a 2-reaction model is trivially contained in a
# genome-scale model, which would chain unrelated models into one giant cluster.
# Containment linkage therefore also requires comparable set sizes.
DEFAULT_MIN_SIZE_RATIO = 0.5
# Containment linkage also needs a meaningful absolute overlap: with one or two
# shared identifiers, "containment" is coincidence rather than shared lineage.
# Genuine duplicates of tiny models are still caught by the Jaccard rule.
DEFAULT_MIN_CONTAINMENT_OVERLAP = 5

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("build_benchmark")


# ---------------------------------------------------------------------------
# Manifest and provenance
# ---------------------------------------------------------------------------

def load_model_ids(manifest_path: Path) -> List[str]:
    ids: List[str] = []
    with manifest_path.open(encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            ids.append(line)
    return ids


def _rel(path: Path) -> str:
    """Repo-relative POSIX path when possible, else the absolute path.

    Synthetic test fixtures live outside the repository, so this must not raise.
    """
    try:
        return str(path.relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def load_registry(registry_path: Path) -> Dict[str, Dict[str, object]]:
    """Map accession -> provenance entry from the download registry."""
    if not registry_path.exists():
        logger.warning("No download registry at %s; provenance cannot be verified", registry_path)
        return {}
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    return {str(e.get("model_id")): e for e in payload.get("models", [])}


# ---------------------------------------------------------------------------
# SBML reading helpers (libSBML only, for determinism and speed)
# ---------------------------------------------------------------------------

def _strip_tags(text: str) -> str:
    out = re.sub(r"<[^>]*>", " ", text or "")
    out = re.sub(r"\s+", " ", out)
    return out.strip()


def read_sbml(model_file: Path) -> Tuple[Optional[libsbml.Model], str]:
    """Return ``(model, error)``; ``model`` is None when parsing fails."""
    reader = libsbml.SBMLReader()
    document = reader.readSBML(str(model_file))
    if document is None:
        return None, "libsbml returned no document"
    n_err = document.getNumErrors()
    fatal = []
    for i in range(n_err):
        err = document.getError(i)
        if err.getSeverity() >= libsbml.LIBSBML_SEV_ERROR:
            fatal.append(f"{err.getErrorId()}: {err.getShortMessage()}")
    model = document.getModel()
    if model is None:
        detail = "; ".join(fatal[:5]) or "no model element"
        return None, detail
    return model, ""


def format_equation(reaction: libsbml.Reaction) -> str:
    """Deterministic ``A + 2 B => C`` rendering from libSBML stoichiometry."""

    def side(refs: Sequence[libsbml.SpeciesReference]) -> str:
        terms = []
        for ref in refs:
            sid = ref.getSpecies()
            stoich = ref.getStoichiometry() if ref.isSetStoichiometry() else 1.0
            if stoich is None:
                stoich = 1.0
            if abs(stoich - 1.0) < 1e-9:
                terms.append(str(sid))
            elif abs(stoich - round(stoich)) < 1e-9:
                terms.append(f"{int(round(stoich))} {sid}")
            else:
                terms.append(f"{stoich:g} {sid}")
        return " + ".join(terms)

    reactants = [reaction.getReactant(i) for i in range(reaction.getNumReactants())]
    products = [reaction.getProduct(i) for i in range(reaction.getNumProducts())]
    arrow = "<=>" if reaction.getReversible() else "=>"
    return f"{side(reactants)} {arrow} {side(products)}".strip()


def libsbml_ssx_reaction_ids(model: libsbml.Model) -> Set[str]:
    """Reactions with no reactants or no products (source/sink/exchange style)."""
    out: Set[str] = set()
    for i in range(model.getNumReactions()):
        rxn = model.getReaction(i)
        if rxn.getNumReactants() == 0 or rxn.getNumProducts() == 0:
            out.add(str(rxn.getId()))
    return out


def raw_kegg_reaction_mentions(model: libsbml.Model) -> Tuple[int, List[str]]:
    """Count reactions whose raw annotation mentions ``kegg.reaction``.

    Returns the count plus a few example annotation snippets, so a URI-form
    mismatch can be diagnosed without re-reading the file by hand.
    """
    count = 0
    examples: List[str] = []
    for i in range(model.getNumReactions()):
        rxn = model.getReaction(i)
        if not rxn.isSetAnnotation():
            continue
        ann = rxn.getAnnotation().toXMLString()
        if RAW_KEGG_MENTION_RE.search(ann):
            count += 1
            if len(examples) < 3:
                for line in ann.splitlines():
                    if RAW_KEGG_MENTION_RE.search(line):
                        examples.append(line.strip()[:200])
                        break
    return count, examples


def misplaced_kegg_reaction_mentions(model: libsbml.Model) -> int:
    """Count *species* carrying a ``kegg.reaction`` annotation.

    Reaction identifiers on a species are a curation error in the source model.
    They matter here because the manifest was assembled by grepping raw file
    bytes for ``kegg.reaction``, so such models were selected as
    "KEGG-annotated" despite having no reaction-level ground truth at all.
    """
    count = 0
    for i in range(model.getNumSpecies()):
        sp = model.getSpecies(i)
        if sp.isSetAnnotation() and RAW_KEGG_MENTION_RE.search(sp.getAnnotation().toXMLString()):
            count += 1
    return count


# ---------------------------------------------------------------------------
# Species annotations
# ---------------------------------------------------------------------------

def extract_species_annotations(model_file: Path) -> Tuple[pd.DataFrame, str]:
    """Species annotation rows plus source label (chebi/kegg_compound/mixed/none)."""
    chebi = find_species_with_chebi_annotations(str(model_file))
    kegg, _ = find_species_with_annotations_and_qualifiers(
        str(model_file), DatabaseID.KEGG.value
    )

    rows: List[Dict[str, object]] = []
    all_ids = sorted(set(chebi.keys()) | set(kegg.keys()))
    for sid in all_ids:
        sid_str = str(sid)
        ch_list = chebi.get(sid) if chebi else None
        if ch_list:
            for cid in sorted({str(c).strip() for c in ch_list if str(c).strip()}):
                rows.append(
                    {
                        "species_id": sid_str,
                        "annotation": f"CHEBI:{cid.split(':', 1)[-1]}",
                        "source": "chebi",
                    }
                )
            continue
        kg_list = kegg.get(sid) if kegg else None
        if kg_list:
            for kid in sorted({str(k).strip() for k in kg_list if str(k).strip()}):
                rows.append(
                    {"species_id": sid_str, "annotation": kid, "source": "kegg_compound"}
                )

    if not rows:
        return pd.DataFrame(columns=["species_id", "annotation", "source"]), "none"

    has_chebi = any(r["source"] == "chebi" for r in rows)
    has_kegg = any(r["source"] == "kegg_compound" for r in rows)
    source = "mixed" if (has_chebi and has_kegg) else ("chebi" if has_chebi else "kegg_compound")
    return pd.DataFrame(rows), source


# ---------------------------------------------------------------------------
# Candidate generation (optional)
# ---------------------------------------------------------------------------

def run_candidates(model_file: Path, species_df: pd.DataFrame, work_dir: Path) -> pd.DataFrame:
    from core import annotate_model  # heavy import; only when requested

    if species_df.empty:
        return pd.DataFrame()

    rec = species_df.rename(columns={"species_id": "id"}).assign(match_score=1.0)[
        ["id", "annotation", "match_score"]
    ]
    work_dir.mkdir(parents=True, exist_ok=True)
    species_csv = work_dir / f"{model_file.stem}__species.csv"
    rec.to_csv(species_csv, index=False)
    reloaded = pd.read_csv(species_csv, encoding="utf-8-sig")

    import os

    cwd = Path.cwd()
    try:
        os.chdir(work_dir)
        df, _ = annotate_model(
            model_file=str(model_file),
            method="rulebased",
            entity_type="reaction",
            database="kegg",
            species_recommendations_df=reloaded,
            evaluate_candidates=False,
            include_exchange_reactions=False,
        )
    finally:
        os.chdir(cwd)
    return df if df is not None else pd.DataFrame()


# ---------------------------------------------------------------------------
# Per-model processing
# ---------------------------------------------------------------------------

class ModelResult:
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.status = "included"
        self.reaction_rows: List[Dict] = []
        self.species_rows: List[Dict] = []
        self.exclusions: List[Dict] = []
        self.pipeline_failures: List[Dict] = []
        self.context: Optional[Dict] = None
        self.diagnostics: Optional[Dict] = None
        self.summary: Dict = {}
        self.gt_id_set: Set[str] = set()


def process_model(
    model_id: str,
    model_file: Path,
    provenance: Dict[str, object],
    *,
    with_candidates: bool,
    work_dir: Path,
    registry_available: bool = False,
) -> ModelResult:
    res = ModelResult(model_id)

    # --- pipeline gate: the file must exist -----------------------------
    prov_status = str(provenance.get("status", "")) if provenance else ""

    if not model_file.exists():
        res.status = "pipeline_failure"
        res.pipeline_failures.append(
            {
                "model_id": model_id,
                "failure_type": "file_missing",
                "detail": f"Expected SBML at {_rel(model_file)}; "
                f"download status={prov_status or 'unknown'}",
                "actionable": "Re-run download_biomodels.py; investigate API response.",
            }
        )
        res.summary = {"model_id": model_id, "status": res.status}
        return res

    # --- parse ----------------------------------------------------------
    model, parse_error = read_sbml(model_file)
    if model is None:
        res.status = "pipeline_failure"
        res.pipeline_failures.append(
            {
                "model_id": model_id,
                "failure_type": "parse_error",
                "detail": parse_error,
                "actionable": "Inspect the SBML individually; may need libSBML upgrade or upstream fix.",
            }
        )
        res.summary = {"model_id": model_id, "status": res.status}
        return res

    # --- provenance checks (after the file itself is known to be readable) ---
    if registry_available:
        if not provenance:
            res.pipeline_failures.append(
                {
                    "model_id": model_id,
                    "failure_type": "provenance_missing",
                    "detail": "No entry in model_registry.json for this accession.",
                    "actionable": "Re-run download_biomodels.py to regenerate the registry.",
                }
            )
        elif not provenance.get("checksum_verified"):
            res.pipeline_failures.append(
                {
                    "model_id": model_id,
                    "failure_type": "checksum_unverified",
                    "detail": f"download status={prov_status}; "
                    f"upstream={provenance.get('upstream_sha256')} "
                    f"local={provenance.get('local_sha256')}",
                    "actionable": "Re-download with --force; confirm upstream file identity.",
                }
            )

    num_reactions_total = model.getNumReactions()
    raw_mentions, raw_examples = raw_kegg_reaction_mentions(model)
    misplaced_mentions = misplaced_kegg_reaction_mentions(model)

    try:
        gt_map_raw, _ = find_reactions_with_kegg_annotations(str(model_file))
    except Exception as exc:
        res.status = "pipeline_failure"
        res.pipeline_failures.append(
            {
                "model_id": model_id,
                "failure_type": "annotation_extraction_error",
                "detail": str(exc),
                "actionable": "Inspect annotation extraction for this model.",
            }
        )
        res.summary = {"model_id": model_id, "status": res.status}
        return res

    # Preserve every ground-truth ID, deduplicated and ordered deterministically.
    gt_map: Dict[str, List[str]] = {}
    for rxn_id, kegg_ids in gt_map_raw.items():
        ids = sorted({str(k).strip() for k in kegg_ids if str(k).strip()})
        if ids:
            gt_map[str(rxn_id)] = ids

    res.diagnostics = {
        "model_id": model_id,
        "num_reactions_total": num_reactions_total,
        "reactions_with_raw_kegg_mention": raw_mentions,
        "reactions_with_extracted_kegg": len(gt_map),
        "unextracted_mentions": max(0, raw_mentions - len(gt_map)),
        "parser_discrepancy": raw_mentions > len(gt_map),
        "species_with_kegg_reaction_annotation": misplaced_mentions,
        "misplaced_annotations_only": raw_mentions == 0 and misplaced_mentions > 0,
        "example_annotation_lines": " || ".join(raw_examples),
    }

    # --- model-level scientific exclusion -------------------------------
    if not gt_map:
        res.status = "excluded_no_ground_truth"
        res.exclusions.append(
            {
                "model_id": model_id,
                "reaction_id": "",
                "exclusion_level": "model",
                "reason": "no_kegg_reaction_annotations",
                "detail": f"{num_reactions_total} reactions parsed; "
                f"{raw_mentions} reaction-level kegg.reaction mentions; "
                f"{misplaced_mentions} species carry a kegg.reaction annotation "
                f"(manifest selected this model by raw file grep)",
            }
        )
        res.summary = {
            "model_id": model_id,
            "status": res.status,
            "num_reactions_total": num_reactions_total,
            "num_ground_truth_reactions": 0,
            "num_eval_reactions": 0,
        }
        return res

    # --- context --------------------------------------------------------
    model_title = model.getName() if model.isSetName() else (model.getId() or "")
    model_notes = _strip_tags(model.getNotesString()) if model.isSetNotes() else ""
    res.context = {
        "model_id": model_id,
        "model_title": model_title,
        "model_notes": model_notes,
        "sbml_level": model.getLevel(),
        "sbml_version": model.getVersion(),
        "num_species": model.getNumSpecies(),
        "num_reactions_total": num_reactions_total,
    }

    # --- equations and SSX ---------------------------------------------
    equations: Dict[str, str] = {}
    for i in range(num_reactions_total):
        rxn = model.getReaction(i)
        equations[str(rxn.getId())] = format_equation(rxn)

    ssx_libsbml = libsbml_ssx_reaction_ids(model)
    ssx_pipeline: Set[str] = set()
    ssx_method = "libsbml"
    try:
        ssx_pipeline = exchange_constraint_skipped_reaction_ids(str(model_file))
        ssx_method = "antimony_pipeline"
    except Exception as exc:
        logger.warning("%s: antimony SSX detection failed (%s); using libSBML", model_id, exc)

    ssx_authoritative = ssx_pipeline if ssx_method == "antimony_pipeline" else ssx_libsbml
    ssx_disagreement = sorted(ssx_libsbml.symmetric_difference(ssx_pipeline)) if ssx_pipeline else []
    res.diagnostics["ssx_detection_method"] = ssx_method
    res.diagnostics["ssx_count_libsbml"] = len(ssx_libsbml)
    res.diagnostics["ssx_count_pipeline"] = len(ssx_pipeline)
    res.diagnostics["ssx_disagreement_ids"] = ";".join(ssx_disagreement[:20])

    # --- species --------------------------------------------------------
    species_df, species_source = extract_species_annotations(model_file)
    for _, row in species_df.iterrows():
        res.species_rows.append({"model_id": model_id, **row.to_dict()})

    # --- optional candidates -------------------------------------------
    cand_by_rxn: Dict[str, List[Dict]] = defaultdict(list)
    if with_candidates and species_source != "none":
        try:
            cdf = run_candidates(model_file, species_df, work_dir)
            if not cdf.empty and {"id", "annotation"}.issubset(cdf.columns):
                score_col = "match_score" if "match_score" in cdf.columns else None
                relax_col = "relax_level" if "relax_level" in cdf.columns else None
                for rxn_id, group in cdf.groupby("id", sort=False):
                    for rank, (_, crow) in enumerate(group.iterrows(), start=1):
                        item = {
                            "candidate_kegg": str(crow.get("annotation", "")),
                            "candidate_rank": rank,
                        }
                        if score_col:
                            item["heuristic_score"] = crow.get(score_col)
                        if relax_col:
                            item["relax_level"] = crow.get(relax_col)
                        cand_by_rxn[str(rxn_id)].append(item)
        except Exception as exc:
            logger.warning("%s: candidate generation failed: %s", model_id, exc)

    # --- reaction rows --------------------------------------------------
    n_eval = 0
    n_ssx = 0
    n_invalid = 0
    for rxn_id in sorted(gt_map):
        gt_all = gt_map[rxn_id]
        gt_valid = [g for g in gt_all if VALID_KEGG_REACTION_RE.match(g)]
        gt_invalid = [g for g in gt_all if not VALID_KEGG_REACTION_RE.match(g)]
        is_ssx = rxn_id in ssx_authoritative

        if not gt_valid:
            reason = "invalid_ground_truth_id"
            n_invalid += 1
            res.exclusions.append(
                {
                    "model_id": model_id,
                    "reaction_id": rxn_id,
                    "exclusion_level": "reaction",
                    "reason": reason,
                    "detail": f"ground truth IDs failed R##### validation: {';'.join(gt_all)}",
                }
            )
        elif is_ssx:
            reason = "exchange_ssx"
            n_ssx += 1
            res.exclusions.append(
                {
                    "model_id": model_id,
                    "reaction_id": rxn_id,
                    "exclusion_level": "reaction",
                    "reason": reason,
                    "detail": "Empty reactant or product side; rule-based matcher skips it "
                    "when include_exchange_reactions=False.",
                }
            )
        else:
            reason = ""
            n_eval += 1

        cands = cand_by_rxn.get(rxn_id, [])
        res.gt_id_set.update(gt_valid)
        res.reaction_rows.append(
            {
                "model_id": model_id,
                "reaction_id": rxn_id,
                "reaction_equation": equations.get(rxn_id, ""),
                "ground_truth_kegg_all": ";".join(gt_valid),
                "ground_truth_kegg_primary": gt_valid[0] if gt_valid else "",
                "num_ground_truth_ids": len(gt_valid),
                "invalid_ground_truth_ids": ";".join(gt_invalid),
                "species_source": species_source,
                "num_species_annotations": len(species_df),
                "is_exchange_ssx": is_ssx,
                "included_in_eval": reason == "",
                "exclusion_reason": reason,
                "num_candidates": len(cands),
                "candidates_json": json.dumps(cands, sort_keys=True) if cands else "",
            }
        )

    res.summary = {
        "model_id": model_id,
        "status": res.status,
        "num_reactions_total": num_reactions_total,
        "num_ground_truth_reactions": len(gt_map),
        "num_eval_reactions": n_eval,
        "num_ssx_excluded": n_ssx,
        "num_invalid_gt_excluded": n_invalid,
        "species_source": species_source,
        "num_species_annotations": len(species_df),
        "zero_eval_reactions": n_eval == 0,
    }
    return res


# ---------------------------------------------------------------------------
# Duplicate clustering
# ---------------------------------------------------------------------------

def cluster_models(
    gt_sets: Dict[str, Set[str]],
    threshold: float,
    containment_threshold: float = DEFAULT_CONTAINMENT_THRESHOLD,
    min_size_ratio: float = DEFAULT_MIN_SIZE_RATIO,
    min_containment_overlap: int = DEFAULT_MIN_CONTAINMENT_OVERLAP,
) -> Tuple[Dict[str, str], pd.DataFrame]:
    """Union-find clustering on overlap of ground-truth KEGG ID sets.

    Two linkage rules are applied, because model variants relate in two ways:

    - **Jaccard** ``|A∩B| / |A∪B|`` catches variants of near-identical scope,
      such as a knockout of an otherwise unchanged model.
    - **Containment** ``|A∩B| / min(|A|,|B|)`` catches variants where one model
      extends another. Jaccard penalises the size difference and would split
      them: the Smallbone2013 yeast variants share 189 reactions, yet score
      only 0.86 by Jaccard because one model adds 23 further reactions.
      Containment is gated on ``min_size_ratio`` because a two-reaction model is
      trivially contained in a genome-scale one; without that guard, unrelated
      models chain together into a single cluster through the large models.

    Cluster IDs are ``CLU_<lexicographically smallest member>`` so they stay
    stable across rebuilds as long as cluster membership is unchanged. Every
    model receives a cluster ID (singletons included) so train/test splitting
    can partition on ``cluster_id`` and keep related variants together.
    """
    model_ids = sorted(gt_sets)
    parent = {m: m for m in model_ids}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            if ra < rb:
                parent[rb] = ra
            else:
                parent[ra] = rb

    pair_rows: List[Dict[str, object]] = []
    for i, a in enumerate(model_ids):
        sa = gt_sets[a]
        if not sa:
            continue
        for b in model_ids[i + 1 :]:
            sb = gt_sets[b]
            if not sb:
                continue
            inter = len(sa & sb)
            if inter == 0:
                continue
            jaccard = inter / len(sa | sb)
            containment = inter / min(len(sa), len(sb))
            size_ratio = min(len(sa), len(sb)) / max(len(sa), len(sb))
            rules = []
            if jaccard >= threshold:
                rules.append("jaccard")
            if (
                containment >= containment_threshold
                and size_ratio >= min_size_ratio
                and inter >= min_containment_overlap
            ):
                rules.append("containment")
            if rules:
                union(a, b)
                pair_rows.append(
                    {
                        "model_a": a,
                        "model_b": b,
                        "jaccard": round(jaccard, 4),
                        "containment": round(containment, 4),
                        "size_ratio": round(size_ratio, 4),
                        "shared_ids": inter,
                        "linkage_rule": "+".join(rules),
                    }
                )

    members: Dict[str, List[str]] = defaultdict(list)
    for m in model_ids:
        members[find(m)].append(m)

    cluster_of: Dict[str, str] = {}
    group_rows: List[Dict[str, object]] = []
    for root in sorted(members):
        group = sorted(members[root])
        cluster_id = f"CLU_{group[0]}"
        for m in group:
            cluster_of[m] = cluster_id
        if len(group) > 1:
            union_ids: Set[str] = set()
            inter_ids: Optional[Set[str]] = None
            for m in group:
                union_ids |= gt_sets[m]
                inter_ids = gt_sets[m] if inter_ids is None else (inter_ids & gt_sets[m])
            group_rows.append(
                {
                    "cluster_id": cluster_id,
                    "group_size": len(group),
                    "model_ids": ";".join(group),
                    "union_ground_truth_ids": len(union_ids),
                    "shared_ground_truth_ids": len(inter_ids or set()),
                    "min_pairwise_jaccard": min(
                        (
                            r["jaccard"]
                            for r in pair_rows
                            if r["model_a"] in group and r["model_b"] in group
                        ),
                        default=None,
                    ),
                    "linkage_rules": "+".join(
                        sorted(
                            {
                                rule
                                for r in pair_rows
                                if r["model_a"] in group and r["model_b"] in group
                                for rule in str(r["linkage_rule"]).split("+")
                            }
                        )
                    ),
                }
            )

    return cluster_of, pd.DataFrame(group_rows)


# ---------------------------------------------------------------------------
# Invariants
# ---------------------------------------------------------------------------

def check_invariants(
    *,
    manifest_ids: List[str],
    registry: Dict[str, Dict[str, object]],
    results: List[ModelResult],
    reactions_df: pd.DataFrame,
    exclusions_df: pd.DataFrame,
    cluster_of: Dict[str, str],
) -> Dict[str, object]:
    checks: List[Dict[str, object]] = []

    def add(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    unique_ids = set(manifest_ids)
    add(
        "manifest_has_75_unique_accessions",
        len(manifest_ids) == 75 and len(unique_ids) == 75,
        f"{len(manifest_ids)} entries, {len(unique_ids)} unique",
    )

    downloaded = [m for m in manifest_ids if (m in registry)]
    with_prov = [
        m
        for m in downloaded
        if registry[m].get("local_sha256") and registry[m].get("download_url")
    ]
    verified = [m for m in downloaded if registry[m].get("checksum_verified")]
    add(
        "every_registry_entry_has_provenance_and_sha256",
        len(with_prov) == len(downloaded) and len(downloaded) > 0,
        f"{len(with_prov)}/{len(downloaded)} registry entries carry URL + local SHA-256",
    )
    add(
        "every_downloaded_file_checksum_verified",
        len(verified) == len(downloaded) and len(downloaded) > 0,
        f"{len(verified)}/{len(downloaded)} match upstream SHA-256",
    )

    parse_failures = [
        r.model_id
        for r in results
        if any(f["failure_type"] == "parse_error" for f in r.pipeline_failures)
    ]
    included_models = [r.model_id for r in results if r.status == "included"]
    add(
        "every_included_model_parses",
        not (set(parse_failures) & set(included_models)),
        f"{len(parse_failures)} parse failures, none among {len(included_models)} included models",
    )

    if reactions_df.empty:
        add("every_included_reaction_has_valid_ground_truth", False, "no reaction rows")
    else:
        inc = reactions_df[reactions_df["included_in_eval"]]
        bad = inc[(inc["num_ground_truth_ids"] < 1) | (inc["ground_truth_kegg_primary"] == "")]
        add(
            "every_included_reaction_has_valid_ground_truth",
            len(bad) == 0,
            f"{len(bad)} of {len(inc)} evaluable reactions lack a valid KEGG ID",
        )

        malformed = inc[
            ~inc["ground_truth_kegg_all"].fillna("").apply(
                lambda s: all(VALID_KEGG_REACTION_RE.match(x) for x in s.split(";") if x)
            )
        ]
        add(
            "all_included_ground_truth_ids_wellformed",
            len(malformed) == 0,
            f"{len(malformed)} evaluable reactions carry a malformed ID",
        )

    # Reconciliation: total ground-truth reactions == evaluable + reaction-level exclusions
    total_gt = len(reactions_df)
    n_eval = int(reactions_df["included_in_eval"].sum()) if not reactions_df.empty else 0
    rxn_excl = (
        exclusions_df[exclusions_df["exclusion_level"] == "reaction"]
        if not exclusions_df.empty
        else pd.DataFrame()
    )
    add(
        "reaction_records_reconcile",
        total_gt == n_eval + len(rxn_excl),
        f"{total_gt} ground-truth reactions = {n_eval} evaluable + {len(rxn_excl)} excluded",
    )

    model_excl = (
        exclusions_df[exclusions_df["exclusion_level"] == "model"]
        if not exclusions_df.empty
        else pd.DataFrame()
    )
    n_pipeline = len({r.model_id for r in results if r.status == "pipeline_failure"})
    accounted = len(included_models) + len(model_excl) + n_pipeline
    add(
        "model_records_reconcile",
        accounted == len(manifest_ids),
        f"{len(manifest_ids)} manifest = {len(included_models)} included "
        f"+ {len(model_excl)} scientifically excluded + {n_pipeline} pipeline failures",
    )

    add(
        "every_included_model_has_cluster_id",
        all(m in cluster_of for m in included_models),
        f"{sum(1 for m in included_models if m in cluster_of)}/{len(included_models)} assigned",
    )

    if not reactions_df.empty:
        dup_keys = reactions_df.duplicated(subset=["model_id", "reaction_id"]).sum()
        add(
            "reaction_keys_unique",
            dup_keys == 0,
            f"{dup_keys} duplicate (model_id, reaction_id) keys",
        )

    return {
        "all_passed": all(c["passed"] for c in checks),
        "num_passed": sum(1 for c in checks if c["passed"]),
        "num_checks": len(checks),
        "checks": checks,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, payload: object) -> None:
    """Write JSON with LF endings on every platform.

    ``Path.write_text`` translates ``\\n`` to ``\\r\\n`` on Windows, which would
    make the recorded SHA-256 digests platform-dependent and stop them matching
    the LF-normalised blobs stored by git.
    """
    text = json.dumps(payload, indent=2) + "\n"
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


def write_csv(df: pd.DataFrame, path: Path, sort_by: Optional[List[str]] = None) -> None:
    """Deterministic CSV: stable sort, fixed line terminator, no index."""
    if sort_by and not df.empty:
        present = [c for c in sort_by if c in df.columns]
        if present:
            df = df.sort_values(present, kind="mergesort").reset_index(drop=True)
    df.to_csv(path, index=False, lineterminator="\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--with-candidates", action="store_true")
    parser.add_argument("--duplicate-threshold", type=float, default=DEFAULT_DUPLICATE_THRESHOLD)
    parser.add_argument(
        "--containment-threshold", type=float, default=DEFAULT_CONTAINMENT_THRESHOLD
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    manifest_ids = load_model_ids(args.manifest)
    registry = load_registry(args.registry)
    model_ids = manifest_ids[: args.limit] if args.limit is not None else manifest_ids
    work_dir = args.work_dir or (args.output_dir / "_work")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results: List[ModelResult] = []
    for i, model_id in enumerate(model_ids, start=1):
        logger.info("[%d/%d] %s", i, len(model_ids), model_id)
        results.append(
            process_model(
                model_id,
                args.models_dir / f"{model_id}.xml",
                registry.get(model_id, {}),
                with_candidates=args.with_candidates,
                work_dir=work_dir,
                registry_available=bool(registry),
            )
        )

    reactions_df = pd.DataFrame([r for res in results for r in res.reaction_rows])
    species_df = pd.DataFrame([r for res in results for r in res.species_rows])
    exclusions_df = pd.DataFrame([e for res in results for e in res.exclusions])
    failure_rows = [f for res in results for f in res.pipeline_failures]
    if not registry:
        # One global row rather than a per-model failure for every accession.
        failure_rows.append(
            {
                "model_id": "",
                "failure_type": "registry_missing",
                "detail": f"No download registry at {_rel(args.registry)}; "
                "provenance and checksums are unverified for this build.",
                "actionable": "Run download_biomodels.py before building the benchmark.",
            }
        )
    failures_df = pd.DataFrame(failure_rows)
    context_df = pd.DataFrame([res.context for res in results if res.context])
    diagnostics_df = pd.DataFrame([res.diagnostics for res in results if res.diagnostics])

    gt_sets = {res.model_id: res.gt_id_set for res in results if res.status == "included"}
    cluster_of, duplicates_df = cluster_models(
        gt_sets, args.duplicate_threshold, args.containment_threshold
    )

    summary_rows = []
    for res in results:
        row = dict(res.summary)
        row["cluster_id"] = cluster_of.get(res.model_id, "")
        row["has_pipeline_failure"] = bool(res.pipeline_failures)
        summary_rows.append(row)
    models_df = pd.DataFrame(summary_rows)

    clusters_df = pd.DataFrame(
        [{"model_id": m, "cluster_id": c} for m, c in sorted(cluster_of.items())]
    )

    write_csv(reactions_df, args.output_dir / "reactions.csv", ["model_id", "reaction_id"])
    write_csv(species_df, args.output_dir / "species_annotations.csv", ["model_id", "species_id", "annotation"])
    write_csv(exclusions_df, args.output_dir / "exclusions.csv", ["exclusion_level", "model_id", "reaction_id"])
    write_csv(failures_df, args.output_dir / "pipeline_failures.csv", ["failure_type", "model_id"])
    write_csv(models_df, args.output_dir / "model_summary.csv", ["model_id"])
    write_csv(context_df, args.output_dir / "model_context.csv", ["model_id"])
    write_csv(diagnostics_df, args.output_dir / "parser_diagnostics.csv", ["model_id"])
    write_csv(duplicates_df, args.output_dir / "duplicate_groups.csv", ["cluster_id"])
    write_csv(clusters_df, args.output_dir / "model_clusters.csv", ["cluster_id", "model_id"])

    included = [r for r in results if r.status == "included"]
    excluded_models = [r for r in results if r.status == "excluded_no_ground_truth"]
    pipeline_failed = [r for r in results if r.status == "pipeline_failure"]
    n_eval = int(reactions_df["included_in_eval"].sum()) if not reactions_df.empty else 0

    summary = {
        "benchmark_version": BENCHMARK_VERSION,
        "with_candidates": args.with_candidates,
        "duplicate_threshold": args.duplicate_threshold,
        "containment_threshold": args.containment_threshold,
        "manifest_models": len(manifest_ids),
        "manifest_unique_accessions": len(set(manifest_ids)),
        "models_processed": len(model_ids),
        "models_included": len(included),
        "models_excluded_no_ground_truth": len(excluded_models),
        "models_pipeline_failure": len(pipeline_failed),
        "models_with_zero_eval_reactions": sum(
            1 for r in included if r.summary.get("zero_eval_reactions")
        ),
        "total_ground_truth_reactions": len(reactions_df),
        "evaluable_reactions": n_eval,
        "reactions_excluded_ssx": sum(int(r.summary.get("num_ssx_excluded", 0)) for r in included),
        "reactions_excluded_invalid_gt": sum(
            int(r.summary.get("num_invalid_gt_excluded", 0)) for r in included
        ),
        "reactions_with_multiple_ground_truth_ids": (
            int((reactions_df["num_ground_truth_ids"] > 1).sum()) if not reactions_df.empty else 0
        ),
        "duplicate_clusters_multi_member": len(duplicates_df),
        "models_in_multi_member_clusters": (
            int(duplicates_df["group_size"].sum()) if not duplicates_df.empty else 0
        ),
        "distinct_clusters": len(set(cluster_of.values())),
        "models_with_parser_discrepancy": (
            int(diagnostics_df["parser_discrepancy"].sum()) if not diagnostics_df.empty else 0
        ),
        "models_with_misplaced_annotations_only": (
            int(diagnostics_df["misplaced_annotations_only"].sum())
            if not diagnostics_df.empty
            else 0
        ),
        "historical_reference": {
            "models": 68,
            "reactions": 4379,
            "note": "Clue only, not an acceptance criterion. Differences are documented "
            "in benchmark/data/RECONCILIATION.md rather than engineered away.",
        },
    }
    write_json(args.output_dir / "benchmark_summary.json", summary)

    invariants = check_invariants(
        manifest_ids=manifest_ids,
        registry=registry,
        results=results,
        reactions_df=reactions_df,
        exclusions_df=exclusions_df,
        cluster_of=cluster_of,
    )
    write_json(args.output_dir / "invariants.json", invariants)

    artifacts = [
        "reactions.csv",
        "model_summary.csv",
        "model_context.csv",
        "model_clusters.csv",
        "exclusions.csv",
        "pipeline_failures.csv",
        "duplicate_groups.csv",
        "species_annotations.csv",
        "parser_diagnostics.csv",
        "benchmark_summary.json",
        "invariants.json",
    ]
    version = {
        "benchmark_version": BENCHMARK_VERSION,
        "manifest_sha256": sha256_of(args.manifest),
        "registry_sha256": sha256_of(args.registry) if args.registry.exists() else None,
        "artifact_sha256": {
            name: sha256_of(args.output_dir / name)
            for name in artifacts
            if (args.output_dir / name).exists()
        },
        "counts": {
            "models_included": summary["models_included"],
            "total_ground_truth_reactions": summary["total_ground_truth_reactions"],
            "evaluable_reactions": summary["evaluable_reactions"],
        },
        "invariants_all_passed": invariants["all_passed"],
    }
    write_json(args.output_dir / "VERSION.json", version)

    logger.info("Summary: %s", json.dumps({k: v for k, v in summary.items() if k != "historical_reference"}))
    logger.info(
        "Invariants: %d/%d passed", invariants["num_passed"], invariants["num_checks"]
    )
    for check in invariants["checks"]:
        if not check["passed"]:
            logger.warning("INVARIANT FAILED %s: %s", check["check"], check["detail"])
    logger.info("reactions.csv sha256=%s", version["artifact_sha256"].get("reactions.csv"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
