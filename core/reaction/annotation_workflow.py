import logging
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, NamedTuple, Optional, Tuple, Union

import pandas as pd

from core.annotation_workflow import _generate_recommendation_table
from core.database_search import _get_kegg_recommendations_rulebased
from core.llm_interface import query_llm
from core.model_info import (
    extract_model_info,
    extract_reactions_from_sbml,
    find_species_with_annotations_and_qualifiers,
    find_species_with_chebi_annotations,
    get_all_reaction_ids,
    map_reaction_ids_to_stoichiometry_strings,
)

from .amendment import LikelihoodCalculator, update_participant_likelihoods
from .matching import map_reactions_to_kegg
from .scoring import SimilarityCalculator
from .amendment_config import CofactorConfig, ConvergenceConfig, MatchingConfig
from .kegg_features import KEGGReactionFeatures, REF_KEGG_REACTION_FEATURES
from .relaxation_workflow import map_reactions_to_kegg_with_relaxation
from .species_probability import init_species_probs_from_dict
from .utils import check_environment, extract_reaction_participants, map_chebi_to_kegg
from utils.constants import DatabaseID, EntityType

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class KeggAnnotationWorkflowResult(NamedTuple):
    """DataFrames from :func:`run_kegg_annotation_workflow_rulebased` (ChEBI→KEGG map, KEGG reaction
    candidates, scored candidates, updated participant likelihoods)."""

    high_score_recommendations: pd.DataFrame
    kegg_recommendations: pd.DataFrame
    scored_reactions: pd.DataFrame
    updated_participants: pd.DataFrame


def run_kegg_annotation_workflow_rulebased(
    model_file: str,
    recommendations_df: pd.DataFrame,
    existing_annotations: Optional[Dict[str, List[str]]] = None,
    kegg_features_file: str = REF_KEGG_REACTION_FEATURES,
    entity_type: str = "reaction",
    database: str = "kegg",
    cofactor_config: Optional[CofactorConfig] = None,
    convergence_config: Optional[ConvergenceConfig] = None,
    matching_config: Optional[MatchingConfig] = None,
    evaluate_candidates: bool = False,
    include_exchange_reactions: bool = False,
) -> Optional[KeggAnnotationWorkflowResult]:
    """Run the complete KEGG annotation workflow.

    Returns:
        KeggAnnotationWorkflowResult with four DataFrames, or ``None`` if the
        environment check fails (see :func:`~core.reaction.utils.check_environment`).
    """
    if cofactor_config is None:
        cofactor_config = CofactorConfig()
    if convergence_config is None:
        convergence_config = ConvergenceConfig()
    if matching_config is None:
        matching_config = MatchingConfig()
    if existing_annotations is None:
        existing_annotations = {}

    if not check_environment(model_file):
        logger.error("Environment check failed. Please fix the issues and try again.")
        return None

    logger.info("Analyzing model: %s", model_file)

    reaction_ids = get_all_reaction_ids(model_file)
    model_info = extract_model_info(model_file, reaction_ids, entity_type)

    logger.info("Step 2: Map ChEBI IDs to KEGG Compound IDs")
    _, high_score_recommendations = map_chebi_to_kegg(recommendations_df)

    logger.info("\nSample of ChEBI to KEGG mapping:")
    if not high_score_recommendations.empty:
        sample_cols = ["id", "display_name", "annotation", "KEGG_ID", "match_score"]
        sample_cols = [c for c in sample_cols if c in high_score_recommendations.columns]
        logger.info(
            high_score_recommendations[
                sample_cols
            ].head()
        )

    logger.info("Step 3: Begin rule-based matching to identify reactions")
    # ``map_chebi_to_kegg`` returns an empty DataFrame with no columns when every
    # species ChEBI term fails ChEBI→KEGG mapping; avoid ``df["id"]`` on schema-less
    # frames.
    if high_score_recommendations.empty or "id" not in high_score_recommendations.columns:
        mapped_species_ids: List[str] = []
        logger.warning(
            "No high-score ChEBI→KEGG compound mappings (species evidence); "
            "skipping relaxation-based reaction matching."
        )
    else:
        mapped_species_ids = list(high_score_recommendations["id"].astype(str).unique())

    reactions, _ = extract_reactions_from_sbml(model_file, mapped_species_ids)

    if high_score_recommendations.empty or "id" not in high_score_recommendations.columns:
        match_results: List[Any] = []
        _species_relax_levels: Dict[str, int] = {}
    else:
        _, match_results, _species_relax_levels = map_reactions_to_kegg_with_relaxation(
            reactions,
            reaction_ids,
            high_score_recommendations,
            spectators=False,
            cofactors_to_ignore=cofactor_config.kegg_ids,
            top_k=None,
            evaluate_candidates=bool(evaluate_candidates),
            include_exchange_reactions=bool(include_exchange_reactions),
        )

    kegg_recommendations_df = _generate_recommendation_table(
        model_file,
        match_results,
        existing_annotations,
        model_info,
        entity_type,
        database,
        {},
    )

    if not evaluate_candidates:
        logger.info("Generation-only mode: skipping scoring/participant update steps.")
        return KeggAnnotationWorkflowResult(
            high_score_recommendations=high_score_recommendations,
            kegg_recommendations=kegg_recommendations_df,
            scored_reactions=pd.DataFrame(),
            updated_participants=pd.DataFrame(),
        )

    kegg_recommendations_df["match_score_norm"] = (
        kegg_recommendations_df["match_score"]
        / kegg_recommendations_df.groupby("id")["match_score"].transform("sum")
    )

    reaction_participants = extract_reaction_participants(model_info, recommendations_df)

    kegg_features = KEGGReactionFeatures.load_from_file(kegg_features_file)

    kegg_recommendations_df["participants"] = kegg_recommendations_df["annotation"].apply(
        kegg_features.get_participants
    )
    kegg_recommendations_df["participant_ids"] = kegg_recommendations_df["annotation"].apply(
        kegg_features.get_participant_ids
    )

    merged_participants = kegg_recommendations_df.groupby("id")["participants"].agg("; ".join)
    counters = merged_participants.apply(
        lambda s: Counter(p.strip() for p in s.split(";") if p.strip())
    )

    similarity_calc = SimilarityCalculator(matching_config)
    init_probs = init_species_probs_from_dict(
        reaction_participants, counters, similarity_calc.is_plausible_match
    )

    likelihood_calc = LikelihoodCalculator(cofactor_config, matching_config, convergence_config)
    scored_df = likelihood_calc.compute_reaction_likelihoods(init_probs, kegg_recommendations_df)

    updated_participants_df, _ = update_participant_likelihoods(
        high_score_recommendations,
        scored_df,
        model_file,
        model_info=model_info,
        kegg_features=kegg_features,
        reactions=reactions,
        reaction_ids=reaction_ids,
        entity_type=entity_type,
        database=database,
        cofactor_config=cofactor_config,
        convergence_config=convergence_config,
    )

    logger.info("\nSample of participants with updated likelihoods after convergence:")
    if not updated_participants_df.empty:
        logger.info(
            updated_participants_df[["id", "display_name", "KEGG_ID", "participant_likelihood"]].head()
        )

    updated_participants_df.sort_values(by="participant_likelihood", ascending=False, inplace=True)
    scored_df.sort_values(by="likelihood", ascending=False, inplace=True)

    logger.info("KEGG annotation workflow completed successfully.")
    return KeggAnnotationWorkflowResult(
        high_score_recommendations=high_score_recommendations,
        kegg_recommendations=kegg_recommendations_df,
        scored_reactions=scored_df,
        updated_participants=updated_participants_df,
    )


# ---------------------------------------------------------------------------
# LLM-based ranking of rule-based KEGG annotation candidates
# ---------------------------------------------------------------------------

def _strip_kegg_prefix(raw) -> str:
    """Return the bare KEGG id (e.g. ``R01600``) from values like ``KEGG:R01600``."""
    if raw is None or (isinstance(raw, float) and raw != raw):
        return ""
    s = str(raw).strip()
    if not s or s.lower() == "nan":
        return ""
    if s.upper().startswith("KEGG:"):
        return s[5:].strip()
    return s


def _build_reaction_annotation_choices(
    sub_df: pd.DataFrame,
    allowed_kegg_ids: Optional[set] = None,
) -> str:
    """Build a newline-separated string of ``R#####: <definition>`` for an LLM prompt.

    If *allowed_kegg_ids* is given, only candidates whose KEGG id is in that
    set are emitted (used to restrict the prompt to the IUBMB-leaf
    representatives selected by :func:`_group_candidates_by_iubmb`).
    """
    lines: list[str] = []
    seen: set[tuple[str, str]] = set()
    for _, row in sub_df.iterrows():
        rid = _strip_kegg_prefix(row.get("annotation", ""))
        if not rid:
            continue
        if allowed_kegg_ids is not None and rid not in allowed_kegg_ids:
            continue
        definition = row.get("reaction_definition", "")
        if definition is None or (isinstance(definition, float) and definition != definition):
            definition = ""
        else:
            definition = str(definition).strip()
        key = (rid, definition)
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"{rid}: {definition}")
    return "\n".join(lines)


def _group_candidates_by_iubmb(
    ordered_kegg_ids: List[str],
    kegg_features: "KEGGReactionFeatures",
) -> Tuple[Dict[str, List[str]], Dict[str, str], List[str]]:
    """Cluster KEGG reaction candidates by their IUBMB BRITE hierarchy.

    Uses the per-reaction IUBMB hierarchy block (``[BR:br08202]``) to figure
    out which candidates collapse together. Two candidates land in the same
    group when their (ancestors ∪ {self}) sets overlap — a transitive closure
    that connects e.g. ``R00299`` and ``R01786`` because both list ``R02848``
    as their IUBMB parent under EC 2.7.1.1.

    For each group we report:
        * ``brite_group_members`` — every candidate that ended up in the group,
          in input order. Members are the only candidates that share an IUBMB
          subtree; reactions that lack a ``br08202`` block (e.g. ``R10049``)
          stay in their own singleton group and are never collapsed with
          biochemically distinct neighbors.
        * ``brite_iubmb_parent`` — the topmost element of the group's
          ancestor closure. This is whichever R-number sits above every other
          chain element when we read each candidate's IUBMB subblock as a
          root-to-self chain. The parent may itself be a candidate
          (e.g. ``R01068`` for the aldolase pair) or a non-candidate
          (e.g. ``R02848`` for the hexokinase pair when only ``R00299``
          and ``R01786`` are in the input).
        * Leaves — candidates that no other candidate has as an ancestor.
          These are the "most specific" entries in the candidate set and are
          the ones we hand to the LLM.

    Args:
        ordered_kegg_ids: Bare KEGG reaction ids (e.g. ``["R00299", "R01600"]``)
            in display order. Order matters for stable output ordering only.
        kegg_features: Loaded :class:`KEGGReactionFeatures` instance.

    Returns:
        ``(kid_to_members, kid_to_iubmb_parent, leaves_in_order)``:
            * ``kid_to_members[kid]`` — ordered member list of *kid*'s group.
            * ``kid_to_iubmb_parent[kid]`` — IUBMB parent R-number for *kid*'s
              group (empty string if neither *kid* nor any group-mate has a
              ``br08202`` block).
            * ``leaves_in_order`` — group leaves across all groups, in input
              order. These are the candidates the LLM should rank.
    """
    if not ordered_kegg_ids:
        return {}, {}, []

    cand_to_chains: Dict[str, Tuple[Tuple[str, ...], ...]] = {
        kid: kegg_features.get_iubmb_chains(f"KEGG:{kid}")
        for kid in ordered_kegg_ids
    }
    cand_to_ancestors: Dict[str, FrozenSet[str]] = {
        kid: kegg_features.get_iubmb_ancestors(f"KEGG:{kid}")
        for kid in ordered_kegg_ids
    }
    cand_to_chain_set: Dict[str, set] = {
        kid: set(cand_to_ancestors[kid]) | {kid} for kid in ordered_kegg_ids
    }

    parent: Dict[str, str] = {c: c for c in ordered_kegg_ids}

    def _find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(a: str, b: str) -> None:
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[rb] = ra

    elem_to_candidates: Dict[str, List[str]] = {}
    for kid in ordered_kegg_ids:
        for elem in cand_to_chain_set[kid]:
            elem_to_candidates.setdefault(elem, []).append(kid)
    for cands_sharing in elem_to_candidates.values():
        first = cands_sharing[0]
        for other in cands_sharing[1:]:
            _union(first, other)

    group_to_members: Dict[str, List[str]] = {}
    for c in ordered_kegg_ids:
        root = _find(c)
        group_to_members.setdefault(root, []).append(c)

    kid_to_members: Dict[str, List[str]] = {}
    kid_to_iubmb_parent: Dict[str, str] = {}
    is_leaf: Dict[str, bool] = {c: True for c in ordered_kegg_ids}

    first_idx = {kid: i for i, kid in enumerate(ordered_kegg_ids)}

    for members in group_to_members.values():
        # Build the closure of every chain element seen across this group's
        # candidates and an `ancestor_of` relation (a, b) meaning "a appears
        # before b in some candidate's IUBMB chain".
        closure: set = set()
        ancestor_of: set = set()
        elem_first_seen: Dict[str, int] = {}
        for m in members:
            for chain_idx, chain in enumerate(cand_to_chains[m]):
                for pos, node in enumerate(chain):
                    if node not in elem_first_seen:
                        elem_first_seen[node] = (
                            first_idx[m] * 10000 + chain_idx * 100 + pos
                        )
                    closure.add(node)
                    for descendant in chain[pos + 1 :]:
                        ancestor_of.add((node, descendant))
            closure.add(m)

        # IUBMB parent: a closure element with no element above it in the
        # closure. Tie-break by earliest appearance across all chains.
        topmost = [
            e for e in closure
            if not any((other, e) in ancestor_of for other in closure if other != e)
        ]
        if topmost:
            iubmb_parent = min(topmost, key=lambda e: elem_first_seen.get(e, 10**9))
        else:
            iubmb_parent = ""

        # Leaves: candidates that no other candidate has as an ancestor.
        ancestor_union: set = set()
        for m in members:
            ancestor_union |= cand_to_ancestors[m]
        for m in members:
            if m in ancestor_union:
                is_leaf[m] = False
            kid_to_members[m] = members
            kid_to_iubmb_parent[m] = iubmb_parent

    leaves_in_order = [c for c in ordered_kegg_ids if is_leaf.get(c, True)]
    return kid_to_members, kid_to_iubmb_parent, leaves_in_order


def rank_kegg_annotations_with_llm(
    model_file: str,
    recommendations_df: pd.DataFrame,
    llm_model: str = "Llama-3.3-70B-Instruct",
    kegg_features_file: str = REF_KEGG_REACTION_FEATURES,
    top_k: int = 10,
    csv_path: Optional[str] = None,
    only_rank_meaningful = True,
) -> pd.DataFrame:
    """Re-rank KEGG reaction candidates using an LLM and return a filtered DataFrame.

    For each reaction in *model_file* that has candidate annotations in
    *recommendations_df*, candidates are first collapsed by their IUBMB
    BRITE hierarchy: variants like ``R00299`` (D-Glucose hexokinase) and
    ``R01786`` (alpha-D-Glucose hexokinase) — both children of IUBMB parent
    ``R02848`` — are grouped together and only the leaves of each group
    (the most-specific entries in the candidate set) are sent to the LLM.
    The LLM is asked to select the best-matching KEGG ids, and the returned
    DataFrame contains the LLM-selected rows in ranked order.

    Args:
        model_file: Path to the SBML model file.
        recommendations_df: Recommendations table (as produced by
            ``annotate_model`` or loaded from its CSV output).
        llm_model: LLM model identifier forwarded to :func:`query_llm`.
        kegg_features_file: Path to the KEGG reaction features lzma file.
        top_k: Maximum number of KEGG ids to keep per reaction from the LLM
            response.
        csv_path: If given, the enriched recommendations table — augmented
            with ``reaction_definition``, ``brite_group_members`` (every
            candidate folded into the same IUBMB group, semicolon-joined)
            and ``brite_iubmb_parent`` (the topmost IUBMB ancestor of the
            group) — is saved to this path before ranking begins.

    Returns:
        A DataFrame filtered to the LLM-selected KEGG ids, in ranked order.
        Carries the ``reaction_definition``, ``brite_group_members`` and
        ``brite_iubmb_parent`` columns. A copy is also saved to
        ``<csv_stem>_llm_ranked.csv`` next to *csv_path* (or next to
        *model_file* if *csv_path* is not provided).
    """
    from utils.constants import REACTION_ANNOTATION_RANKING_PROMPT

    result_df = recommendations_df.copy()

    kegg_features = KEGGReactionFeatures.load_from_file(kegg_features_file)
    # Some models may yield zero candidates (or an unexpected recommendations_df shape).
    # In that case, return an empty ranked dataframe instead of failing.
    if result_df.empty or "annotation" not in result_df.columns or "id" not in result_df.columns:
        if "reaction_definition" not in result_df.columns:
            result_df["reaction_definition"] = pd.Series(dtype="object")

        base = Path(csv_path) if csv_path else Path(f"{Path(model_file).name}_recommendations")
        ranked_out_path = base.with_name(base.stem + "_llm_ranked.csv")
        result_df.iloc[0:0].copy().to_csv(ranked_out_path, index=False)
        logger.info("LLM-ranked recommendations saved to %s", ranked_out_path)
        return result_df.iloc[0:0].copy()

    result_df["reaction_definition"] = result_df["annotation"].map(kegg_features.get_definition)

    # Cluster candidates by their IUBMB BRITE hierarchy (``[BR:br08202]``),
    # per reaction. Each row gets:
    #   * ``brite_group_members`` — semicolon-joined list of every candidate
    #     in the IUBMB group (so the user can see which KEGG entries we
    #     collapsed away before LLM ranking).
    #   * ``brite_iubmb_parent`` — the topmost IUBMB ancestor of the group
    #     (e.g. ``R02848`` for hexokinase variants, ``R01068`` for the
    #     aldolase pair). Empty if no group member has a ``br08202`` block.
    # Only the leaves of each IUBMB group — the most-specific entries the
    # rule-based pipeline produced — are forwarded to the LLM.
    brite_kid_to_members: Dict[str, Dict[str, List[str]]] = {}
    brite_kid_to_parent: Dict[str, Dict[str, str]] = {}
    brite_leaves: Dict[str, List[str]] = {}
    for reaction_id, sub in result_df.groupby("id", sort=False):
        ordered: List[str] = []
        seen_kids: set = set()
        for ann in sub["annotation"].tolist():
            kid = _strip_kegg_prefix(ann)
            if kid and kid not in seen_kids:
                ordered.append(kid)
                seen_kids.add(kid)
        kid_to_members, kid_to_parent, leaves = _group_candidates_by_iubmb(
            ordered, kegg_features
        )
        brite_kid_to_members[str(reaction_id)] = kid_to_members
        brite_kid_to_parent[str(reaction_id)] = kid_to_parent
        brite_leaves[str(reaction_id)] = leaves

    def _members_for_row(row: pd.Series) -> str:
        rxn_id = str(row.get("id", ""))
        kid = _strip_kegg_prefix(row.get("annotation", ""))
        if not kid:
            return ""
        members = brite_kid_to_members.get(rxn_id, {}).get(kid, [kid])
        return ";".join(members)

    def _parent_for_row(row: pd.Series) -> str:
        rxn_id = str(row.get("id", ""))
        kid = _strip_kegg_prefix(row.get("annotation", ""))
        if not kid:
            return ""
        return brite_kid_to_parent.get(rxn_id, {}).get(kid, "")

    result_df["brite_group_members"] = result_df.apply(_members_for_row, axis=1)
    result_df["brite_iubmb_parent"] = result_df.apply(_parent_for_row, axis=1)

    if csv_path is not None:
        result_df.to_csv(csv_path, index=False)
        logger.info(
            "%s updated with KEGG DEFINITIONs and BRITE IUBMB hierarchy columns",
            csv_path,
        )

    reaction_ids = get_all_reaction_ids(model_file)
    id_to_equation = map_reaction_ids_to_stoichiometry_strings(model_file)
    model_context = "\n".join(id_to_equation.get(rid, str(rid)) for rid in reaction_ids)

    if only_rank_meaningful:
        prompt = f"""
        You are evaluating whether a biochemical reaction network uses biologically meaningful reaction and species names.

Criteria for YES:

* At least some species names correspond to recognizable biochemical entities, metabolites, macromolecules, or chemical abbreviations (e.g. ATP, ADP, IMP, DNA, PRPP, glucose).
* Reactions resemble plausible biochemical transformations or transport/degradation processes.
* The model appears interpretable by a human familiar with biochemistry.

Criteria for NO:

* Species and reactions are anonymous placeholders or opaque IDs (e.g. s1, c03493, x103, R13842, etc.) with no biochemical meaning.
* Reactions cannot be interpreted biologically from the names alone.
* The network appears synthetically generated or unlabeled.

Return ONLY:
TRUE
or
FALSE

Model:
{model_context}

        """
        response_text = query_llm(prompt, model=llm_model, entity_type=EntityType.REACTION)
        logger.info("LLM meaningful evaluation: %s", response_text)
        meaningful = response_text.strip().lower() == "true"
    else:
        meaningful = False

    ranked_reaction_ids: list[str] = []
    ranked_responses: list[list[str]] = []

    for reaction_id in reaction_ids:
        model_reaction = id_to_equation.get(reaction_id, reaction_id)
        logger.info("Ranking candidates for %s", model_reaction)

        sub = result_df[result_df["id"] == reaction_id]
        # Show the LLM only the IUBMB-leaf candidates — the most-specific
        # entries within each BRITE group. Non-leaf members (e.g. R00299 when
        # R01786 is also a candidate) are still recorded per row in
        # `brite_group_members` for the final output.
        leaf_ids = set(brite_leaves.get(str(reaction_id), []))
        reaction_annotation_choices = _build_reaction_annotation_choices(
            sub, allowed_kegg_ids=leaf_ids if leaf_ids else None
        )
        if not reaction_annotation_choices.strip():
            continue

        if meaningful:
            prompt = REACTION_ANNOTATION_RANKING_PROMPT.format(
                model_context=model_context,
                model_reaction=model_reaction,
                reaction_annotation_choices=reaction_annotation_choices,
            )
        else:
            prompt = REACTION_ANNOTATION_RANKING_PROMPT.format(
                model_context="",
                model_reaction=model_reaction,
                reaction_annotation_choices=reaction_annotation_choices,
            )


        response_text = query_llm(prompt, model=llm_model, entity_type=EntityType.REACTION)
        response_lines = [ln.strip() for ln in (response_text or "").splitlines() if ln.strip()]

        logger.info("%s -> %s", reaction_id, response_lines)

        if len(response_lines) == 1 and response_lines[0] == "UNK":
            continue

        ranked_reaction_ids.append(reaction_id)
        ranked_responses.append(response_lines[:top_k])

    logger.info("Collected LLM rankings for %d reactions", len(ranked_responses))

    ranked_rows: list[pd.DataFrame] = []
    for reaction_id, kegg_ids in zip(ranked_reaction_ids, ranked_responses):
        for kegg_id in kegg_ids:
            if not kegg_id:
                continue
            mask = (
                (result_df["id"] == reaction_id)
                & (result_df["annotation"].astype(str).str.upper() == f"KEGG:{kegg_id}".upper())
            )
            rows = result_df[mask]
            if rows.empty:
                continue
            ranked_rows.append(rows.iloc[[0]])

    ranked_df = pd.concat(ranked_rows, ignore_index=True) if ranked_rows else result_df.iloc[0:0].copy()

    base = Path(csv_path) if csv_path else Path(f"{Path(model_file).name}_recommendations")
    ranked_out_path = base.with_name(base.stem + "_llm_ranked.csv")
    ranked_df.to_csv(ranked_out_path, index=False)
    logger.info("LLM-ranked recommendations saved to %s", ranked_out_path)

    return ranked_df


# ---------------------------------------------------------------------------
# Reaction-KEGG curation entry point (no LLM)
# ---------------------------------------------------------------------------

def curate_reactions_kegg_rulebased(
    model_file: str,
    existing_annotations: Dict[str, List[str]],
    qualifier_annotations: Dict[str, Dict[str, str]],
    specs_to_evaluate: List[str],
    *,
    evaluate_candidates: bool = False,
    include_exchange_reactions: bool = False,
    llm_model: str = "",
    top_k: Optional[int] = None,
    tax_id: Optional[str] = None,
    start_time: Optional[float] = None,
) -> Union["AnnotationResult", Tuple[pd.DataFrame, Dict[str, Any]]]:
    """Curate KEGG reaction annotations using the rule-based pipeline.

    Tries species ChEBI annotations first (full ChEBI->KEGG mapping + ontology
    relaxation via :func:`run_kegg_annotation_workflow_rulebased`). If no ChEBI
    species annotations are present, falls back to species annotated directly
    with KEGG compound IDs (no ontology relaxation), feeding the species->KEGG
    table straight into :func:`_get_kegg_recommendations_rulebased`.

    The result is filtered to ``specs_to_evaluate`` (the curation target set),
    saved as ``<model_stem>_recommendations.csv``, and wrapped in an
    :class:`AnnotationResult` with an empty LLM conversation (this path is
    LLM-free).

    Returns an :class:`AnnotationResult` on success, or an
    ``(empty_df, {"error": ...})`` tuple on early-exit error cases (to match
    the error-return convention of :func:`curate_single_model`).
    """
    from core.feedback import AnnotationResult, build_initial_conversation

    if start_time is None:
        start_time = time.time()

    species_chebi = find_species_with_chebi_annotations(model_file)
    kegg_recommendations_df: Optional[pd.DataFrame] = None

    if species_chebi:
        # Preferred path: ChEBI species annotations drive the full ChEBI->KEGG
        # mapping + ontology relaxation pipeline.
        rows = [
            {"id": str(sid), "annotation": str(chebi), "match_score": 1.0}
            for sid, chebis in species_chebi.items()
            for chebi in chebis
            if chebi
        ]
        species_recommendations_df = pd.DataFrame(rows)
        if species_recommendations_df.empty:
            return pd.DataFrame(), {"error": "Empty species ChEBI table"}

        result = run_kegg_annotation_workflow_rulebased(
            model_file=model_file,
            recommendations_df=species_recommendations_df,
            existing_annotations=existing_annotations,
            evaluate_candidates=bool(evaluate_candidates),
            include_exchange_reactions=bool(include_exchange_reactions),
        )
        if result is None:
            return pd.DataFrame(), {"error": "Rulebased KEGG reaction curation failed"}
        kegg_recommendations_df = result.kegg_recommendations
    else:
        # Fallback: species annotated with KEGG compound IDs directly.
        # ChEBI-hierarchy relaxation isn't applicable, so we bypass
        # ``map_reactions_to_kegg_with_relaxation`` and feed a direct
        # species->KEGG mapping into the rule-based candidate generator.
        species_kegg, _ = find_species_with_annotations_and_qualifiers(
            model_file, DatabaseID.KEGG.value
        )
        if not species_kegg:
            logger.warning(
                "No existing ChEBI or KEGG-compound species annotations found; "
                "cannot run rulebased KEGG reaction curation."
            )
            return pd.DataFrame(), {
                "error": "No existing ChEBI or KEGG-compound species annotations found"
            }

        logger.info(
            "No ChEBI species annotations found; falling back to existing KEGG "
            "compound species annotations (%d species).",
            len(species_kegg),
        )

        id_rows = [
            {"id": str(sid), "KEGG_ID": str(kegg_id)}
            for sid, kegg_ids in species_kegg.items()
            for kegg_id in kegg_ids
            if kegg_id
        ]
        id_df = pd.DataFrame(id_rows)
        if id_df.empty:
            return pd.DataFrame(), {"error": "Empty species KEGG compound table"}

        reaction_ids = get_all_reaction_ids(model_file)
        rxn_list, _ = extract_reactions_from_sbml(model_file, list(id_df["id"].unique()))

        normalized_reactions = map_reactions_to_kegg(
            rxn_list, reaction_ids, id_df, spectators=False
        )

        cofactor_config = CofactorConfig()
        match_results = _get_kegg_recommendations_rulebased(
            normalized_reactions,
            cofactors_to_ignore=cofactor_config.kegg_ids,
            top_k=None,
            spectators=False,
            evaluate_candidates=bool(evaluate_candidates),
            include_exchange_reactions=bool(include_exchange_reactions),
        )

        rxn_model_info = extract_model_info(model_file, reaction_ids, EntityType.REACTION)
        kegg_recommendations_df = _generate_recommendation_table(
            model_file,
            match_results,
            existing_annotations,
            rxn_model_info,
            EntityType.REACTION.value,
            DatabaseID.KEGG.value,
            {},
        )

    # Filter output down to reactions that had existing KEGG annotations (curation target set).
    kegg_recommendations_df = kegg_recommendations_df[
        kegg_recommendations_df["id"].astype(str).isin(set(map(str, specs_to_evaluate)))
    ].copy()

    total_time = time.time() - start_time
    has_prediction = kegg_recommendations_df["annotation"].astype(str).str.strip() != ""
    metrics = {
        "total_time": total_time,
        "total_entities": len(specs_to_evaluate),
        "entities_with_predictions": int(has_prediction.sum()),
        "annotation_rate": float(has_prediction.mean() if len(kegg_recommendations_df) else 0.0),
    }

    csv_path = f"{Path(model_file).name}_recommendations.csv"
    kegg_recommendations_df.to_csv(csv_path, index=False)
    print(f"Recommendations saved to {csv_path}")
    logger.info(
        "Curation completed in %.2fs – %d recommendations",
        total_time,
        len(kegg_recommendations_df),
    )

    return AnnotationResult(
        kegg_recommendations_df,
        metrics,
        model_file=model_file,
        conversation_history=build_initial_conversation("", "", ""),
        entities_to_evaluate=specs_to_evaluate,
        entity_type=EntityType.REACTION,
        database=DatabaseID.KEGG,
        method="rulebased",
        llm_model=llm_model,
        top_k=top_k,
        tax_id=tax_id,
        existing_annotations=existing_annotations,
        qualifier_annotations=qualifier_annotations,
        model_info=extract_model_info(model_file, specs_to_evaluate, EntityType.REACTION),
        csv_path=csv_path,
    )
