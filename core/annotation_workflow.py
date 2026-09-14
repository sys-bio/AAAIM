"""
Annotation Workflow for AAAIM

Main interface for annotating a single model that has no or limited existing annotations.
Provides the primary function that users will call to get recommendation tables
for all species in a model.
"""

import logging
import time
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from core.data_types import Recommendation
from core.database_search import (
    extract_classifications,
    get_species_recommendations_direct,
    get_species_recommendations_rag,
    load_chebi_label_dict,
    load_kegg_label_dict,
    load_ncbigene_label_dict,
    load_uniprot_label_dict,
)
from core.feedback import AnnotationResult, build_initial_conversation
from core.llm_interface import (
    format_component_curated_name,
    get_system_prompt,
    parse_llm_response,
    query_llm,
    query_llm_message,
    summarize_usage_records,
)
from utils.constants import (
    SPECIES_ANNOTATION_RANKING_PROMPT,
    DatabaseID,
    EntityType,
    get_database_for_entity_type,
)
from core.model_info import (
    extract_model_info,
    find_reactions_with_kegg_annotations,
    find_species_with_annotations_and_qualifiers,
    format_prompt,
    get_all_reaction_ids,
    get_all_species_ids,
    get_species_display_names,
)


logger = logging.getLogger(__name__)

# Suppress pandas FutureWarning noise (e.g., concat dtype changes)
warnings.filterwarnings("ignore", category=FutureWarning)

SUPPORTED_SEARCH_DATABASES = {
    DatabaseID.CHEBI.value,
    DatabaseID.NCBIGENE.value,
    DatabaseID.UNIPROT.value,
    DatabaseID.KEGG.value,
}


def _normalize_entity_type(entity_type: str | EntityType) -> EntityType:
    if isinstance(entity_type, EntityType):
        return entity_type
    try:
        return EntityType(entity_type)
    except ValueError:
        logger.warning(f"Unknown entity type {entity_type}, using chemical")
        return EntityType.CHEMICAL


def _normalize_databases(
    database: str | DatabaseID | Sequence[str | DatabaseID],
) -> List[DatabaseID]:
    """Normalize database input to a non-empty list of DatabaseID enums."""
    items = database if isinstance(database, (list, tuple)) else [database]
    databases: List[DatabaseID] = []
    for item in items:
        parsed = None
        if isinstance(item, DatabaseID):
            parsed = item
        elif isinstance(item, str):
            try:
                parsed = DatabaseID(item.lower())
            except ValueError:
                logger.warning(f"Unknown database {item}, skipping")
        else:
            logger.warning(f"Unknown database {item}, skipping")
        if parsed is not None and parsed not in databases:
            databases.append(parsed)
    if not databases:
        logger.warning("No valid databases provided, using chebi")
        return [DatabaseID.CHEBI]
    return databases


def _database_log_label(databases: List[DatabaseID]) -> str:
    return ", ".join(db.value for db in databases)


def _empty_recommendation(species_id: str, synonyms_dict: Dict[str, List[str]]) -> Recommendation:
    return Recommendation(
        id=species_id,
        synonyms=synonyms_dict.get(species_id, []),
        candidates=[],
        candidate_names=[],
        match_score=[],
    )


def _search_one_database(
    species_list: List[str],
    synonyms_dict: Dict[str, List[str]],
    database: str,
    method: str,
    top_k: int,
    tax_id: str = None,
    model_info: Dict[str, Any] = None,
    model_type: str = None,
) -> List[Recommendation]:
    """Search a single database and return Recommendation objects."""
    if database not in SUPPORTED_SEARCH_DATABASES:
        logger.error(f"Database {database} not yet supported")
        return []

    if method == "direct":
        return get_species_recommendations_direct(
            species_list, synonyms_dict, database=database, tax_id=tax_id, top_k=top_k
        )
    if method == "rag":
        kwargs: Dict[str, Any] = {"database": database, "tax_id": tax_id, "top_k": top_k}
        if model_type:
            kwargs["model_type"] = model_type
        if database == DatabaseID.KEGG.value and model_info:
            reaction_definitions = [i.split(":")[1] for i in model_info.get("reactions", [])]
            kwargs["reaction_participants"] = [
                extract_classifications(i, "definition") for i in reaction_definitions
            ]
        return get_species_recommendations_rag(species_list, synonyms_dict, **kwargs)

    logger.error(f"Invalid method: {method}")
    return []


def _extend_search_hits(
    species_id: str,
    recs: List[Recommendation],
    database: str,
    all_candidates: List[str],
    all_candidate_names: List[str],
    all_scores: List[float],
    candidate_databases: Dict[Tuple[str, str], str],
) -> None:
    """Append one-database hits onto a complex recommendation."""
    for rec in recs:
        if rec.id != species_id:
            continue
        all_candidates.extend(rec.candidates)
        all_candidate_names.extend(rec.candidate_names)
        all_scores.extend(rec.match_score)
        for candidate in rec.candidates:
            candidate_databases[(species_id, candidate)] = database


def _search_complex_species(
    species_id: str,
    synonyms_dict: Dict[str, List[str]],
    allowed_names: List[str],
    method: str,
    top_k: int,
    tax_id: str = None,
    model_info: Dict[str, Any] = None,
    components: Optional[List[Tuple[str, List[str]]]] = None,
    model_type: str = None,
) -> Tuple[Recommendation, Dict[Tuple[str, str], str]]:
    """Search a complex: per-component DB if typed, else every allowed database."""
    candidate_databases: Dict[Tuple[str, str], str] = {}
    all_candidates: List[str] = []
    all_candidate_names: List[str] = []
    all_scores: List[float] = []

    if components:
        for comp_type, names in components:
            db = get_database_for_entity_type(comp_type, allowed_names)
            if db is None:
                logger.warning(
                    f"No database for complex component type '{comp_type}' "
                    f"in {allowed_names} ({species_id})"
                )
                continue
            recs = _search_one_database(
                [species_id], {species_id: names}, db, method, top_k,
                tax_id, model_info, model_type,
            )
            _extend_search_hits(
                species_id, recs, db,
                all_candidates, all_candidate_names, all_scores, candidate_databases,
            )
    else:
        for db in allowed_names:
            recs = _search_one_database(
                [species_id], synonyms_dict, db, method, top_k,
                tax_id, model_info, model_type,
            )
            _extend_search_hits(
                species_id, recs, db,
                all_candidates, all_candidate_names, all_scores, candidate_databases,
            )

    return Recommendation(
        id=species_id,
        synonyms=synonyms_dict.get(species_id, []),
        candidates=all_candidates,
        candidate_names=all_candidate_names,
        match_score=all_scores,
    ), candidate_databases


def _search_databases(
    entities: List[str],
    synonyms_dict: Dict[str, List[str]],
    entity_type: EntityType,
    databases: List[DatabaseID],
    method: str,
    top_k: int,
    tax_id: str = None,
    entity_type_dict: Optional[Dict[str, str]] = None,
    model_info: Dict[str, Any] = None,
    component_dict: Optional[Dict[str, List[Tuple[str, List[str]]]]] = None,
) -> Tuple[List[Recommendation], Dict[str, str], Dict[Tuple[str, str], str]]:
    """Search the appropriate database(s) for each entity.

    Returns:
        recommendations, species_id -> database name, (species_id, candidate) -> database name
    """
    species_database: Dict[str, str] = {}
    candidate_databases: Dict[Tuple[str, str], str] = {}
    allowed_names = [db.value for db in databases]
    component_dict = component_dict or {}

    if entity_type == EntityType.AUTO:
        logger.info(f">>>Step 4: Searching databases {_database_log_label(databases)}...<<<")
        entity_type_dict = entity_type_dict or {}
        species_by_type: Dict[str, List[str]] = {}
        for species_id in entities:
            detected_type = entity_type_dict.get(species_id, "unknown")
            species_by_type.setdefault(detected_type, []).append(species_id)

        logger.info(
            f"Detected entity types: {dict((k, len(v)) for k, v in species_by_type.items())}"
        )

        all_recommendations: List[Recommendation] = []
        for detected_type, species_list in species_by_type.items():
            if detected_type == "unknown":
                logger.warning(
                    f"There are {len(species_list)} species with unknown entity type: {species_list}"
                )
                for species_id in species_list:
                    all_recommendations.append(_empty_recommendation(species_id, synonyms_dict))
                continue

            if detected_type == "complex":
                typed = sum(1 for sid in species_list if component_dict.get(sid))
                logger.info(
                    f"Searching {len(species_list)} complex entities "
                    f"({typed} with per-component types) in {allowed_names}"
                )
                for species_id in species_list:
                    rec, cand_dbs = _search_complex_species(
                        species_id, synonyms_dict, allowed_names, method, top_k,
                        tax_id, model_info, component_dict.get(species_id),
                    )
                    candidate_databases.update(cand_dbs)
                    all_recommendations.append(rec)
                continue

            target_database = get_database_for_entity_type(detected_type, allowed_names)
            if target_database is None:
                logger.warning(
                    f"No valid database found for entity type '{detected_type}' "
                    f"in {allowed_names} for {len(species_list)} species"
                )
                for species_id in species_list:
                    all_recommendations.append(_empty_recommendation(species_id, synonyms_dict))
                continue

            logger.info(
                f"Searching {target_database} for {len(species_list)} {detected_type} entities"
            )
            group_recs = _search_one_database(
                species_list, synonyms_dict, target_database, method, top_k, tax_id, model_info
            )
            for rec in group_recs:
                species_database[rec.id] = target_database
            all_recommendations.extend(group_recs)

        return all_recommendations, species_database, candidate_databases

    if len(databases) > 1:
        logger.warning(
            f"Multiple databases provided but entity_type is not 'auto'. "
            f"Using first database: {databases[0].value}"
        )

    database_name = databases[0].value
    logger.info(f">>>Step 4: Searching {database_name} database...<<<")
    recommendations = _search_one_database(
        entities, synonyms_dict, database_name, method, top_k, tax_id, model_info
    )
    for rec in recommendations:
        species_database[rec.id] = database_name
    return recommendations, species_database, candidate_databases


def _collect_existing_annotations(
    model_file: str,
    entity_type: EntityType,
    databases: List[DatabaseID],
) -> Tuple[Dict[str, List[str]], Dict[str, Dict[str, str]], Dict[Tuple[str, str], str]]:
    """Collect existing annotations for metrics, keyed by species ID."""
    existing_annotations: Dict[str, List[str]] = {}
    qualifier_annotations: Dict[str, Dict[str, str]] = {}
    existing_annotation_databases: Dict[Tuple[str, str], str] = {}

    if entity_type == EntityType.AUTO:
        dbs_to_search = databases
    else:
        dbs_to_search = databases[:1]

    for db in dbs_to_search:
        if entity_type != EntityType.AUTO:
            supported = (
                (entity_type == EntityType.CHEMICAL and db == DatabaseID.CHEBI)
                or (entity_type == EntityType.GENE and db == DatabaseID.NCBIGENE)
                or (entity_type == EntityType.PROTEIN and db == DatabaseID.UNIPROT)
                or (entity_type == EntityType.REACTION and db == DatabaseID.KEGG)
            )
            if not supported:
                logger.warning(
                    f"Entity type {entity_type.value} with database {db.value} not yet supported"
                )
                continue

        if db == DatabaseID.KEGG:
            anns, quals = find_reactions_with_kegg_annotations(model_file)
        elif db in (DatabaseID.CHEBI, DatabaseID.NCBIGENE, DatabaseID.UNIPROT):
            anns, quals = find_species_with_annotations_and_qualifiers(model_file, db.value)
        else:
            logger.warning(f"Database {db.value} not yet supported for existing annotation lookup")
            continue

        for species_id, ids in anns.items():
            existing_annotations.setdefault(species_id, []).extend(ids)
            for annotation_id in ids:
                existing_annotation_databases[(species_id, annotation_id)] = db.value
        for species_id, qualifier_map in quals.items():
            qualifier_annotations.setdefault(species_id, {}).update(qualifier_map)

    if existing_annotations:
        logger.info(f"Found {len(existing_annotations)} entities with existing annotations")
    return existing_annotations, qualifier_annotations, existing_annotation_databases


def _resolve_annotate(annotate: str, entity_type: EntityType, method: str) -> str:
    """Return ``species``, ``reactions``, or ``both``.

    ``entity_type="reaction"`` or ``method="rulebased"`` still selects reactions
    when the caller leaves ``annotate`` at the default (``"species"``).
    """
    value = (annotate or "species").strip().lower()
    if value in ("reaction", "reactions"):
        return "reactions"
    if value == "both":
        return "both"
    if value != "species":
        logger.warning("Unknown annotate=%r, using species", annotate)
    if entity_type == EntityType.REACTION or method == "rulebased":
        return "reactions"
    return "species"


def _chebi_rows(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Keep recommendation rows whose annotation is a ChEBI identifier."""
    if df is None or df.empty or "annotation" not in df.columns:
        return pd.DataFrame(columns=["id", "annotation"])
    out = df.copy()
    if "id" in out.columns:
        out = out[out["id"].astype(str) != "Reason:"]
    mask = out["annotation"].astype(str).str.upper().str.startswith("CHEBI:")
    return out.loc[mask].reset_index(drop=True)


def _species_recommendations_from_model(model_file: str) -> pd.DataFrame:
    """Build a species recommendation table from ChEBI annotations already in the SBML."""
    annotations, _ = find_species_with_annotations_and_qualifiers(model_file, DatabaseID.CHEBI.value)
    names = get_species_display_names(model_file)
    rows = []
    for species_id, ids in annotations.items():
        for ann in ids:
            s = str(ann).strip()
            if not s:
                continue
            if not s.upper().startswith("CHEBI:"):
                s = f"CHEBI:{s}"
            rows.append({
                "id": species_id,
                "display_name": names.get(species_id, species_id),
                "annotation": s,
                "match_score": 1.0,
            })
    return pd.DataFrame(rows)


def _silence_internal_logs() -> None:
    logging.getLogger("core").setLevel(logging.WARNING)


def _vprint(verbose: bool, msg: str) -> None:
    if verbose:
        print(msg)


def _apply_reason_comments(
    df: pd.DataFrame,
    reason: str | Dict[str, str],
) -> pd.DataFrame:
    """Put LLM reason text in ``comment``, once, on the first row of each key."""
    df = df.copy()
    if df.empty:
        df["comment"] = pd.Series(dtype=str)
        return df
    df["comment"] = ""
    if not reason:
        return df
    if isinstance(reason, dict):
        leftover: List[str] = []
        for eid, text in reason.items():
            if not text:
                continue
            idxs = df.index[df["id"] == eid]
            if len(idxs):
                df.at[idxs[0], "comment"] = text
            else:
                leftover.append(text)
        if leftover:
            empty = df.index[df["comment"] == ""]
            target = empty[0] if len(empty) else df.index[0]
            extra = " ".join(leftover)
            existing = df.at[target, "comment"]
            df.at[target, "comment"] = f"{existing} {extra}".strip() if existing else extra
        return df
    df.at[df.index[0], "comment"] = reason
    return df


def _extract_reason_comments(df: pd.DataFrame) -> Dict[str, str]:
    """Collect non-empty comment values, one per entity id."""
    if df.empty or "comment" not in df.columns or "id" not in df.columns:
        return {}
    comments: Dict[str, str] = {}
    for eid, group in df.groupby("id", sort=False):
        if str(eid) == "Reason:":
            continue
        vals = [c for c in group["comment"].fillna("").astype(str) if c.strip() and c != "nan"]
        if vals:
            comments[str(eid)] = vals[0]
    return comments


def _print_run_summary(
    df: pd.DataFrame,
    entity_word: str = "species",
    reason: str | Dict[str, str] = "",
) -> None:
    """Print annotation counts and the LLM reason after a run."""
    if df is None or df.empty:
        print(f"Found 0 annotations for 0 {entity_word}.")
    else:
        work = df[df["id"].astype(str) != "Reason:"] if "id" in df.columns else df
        n_entities = int(work["id"].nunique()) if "id" in work.columns else 0
        if "annotation" in work.columns:
            has_ann = work["annotation"].astype(str).str.strip()
            pred = work[~has_ann.isin(["", "nan"])]
        else:
            pred = work.iloc[0:0]
        n_ann = len(pred)
        n_with = int(pred["id"].nunique()) if "id" in pred.columns and not pred.empty else 0
        print(f"Found {n_ann} annotations for {n_with} of {n_entities} {entity_word}.")

    texts: List[str] = []
    if isinstance(reason, dict):
        texts = [t for t in reason.values() if t]
    elif reason:
        texts = [reason]
    if texts:
        if len(texts) == 1:
            print(f"LLM Reason: {texts[0]}")
        else:
            print("LLM Reason:")
            for text in texts:
                print(f"  {text}")


def _build_species_annotation_choices(sub_df: pd.DataFrame) -> str:
    lines: List[str] = []
    seen: set[str] = set()
    for _, row in sub_df.iterrows():
        ann = str(row.get("annotation", "")).strip()
        if not ann or ann.lower() == "nan":
            continue
        key = ann.upper()
        if key in seen:
            continue
        seen.add(key)
        label = row.get("annotation_label", "")
        if label is None or (isinstance(label, float) and label != label):
            label = ""
        lines.append(f"{ann}: {label}".rstrip())
    return "\n".join(lines)


def _ranking_notes_block(notes: Optional[str]) -> str:
    text = (notes or "").strip()
    return f"Model notes:\n{text}\n\n" if text else ""


def _notes_plus_message(notes: Optional[str], message: str = "") -> str:
    text = (notes or "").strip()
    msg = (message or "").strip()
    if not msg:
        return text
    return f"{text}\n\nUser message:\n{msg}".strip()


def _parse_ranked_id_lines(response_text: Optional[str]) -> Dict[str, List[str]]:
    """Parse ``entity_id: ID[, ID...]`` lines from a batched ranking response."""
    parsed: Dict[str, List[str]] = {}
    for line in (response_text or "").splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        entity_id, rest = line.split(":", 1)
        entity_id = entity_id.strip()
        if not entity_id:
            continue
        ids: List[str] = []
        for part in rest.split(","):
            tok = part.strip().split()[0] if part.strip() else ""
            if tok and tok.upper() != "UNK":
                ids.append(tok)
        parsed[entity_id] = ids
    return parsed


def _species_ranking_context(species_id: str, sub_df: pd.DataFrame) -> str:
    display = ""
    curated = ""
    if "display_name" in sub_df.columns and not sub_df.empty:
        display = str(sub_df["display_name"].iloc[0] or "")
    if "curated_name" in sub_df.columns and not sub_df.empty:
        curated = str(sub_df["curated_name"].iloc[0] or "")
    context = f"{species_id}: {display}" if display else species_id
    if curated and curated not in (display, species_id, "nan"):
        context += f"\nSynonyms: {curated}"
    return context


def rank_species_annotations_with_llm(
    model_file: str,
    recommendations_df: pd.DataFrame,
    llm_model: str = "gpt-4o-mini",
    n_return: int = 3,
    model_notes: str = "",
    max_completion_tokens: Optional[int] = None,
    usage_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> pd.DataFrame:
    """Re-rank species candidates with an LLM and keep at most *n_return* IDs per species.

    Species with ``n_return`` or fewer candidates are left as-is. Remaining
    species are ranked in a single LLM call.
    """
    if recommendations_df.empty or "annotation" not in recommendations_df.columns:
        return recommendations_df

    result_df = recommendations_df.copy()
    reason_df = result_df[result_df["id"] == "Reason:"] if "id" in result_df.columns else result_df.iloc[0:0]
    work_df = result_df[result_df["id"] != "Reason:"] if "id" in result_df.columns else result_df
    comments = _extract_reason_comments(work_df)
    notes_block = _ranking_notes_block(model_notes)

    ranked_rows: List[pd.DataFrame] = []
    to_rank: List[Tuple[str, pd.DataFrame, str]] = []
    for species_id in work_df["id"].unique():
        sub = work_df[work_df["id"] == species_id]
        if "type" in sub.columns and str(sub["type"].iloc[0]).lower() == "complex":
            ranked_rows.append(sub)
            continue
        choices = _build_species_annotation_choices(sub)
        if not choices.strip() or choices.count("\n") + 1 <= n_return:
            ranked_rows.append(sub)
            continue
        to_rank.append((str(species_id), sub, choices))

    if to_rank:
        entities = "\n\n".join(
            f"{_species_ranking_context(sid, sub)}\n{choices}"
            for sid, sub, choices in to_rank
        )
        prompt = SPECIES_ANNOTATION_RANKING_PROMPT.format(
            n_return=n_return,
            model_notes=notes_block,
            entities=entities,
        )
        parsed = _parse_ranked_id_lines(
            query_llm(
                prompt,
                model=llm_model,
                entity_type=EntityType.CHEMICAL,
                max_completion_tokens=max_completion_tokens,
                usage_callback=usage_callback,
            )
        )
        for sid, sub, _choices in to_rank:
            selected = parsed.get(sid)
            if selected is None:
                ranked_rows.append(sub)
                continue
            if not selected:
                empty = sub[sub["annotation"].astype(str).str.strip().isin(["", "nan"])]
                ranked_rows.append(empty if not empty.empty else sub.iloc[0:0])
                continue
            ann_upper = sub["annotation"].astype(str).str.strip().str.upper()
            for ann_id in selected[:n_return]:
                rows = sub[ann_upper == ann_id.upper()]
                if not rows.empty:
                    ranked_rows.append(rows.iloc[[0]])

    ranked_df = pd.concat(ranked_rows, ignore_index=True) if ranked_rows else work_df.iloc[0:0].copy()
    if comments:
        ranked_df = _apply_reason_comments(ranked_df, comments)
    elif "comment" not in ranked_df.columns:
        ranked_df["comment"] = ""
    if not reason_df.empty:
        ranked_df = pd.concat([reason_df, ranked_df], ignore_index=True)
    return ranked_df


def _load_species_recommendations(model_file: str, species_recommendations_df) -> pd.DataFrame:
    """Resolve species ChEBI rows from a DataFrame, CSV path, or the model itself."""
    if species_recommendations_df is None:
        df = _species_recommendations_from_model(model_file)
    elif isinstance(species_recommendations_df, (str, Path)):
        df = pd.read_csv(species_recommendations_df)
    else:
        df = species_recommendations_df
    return _chebi_rows(df)


def _output_csv(save_to: Optional[str], model_file: str, kind: str) -> str:
    return f"{save_to or Path(model_file).name}_{kind}.csv"


def annotate_single_model(
    model_file: str,
    llm_model: str = "gpt-4o-mini",
    method: str = "direct",
    top_k: int = 3,
    n_return: int = 3,
    max_entities: int = None,
    entity_type: str | EntityType = EntityType.CHEMICAL,
    database: str | DatabaseID | Sequence[str | DatabaseID] = DatabaseID.CHEBI,
    tax_id: str = None,
    chunk_size: int = 50,
    species_recommendations_df = None,
    annotate: str = "species",
    save_to: Optional[str] = None,
    verbose: bool = False,
    em_max_iterations: int = 5,
    message: str = "",
    max_completion_tokens: Optional[int] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Annotate a single model that has no or limited existing annotations.

    ``annotate`` selects the target:
        ``"species"`` — species only (default)
        ``"reactions"`` — KEGG reactions only (needs species ChEBI annotations)
        ``"both"`` — species first, then reactions from those recommendations

    Args:
        model_file: Path to SBML model file
        llm_model: LLM model to use ("gpt-4o-mini" or an OpenRouter "meta-llama/..." model)
        method: Method to use for database search ("direct", "rag"); reactions use
            rule-based KEGG matching plus LLM ranking
        top_k: Number of database candidates to retrieve per entity (direct/RAG).
            Synonym generation is fixed at 3.
        n_return: Number of IDs the final LLM ranking keeps per entity
            (species and reactions). Default 3. Ranking is skipped when a
            species or reaction has ``n_return`` or fewer candidates.
        max_entities: Maximum number of entities to annotate (None for all)
        entity_type: Type of entities to annotate ("chemical", "gene", "protein",
            "auto", "reaction")
        database: Target database ("chebi", "ncbigene", "uniprot") or a list of
            databases when entity_type is "auto"
        tax_id: For gene/protein annotations, the organism's tax_id for species-specific lookup
        chunk_size: Size of chunks to split large models into (default: 50, None for no chunking)
        species_recommendations_df: Species recommendation table or CSV path. Used
            when annotating reactions; if omitted, ChEBI terms already in the model
            are used
        annotate: ``"species"``, ``"reactions"``, or ``"both"``
        save_to: Output file prefix. Species go to ``<save_to>_species.csv``
            and reactions to ``<save_to>_reactions.csv``. Default is the
            model filename (e.g. ``model.xml_species.csv``).
        verbose: If True, print a short progress summary. Default False.
        em_max_iterations: Reaction EM rematch rounds (default 5). Use 0 to skip
            EM, or 1–2 for a faster run.
        message: Optional user note included in LLM prompts.
        max_completion_tokens: Optional per-request output-token ceiling passed
            to the LLM provider. Token usage is recorded in ``metrics['llm_usage']``.

    Returns:
        AnnotationResult. Species tables are on ``species_recommendations_df``;
        reaction tables are on ``reaction_recommendations_df``.
        ``recommendations_df`` is the primary table for the mode that ran
        (species-only or reactions).
    """
    start_time = time.time()
    _silence_internal_logs()

    # Ensure these locals always exist (rulebased path doesn't query the LLM).
    all_prompts: List[str] = []
    all_responses: List[str] = []
    assistant_messages: List[Dict[str, Any]] = []
    llm_usage_records: List[Dict[str, Any]] = []
    system_prompt: str = ""

    # logger.info(f"Starting annotation for model: {model_file}")
    # logger.info(f"Using LLM model: {llm_model}")
    # logger.info(f"Using method: {method} for database search")
    entity_type = _normalize_entity_type(entity_type)
    databases = _normalize_databases(database)
    if entity_type != EntityType.AUTO and len(databases) > 1:
        logger.warning(
            f"Multiple databases provided but entity_type is not 'auto'. "
            f"Using first database: {databases[0].value}"
        )
        databases = databases[:1]
    logger.info(f"Entity type: {entity_type.value}, Database: {_database_log_label(databases)}")
    if tax_id:
        logger.info(f"Using organism-specific search for tax_id: {tax_id}")

    annotate = _resolve_annotate(annotate, entity_type, method)
    csv_path = _output_csv(
        save_to, model_file, "reactions" if annotate == "reactions" else "species"
    )

    if annotate == "both":
        _vprint(verbose, f"Annotating species and reactions: {Path(model_file).name}")
        species_entity_type = (
            EntityType.CHEMICAL if entity_type == EntityType.REACTION else entity_type
        )
        species_method = "direct" if method == "rulebased" else method
        species_database: str | DatabaseID | Sequence[str | DatabaseID] = database
        if entity_type == EntityType.REACTION:
            species_database = DatabaseID.CHEBI
            species_method = "direct"
        species_result = annotate_single_model(
            model_file,
            llm_model=llm_model,
            method=species_method,
            top_k=top_k,
            n_return=n_return,
            max_entities=max_entities,
            entity_type=species_entity_type,
            database=species_database,
            tax_id=tax_id,
            chunk_size=chunk_size,
            annotate="species",
            save_to=save_to,
            verbose=verbose,
            message=message,
            max_completion_tokens=max_completion_tokens,
        )
        if not hasattr(species_result, "species_recommendations_df"):
            return species_result
        species_df = species_result.species_recommendations_df
        chebi_df = _chebi_rows(species_df)
        if chebi_df.empty:
            logger.warning("No ChEBI species annotations to drive reaction annotation")
            species_result.reaction_recommendations_df = pd.DataFrame()
            return species_result
        reaction_result = annotate_single_model(
            model_file,
            llm_model=llm_model,
            method="rulebased",
            top_k=top_k,
            n_return=n_return,
            max_entities=max_entities,
            entity_type=EntityType.REACTION,
            database=DatabaseID.KEGG,
            tax_id=tax_id,
            chunk_size=chunk_size,
            species_recommendations_df=chebi_df,
            annotate="reactions",
            save_to=save_to,
            verbose=verbose,
            em_max_iterations=em_max_iterations,
            message=message,
            max_completion_tokens=max_completion_tokens,
        )
        if hasattr(reaction_result, "recommendations_df"):
            reaction_result.species_recommendations_df = species_df
            reaction_result.metrics = {**species_result.metrics, "reaction": reaction_result.metrics}
            return reaction_result
        species_result.reaction_recommendations_df = pd.DataFrame()
        species_result.metrics = {**species_result.metrics, "reaction": reaction_result[1]}
        return species_result

    if annotate == "reactions":
        entity_type = EntityType.REACTION
        databases = [DatabaseID.KEGG]
        method = "rulebased"
        species_recommendations_df = _load_species_recommendations(
            model_file, species_recommendations_df
        )
        if species_recommendations_df.empty:
            logger.error(
                "No species ChEBI annotations available for reaction annotation. "
                "Run annotate='species' first or pass species_recommendations_df."
            )
            return pd.DataFrame(), {
                "error": "No species ChEBI annotations available for reaction annotation"
            }

    # Always define a system prompt so conversation history construction can't fail.
    system_prompt = get_system_prompt(entity_type)

    if entity_type == EntityType.REACTION:
        # Step 1: Get reactions from model
        logger.info(">>>Step 1: Getting reactions from model...<<<")
        all_entity_ids = get_all_reaction_ids(model_file)

        if not all_entity_ids:
            logger.warning("No reactions found in model")
            return pd.DataFrame(), {"error": "No reactions found in model"}

        logger.info(f"Found {len(all_entity_ids)} reactions in model")
    else:
        # Step 1: Get species from model
        logger.info(">>>Step 1: Getting species from model...<<<")
        all_entity_ids = get_all_species_ids(model_file, entity_type.value)

        if not all_entity_ids:
            logger.warning("No species found in model")
            return pd.DataFrame(), {"error": "No species found in model"}

        logger.info(f"Found {len(all_entity_ids)} species in model")

    # Check for existing annotations (for metrics calculation)
    existing_annotations, qualifier_annotations, existing_annotation_databases = (
        _collect_existing_annotations(model_file, entity_type, databases)
    )

    if max_entities:
        entities_to_evaluate = all_entity_ids[:max_entities]
        logger.info(f"Selected {max_entities} entities for annotation")
    else:
        entities_to_evaluate = all_entity_ids
        logger.info(f"Annotate all {len(entities_to_evaluate)} entities")

    if annotate == "reactions":
        _vprint(verbose, f"Reactions: {len(entities_to_evaluate)} in model, ranking with {llm_model}")
    else:
        _vprint(verbose, f"Species: {len(entities_to_evaluate)} entities, {llm_model}")

    # Step 2: Extract model context
    logger.info(">>>Step 2: Extracting model context...<<<")

    model_info = extract_model_info(model_file, entities_to_evaluate, entity_type.value)

    if not model_info:
        logger.error("Failed to extract model context")
        return pd.DataFrame(), {"error": "Failed to extract model context"}

    logger.info(f"Extracted context for model: {model_info['model_name']}")

    reason: str | Dict[str, str] = ""

    # Step 3
    if method == 'rulebased':
        logger.info(f">>>Step 3: Rule-based search of KEGG Reaction Annotations...<<<")
        from core.reaction.annotation_workflow import (
            rank_kegg_annotations_with_llm,
            run_kegg_annotation_workflow_rulebased,
        )
        search_start = time.time()
        from core.reaction.amendment_config import ConvergenceConfig
        kegg_annotation_workflow_result = run_kegg_annotation_workflow_rulebased(
            model_file,
            species_recommendations_df,
            existing_annotations=existing_annotations,
            convergence_config=ConvergenceConfig(max_iterations=em_max_iterations),
        )
        search_time = time.time() - search_start
        logger.info(f"Rule-based search completed in {search_time:.2f}s")

        if kegg_annotation_workflow_result is None:
            return pd.DataFrame(), {"error": "KEGG reaction annotation failed"}

        kegg_df = kegg_annotation_workflow_result.kegg_recommendations
        if kegg_df.empty:
            _vprint(verbose, "No KEGG reaction candidates found; skipping LLM ranking.")
            llm_time = 0.0
            recommendations_df = kegg_df
        else:
            llm_start = time.time()
            ranked_df = rank_kegg_annotations_with_llm(
                model_file,
                kegg_df,
                llm_model=llm_model,
                n_return=n_return,
                csv_path=csv_path,
                model_notes=_notes_plus_message(
                    (model_info or {}).get("model_notes", ""), message
                ),
            )
            llm_time = time.time() - llm_start
            recommendations_df = ranked_df if not ranked_df.empty else kegg_df

    else:
        # Format prompt for LLM
        logger.info(f">>>Step 3: Querying LLM ({llm_model})...<<<")

        # Track conversation context for potential feedback rounds
        all_prompts = []
        all_responses = []
        assistant_messages = []
        system_prompt = get_system_prompt(entity_type)
        entity_type_dict: Dict[str, str] = {}
        component_dict: Dict[str, List[Tuple[str, List[str]]]] = {}

        if chunk_size and len(entities_to_evaluate) > chunk_size:
            logger.info(f"Breaking {len(entities_to_evaluate)} entities into chunks of {chunk_size}")

            # Break down large models into chunks
            species_chunks = []
            for i in range(0, len(entities_to_evaluate), chunk_size):
                chunk = entities_to_evaluate[i:i + chunk_size]
                species_chunks.append(chunk)

            # Process each chunk and accumulate results
            all_synonyms_dict = {}
            all_entity_type_dict = {}
            all_component_dict: Dict[str, List[Tuple[str, List[str]]]] = {}
            reason_by_id: Dict[str, str] = {}
            total_llm_time = 0

            for chunk_idx, chunk in enumerate(species_chunks):
                logger.info(f"Processing chunk {chunk_idx + 1}/{len(species_chunks)} ({len(chunk)} entities)")

                # Format prompt for this chunk
                prompt = format_prompt(model_file, chunk, entity_type, message=message)

                if not prompt:
                    logger.error(f"Failed to format prompt for chunk {chunk_idx + 1}")
                    continue

                all_prompts.append(prompt)

                llm_start = time.time()
                try:
                    assistant_message = query_llm_message(
                        prompt,
                        system_prompt,
                        model=llm_model,
                        entity_type=entity_type,
                        max_completion_tokens=max_completion_tokens,
                        usage_callback=llm_usage_records.append,
                    )
                    result = assistant_message.get("content") if assistant_message else ""
                    chunk_llm_time = time.time() - llm_start
                    total_llm_time += chunk_llm_time

                    if not result:
                        logger.error(f"No response from LLM for chunk {chunk_idx + 1}")
                        continue

                    all_responses.append(result)
                    assistant_messages.append(assistant_message)
                    logger.info(f"Chunk {chunk_idx + 1} LLM response received in {chunk_llm_time:.2f}s")

                except Exception as e:
                    logger.error(f"LLM query failed for chunk {chunk_idx + 1}: {e}")
                    continue

                # Parse LLM response
                chunk_synonyms_dict, chunk_entity_type_dict, chunk_reason, chunk_component_dict = parse_llm_response(result, entity_type)

                # Accumulate synonyms and detected entity types
                all_synonyms_dict.update(chunk_synonyms_dict)
                all_entity_type_dict.update(chunk_entity_type_dict)
                all_component_dict.update(chunk_component_dict)

                if chunk_reason and chunk:
                    prefix = f"Chunk {chunk_idx + 1}: " if len(species_chunks) > 1 else ""
                    reason_by_id[chunk[0]] = f"{prefix}{chunk_reason}"

            reason = reason_by_id

            # Use accumulated synonyms
            synonyms_dict = all_synonyms_dict
            entity_type_dict = all_entity_type_dict
            component_dict = all_component_dict
            llm_time = total_llm_time

        else:
            # Single prompt for all entities
            prompt = format_prompt(model_file, entities_to_evaluate, entity_type, message=message)

            if not prompt:
                logger.error("Failed to format prompt")
                return pd.DataFrame(), {"error": "Failed to format prompt"}

            all_prompts.append(prompt)

            llm_start = time.time()
            try:
                assistant_message = query_llm_message(
                    prompt,
                    system_prompt,
                    model=llm_model,
                    entity_type=entity_type,
                    max_completion_tokens=max_completion_tokens,
                    usage_callback=llm_usage_records.append,
                )
                result = assistant_message.get("content") if assistant_message else ""
                llm_time = time.time() - llm_start

                if not result:
                    logger.error("No response from LLM")
                    return pd.DataFrame(), {"error": "No response from LLM"}

                all_responses.append(result)
                assistant_messages.append(assistant_message)
                logger.info(f"LLM response received in {llm_time:.2f}s")

            except Exception as e:
                logger.error(f"LLM query failed: {e}")
                return pd.DataFrame(), {"error": f"LLM query failed: {e}"}

            # Parse LLM response
            synonyms_dict, entity_type_dict, reason, component_dict = parse_llm_response(result, entity_type)

        if not synonyms_dict:
            logger.error("Failed to parse LLM response")
            return pd.DataFrame(), {"error": "Failed to parse LLM response"}

        logger.info(f"Parsed synonyms for {len(synonyms_dict)} entities")

        # Step 4: Search database
        if method not in ("direct", "rag"):
            logger.error(f"Invalid method: {method}")
            return pd.DataFrame(), {"error": f"Invalid method: {method}"}
        if entity_type != EntityType.AUTO and databases[0].value not in SUPPORTED_SEARCH_DATABASES:
            logger.error(f"Database {databases[0].value} not yet supported")
            return pd.DataFrame(), {"error": f"Database {databases[0].value} not yet supported"}

        search_start = time.time()
        recommendations, species_database, candidate_databases = _search_databases(
            entities_to_evaluate,
            synonyms_dict,
            entity_type=entity_type,
            databases=databases,
            method=method,
            top_k=top_k,
            tax_id=tax_id,
            entity_type_dict=entity_type_dict,
            model_info=model_info,
            component_dict=component_dict,
        )
        search_time = time.time() - search_start
        logger.info(f"Database search completed in {search_time:.2f}s")

        # Generate recommendation table
        logger.info(">>>Step 5: Generating recommendation table...<<<")
        recommendations_df = _generate_recommendation_table(
            model_file, recommendations, existing_annotations, model_info,
            entity_type.value, [db.value for db in databases], qualifier_annotations,
            synonyms_dict=synonyms_dict, reason=reason,
            entity_type_dict=entity_type_dict,
            species_database=species_database,
            candidate_databases=candidate_databases,
            existing_annotation_databases=existing_annotation_databases,
            component_dict=component_dict,
        )

        if top_k > n_return:
            logger.info(">>>Step 6: Ranking species candidates with LLM...<<<")
            rank_start = time.time()
            ranked_df = rank_species_annotations_with_llm(
                model_file,
                recommendations_df,
                llm_model=llm_model,
                n_return=n_return,
                model_notes=_notes_plus_message(
                    (model_info or {}).get("model_notes", ""), message
                ),
                max_completion_tokens=max_completion_tokens,
                usage_callback=llm_usage_records.append,
            )
            llm_time += time.time() - rank_start
            if not ranked_df.empty:
                recommendations_df = ranked_df
        else:
            logger.info("Skipping species LLM ranking (top_k <= n_return)")

    # Step 10: Calculate metrics
    total_time = time.time() - start_time

    metrics = _calculate_metrics(
        recommendations_df, existing_annotations, max_entities, len(all_entity_ids), total_time, llm_time, search_time
    )
    metrics["llm_usage"] = summarize_usage_records(llm_usage_records)

    if not recommendations_df.empty and "id" in recommendations_df.columns:
        recommendations_df = recommendations_df[recommendations_df["id"] != "Reason:"].reset_index(drop=True)

    recommendations_df.to_csv(csv_path, index=False)
    print(f"Saved {len(recommendations_df)} recommendations to {csv_path}")
    _print_run_summary(
        recommendations_df,
        entity_word="reactions" if annotate == "reactions" else "species",
        reason=reason,
    )
    _vprint(verbose, f"Finished in {total_time:.1f}s")
    # logger.info(f"Annotation completed in {total_time:.2f}s – {len(recommendations_df)} recommendations")

    combined_prompt = "\n\n".join(all_prompts)
    combined_response = "\n\n".join(all_responses)
    combined_assistant_message = assistant_messages[0] if len(assistant_messages) == 1 else None

    result = AnnotationResult(
        recommendations_df, metrics,
        model_file=model_file,
        conversation_history=build_initial_conversation(
            system_prompt,
            combined_prompt,
            combined_response,
            assistant_message=combined_assistant_message,
        ),
        entities_to_evaluate=entities_to_evaluate,
        entity_type=entity_type,
        database=[db.value for db in databases],
        method=method,
        llm_model=llm_model,
        top_k=top_k,
        n_return=n_return,
        tax_id=tax_id,
        existing_annotations=existing_annotations,
        qualifier_annotations=qualifier_annotations,
        model_info=model_info,
        csv_path=csv_path,
    )
    if annotate == "reactions":
        result.reaction_recommendations_df = recommendations_df
        result.species_recommendations_df = species_recommendations_df
    else:
        result.species_recommendations_df = recommendations_df
        result.reaction_recommendations_df = None
    return result


def _database_name(database: str | DatabaseID) -> str:
    return database.value if isinstance(database, DatabaseID) else str(database)


def _entity_type_name(entity_type: str | EntityType) -> str:
    return entity_type.value if isinstance(entity_type, EntityType) else str(entity_type)


def _normalize_database_names(
    database: str | DatabaseID | Sequence[str | DatabaseID],
) -> List[str]:
    items = database if isinstance(database, (list, tuple)) else [database]
    names = []
    for item in items:
        name = _database_name(item)
        if name not in names:
            names.append(name)
    return names or [DatabaseID.CHEBI.value]


def _resolve_row_database(
    species_id: str,
    candidate: Optional[str],
    default_databases: List[str],
    species_database: Dict[str, str],
    candidate_databases: Dict[Tuple[str, str], str],
    existing_annotation_databases: Dict[Tuple[str, str], str],
) -> str:
    if candidate is not None:
        if (species_id, candidate) in candidate_databases:
            return candidate_databases[(species_id, candidate)]
        if (species_id, candidate) in existing_annotation_databases:
            return existing_annotation_databases[(species_id, candidate)]
    if species_id in species_database:
        return species_database[species_id]
    return default_databases[0]


def _load_label_dicts(database_names: List[str]) -> Dict[str, Dict[str, str]]:
    label_dicts: Dict[str, Dict[str, str]] = {}
    for name in database_names:
        if name in label_dicts:
            continue
        if name == DatabaseID.CHEBI.value:
            label_dicts[name] = load_chebi_label_dict()
        elif name == DatabaseID.NCBIGENE.value:
            label_dicts[name] = load_ncbigene_label_dict()
        elif name == DatabaseID.UNIPROT.value:
            label_dicts[name] = load_uniprot_label_dict()
        elif name == DatabaseID.KEGG.value:
            label_dicts[name] = load_kegg_label_dict()
        else:
            label_dicts[name] = {}
    return label_dicts


def _generate_recommendation_table(model_file: str,
                                 recommendations: List[Recommendation],
                                 existing_annotations: Dict[str, List[str]],
                                 model_info: Dict[str, Any],
                                 entity_type: str = "chemical",
                                 database: str | DatabaseID | Sequence[str | DatabaseID] = DatabaseID.CHEBI.value,
                                 qualifier_annotations: Dict[str, List[str]] = None,
                                 synonyms_dict: Dict[str, List[str]] = None,
                                 reason: str | Dict[str, str] = "",
                                 entity_type_dict: Optional[Dict[str, str]] = None,
                                 species_database: Optional[Dict[str, str]] = None,
                                 candidate_databases: Optional[Dict[Tuple[str, str], str]] = None,
                                 existing_annotation_databases: Optional[Dict[Tuple[str, str], str]] = None,
                                 component_dict: Optional[Dict[str, List[Tuple[str, List[str]]]]] = None) -> pd.DataFrame:
    """
    Generate AMAS-compatible recommendation table.

    Args:
        model_file: Path to model file
        recommendations: List of Recommendation or ReactionRecommendation objects
        existing_annotations: Dictionary of existing annotations (may be empty)
        model_info: Model information dictionary
        entity_type: Type of entity being annotated ("auto" or a specific type)
        database: Database or list of databases used for search
        qualifier_annotations: Dictionary of qualifier annotations
        synonyms_dict: Dictionary mapping species IDs to LLM-suggested synonyms
        reason: LLM reasoning text, or per-entity map for chunked runs
        entity_type_dict: Optional mapping of species IDs to detected entity types
        species_database: Optional mapping of species IDs to the database searched
        candidate_databases: Optional mapping of (species_id, candidate) to database
        existing_annotation_databases: Optional mapping of (species_id, annotation) to database

    Returns:
        DataFrame in AMAS format
    """
    rows = []
    filename = Path(model_file).name
    if synonyms_dict is None:
        synonyms_dict = {}
    if qualifier_annotations is None:
        qualifier_annotations = {}
    if entity_type_dict is None:
        entity_type_dict = {}
    if species_database is None:
        species_database = {}
    if candidate_databases is None:
        candidate_databases = {}
    if existing_annotation_databases is None:
        existing_annotation_databases = {}
    if component_dict is None:
        component_dict = {}

    default_entity_type = _entity_type_name(entity_type)
    default_databases = _normalize_database_names(database)
    display_names = model_info.get("display_names", {}) if model_info else {}

    needed_databases = set(default_databases)
    needed_databases.update(species_database.values())
    needed_databases.update(candidate_databases.values())
    needed_databases.update(existing_annotation_databases.values())
    label_dicts = _load_label_dicts(list(needed_databases))

    def row_type(species_id: str) -> str:
        return entity_type_dict.get(species_id, default_entity_type)

    def row_database(species_id: str, candidate: Optional[str] = None) -> str:
        return _resolve_row_database(
            species_id, candidate, default_databases,
            species_database, candidate_databases, existing_annotation_databases,
        )

    def label_for(db_id: str, database_name: str) -> str:
        return label_dicts.get(database_name, {}).get(db_id, db_id)

    def curated_for(species_id: str) -> str:
        comps = component_dict.get(species_id)
        if comps:
            return format_component_curated_name(comps)
        return ", ".join(synonyms_dict.get(species_id) or [])

    seen_pairs = set()

    for rec in recommendations:
        curated_name = curated_for(rec.id)
        rec_type = row_type(rec.id)

        if not rec.candidates:
            if rec.id in qualifier_annotations and qualifier_annotations[rec.id]:
                all_qualifiers = list(qualifier_annotations[rec.id].values())
                specific_qualifier = ', '.join(all_qualifiers) if all_qualifiers else 'is'
            else:
                specific_qualifier = 'is'

            rec_db = row_database(rec.id)
            label = label_for(rec.id, rec_db)

            row = {
                'file': filename,
                'type': rec_type,
                'id': rec.id,
                'display_name': display_names.get(rec.id, rec.id),
                'curated_name': curated_name,
                'annotation': '',
                'annotation_label': label,
                'match_score': 0.0,
                'status': '',
                'update_annotation': 'ignore',
                'qualifier': specific_qualifier
            }
            rows.append(row)
            continue

        for i, candidate in enumerate(rec.candidates):
            rec_db = row_database(rec.id, candidate)
            candidate_display = f"{rec_db.upper()}:{candidate}"
            is_existing = candidate in existing_annotations.get(rec.id, [])
            match_score = rec.match_score[i]

            if is_existing:
                status = 'original and predicted'
                update_action = 'keep'
            else:
                status = 'predicted only'
                if i == 0 and match_score > 0.5:
                    update_action = 'add'
                else:
                    update_action = 'ignore'

            if is_existing and qualifier_annotations:
                specific_qualifier = qualifier_annotations.get(rec.id, {}).get(candidate, 'is')
            else:
                specific_qualifier = 'is'

            row = {
                'file': filename,
                'type': rec_type,
                'id': rec.id,
                'display_name': display_names.get(rec.id, rec.id),
                'curated_name': curated_name,
                'annotation': candidate_display,
                'annotation_label': rec.candidate_names[i] if i < len(rec.candidate_names) else candidate,
                'match_score': match_score,
                'status': status,
                'update_annotation': update_action,
                'qualifier': specific_qualifier
            }

            rows.append(row)
            seen_pairs.add((rec.id, candidate))

    # Add rows for existing annotations not predicted
    if existing_annotations:
        for species_id, ann_list in existing_annotations.items():
            for ann in ann_list:
                if (species_id, ann) not in seen_pairs:
                    rec_db = row_database(species_id, ann)
                    candidate_display = f"{rec_db.upper()}:{ann}"
                    curated_name = curated_for(species_id)

                    if qualifier_annotations:
                        specific_qualifier = qualifier_annotations.get(species_id, {}).get(ann, 'is')
                    else:
                        specific_qualifier = 'is'

                    row = {
                        'file': filename,
                        'type': row_type(species_id),
                        'id': species_id,
                        'display_name': display_names.get(species_id, species_id),
                        'curated_name': curated_name,
                        'annotation': candidate_display,
                        'annotation_label': label_for(ann, rec_db),
                        'match_score': None,
                        'status': 'original only',
                        'update_annotation': 'keep',
                        'qualifier': specific_qualifier
                    }
                    rows.append(row)

    df = pd.DataFrame(rows)

    if not df.empty and 'id' in df.columns:
        status_order = {'original and predicted': 0, 'original only': 1, 'predicted only': 2, '': 3}
        df['_status_order'] = df['status'].map(status_order).fillna(3)
        df = df.sort_values(by=['id', '_status_order']).reset_index(drop=True)
        df = df.drop(columns=['_status_order'])

    return _apply_reason_comments(df, reason)



def _calculate_metrics(recommendations_df: pd.DataFrame,
                      existing_annotations: Dict[str, List[str]],
                      max_entities: int,
                      total_species: int,
                      total_time: float,
                      llm_time: float,
                      search_time: float) -> Dict[str, Any]:
    """
    Calculate evaluation metrics for annotation workflow.

    Args:
        recommendations_df: Recommendation DataFrame
        existing_annotations: Dictionary of existing annotations (may be empty)
        max_entities: Maximum number of entities to annotate (None for all)
        total_species: Total number of species in the model
        total_time: Total processing time
        llm_time: LLM query time
        search_time: Database search time

    Returns:
        Dictionary with metrics
    """
    if recommendations_df.empty:
        return {
            'total_entities': max_entities,
            'entities_with_predictions': 0,
            'annotation_rate': 0.0,
            'total_predictions': 0,
            'matches': 0,
            'accuracy': np.nan,
            'total_time': total_time,
            'llm_time': llm_time,
            'search_time': search_time
        }

    if max_entities is None:
        max_entities = total_species

    # Filter out Reason row for metrics calculation
    df = recommendations_df[recommendations_df['id'] != 'Reason:'] if not recommendations_df.empty else recommendations_df

    entities_with_predictions = df[df['annotation'] != '']['id'].nunique()
    annotation_rate = entities_with_predictions / max_entities if max_entities > 0 else np.nan

    # Calculate accuracy based on existing annotations
    total_predictions = len(df[df['annotation'] != ''])
    matches = len(df[df['status'] == 'original and predicted'])

    # Accuracy = matches / entities with existing annotations
    entities_with_existing = len(existing_annotations)
    if not existing_annotations:
        accuracy = np.nan
    else:
        accuracy = matches / entities_with_existing if entities_with_existing > 0 else np.nan

    return {
        'total_entities': max_entities,
        'entities_with_predictions': entities_with_predictions,
        'annotation_rate': annotation_rate,
        'total_predictions': total_predictions,
        'matches': matches,
        'accuracy': accuracy,
        'total_time': total_time,
        'llm_time': llm_time,
        'search_time': search_time
    }


def print_results(results_df: pd.DataFrame):
    """
    Print evaluation results summary.
    Adapted from AMAS test_LLM_synonyms_plain.ipynb for annotation workflow

    Args:
        results_df: DataFrame with evaluation results
    """
    if results_df.empty:
        print("No results to display")
        return

    print("Number of models assessed: %d" % results_df['model'].nunique())
    print("Number of models with predictions: %d" % results_df[results_df['annotation'] != '']['model'].nunique())

    # Calculate per-model averages - handle NaN accuracy values
    results_df = results_df[results_df['id'] != 'Reason:'].copy()
    results_df['_is_match'] = (results_df['status'] == 'original and predicted').astype(int)
    model_accuracies = results_df.groupby('model')['_is_match'].mean()
    valid_accuracies = model_accuracies[~pd.isna(model_accuracies)]

    if len(valid_accuracies) > 0:
        print("Average accuracy (per model, where existing annotations available): %.02f" % valid_accuracies.mean())
    else:
        print("Average accuracy: N/A (no existing annotations)")

    mean_processing_time = results_df.groupby('model')['total_time'].first().mean()
    print("Ave. total time (per model): %.02f" % mean_processing_time)

    num_elements = results_df.groupby('model').size().mean()
    mean_processing_time_per_element = mean_processing_time / num_elements
    print("Ave. total time (per element, per model): %.02f" % mean_processing_time_per_element)

    # LLM time
    mean_llm_time = results_df.groupby('model')['llm_time'].first().mean()
    print("Ave. LLM time (per model): %.02f" % mean_llm_time)

    mean_llm_time_per_element = mean_llm_time / num_elements
    print("Ave. LLM time (per element, per model): %.02f" % mean_llm_time_per_element)

    # Average number of predictions per species
    average_predictions = results_df[results_df['annotation'] != ''].groupby('model').size().mean()
    print(f"Average number of predictions per model: {average_predictions}")


# Main interface function for users
def annotate_model(model_file: str, **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Annotate species, reactions, or both in an SBML model.

    Pass ``annotate="species"`` (default), ``"reactions"``, or ``"both"``.
    Set ``verbose=True`` for a short progress summary.
    ``top_k`` is the species retrieval pool; ``n_return`` (default 3) is
    how many IDs the final LLM ranking keeps.
    ``save_to`` is the output file prefix (``<save_to>_species.csv`` /
    ``<save_to>_reactions.csv``).
    Other keyword arguments are forwarded to :func:`annotate_single_model`
    (including ``message`` for extra LLM prompt text).
    """
    return annotate_single_model(model_file, **kwargs)
