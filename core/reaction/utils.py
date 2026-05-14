"""Workflow utilities: environment checks, ChEBI→KEGG mapping, participant extraction."""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Mapping, Optional, Set, Tuple

import pandas as pd

from ..database_search import get_available_databases, load_chebi2kegg_dict
from .hierarchy_relaxation import (
    chebi_best_kegg_ids_with_ontology_fallback,
    load_chebi_parent_map,
)
from .kegg_compound_ids import parse_kegg_compound_id
from .kegg_definition import extract_classifications

logger = logging.getLogger(__name__)


def check_environment(model_file: str) -> bool:
    """Check if the environment is properly configured."""
    available_dbs = get_available_databases()
    logger.info(f"Available databases: {available_dbs}")
    
    all_ok = True
    
    if "chebi" not in available_dbs:
        logger.error("ChEBI chemical database not available!")
        logger.error("Please ensure ChEBI reference files are present in data/chebi/")
        all_ok = False
    
    if "kegg" not in available_dbs:
        logger.error("KEGG reaction database not available!")
        logger.error("Please ensure KEGG reference files are present in data/kegg/")
        all_ok = False
    
    if not os.path.exists(model_file):
        logger.error(f"Model file not found: {model_file}")
        logger.error("Please provide a valid SBML model file.")
        all_ok = False
    
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("OPENROUTER_API_KEY"):
        logger.warning("No API keys found in environment.")
        logger.warning("Set OPENAI_API_KEY or OPENROUTER_API_KEY to use LLM features.")
    
    return all_ok


def map_chebi_to_kegg(
    recommendations_df: pd.DataFrame,
    *,
    parent_map: Optional[Mapping[str, Set[str]]] = None,
    max_ancestor_depth: int = 2,
    max_descendant_depth: int = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Map species ``annotation`` values to KEGG compound IDs via
    ``load_chebi2kegg_dict`` when those values are ChEBI terms, or treat bare
    KEGG compound ids (``C#####``) as identity mappings when species were
    annotated with KEGG compounds instead of ChEBI.

    Duplicate rows are emitted when one ChEBI maps to multiple KEGG compounds.
    For reaction matching with optional upward relaxation along ChEBI ``is_a``,
    use ``hierarchy_relaxation.normalize_chebi`` / ``normalize_reaction`` instead.
    """
    chebi_to_kegg_map = load_chebi2kegg_dict()

    # Lazy-load / reuse ChEBI hierarchy inputs for the ontology fallback.
    ontology_parent_map: Optional[Mapping[str, Set[str]]] = parent_map
    ontology_cache: Dict[str, List[str]] = {}

    def _kegg_ids_for_chebi(seed_chebi_id: str) -> List[str]:
        """Direct hits, then bounded up-then-down ontology fallback."""
        nonlocal ontology_parent_map

        s = str(seed_chebi_id).strip()
        if not s:
            return []

        cached = ontology_cache.get(s)
        if cached is not None:
            return cached

        if ontology_parent_map is None:
            ontology_parent_map = load_chebi_parent_map()

        ids = chebi_best_kegg_ids_with_ontology_fallback(
            s,
            chebi_to_kegg_map,
            ontology_parent_map,  # type: ignore[arg-type]
            max_ancestor_depth=max_ancestor_depth,
            max_descendant_depth=max_descendant_depth,
        )
        ontology_cache[s] = ids
        return ids
    
    expanded_rows = []
    existing_mappings = {}
    has_reaction_id = "reaction_id" in recommendations_df.columns
    # Extend de-duplication/keying so reaction-scoped participant rows don't collapse.
    dedup_cols = ["id", "annotation", "KEGG_ID"] + (["reaction_id"] if has_reaction_id else [])
    
    if 'KEGG_ID' in recommendations_df.columns and 'participant_likelihood' in recommendations_df.columns:
        for _, row in recommendations_df.iterrows():
            if pd.notna(row['KEGG_ID']) and row['KEGG_ID'] != '':
                key = (row['id'], row['annotation'], row['KEGG_ID']) + (
                    (row['reaction_id'],) if has_reaction_id else tuple()
                )
                existing_mappings[key] = row['participant_likelihood']
    
    if not recommendations_df.empty and 'annotation' in recommendations_df.columns:
        for _, row in recommendations_df.iterrows():
            chebi_id = row['annotation']

            direct_kegg = parse_kegg_compound_id(chebi_id)
            if direct_kegg:
                row_copy = row.copy()
                row_copy["KEGG_ID"] = direct_kegg
                expanded_rows.append(row_copy)
                continue

            kegg_ids = _kegg_ids_for_chebi(chebi_id)

            if not kegg_ids:
                row_copy = row.copy()
                row_copy['KEGG_ID'] = ""
                expanded_rows.append(row_copy)
            else:
                for kegg_id in kegg_ids:
                    if kegg_id:
                        row_copy = row.copy()
                        row_copy['KEGG_ID'] = kegg_id
                        key = (row['id'], row['annotation'], kegg_id) + (
                            (row['reaction_id'],) if has_reaction_id else tuple()
                        )
                        if key in existing_mappings:
                            row_copy['participant_likelihood'] = existing_mappings[key]
                        expanded_rows.append(row_copy)
        
        expanded_df = pd.DataFrame(expanded_rows)
        
        if expanded_df.empty:
            recommendations_df['KEGG_ID'] = ""
            empty_cols = list(recommendations_df.columns)
            return recommendations_df, pd.DataFrame(columns=empty_cols)
        
        combined_df = pd.concat([recommendations_df, expanded_df]).drop_duplicates(
            subset=dedup_cols
        )
    else:
        recommendations_df['KEGG_ID'] = ""
        empty_cols = list(recommendations_df.columns)
        return recommendations_df, pd.DataFrame(columns=empty_cols)
    
    filtered_df = combined_df[
        combined_df['KEGG_ID'].notna() &
        (combined_df['KEGG_ID'] != '')
    ]
    
    if not filtered_df.empty:
        high_score_recommendations = filtered_df[
            filtered_df['match_score'] == filtered_df.groupby('id')['match_score'].transform('max')
        ].reset_index(drop=True)
    else:
        high_score_recommendations = pd.DataFrame(columns=list(combined_df.columns))
    
    logger.info(f"Expanded {len(recommendations_df)} ChEBI entries to {len(expanded_df)} KEGG mappings")
    logger.info(f"Found {len(filtered_df)} valid KEGG mappings")
    logger.info(f"Selected {len(high_score_recommendations)} high-score recommendations")
    
    return filtered_df, high_score_recommendations


def extract_reaction_participants(
    model_info: Dict,
    recommendations_df: pd.DataFrame
) -> Dict[str, List[str]]:
    """Extract participant names for each reaction."""
    reaction_participants = {}

    label_by_id: Dict[str, str] = {}
    if 'annotation_label' in recommendations_df.columns:
        for tup in recommendations_df[['id', 'annotation_label']].drop_duplicates('id').itertuples():
            label_by_id[tup.id] = tup.annotation_label

    for reaction in model_info['reactions']:
        reaction_id = reaction.split(':')[0].strip()
        participant_str = extract_classifications(reaction, 'definition')

        participant_names = []
        for participant in participant_str.split('; '):
            label = label_by_id.get(participant)
            if label is not None:
                participant_names.append(label)

        reaction_participants[reaction_id] = participant_names

    return reaction_participants