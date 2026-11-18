"""AAAIM core package — high-level model annotation/curation and shared building blocks."""

from .annotation_workflow import (
    annotate_model,
    annotate_single_model,
    print_results,
)
from .curation_workflow import curate_model, curate_single_model
from .identity_support import assess_species_identity_support
from .qualifier_workflow import predict_species_qualifiers
from .feedback import AnnotationResult
from .reaction.retrieval import RetrievalResources
from .reaction.writer import write_reaction_annotations

from .database_search import (
    Recommendation,
    get_available_databases,
    get_species_recommendations_direct,
    load_chebi2kegg_dict,
    load_kegg_reaction_features_dict,
    score_model_against_kegg_reaction,
    search_database,
)
from .llm_interface import (
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_CHEMICAL,
    SYSTEM_PROMPT_GENE,
    get_system_prompt,
    parse_llm_response,
    query_llm,
    query_llm_message,
    query_llm_message_with_history,
    query_llm_with_history,
)

from .model_info import (
    detect_model_format,
    extract_model_info,
    find_species_with_annotations_and_qualifiers,
    find_species_with_chebi_annotations,
    find_species_with_ncbigene_annotations,
    format_prompt,
    get_all_species_ids,
    get_species_display_names,
)
from .reaction.deduplication import Reaction, deduplicate_reactions
from .reaction.hierarchy_relaxation import (
    compute_relaxation_penalty,
    detect_problematic_metabolites,
    detect_unmapped_metabolites,
    detect_unmapped_species_ids,
    expand_chebi_with_metadata,
    get_ancestors,
    iter_chebi_for_species,
    load_chebi_child_map,
    load_chebi_parent_map,
    merge_chebi_to_kegg_mapping,
    normalize_chebi,
    normalize_reaction,
    parse_chebi_obo,
    progressive_normalization,
    select_metabolites_to_relax,
)
from .reaction.scoring import (
    unified_reaction_objective,
    unified_reaction_objective_weighted,
)
from .reaction.relaxation_workflow import (
    map_reactions_to_kegg_with_relaxation,
    should_continue_iteration,
)
from .reaction.matching import (
    normalize_reactions,
)
from .update_model import update_annotation

__all__ = [
    "RetrievalResources",
    "write_reaction_annotations",
    # Annotation / curation
    "annotate_model",
    "annotate_single_model",
    "curate_model",
    "curate_single_model",
    "print_results",
    "update_annotation",
    "assess_species_identity_support",
    "predict_species_qualifiers",
    "AnnotationResult",
    
    # Model + LLM
    "detect_model_format",
    "extract_model_info",
    "find_species_with_annotations_and_qualifiers",
    "find_species_with_chebi_annotations",
    "find_species_with_ncbigene_annotations",
    "format_prompt",
    "get_all_species_ids",
    "get_species_display_names",
    "SYSTEM_PROMPT",
    "SYSTEM_PROMPT_CHEMICAL",
    "SYSTEM_PROMPT_GENE",
    "get_system_prompt",
    "query_llm",
    "query_llm_message",
    "query_llm_message_with_history",
    "query_llm_with_history",
    "parse_llm_response",
    
    # Database search
    "Recommendation",
    "get_available_databases",
    "get_species_recommendations_direct",
    "load_chebi2kegg_dict",
    "load_kegg_reaction_features_dict",
    "score_model_against_kegg_reaction",
    "search_database",

    # Reaction ↔ KEGG matching
    "map_reactions_to_kegg_with_relaxation",
    "normalize_reactions",
    "Reaction",
    "deduplicate_reactions",
    
    # ChEBI hierarchy / relaxation
    "compute_relaxation_penalty",
    "expand_chebi_with_metadata",
    "iter_chebi_for_species",
    "load_chebi_parent_map",
    "load_chebi_child_map",
    "merge_chebi_to_kegg_mapping",
    "normalize_chebi",
    "normalize_reaction",
    "parse_chebi_obo",
    "progressive_normalization",
    "get_ancestors",
    "detect_unmapped_metabolites",
    "detect_unmapped_species_ids",
    "detect_problematic_metabolites",
    "select_metabolites_to_relax",
    "should_continue_iteration",
    "unified_reaction_objective",
    "unified_reaction_objective_weighted",
]
