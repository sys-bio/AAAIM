"""
Curation Workflow for AAAIM

Main interface for curating a single model.
Provides the primary function that users will call to get recommendation tables
for models that already have existing annotations.
"""

import time
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import logging
import warnings

from utils.constants import DatabaseID, EntityType
from core.model_info import (
    find_species_with_annotations_and_qualifiers,
    find_reactions_with_kegg_annotations,
    extract_model_info,
    format_prompt,
)
from core.llm_interface import get_system_prompt, query_llm, parse_llm_response
from core.data_types import Recommendation
from core.database_search import get_species_recommendations_direct, get_species_recommendations_rag, load_uniprot_label_dict, load_ncbigene_label_dict, load_chebi_label_dict

logger = logging.getLogger(__name__)

# Suppress pandas FutureWarning noise (e.g., concat dtype changes)
warnings.filterwarnings("ignore", category=FutureWarning)

def curate_single_model(model_file: str, 
                  llm_model: str = "Llama-3.3-70B-Instruct",
                  method: str = "direct",
                  top_k: int = 3,
                  max_entities: int = None,
                  entity_type: str | EntityType = EntityType.CHEMICAL,
                  database: str | DatabaseID = DatabaseID.CHEBI,
                  tax_id: str = None,
                  chunk_size: int = 50,
                  *,
                  evaluate_candidates: bool = False,
                  include_exchange_reactions: bool = False,
                  ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    This is the main function users will call to get curation recommendations
    for a model that already has existing annotations.
    
    Args:
        model_file: Path to SBML model file
        llm_model: LLM model to use ("gpt-4o-mini", "Llama-3.3-70B-Instruct")
        method: Method to use for database search ("direct", "rag")
        top_k: Number of top candidates to return per species
        max_entities: Maximum number of entities to annotate (None for all)
        entity_type: Type of entities to annotate ("chemical", "gene", "protein", "auto")
        database: Target database ("chebi", "ncbigene", "uniprot")
        tax_id: For gene/protein annotations, the organism's tax_id for species-specific lookup
        chunk_size: Size of chunks to split large models into (default: 50, None for no chunking)
        
    Returns:
        Tuple of (recommendations_df, metrics_dict)
        - recommendations_df: AMAS-compatible DataFrame with annotation recommendations
        - metrics_dict: Dictionary with evaluation metrics and timing information
    """
    start_time = time.time()
    
    logger.info(f"Starting curation for model: {model_file}")
    logger.info(f"Using LLM model: {llm_model}")
    logger.info(f"Using method: {method} for database search")
    if isinstance(entity_type, str):
        try:
            entity_type = EntityType(entity_type)
        except ValueError:
            logger.warning(f"Unknown entity type {entity_type}, using chemical")
            entity_type = EntityType.CHEMICAL
    if isinstance(database, str):
        try:
            database = DatabaseID(database)
        except ValueError:
            logger.warning(f"Unknown database {database}, using chebi")
            database = DatabaseID.CHEBI
    logger.info(f"Entity type: {entity_type.value}, Database: {database.value}")
    if tax_id:
        logger.info(f"Using organism-specific search for tax_id: {tax_id}")
    
    # Step 1: Find existing annotations
    logger.info(">>>Step 1: Finding existing annotations...<<<")
    if entity_type == EntityType.CHEMICAL and database == DatabaseID.CHEBI:
        existing_annotations, qualifier_annotations = find_species_with_annotations_and_qualifiers(model_file, DatabaseID.CHEBI.value)
        logger.info(f"Found {len(existing_annotations)} entities with existing annotations")
    elif entity_type == EntityType.GENE and database == DatabaseID.NCBIGENE:
        existing_annotations, qualifier_annotations = find_species_with_annotations_and_qualifiers(model_file, DatabaseID.NCBIGENE.value)
        logger.info(f"Found {len(existing_annotations)} entities with existing annotations")
    elif entity_type == EntityType.PROTEIN and database == DatabaseID.UNIPROT:
        existing_annotations, qualifier_annotations = find_species_with_annotations_and_qualifiers(model_file, DatabaseID.UNIPROT.value)
        logger.info(f"Found {len(existing_annotations)} entities with existing annotations")
    elif entity_type == EntityType.REACTION and database == DatabaseID.KEGG:
        existing_annotations, qualifier_annotations = find_reactions_with_kegg_annotations(model_file)
        logger.info(f"Found {len(existing_annotations)} reactions with existing annotations")
    else:
        # Future: support other entity types and databases
        logger.warning(f"Entity type {entity_type.value} with database {database.value} not yet supported")
        existing_annotations = {}
        qualifier_annotations = {}
    
    if not existing_annotations:
        logger.warning("No existing annotations found in model")
        return pd.DataFrame(), {"error": "No existing annotations found"}
    
    # Select entities to evaluate
    if max_entities:
        specs_to_evaluate = list(existing_annotations.keys())[:max_entities]
        logger.info(f"Selected {len(specs_to_evaluate)} entities for curation")
    else:
        specs_to_evaluate = list(existing_annotations.keys())
        logger.info(f"Curation all {len(specs_to_evaluate)} entities")

    # Special-case: curate reaction->KEGG using the rulebased workflow.
    # This path is LLM-free; it uses existing species annotations (ChEBI, or
    # KEGG-compound as a fallback) as the metabolite evidence for KEGG
    # reaction matching. See :func:`curate_reactions_kegg_rulebased` for the
    # full logic.
    if entity_type == EntityType.REACTION and database == DatabaseID.KEGG:
        from core.reaction.annotation_workflow import curate_reactions_kegg_rulebased

        return curate_reactions_kegg_rulebased(
            model_file,
            existing_annotations,
            qualifier_annotations,
            specs_to_evaluate,
            evaluate_candidates=bool(evaluate_candidates),
            include_exchange_reactions=bool(include_exchange_reactions),
            llm_model=llm_model,
            top_k=top_k,
            tax_id=tax_id,
            start_time=start_time,
        )
    
    # Extract model context
    logger.info(">>>Step 2: Extracting model context...<<<")
    model_info = extract_model_info(model_file, specs_to_evaluate, entity_type)
    
    if not model_info:
        logger.error("Failed to extract model context")
        return pd.DataFrame(), {"error": "Failed to extract model context"}
    
    logger.info(f"Extracted context for model: {model_info['model_name']}")
    
    # Format prompt for LLM
    logger.info(">>>Step 3: Querying LLM ({llm_model})...<<<")
    
    # Track conversation context for potential feedback rounds
    all_prompts = []
    all_responses = []
    system_prompt = get_system_prompt(entity_type)

    if chunk_size and len(specs_to_evaluate) > chunk_size:
        logger.info(f"Breaking {len(specs_to_evaluate)} entities into chunks of {chunk_size}")
        
        # Break down large models into chunks
        species_chunks = []
        for i in range(0, len(specs_to_evaluate), chunk_size):
            chunk = specs_to_evaluate[i:i + chunk_size]
            species_chunks.append(chunk)
        
        # Process each chunk and accumulate results
        all_synonyms_dict = {}
        all_reasons = []
        total_llm_time = 0
        
        for chunk_idx, chunk in enumerate(species_chunks):
            logger.info(f"Processing chunk {chunk_idx + 1}/{len(species_chunks)} ({len(chunk)} entities)")
            
            # Format prompt for this chunk
            prompt = format_prompt(model_file, chunk, entity_type, top_k)
            
            if not prompt:
                logger.error(f"Failed to format prompt for chunk {chunk_idx + 1}")
                continue
            
            all_prompts.append(prompt)
            
            llm_start = time.time()
            try:
                result = query_llm(prompt, system_prompt, model=llm_model, entity_type=entity_type)
                chunk_llm_time = time.time() - llm_start
                total_llm_time += chunk_llm_time
                
                if not result:
                    logger.error(f"No response from LLM for chunk {chunk_idx + 1}")
                    continue
                
                all_responses.append(result)
                logger.info(f"Chunk {chunk_idx + 1} LLM response received in {chunk_llm_time:.2f}s")
                
            except Exception as e:
                logger.error(f"LLM query failed for chunk {chunk_idx + 1}: {e}")
                continue
            
            # Parse LLM response
            chunk_synonyms_dict, chunk_entity_type_dict, chunk_reason = parse_llm_response(result, entity_type)
            
            # Accumulate synonyms
            all_synonyms_dict.update(chunk_synonyms_dict)
            
            # Accumulate reasons
            if chunk_reason:
                all_reasons.append(f"Chunk {chunk_idx + 1}: {chunk_reason}")
        
        # Combine all reasons
        if all_reasons:
            reason = ' '.join(all_reasons)
        else:
            reason = ""
        
        # Use accumulated synonyms
        synonyms_dict = all_synonyms_dict
        llm_time = total_llm_time
        
    else:
        # Single prompt for all entities
        prompt = format_prompt(model_file, specs_to_evaluate, entity_type, top_k)
        
        if not prompt:
            logger.error("Failed to format prompt")
            return pd.DataFrame(), {"error": "Failed to format prompt"}
        
        all_prompts.append(prompt)
        
        llm_start = time.time()
        try:
            result = query_llm(prompt, system_prompt, model=llm_model, entity_type=entity_type)
            llm_time = time.time() - llm_start
            
            if not result:
                logger.error("No response from LLM")
                return pd.DataFrame(), {"error": "No response from LLM"}
            
            all_responses.append(result)
            logger.info(f"LLM response received in {llm_time:.2f}s")
            
        except Exception as e:
            logger.error(f"LLM query failed: {e}")
            return pd.DataFrame(), {"error": f"LLM query failed: {e}"}
        
        # Parse LLM response
        synonyms_dict, entity_type_dict, reason = parse_llm_response(result, entity_type)
    
    if not synonyms_dict:
        logger.error("Failed to parse LLM response")
        return pd.DataFrame(), {"error": "Failed to parse LLM response"}
    
    logger.info(f"Parsed synonyms for {len(synonyms_dict)} entities")

    if reason:
        print(f"LLM Reason: {reason}")

    # Search database
    logger.info(f">>>Step 4: Searching {database.value} database...<<<")
    search_start = time.time()
    
    if database == DatabaseID.CHEBI:
        if method == "direct":
            recommendations = get_species_recommendations_direct(specs_to_evaluate, synonyms_dict, database=DatabaseID.CHEBI.value, top_k=top_k)
        elif method == "rag":
            recommendations = get_species_recommendations_rag(specs_to_evaluate, synonyms_dict, database=DatabaseID.CHEBI.value)
        else:
            logger.error(f"Invalid method: {method}")
            return pd.DataFrame(), {"error": f"Invalid method: {method}"}
    elif database == DatabaseID.NCBIGENE:
        if method == "direct":
            recommendations = get_species_recommendations_direct(specs_to_evaluate, synonyms_dict, database=DatabaseID.NCBIGENE.value, tax_id=tax_id, top_k=top_k)
        elif method == "rag":
            recommendations = get_species_recommendations_rag(specs_to_evaluate, synonyms_dict, database=DatabaseID.NCBIGENE.value, tax_id=tax_id)
        else:
            logger.error(f"Invalid method: {method}")
            return pd.DataFrame(), {"error": f"Invalid method: {method}"}
    elif database == DatabaseID.UNIPROT:
        if method == "direct":
            recommendations = get_species_recommendations_direct(specs_to_evaluate, synonyms_dict, database=DatabaseID.UNIPROT.value, tax_id=tax_id, top_k=top_k)
        elif method == "rag":
            recommendations = get_species_recommendations_rag(specs_to_evaluate, synonyms_dict, database=DatabaseID.UNIPROT.value, tax_id=tax_id)
        else:
            logger.error(f"Invalid method: {method}")
            return pd.DataFrame(), {"error": f"Invalid method: {method}"}
    else:
        # Future: support other databases
        logger.error(f"Database {database.value} not yet supported")
        return pd.DataFrame(), {"error": f"Database {database.value} not yet supported"}
    
    search_time = time.time() - search_start
    logger.info(f"Database search completed in {search_time:.2f}s")
    
    # Generate recommendation table
    logger.info(">>>Step 5: Generating recommendation table...<<<")
    recommendations_df = _generate_recommendation_table(
        model_file, recommendations, existing_annotations, model_info, entity_type.value, database.value, qualifier_annotations,
        synonyms_dict=synonyms_dict, reason=reason
    )
    
    # Step 9: Calculate metrics
    total_time = time.time() - start_time
    metrics = _calculate_metrics(
        recommendations_df, existing_annotations, max_entities, total_time, llm_time, search_time
    )

    csv_path = f"{Path(model_file).name}_recommendations.csv"
    recommendations_df.to_csv(csv_path, index=False)
    print(f"Recommendations saved to {csv_path}")
    logger.info(f"Curation completed in {total_time:.2f}s – {len(recommendations_df)} recommendations")

    from core.feedback import AnnotationResult, build_initial_conversation
    combined_prompt = "\n\n".join(all_prompts)
    combined_response = "\n\n".join(all_responses)

    return AnnotationResult(
        recommendations_df, metrics,
        model_file=model_file,
        conversation_history=build_initial_conversation(system_prompt, combined_prompt, combined_response),
        entities_to_evaluate=specs_to_evaluate,
        entity_type=entity_type,
        database=database,
        method=method,
        llm_model=llm_model,
        top_k=top_k,
        tax_id=tax_id,
        existing_annotations=existing_annotations,
        qualifier_annotations=qualifier_annotations,
        model_info=model_info,
        csv_path=csv_path,
    )

def _generate_recommendation_table(model_file: str, 
                                 recommendations: List[Recommendation],
                                 existing_annotations: Dict[str, List[str]],
                                 model_info: Dict[str, Any],
                                 entity_type: str = "chemical",
                                 database: str = DatabaseID.CHEBI.value,
                                 qualifier_annotations: Dict[str, List[str]] = None,
                                 synonyms_dict: Dict[str, List[str]] = None,
                                 reason: str = "") -> pd.DataFrame:
    """
    Generate AMAS-compatible recommendation table.
    
    Args:
        model_file: Path to model file
        recommendations: List of Recommendation objects
        existing_annotations: Dictionary of existing annotations
        model_info: Model information dictionary
        entity_type: Type of entity being annotated
        database: Database being used for search
        qualifier_annotations: Dictionary of qualifier annotations
        synonyms_dict: Dictionary mapping species IDs to LLM-suggested synonyms
        reason: LLM reasoning text
        
    Returns:
        DataFrame in AMAS format
    """
    rows = []
    filename = Path(model_file).name
    if synonyms_dict is None:
        synonyms_dict = {}
    if qualifier_annotations is None:
        qualifier_annotations = {}

    seen_pairs = set()
    for rec in recommendations:
        curated_name = synonyms_dict.get(rec.id, [""])[0]

        if not rec.candidates:
            if qualifier_annotations and rec.id in qualifier_annotations and qualifier_annotations[rec.id]:
                all_qualifiers = list(qualifier_annotations[rec.id].values())
                specific_qualifier = ', '.join(all_qualifiers) if all_qualifiers else 'is'
            else:
                specific_qualifier = 'is'
            
            row = {
                'file': filename,
                'type': entity_type,
                'id': rec.id,
                'display_name': model_info["display_names"].get(rec.id, rec.id),
                'curated_name': curated_name,
                'annotation': '',
                'annotation_label': '',
                'match_score': 0.0,
                'status': '',
                'update_annotation': 'ignore',
                'qualifier': specific_qualifier
            }
            rows.append(row)
            continue
        for i, candidate in enumerate(rec.candidates):
            candidate_display = f"{database.upper()}:{candidate}"
            is_existing = candidate in existing_annotations.get(rec.id, [])
            match_score = rec.match_score[i]

            if is_existing:
                status = 'original and predicted'
                update_action = 'keep'
            else:
                status = 'predicted only'
                update_action = 'ignore'

            if is_existing and qualifier_annotations:
                specific_qualifier = qualifier_annotations.get(rec.id, {}).get(candidate, 'is')
            else:
                specific_qualifier = 'is'
            
            row = {
                'file': filename,
                'type': entity_type,
                'id': rec.id,
                'display_name': model_info["display_names"].get(rec.id, rec.id),
                'curated_name': curated_name,
                'annotation': candidate_display,
                'annotation_label': rec.candidate_names[i],
                'match_score': match_score,
                'status': status,
                'update_annotation': update_action,
                'qualifier': specific_qualifier
            }
            rows.append(row)
            seen_pairs.add((rec.id, candidate))

    # Add rows for existing annotations not predicted
    if database == DatabaseID.CHEBI.value:
        lbl_dict = load_chebi_label_dict()
    elif database == DatabaseID.NCBIGENE.value:
        lbl_dict = load_ncbigene_label_dict()
    elif database == DatabaseID.UNIPROT.value:
        lbl_dict = load_uniprot_label_dict()
    else:
        lbl_dict = {}

    for species_id, ann_list in existing_annotations.items():
        for ann in ann_list:
            if (species_id, ann) not in seen_pairs:
                candidate_display = f"{database.upper()}:{ann}"
                curated_name = synonyms_dict.get(species_id, [""])[0]

                if qualifier_annotations:
                    specific_qualifier = qualifier_annotations.get(species_id, {}).get(ann, 'is')
                else:
                    specific_qualifier = 'is'

                label = lbl_dict.get(ann, ann)

                row = {
                    'file': filename,
                    'type': entity_type,
                    'id': species_id,
                    'display_name': model_info["display_names"].get(species_id, species_id),
                    'curated_name': curated_name,
                    'annotation': candidate_display,
                    'annotation_label': label,
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

    if reason:
        reason_row = pd.DataFrame([{
            'file': filename, 'type': '', 'id': 'Reason:',
            'display_name': reason, 'curated_name': '',
            'annotation': '', 'annotation_label': '',
            'match_score': None, 'status': '',
            'update_annotation': '', 'qualifier': ''
        }])
        if df.empty:
            df = reason_row
        else:
            reason_row = reason_row.reindex(columns=df.columns)
            df = pd.concat([reason_row, df], ignore_index=True)

    return df

def _calculate_metrics(recommendations_df: pd.DataFrame,
                      existing_annotations: Dict[str, List[str]],
                      max_entities: int,
                      total_time: float,
                      llm_time: float,
                      search_time: float) -> Dict[str, Any]:
    """
    Calculate evaluation metrics.
    
    Args:
        recommendations_df: Recommendation DataFrame
        existing_annotations: Dictionary of existing annotations
        max_entities: Maximum number of entities to annotate (None for all)
        total_time: Total processing time
        llm_time: LLM query time
        search_time: Database search time
        
    Returns:
        Dictionary with metrics
    """
    if recommendations_df.empty:
        return {
            'total_entities': 0,
            'entities_with_predictions': 0,
            'annotation_rate': 0.0,
            'total_predictions': 0,
            'matches': 0,
            'accuracy': 0.0,
            'total_time': total_time,
            'llm_time': llm_time,
            'search_time': search_time
        }
    
    if max_entities is None:
        max_entities = len(existing_annotations)

    # Filter out Reason row for metrics calculation
    df = recommendations_df[recommendations_df['id'] != 'Reason:'] if not recommendations_df.empty else recommendations_df

    entities_with_predictions = df[df['annotation'] != '']['id'].nunique()
    annotation_rate = entities_with_predictions / max_entities if max_entities > 0 else 0.0
    
    # Accuracy = matches / entities with existing annotations
    total_predictions = len(df[df['annotation'] != ''])
    matches = len(df[df['status'] == 'original and predicted'])
    entities_with_existing = len(existing_annotations)
    accuracy = matches / entities_with_existing if entities_with_existing > 0 else 0
    
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
    
    Args:
        results_df: DataFrame with evaluation results
    """
    if results_df.empty:
        print("No results to display")
        return
    
    print("Number of models assessed: %d" % results_df['model'].nunique())
    print("Number of models with predictions: %d" % results_df[results_df['annotation'] != '']['model'].nunique())
    
    # Calculate per-model averages
    results_df = results_df[results_df['id'] != 'Reason:'].copy()
    results_df['_is_match'] = (results_df['status'] == 'original and predicted').astype(int)
    model_accuracy = results_df.groupby('model')['_is_match'].mean().mean()
    print("Average accuracy (per model): %.02f" % model_accuracy)
    
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
def curate_model(model_file: str, **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Main interface function for curating a single model.
    
    This is the primary function users should call for models with existing annotations.
    
    Args:
        model_file: Path to SBML model file
        **kwargs: Additional arguments passed to curate_single_model
        
    Returns:
        Tuple of (recommendations_df, metrics_dict)
    """
    return curate_single_model(model_file, **kwargs) 