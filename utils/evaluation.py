"""
Evaluation Utilities for AAAIM

Internal evaluation functions for testing and validation.
"""

import os
import time
import pandas as pd
import numpy as np
import lzma
import pickle
import re
import warnings
import contextlib
import sys
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import logging
from collections import deque
from threading import Lock
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_distances


class RateLimiter:
    """
    Rate limiter for API calls with support for request count limits.
    """
    def __init__(self, max_requests_per_minute: int = 10, verbose: bool = True):
        """
        Initialize the rate limiter.
        
        Args:
            max_requests_per_minute: Maximum number of requests allowed per minute (default: 10)
            verbose: If True, print rate limiting messages
        """
        self.max_requests_per_minute = max_requests_per_minute
        self.request_timestamps = deque()
        self.lock = Lock()
        self.verbose = verbose
    
    def wait_if_needed(self):
        """
        Wait if we're approaching rate limits.
        Call this before making an API request.
        """
        with self.lock:
            current_time = time.time()
            minute_ago = current_time - 60
            
            # Remove timestamps older than 1 minute
            while self.request_timestamps and self.request_timestamps[0] < minute_ago:
                self.request_timestamps.popleft()
            
            requests_in_last_minute = len(self.request_timestamps)
            
            # Check if we need to wait
            if requests_in_last_minute >= self.max_requests_per_minute:
                # Wait until the oldest request is more than a minute old
                oldest_timestamp = self.request_timestamps[0]
                wait_time = 60 - (current_time - oldest_timestamp) + 0.5  # Add 0.5s buffer
                if wait_time > 0:
                    if self.verbose:
                        print(f"Rate limit reached ({requests_in_last_minute} requests in last minute). Waiting {wait_time:.1f}s...")
                    time.sleep(wait_time)
            
            # Record this request
            self.request_timestamps.append(time.time())
    
    def reset(self):
        """Reset the rate limiter state."""
        with self.lock:
            self.request_timestamps.clear()


# Global rate limiter instance (can be shared across function calls)
_global_rate_limiter: Optional[RateLimiter] = None


def get_rate_limiter(max_requests_per_minute: int = 10, verbose: bool = True) -> RateLimiter:
    """
    Get or create a global rate limiter instance.
    
    Args:
        max_requests_per_minute: Maximum requests per minute
        verbose: If True, print rate limiting messages
        
    Returns:
        RateLimiter instance
    """
    global _global_rate_limiter
    if _global_rate_limiter is None or _global_rate_limiter.max_requests_per_minute != max_requests_per_minute:
        _global_rate_limiter = RateLimiter(max_requests_per_minute, verbose)
    return _global_rate_limiter


def reset_rate_limiter():
    """Reset the global rate limiter."""
    global _global_rate_limiter
    if _global_rate_limiter is not None:
        _global_rate_limiter.reset()

from core.model_info import find_species_with_chebi_annotations, find_species_with_annotations_and_qualifiers, extract_model_info, format_prompt, find_species_with_ncbigene_annotations, find_species_with_uniprot_annotations, get_species_display_names, detect_model_format
from core.llm_interface import SYSTEM_PROMPT, query_llm, parse_llm_response, get_system_prompt
from core.annotation_workflow import _search_complex_species, annotate_model
from core.data_types import Recommendation
from core.database_search import get_species_recommendations_direct, get_species_recommendations_rag, clear_chromadb_cache
from utils.constants import REF_CHEBI2LABEL, REF_NCBIGENE2LABEL, REF_UNIPROT2LABEL, REF_CHEBI2FORMULA, CHEBI_URI_PATTERNS, NCBIGENE_URI_PATTERNS, UNIPROT_URI_PATTERNS, ModelType, EntityType, DatabaseID, get_database_for_entity_type

REF_RESULTS = "/Users/luna/Desktop/CRBM/AMAS_proj/Results/biomd_species_accuracy_AMAS.csv"

logger = logging.getLogger(__name__)

@contextlib.contextmanager
def suppress_outputs(verbose: bool = True):
    """
    Context manager to suppress various outputs when verbose=False.
    """
    if not verbose:
        # Save original stdout/stderr
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        
        # Redirect to devnull for sentence transformers progress bars
        with open(os.devnull, 'w') as devnull:
            try:
                # Set environment variables to suppress progress bars
                os.environ['TOKENIZERS_PARALLELISM'] = 'false'
                os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
                os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
                
                # Suppress tqdm
                import tqdm
                original_tqdm_disable = getattr(tqdm.tqdm, '__init__', None)
                tqdm.tqdm.__init__ = lambda self, *args, **kwargs: original_tqdm_disable(self, *args, **{**kwargs, 'disable': True})
                
                yield
                
            finally:
                # Restore tqdm
                if original_tqdm_disable:
                    tqdm.tqdm.__init__ = original_tqdm_disable
                
                # Restore stdout/stderr
                sys.stdout = original_stdout
                sys.stderr = original_stderr
    else:
        yield

def _configure_verbosity(verbose: bool = True):
    """
    Configure logging and warning verbosity levels.
    
    Args:
        verbose: If True, show all logging. If False, minimize output.
    """
    if not verbose:
        # Set logging to WARNING level for AAAIM modules
        logging.getLogger('core').setLevel(logging.WARNING)
        logging.getLogger('utils').setLevel(logging.WARNING)
        
        # Suppress HTTP request logs
        logging.getLogger('httpx').setLevel(logging.WARNING)
        logging.getLogger('httpcore').setLevel(logging.WARNING)
        logging.getLogger('openai').setLevel(logging.WARNING)
        
        # Suppress ChromaDB logs
        logging.getLogger('chromadb').setLevel(logging.WARNING)
        logging.getLogger('sqlite3').setLevel(logging.WARNING)
        
        # Suppress sentence transformers logs
        logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
        
        # Suppress warnings
        warnings.filterwarnings("ignore")
        
        # Suppress ChromaDB telemetry messages
        os.environ.setdefault('ANONYMIZED_TELEMETRY', 'false')
        
        # Set transformers logging
        try:
            import transformers
            transformers.logging.set_verbosity_error()
        except ImportError:
            pass
    else:
        # Reset to normal logging
        logging.getLogger('core').setLevel(logging.INFO)
        logging.getLogger('utils').setLevel(logging.INFO)
        logging.getLogger('httpx').setLevel(logging.INFO)
        logging.getLogger('httpcore').setLevel(logging.INFO)
        logging.getLogger('openai').setLevel(logging.INFO)
        logging.getLogger('chromadb').setLevel(logging.INFO)
        logging.getLogger('sentence_transformers').setLevel(logging.INFO)
        
        try:
            import transformers
            transformers.logging.set_verbosity_warning()
        except ImportError:
            pass

# Cache for loaded dictionaries
_CHEBI_LABEL_DICT: Optional[Dict[str, str]] = None
_CHEBI_FORMULA_DICT: Optional[Dict[str, str]] = None
_NCBIGENE_LABEL_DICT: Optional[Dict[str, str]] = None
_UNIPROT_LABEL_DICT: Optional[Dict[str, str]] = None

def load_chebi_label_dict() -> Dict[str, str]:
    """
    Load the ChEBI ID to label dictionary.
    
    Returns:
        Dictionary mapping ChEBI IDs to their labels
    """
    global _CHEBI_LABEL_DICT
    
    if _CHEBI_LABEL_DICT is None:
        data_file = Path(__file__).parent.parent / "data" / "chebi" / REF_CHEBI2LABEL
        
        if not data_file.exists():
            raise FileNotFoundError(f"ChEBI label data file not found: {data_file}")
        
        with lzma.open(data_file, 'rb') as f:
            _CHEBI_LABEL_DICT = pickle.load(f)
    
    return _CHEBI_LABEL_DICT

def load_chebi_formula_dict() -> Dict[str, str]:
    """
    Load the ChEBI ID to formula dictionary.
    
    Returns:
        Dictionary mapping ChEBI IDs to their formulas
    """
    global _CHEBI_FORMULA_DICT
    
    if _CHEBI_FORMULA_DICT is None:
        data_file = Path(__file__).parent.parent / "data" / "chebi" / REF_CHEBI2FORMULA
        
        if not data_file.exists():
            raise FileNotFoundError(f"ChEBI formula data file not found: {data_file}")
        
        with lzma.open(data_file, 'rb') as f:
            _CHEBI_FORMULA_DICT = pickle.load(f)
    
    return _CHEBI_FORMULA_DICT

def load_ncbigene_label_dict() -> Dict[str, str]:
    """
    Load the NCBI gene ID to label dictionary.
    
    Returns:
        Dictionary mapping NCBI gene IDs to their labels
    """
    global _NCBIGENE_LABEL_DICT
    
    if _NCBIGENE_LABEL_DICT is None:
        data_file = Path(__file__).parent.parent / "data" / "ncbigene" / REF_NCBIGENE2LABEL
        
        if not data_file.exists():
            raise FileNotFoundError(f"NCBI gene label data file not found: {data_file}")
        
        with lzma.open(data_file, 'rb') as f:
            _NCBIGENE_LABEL_DICT = pickle.load(f)
    
    return _NCBIGENE_LABEL_DICT

def load_uniprot_label_dict(tax_id: str = None) -> Dict[str, str]:
    """
    Load the UniProt ID to label dictionary.
    
    Args:
        tax_id: If provided, loads organism-specific reference file.
                If None, tries to load the combined file
    
    Returns:
        Dictionary mapping UniProt IDs to their labels
    """
    global _UNIPROT_LABEL_DICT
    
    # Use a cache key that includes tax_id to handle multiple organisms
    cache_key = f"uniprot_label_{tax_id or 'combined'}"
    
    # Check if we have this specific version cached
    if not hasattr(load_uniprot_label_dict, '_cache'):
        load_uniprot_label_dict._cache = {}
    
    if cache_key in load_uniprot_label_dict._cache:
        return load_uniprot_label_dict._cache[cache_key]
    
    if tax_id:
        # Load organism-specific file
        data_file = Path(__file__).parent.parent / "data" / "uniprot" / f"uniprot2label_tax{tax_id}.lzma"
    else:
        # Try to load combined file
        data_file = Path(__file__).parent.parent / "data" / "uniprot" / REF_UNIPROT2LABEL
    
    if not data_file.exists():
        if tax_id:
            raise FileNotFoundError(f"UniProt label data file not found for tax_id {tax_id}: {data_file}")
        else:
            raise FileNotFoundError(f"UniProt label data file not found: {data_file}")
    
    with lzma.open(data_file, 'rb') as f:
        label_dict = pickle.load(f)
    
    # Cache the result
    load_uniprot_label_dict._cache[cache_key] = label_dict
    
    return label_dict

def get_recall(ref: Dict[str, List[str]], pred: Dict[str, List[str]], mean: bool = True) -> float:
    """
    Calculate recall metric.
    Replicates tools.getRecall from AMAS.
    
    Args:
        ref: Reference annotations {id: [annotations]}
        pred: Predicted annotations {id: [annotations]}
        mean: If True, return average across all IDs
        
    Returns:
        Recall value(s)
    """
    ref_keys = set(ref.keys())
    pred_keys = set(pred.keys())
    species_to_test = ref_keys.intersection(pred_keys)
    recall = {}
    
    for one_k in species_to_test:
        num_intersection = len(set(ref[one_k]).intersection(pred[one_k]))
        recall[one_k] = num_intersection / len(set(ref[one_k])) if ref[one_k] else 0
    
    if mean:
        return np.round(np.mean([recall[val] for val in recall.keys()]), 3) if recall else 0.0
    else:
        return {val: np.round(recall[val], 3) for val in recall.keys()}

def get_precision(ref: Dict[str, List[str]], pred: Dict[str, List[str]], mean: bool = True) -> float:
    """
    Calculate precision metric.
    Replicates tools.getPrecision from AMAS.
    
    Args:
        ref: Reference annotations {id: [annotations]}
        pred: Predicted annotations {id: [annotations]}
        mean: If True, return average across all IDs
        
    Returns:
        Precision value(s)
    """
    ref_keys = set(ref.keys())
    pred_keys = set(pred.keys())
    precision = {}
    species_to_test = ref_keys.intersection(pred_keys)
    
    for one_k in species_to_test:
        num_intersection = len(set(ref[one_k]).intersection(pred[one_k]))
        num_predicted = len(set(pred[one_k]))
        if num_predicted == 0:
            precision[one_k] = 0.0
        else:
            precision[one_k] = num_intersection / num_predicted
    
    if mean:
        if precision:
            return np.round(np.mean([precision[val] for val in precision.keys()]), 3)
        else:
            return 0.0
    else:
        return {val: np.round(precision[val], 3) for val in precision.keys()}

def get_species_statistics(recommendations: List[Recommendation], 
                          refs_formula: Dict[str, List[str]], 
                          refs_exact: Dict[str, List[str]], 
                          model_mean: bool = False) -> Dict[str, Any]:
    """
    Calculate species statistics including formula and exact-based metrics.
    Replicates getSpeciesStatistics from AMAS test_LLM_synonyms_plain.ipynb
    
    Args:
        recommendations: List of Recommendation objects
        refs_formula: Reference formulas {id: [formulas]}
        refs_exact: Reference ChEBI IDs {id: [chebi_ids]}
        model_mean: If True, return model-level averages
        
    Returns:
        Dictionary with recall and precision statistics
    """
    # Convert recommendations to prediction dictionaries
    preds_exact = {val.id: [k for k in val.candidates] for val in recommendations}
    
    # Convert ChEBI predictions to formulas
    formula_dict = load_chebi_formula_dict()
    preds_formula = {}
    for k in preds_exact.keys():
        formulas = []
        for chebi_id in preds_exact[k]:
            if chebi_id in formula_dict:
                formula = formula_dict[chebi_id]
                if formula:  # Only add non-empty formulas
                    formulas.append(formula)
        preds_formula[k] = formulas
    
    # Calculate metrics
    recall_formula = get_recall(ref=refs_formula, pred=preds_formula, mean=model_mean)
    precision_formula = get_precision(ref=refs_formula, pred=preds_formula, mean=model_mean)
    recall_exact = get_recall(ref=refs_exact, pred=preds_exact, mean=model_mean)
    precision_exact = get_precision(ref=refs_exact, pred=preds_exact, mean=model_mean)
    
    return {
        'recall_formula': recall_formula, 
        'precision_formula': precision_formula, 
        'recall_exact': recall_exact, 
        'precision_exact': precision_exact
    }

def find_species_with_formulas(model_file: str, bqbiol_qualifiers: list = None) -> Tuple[Dict[str, List[str]], Dict[str, Dict[str, str]]]:
    """
    Find species with existing ChEBI annotations that have chemical formulas.
    Replicates the logic from AMAS species_annotation.py exist_annotation_formula.
    
    Args:
        model_file: Path to the SBML model file
        bqbiol_qualifiers: List of bqbiol qualifiers to extract (e.g. ['is', 'isVersionOf', 'hasPart'])

    Returns:
        Tuple of (species_with_formulas, qualifier_annotations) where:
        - species_with_formulas: Dictionary mapping species IDs to their ChEBI annotation IDs (only for species with formulas)
        - qualifier_annotations: Dictionary mapping species IDs to a dict of {annotation_id: qualifier}
    """
    # Get all species with ChEBI annotations and their qualifiers
    existing_annotations, qualifier_annotations = find_species_with_annotations_and_qualifiers(model_file, "chebi", bqbiol_qualifiers)
    
    if not existing_annotations:
        return {}, {}
    
    # Load ChEBI to formula dictionary
    formula_dict = load_chebi_formula_dict()
    
    # Filter to only species that have at least one ChEBI with a formula
    species_with_formulas = {}
    filtered_qualifier_annotations = {}
    
    for species_id, chebi_ids in existing_annotations.items():
        formulas = []
        for chebi_id in chebi_ids:
            if chebi_id in formula_dict:
                formula = formula_dict[chebi_id]
                if formula:  # Only add non-empty formulas
                    formulas.append(formula)
        
        # Only include species that have at least one formula
        if formulas:
            species_with_formulas[species_id] = chebi_ids
            filtered_qualifier_annotations[species_id] = qualifier_annotations.get(species_id, {})
    
    return species_with_formulas, filtered_qualifier_annotations

def find_species_with_gene_annotations(model_file: str, bqbiol_qualifiers: list = None, tax_id: str = None) -> Tuple[Dict[str, List[str]], Dict[str, Dict[str, str]]]:
    """
    Find species with existing NCBI gene annotations.
    
    Args:
        model_file: Path to the SBML model file
        bqbiol_qualifiers: List of bqbiol qualifiers to extract (e.g. ['is', 'isVersionOf', 'hasPart'])
        tax_id: Taxonomy ID
    Returns:
        Tuple of (existing_annotations, qualifier_annotations) where:
        - existing_annotations: Dictionary mapping species IDs to their NCBI gene annotation IDs
        - qualifier_annotations: Dictionary mapping species IDs to a dict of {annotation_id: qualifier}
    """
    # Get all species with NCBI gene annotations and their qualifiers
    existing_annotations, qualifier_annotations = find_species_with_annotations_and_qualifiers(model_file, "ncbigene", bqbiol_qualifiers)
    
    if not existing_annotations:
        return {}, {}
    else:
        if tax_id:
            label_dict = load_ncbigene_label_dict()
            filtered_annotations = {}
            filtered_qualifier_annotations = {}
            
            for species_id, ncbi_ids in existing_annotations.items():
                # Filter out NCBI IDs that don't exist in the label_dict
                valid_ncbi_ids = [ncbi_id for ncbi_id in ncbi_ids if ncbi_id in label_dict]
                if valid_ncbi_ids:
                    filtered_annotations[species_id] = valid_ncbi_ids
                    # Filter qualifier annotations to only include valid NCBI IDs
                    filtered_qualifier_annotations[species_id] = {
                        ncbi_id: qualifier_annotations.get(species_id, {}).get(ncbi_id, 'unknown')
                        for ncbi_id in valid_ncbi_ids
                    }
            
            return filtered_annotations, filtered_qualifier_annotations
    
    # Return all species that have NCBI gene annotations
    return existing_annotations, qualifier_annotations

def find_species_with_protein_annotations(model_file: str, bqbiol_qualifiers: list = None, tax_id: str = None) -> Tuple[Dict[str, List[str]], Dict[str, Dict[str, str]]]:
    """
    Find species with existing UniProt annotations.
    
    Args:
        model_file: Path to the SBML model file
        bqbiol_qualifiers: List of bqbiol qualifiers to extract (e.g. ['is', 'isVersionOf', 'hasPart'])
        tax_id: Taxonomy ID
    Returns:
        Tuple of (existing_annotations, qualifier_annotations) where:
        - existing_annotations: Dictionary mapping species IDs to their UniProt annotation IDs
        - qualifier_annotations: Dictionary mapping species IDs to a dict of {annotation_id: qualifier}
    """
    # Get all species with UniProt annotations and their qualifiers
    existing_annotations, qualifier_annotations = find_species_with_annotations_and_qualifiers(model_file, "uniprot", bqbiol_qualifiers)
    # print(f"Existing annotations: {existing_annotations}")
    
    if not existing_annotations:
        return {}, {}
    else:
        if tax_id:
            label_dict = load_uniprot_label_dict(tax_id)
            filtered_annotations = {}
            filtered_qualifier_annotations = {}
            
            for species_id, uniprot_ids in existing_annotations.items():
                # Filter out UniProt IDs that don't exist in the label dictionary
                valid_uniprot_ids = [uniprot_id for uniprot_id in uniprot_ids if uniprot_id in label_dict]
                if valid_uniprot_ids:
                    filtered_annotations[species_id] = valid_uniprot_ids
                    # Filter qualifier annotations to only include valid UniProt IDs
                    filtered_qualifier_annotations[species_id] = {
                        uniprot_id: qualifier_annotations.get(species_id, {}).get(uniprot_id, 'unknown')
                        for uniprot_id in valid_uniprot_ids
                    }
            
            return filtered_annotations, filtered_qualifier_annotations
    
    # Return all species that have UniProt annotations
    return existing_annotations, qualifier_annotations

def _eval_search_complexes(
    species_list: List[str],
    synonyms_dict: Dict[str, List[str]],
    allowed_databases: List[str],
    method: str,
    top_k: int,
    tax_id: Any,
    model_type: str,
    component_dict: Optional[Dict] = None,
    verbose: bool = True,
) -> List[Recommendation]:
    """Search complex species using per-component types when available."""
    component_dict = component_dict or {}
    recs = []
    for species_id in species_list:
        with suppress_outputs(verbose):
            rec, _cand = _search_complex_species(
                species_id, synonyms_dict, allowed_databases, method, top_k,
                tax_id, None, component_dict.get(species_id), model_type,
            )
        recs.append(rec)
    return recs


def _get_database_for_entity_type(entity_type: str, 
                                   allowed_databases: Optional[List[str]] = None) -> Optional[str]:
    """
    Get the appropriate database for an entity type.
    
    Args:
        entity_type: Detected entity type (chemical, gene, protein, unknown)
        allowed_databases: List of allowed database names provided by user (e.g., ["chebi", "uniprot"])
        
    Returns:
        Database name to use, or None if no valid database found
    """
    target = get_database_for_entity_type(entity_type, allowed_databases)
    if target is None and entity_type.lower() not in ("unknown", EntityType.UNKNOWN.value):
        logger.warning(
            f"No valid database found for entity type '{entity_type}'"
            + (f" in allowed databases: {allowed_databases}" if allowed_databases else "")
        )
    return target

def _bare_id(value) -> str:
    text = "" if value is None else str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    return text.split(":", 1)[1] if ":" in text else text


def _database_names(database) -> List[str]:
    if isinstance(database, str):
        return [database]
    return [str(name) for name in database]


def evaluation_frame_from_annotations(
    recommendations_df: pd.DataFrame,
    existing_annotations: Dict[str, List[str]],
    *,
    model_name: str,
    qualifier_annotations: Dict[str, Dict[str, str]] = None,
    metrics: Dict[str, Any] = None,
    entity_type: str = "chemical",
    database: Union[str, List[str]] = "chebi",
    tax_id: str = None,
    tax_name: str = None,
    formula_dict: Dict[str, str] = None,
    label_dict: Dict[str, str] = None,
) -> pd.DataFrame:
    """Score one annotation table against existing SBML identifiers.

    One row per gold species. Predictions are the ranked identities
    (``original and predicted`` / ``predicted only``), with the database
    prefix removed so they match SBML annotation IDs.
    """
    databases = _database_names(database)
    has_chebi = "chebi" in databases
    if label_dict is None:
        label_dict = {}
        if has_chebi:
            label_dict.update(load_chebi_label_dict())
        if "ncbigene" in databases:
            label_dict.update(load_ncbigene_label_dict())
        if "uniprot" in databases:
            label_dict.update(load_uniprot_label_dict(tax_id))
    if formula_dict is None and has_chebi:
        formula_dict = load_chebi_formula_dict()
    formula_dict = formula_dict or {}
    qualifier_annotations = qualifier_annotations or {}
    metrics = metrics or {}
    if recommendations_df is None or recommendations_df.empty or "id" not in recommendations_df.columns:
        grouped = {}
    else:
        grouped = {
            species_id: frame
            for species_id, frame in recommendations_df.groupby("id", sort=False)
        }

    rows = []
    for species_id, existing_ids in existing_annotations.items():
        sub = grouped.get(species_id, recommendations_df.iloc[0:0])
        predicted_ids = []
        predicted_names = []
        match_scores = []
        if not sub.empty and "annotation" in sub.columns:
            pred = sub[sub["status"].isin(("original and predicted", "predicted only"))].copy()
            pred = pred[pred["annotation"].map(_bare_id) != ""]
            if not pred.empty:
                pred["_identity_rank"] = pd.to_numeric(pred.get("identity_rank"), errors="coerce")
                pred["_candidate_rank"] = pd.to_numeric(pred.get("candidate_rank"), errors="coerce")
                pred = pred.sort_values(
                    ["_identity_rank", "_candidate_rank"], kind="stable", na_position="last"
                )
                for _, rec in pred.iterrows():
                    bare = _bare_id(rec["annotation"])
                    predicted_ids.append(bare)
                    label = rec["annotation_label"] if "annotation_label" in rec.index else ""
                    if not label or str(label).lower() == "nan":
                        label = label_dict.get(bare, bare)
                    predicted_names.append(str(label))
                    score = rec["match_score"] if "match_score" in rec.index else 0.0
                    match_scores.append(0.0 if pd.isna(score) else score)

        existing_bare = [_bare_id(annotation_id) for annotation_id in existing_ids]
        exact_matches = set(predicted_ids) & set(existing_bare)
        recall_exact = len(exact_matches) / len(existing_bare) if existing_bare else 0
        precision_exact = len(exact_matches) / len(predicted_ids) if predicted_ids else 0

        detected = entity_type
        if not sub.empty and "type" in sub.columns and str(sub.iloc[0]["type"]).strip():
            detected = str(sub.iloc[0]["type"])
        if detected == "chemical" and has_chebi:
            def _formulas(ids):
                found = []
                for annotation_id in ids:
                    formula = formula_dict.get(annotation_id) or formula_dict.get(_bare_id(annotation_id))
                    if formula:
                        found.append(formula)
                return found

            existing_formulas = _formulas(existing_bare)
            predicted_formulas = _formulas(predicted_ids)
            if existing_formulas:
                formula_matches = set(predicted_formulas) & set(existing_formulas)
                recall_formula = len(formula_matches) / len(set(existing_formulas))
            else:
                recall_formula = recall_exact
            if predicted_formulas:
                formula_matches = set(predicted_formulas) & set(existing_formulas)
                precision_formula = len(formula_matches) / len(set(predicted_formulas))
            else:
                precision_formula = precision_exact
        else:
            recall_formula = recall_exact
            precision_formula = precision_exact

        if entity_type == "auto":
            accuracy = 1 if (recall_formula if detected == "chemical" else recall_exact) > 0 else 0
        elif entity_type == "chemical":
            accuracy = 1 if recall_formula > 0 else 0
        else:
            accuracy = 1 if recall_exact > 0 else 0

        curated = ""
        if not sub.empty and "curated_name" in sub.columns:
            curated = str(sub.iloc[0]["curated_name"] or "")
            if curated.lower() == "nan":
                curated = ""
        synonyms = [part.strip() for part in curated.split(",") if part.strip()] if curated else []
        reason = ""
        if not sub.empty and "comment" in sub.columns:
            comments = [str(text) for text in sub["comment"] if str(text).strip() and str(text).lower() != "nan"]
            reason = comments[0] if comments else ""
        display_name = ""
        if not sub.empty and "display_name" in sub.columns:
            display_name = sub.iloc[0]["display_name"]
            if pd.isna(display_name):
                display_name = ""
        qualifier_map = qualifier_annotations.get(species_id, {})
        qualifiers = [qualifier_map[annotation_id] for annotation_id in existing_ids if annotation_id in qualifier_map]
        rows.append({
            "model": model_name,
            "species_id": species_id,
            "display_name": display_name,
            "detected_entity_type": detected,
            "synonyms_LLM": synonyms,
            "reason": reason,
            "exist_annotation_id": existing_bare,
            "exist_annotation_name": ", ".join(label_dict.get(annotation_id, annotation_id) for annotation_id in existing_bare) or "NA",
            "predictions": predicted_ids,
            "predictions_names": ", ".join(predicted_names) if predicted_names else "NA",
            "match_score": match_scores,
            "recall_formula": recall_formula,
            "precision_formula": precision_formula,
            "recall_exact": recall_exact,
            "precision_exact": precision_exact,
            "accuracy": accuracy,
            "total_time": metrics.get("total_time", 0),
            "llm_time": metrics.get("llm_time", 0),
            "query_time": metrics.get("search_time", metrics.get("query_time", 0)),
            "tax_id": tax_id,
            "tax_name": tax_name,
            "qualifier": ", ".join(qualifiers) if qualifiers else "is",
        })
    return pd.DataFrame(rows)


def _self_check_evaluation_frame() -> None:
    recommendations = pd.DataFrame([
        {"id": "s1", "type": "chemical", "display_name": "glc", "curated_name": "glucose",
         "annotation": "CHEBI:17234", "annotation_label": "glucose", "status": "original and predicted",
         "match_score": 1.0, "identity_rank": 1, "candidate_rank": 1, "comment": "because"},
        {"id": "s1", "type": "chemical", "display_name": "glc", "curated_name": "glucose",
         "annotation": "CHEBI:4167", "annotation_label": "other", "status": "predicted only",
         "match_score": 0.4, "identity_rank": 2, "candidate_rank": 1, "comment": ""},
        {"id": "s1", "type": "chemical", "display_name": "glc", "curated_name": "glucose",
         "annotation": "CHEBI:999", "annotation_label": "kept", "status": "original only",
         "match_score": None, "identity_rank": "", "candidate_rank": "", "comment": ""},
        {"id": "s3", "type": "chemical", "display_name": "ac", "curated_name": "acetate",
         "annotation": "CHEBI:222", "annotation_label": "acetate", "status": "predicted only",
         "match_score": 0.9, "identity_rank": 1, "candidate_rank": 1, "comment": ""},
    ])
    frame = evaluation_frame_from_annotations(
        recommendations,
        {"s1": ["17234"], "s2": ["15377"], "s3": ["111"]},
        model_name="m.xml",
        qualifier_annotations={"s1": {"17234": "is"}},
        metrics={"total_time": 3, "llm_time": 2, "search_time": 1},
        formula_dict={"17234": "C6", "4167": "C6", "15377": "H2O", "111": "C2", "222": "C2"},
        label_dict={"17234": "glucose", "15377": "water", "111": "acetate"},
    )
    by_id = frame.set_index("species_id")
    assert list(by_id.loc["s1", "predictions"]) == ["17234", "4167"]
    assert by_id.loc["s1", "recall_exact"] == 1
    assert by_id.loc["s1", "precision_exact"] == 0.5
    assert by_id.loc["s1", "precision_formula"] == 1
    assert by_id.loc["s1", "accuracy"] == 1
    assert by_id.loc["s2", "predictions"] == []
    assert by_id.loc["s2", "accuracy"] == 0
    assert by_id.loc["s3", "recall_exact"] == 0
    assert by_id.loc["s3", "recall_formula"] == 1
    assert by_id.loc["s3", "accuracy"] == 1
    assert list(frame["species_id"]) == ["s1", "s2", "s3"]


def _gold_annotations(model_file, entity_type, database, bqbiol_qualifiers, tax_id):
    """Species that already carry the annotations this benchmark scores."""
    if isinstance(database, (list, tuple)) and len(database) == 1 and entity_type != "auto":
        database = database[0]
    existing = {}
    qualifiers = {}
    if entity_type == "auto":
        for db in _database_names(database):
            if db == "chebi":
                anns, quals = find_species_with_formulas(model_file, bqbiol_qualifiers)
            elif db == "ncbigene":
                anns, quals = find_species_with_gene_annotations(model_file, bqbiol_qualifiers, tax_id)
            elif db == "uniprot":
                anns, quals = find_species_with_protein_annotations(model_file, bqbiol_qualifiers, tax_id)
            else:
                continue
            existing.update(anns)
            qualifiers.update(quals)
    elif entity_type == "chemical" and database == "chebi":
        existing, qualifiers = find_species_with_formulas(model_file, bqbiol_qualifiers)
    elif entity_type == "gene" and database == "ncbigene":
        existing, qualifiers = find_species_with_gene_annotations(model_file, bqbiol_qualifiers, tax_id)
    elif entity_type == "protein" and database == "uniprot":
        existing, qualifiers = find_species_with_protein_annotations(model_file, bqbiol_qualifiers, tax_id)
    else:
        return None
    return existing, qualifiers


def evaluate_single_model(model_file: str,
                         llm_model: str = 'meta-llama/llama-3.3-70b-instruct:free',
                         method: str = "direct",
                         top_k: int = 3,
                         n_return: int = None,
                         max_entities: int = None,
                         entity_type: str = "chemical",
                         database: Union[str, List[str]] = "chebi",
                         model_type: str = "default",
                         save_llm_results: bool = True,
                         save_llm_results_folder: str = None,
                         output_dir: str = './results/',
                         verbose: bool = True,
                         tax_id: str = None,
                         tax_name: str = None,
                         bqbiol_qualifiers: list = None,
                         chunk_size: int = 50,
                         max_try: int = 1,
                         rate_limiter: RateLimiter = None,
                         context: bool = True,
                         validation: bool = False,
                         source_fidelity: bool = False,
                         max_completion_tokens: int = None,
                         message: str = "") -> Optional[pd.DataFrame]:
    """
    Score one model by annotating species that already have database IDs.

    Annotation goes through :func:`core.annotation_workflow.annotate_model`,
    so retrieval, ranking, validation, and source fidelity match the use-case
    workflow. ``n_return`` defaults to ``top_k`` (ranking is skipped when the
    two are equal). ``validation`` and ``source_fidelity`` default off.
    """
    _configure_verbosity(verbose)
    try:
        model_name = Path(model_file).name
        if method == "rag" and model_type not in (None, "default"):
            logger.warning(
                "model_type=%s is not used; RAG embeddings stay on the default model",
                model_type,
            )
        if max_try and max_try > 1:
            logger.warning("max_try is ignored; validation allows one structured repair")
        gold = _gold_annotations(model_file, entity_type, database, bqbiol_qualifiers, tax_id)
        if gold is None:
            if verbose:
                logger.warning(f"Entity type {entity_type} with database {database} not yet supported")
            return None
        existing_annotations, qualifier_annotations = gold
        if not existing_annotations:
            if verbose:
                logger.warning(f"No existing annotations found in {model_name}")
            return None
        species_ids = list(existing_annotations)
        if max_entities:
            species_ids = species_ids[:max_entities]
            existing_annotations = {species_id: existing_annotations[species_id] for species_id in species_ids}
            qualifier_annotations = {
                species_id: qualifier_annotations.get(species_id, {}) for species_id in species_ids
            }
        if verbose:
            logger.info(f"Evaluating {len(species_ids)} entities in {model_name}")
        if rate_limiter is not None:
            # ponytail: one slot per model, not per LLM chunk. Provider retries cover 429s.
            rate_limiter.wait_if_needed()

        folder = save_llm_results_folder or time.strftime("annotation_%Y%m%d_%H%M%S")
        save_to = Path(output_dir) / folder / Path(model_file).stem
        save_to.parent.mkdir(parents=True, exist_ok=True)
        recommendations_df, metrics = annotate_model(
            model_file,
            llm_model=llm_model,
            method=method,
            top_k=top_k,
            n_return=top_k if n_return is None else n_return,
            max_entities=None,
            entity_type=entity_type,
            database=database,
            tax_id=tax_id,
            chunk_size=chunk_size,
            annotate="species",
            save_to=str(save_to),
            verbose=verbose,
            message=message,
            max_completion_tokens=max_completion_tokens,
            validation=validation,
            source_fidelity=source_fidelity,
            species_ids=species_ids,
            context=context,
        )
        if not isinstance(metrics, dict) or metrics.get("error"):
            print(f"Failed to evaluate {model_name}: {(metrics or {}).get('error', 'no result')}")
            return None
        return evaluation_frame_from_annotations(
            recommendations_df,
            existing_annotations,
            model_name=model_name,
            qualifier_annotations=qualifier_annotations,
            metrics=metrics,
            entity_type=entity_type,
            database=database,
            tax_id=tax_id,
            tax_name=tax_name,
        )
    except ValueError:
        raise
    except Exception as e:
        print(f"Failed to evaluate {Path(model_file).name}: {e}")
        if verbose:
            logger.exception("Failed to evaluate model %s", model_file)
        return None

def get_model_taxonomy(model_file, tax_dict_file):
    tax_dict_df = pd.read_csv(tax_dict_file)
    model_file_id = model_file.replace('.xml', '')
    if model_file_id in tax_dict_df['id'].values:
        tax_id = tax_dict_df[tax_dict_df['id'] == model_file_id]['tax_id'].values[0]
        tax_name = tax_dict_df[tax_dict_df['id'] == model_file_id]['organism'].values[0]
    else:
        tax_id = None
        tax_name = None
    return tax_id, tax_name

def evaluate_models_in_folder(model_dir: str,
                             num_models: str = 'all',
                             llm_model: str = 'meta-llama/llama-3.3-70b-instruct:free',
                             method: str = "direct",
                             top_k: int = 3,
                             n_return: int = None,
                             max_entities: int = None,
                             entity_type: str = "chemical",
                             database: str = "chebi",
                             model_type: str = "default",
                             save_llm_results: bool = True,
                             save_llm_results_folder: str = None,
                             output_dir: str = './results/',
                             output_file: str = 'evaluation_results.csv',
                             start_at: int = 1,
                             verbose: bool = False,
                             tax_id: str = None,
                             tax_dict_file: str = None,
                             bqbiol_qualifiers: list = None,
                             chunk_size: int = 50,
                             max_try: int = 1,
                             rate_limit_rpm: int = 10,
                             context: bool = True,
                             validation: bool = False,
                             source_fidelity: bool = False,
                             max_completion_tokens: int = None,
                             message: str = "") -> pd.DataFrame:
    """
    Generate species evaluation statistics for multiple models in a directory.
    
    Args:
        model_dir: Directory containing SBML model files
        num_models: Number of models to evaluate ('all' or integer)
        llm_model: LLM model to use
        method: Method to use for database search ("direct", "rag")
        top_k: Identities retrieved per species or complex component
        n_return: Identities kept after ranking. Defaults to top_k, which skips ranking.
        max_entities: Maximum entities per model (None for all)
        entity_type: Type of entities to annotate
        database: Target database
        model_type: Type of embedding model ("default", "openai")
        save_llm_results: Whether to save LLM results
        save_llm_results_folder: Custom folder name for LLM results. If None, uses timestamp.
        output_dir: Directory to save results
        output_file: Name of output CSV file
        start_at: Model index to start at (1-based)
        verbose: If True, show detailed logging. If False, minimize output.
        tax_id: For gene/protein annotations, the organism's tax_id for species-specific lookup
        tax_dict_file: File containing taxonomy information for model files
        bqbiol_qualifiers: List of bqbiol qualifiers to extract (e.g. ['is', 'isVersionOf', 'hasPart'])
        chunk_size: Size of chunks to split large models into, if None, no chunking is done (default: 50)
        max_try: Maximum number of retry attempts for species with empty predictions (default: 1, no retry)
        rate_limit_rpm: Maximum LLM API requests per minute
        context: If True, include full model context (model name, reactions, notes) in prompt.
                 If False, only use display names. (default: True)
        validation: Structured LLM output, cross-database type checks, and ChEBI formula/charge filtering.
        source_fidelity: Keep literal SBML chemical names and prefer their exact ChEBI hits. Requires validation.
        max_completion_tokens: Optional per-request output-token ceiling.
        message: Optional note included in LLM prompts.
        
    Returns:
        Combined DataFrame with all evaluation results
    """
    if source_fidelity and not validation:
        raise ValueError("source_fidelity requires validation=True")

    # Configure verbosity
    _configure_verbosity(verbose)
    
    # Initialize rate limiter for API calls
    rate_limiter = None
    if rate_limit_rpm and rate_limit_rpm > 0:
        rate_limiter = RateLimiter(max_requests_per_minute=rate_limit_rpm, verbose=True)
        print(f"Rate limiting enabled: max {rate_limit_rpm} LLM requests per minute")
    
    if tax_id:
        logger.info(f"Using organism-specific search for tax_id: {tax_id}")
    
    # Clear any existing ChromaDB clients to avoid conflicts
    clear_chromadb_cache()
    
    # Get model files
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.xml') or f.endswith('.sbml')]
    model_files.sort()  # Ensure consistent ordering
    
    # Determine which models to evaluate
    if num_models == 'all':
        num_models = len(model_files)
        model_files = model_files[start_at-1:]
    else:
        num_models = int(min(num_models, len(model_files) - start_at + 1))
        model_files = model_files[start_at-1:start_at+num_models-1]
    
    logger.info(f"Evaluating {len(model_files)} models starting from index {start_at}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    final_file = output_path / output_file

    # Keep rows from models before start_at so a resumed run does not wipe them.
    all_results = []
    if start_at > 1 and final_file.exists():
        previous = pd.read_csv(final_file)
        rerun = set(model_files)
        if "model" in previous.columns and not previous.empty:
            previous = previous[~previous["model"].isin(rerun)]
        if not previous.empty:
            all_results.append(previous)
    
    if llm_model == "meta-llama/llama-3.3-70b-instruct:free":
        llm_name = "llama-3.3-70b-instruct"
    elif llm_model == "meta-llama/llama-3.3-70b-instruct":
        llm_name = "llama-3.3-70b-instruct"
    else:
        llm_name = llm_model

    # Use custom folder name or timestamp-based folder name
    if not save_llm_results_folder:
        timestamp = time.strftime('%Y%m%d_%H%M')
        save_llm_results_folder = f"{llm_name}/{entity_type}/{timestamp}"

    print(f"LLM results will be saved to: {output_dir +  save_llm_results_folder}")
    
    # Save configuration to config.txt
    if save_llm_results:
        config_dir = os.path.join(output_dir, save_llm_results_folder)
        os.makedirs(config_dir, exist_ok=True)
        config_file = os.path.join(config_dir, 'config.txt')
        
        with open(config_file, 'w') as f:
            f.write("Evaluation Configuration\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"model_dir: {model_dir}\n")
            f.write(f"num_models: {num_models}\n")
            f.write(f"llm_model: {llm_model}\n")
            f.write(f"method: {method}\n")
            f.write(f"top_k: {top_k}\n")
            f.write(f"n_return: {top_k if n_return is None else n_return}\n")
            f.write(f"max_entities: {max_entities}\n")
            f.write(f"entity_type: {entity_type}\n")
            f.write(f"database: {database}\n")
            f.write(f"model_type: {model_type}\n")
            f.write(f"output_dir: {output_dir}\n")
            f.write(f"output_file: {output_file}\n")
            f.write(f"start_at: {start_at}\n")
            f.write(f"tax_id: {tax_id}\n")
            f.write(f"tax_dict_file: {tax_dict_file}\n")
            f.write(f"bqbiol_qualifiers: {bqbiol_qualifiers}\n")
            f.write(f"chunk_size: {chunk_size}\n")
            f.write(f"max_try: {max_try}\n")
            f.write(f"rate_limit_rpm: {rate_limit_rpm}\n")
            f.write(f"context: {context}\n")
            f.write(f"validation: {validation}\n")
            f.write(f"source_fidelity: {source_fidelity}\n")
            f.write(f"max_completion_tokens: {max_completion_tokens}\n")
            f.write(f"message: {message}\n")
            f.write(f"\nTimestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        print(f"Saved configuration to {config_file}")

    # Evaluate each model
    for idx, model_file in enumerate(model_files):
        actual_idx = idx + start_at
        print(f"Evaluating {actual_idx}/{start_at + len(model_files) - 1}: {model_file}")
        
        model_path = os.path.join(model_dir, model_file)
        tax_name = None
        if tax_id == 9606 or tax_id == "9606":
            tax_name = "Homo sapiens"
        elif tax_id == 511145 or tax_id == "511145":
            tax_name = "Escherichia coli"
        elif tax_id == 10090 or tax_id == "10090":
            tax_name = "Mus musculus"
        elif tax_id == 10116 or tax_id == "10116":
            tax_name = "Rattus norvegicus"
        if tax_dict_file:
            tax_id, tax_name = get_model_taxonomy(model_file, tax_dict_file)
            if not tax_id:
                logger.warning(f"No tax_id found for {model_file}")
                continue
        
        # Evaluate single model
        result_df = evaluate_single_model(
            model_file=model_path,
            llm_model=llm_model,
            method=method,
            top_k=top_k,
            n_return=n_return,
            max_entities=max_entities,
            entity_type=entity_type,
            database=database,
            model_type=model_type,
            save_llm_results=save_llm_results,
            save_llm_results_folder=save_llm_results_folder,
            output_dir=output_dir,
            verbose=verbose,
            tax_id=tax_id,
            tax_name=tax_name,
            bqbiol_qualifiers=bqbiol_qualifiers,
            chunk_size=chunk_size,
            max_try=max_try,
            rate_limiter=rate_limiter,
            context=context,
            validation=validation,
            source_fidelity=source_fidelity,
            max_completion_tokens=max_completion_tokens,
            message=message,
        )
        
        if result_df is not None:
            all_results.append(result_df)
            pd.concat(all_results, ignore_index=True).to_csv(final_file, index=False)
    
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        combined_df.to_csv(final_file, index=False)
        logger.info(f"Saved final results to: {final_file}")
        return combined_df
    logger.warning("No results generated for any models")
    return pd.DataFrame()

def _convert_format(recommendations: List[Recommendation],
                                   existing_annotations: Dict[str, List[str]],
                                   model_name: str,
                                   synonyms_dict: Dict[str, List[str]],
                                   reason: str,
                                   total_time: float,
                                   llm_time: float,
                                   search_time: float,
                                   entity_type: str = "chemical",
                                   database: Union[str, List[str]] = "chebi",
                                   tax_id: str = None,
                                   tax_name: str = None,
                                   model_file: str = None,
                                   bqbiol_qualifiers: List[str] = None,
                                   qualifier_annotations: Dict[str, List[str]] = None,
                                   entity_type_dict: Dict[str, str] = None) -> pd.DataFrame:
    """
    Convert AAAIM recommendations to evaluation format with LLM results.
    
    Args:
        recommendations: List of Recommendation objects
        existing_annotations: Dictionary of existing annotations
        model_name: Name of the model file
        synonyms_dict: LLM-generated synonyms
        reason: LLM reasoning
        total_time: Total processing time
        llm_time: LLM query time
        search_time: Database search time
        entity_type: Type of entity being annotated
        database: Database being used
        tax_id: For gene/protein annotations, the organism's tax_id
        tax_name: For gene/protein annotations, the organism's tax_name
        model_file: Path to the model file (optional)
        bqbiol_qualifiers: List of bqbiol qualifiers used to extract annotations (optional)
        qualifier_annotations: Dictionary mapping species IDs to their qualifier lists (optional)

    Returns:
        DataFrame in evaluation format
    """
    # Convert database to list
    allowed_databases = [database] if isinstance(database, str) else database
    
    # Load required dictionaries based on database(s)
    label_dict = {}
    formula_dict = {}
    has_chebi = "chebi" in allowed_databases
    has_ncbigene = "ncbigene" in allowed_databases
    has_uniprot = "uniprot" in allowed_databases
    
    # Load all relevant label dictionaries
    if has_chebi:
        label_dict.update(load_chebi_label_dict())
        formula_dict = load_chebi_formula_dict()
    if has_ncbigene:
        label_dict.update(load_ncbigene_label_dict())
    if has_uniprot:
        label_dict.update(load_uniprot_label_dict(tax_id))
    
    # Initialize stats dictionary
    stats = {
        'recall_formula': {},
        'precision_formula': {},
        'recall_exact': {},
        'precision_exact': {}
    }
    
    # Calculate statistics per species based on entity type or database used
    for species_id in existing_annotations.keys():
        existing_ids = existing_annotations.get(species_id, [])
        predicted_ids = []
        detected_type = entity_type_dict.get(species_id, entity_type) if entity_type_dict else entity_type
        
        for rec in recommendations:
            if rec.id == species_id:
                predicted_ids = rec.candidates
                break
        
        # Calculate exact match recall and precision
        if existing_ids:
            matches = set(predicted_ids) & set(existing_ids)
            recall_exact = len(matches) / len(existing_ids)
        else:
            recall_exact = 0
        
        if predicted_ids:
            matches = set(predicted_ids) & set(existing_ids)
            precision_exact = len(matches) / len(predicted_ids)
        else:
            precision_exact = 0
        
        stats['recall_exact'][species_id] = recall_exact
        stats['precision_exact'][species_id] = precision_exact
        
        # For chemicals with ChEBI, also calculate formula-based metrics
        if detected_type == "chemical" and has_chebi:
            # Get formulas for existing ChEBI IDs
            existing_formulas = []
            for chebi_id in existing_ids:
                if chebi_id in formula_dict:
                    formula = formula_dict[chebi_id]
                    if formula:
                        existing_formulas.append(formula)
            
            # Get formulas for predicted ChEBI IDs
            predicted_formulas = []
            for chebi_id in predicted_ids:
                if chebi_id in formula_dict:
                    formula = formula_dict[chebi_id]
                    if formula:
                        predicted_formulas.append(formula)
            
            # Calculate formula-based recall
            if existing_formulas:
                formula_matches = set(predicted_formulas) & set(existing_formulas)
                recall_formula = len(formula_matches) / len(set(existing_formulas))
            else:
                recall_formula = recall_exact  # Fall back to exact if no formulas
            
            # Calculate formula-based precision
            if predicted_formulas:
                formula_matches = set(predicted_formulas) & set(existing_formulas)
                precision_formula = len(formula_matches) / len(set(predicted_formulas))
            else:
                precision_formula = precision_exact  # Fall back to exact if no formulas
            
            stats['recall_formula'][species_id] = recall_formula
            stats['precision_formula'][species_id] = precision_formula
        else:
            # For non-chemical entities, formula metrics equal exact metrics
            stats['recall_formula'][species_id] = recall_exact
            stats['precision_formula'][species_id] = precision_exact
    
    # Get display names from the model file if available
    display_names = {}
    if model_file is not None:
        display_names = get_species_display_names(model_file, entity_type)
    
    # Convert to table format
    result_rows = []
    for idx, rec in enumerate(recommendations):
        species_id = rec.id
        
        # Get existing annotation names
        existing_ids = existing_annotations.get(species_id, [])
        existing_names = [label_dict.get(db_id, db_id) for db_id in existing_ids]
        exist_annotation_name = ', '.join(existing_names) if existing_names else 'NA'
        
        # Get LLM synonyms
        llm_synonyms = synonyms_dict.get(species_id, [])
        
        # Get predictions and their names
        predictions = rec.candidates
        prediction_names = [label_dict.get(db_id, db_id) for db_id in predictions]
        prediction_names = ', '.join(prediction_names) if prediction_names else 'NA'
        
        # Calculate match scores
        match_scores = []
        if rec.match_score and llm_synonyms:
            match_scores = [match_score for match_score in rec.match_score]
        else:
            match_scores = [0.0] * len(predictions)
                
        # Get statistics for this species
        recall_formula = stats['recall_formula'].get(species_id, 0) if isinstance(stats['recall_formula'], dict) else 0
        precision_formula = stats['precision_formula'].get(species_id, 0) if isinstance(stats['precision_formula'], dict) else 0
        recall_exact = stats['recall_exact'].get(species_id, 0) if isinstance(stats['recall_exact'], dict) else 0
        precision_exact = stats['precision_exact'].get(species_id, 0) if isinstance(stats['precision_exact'], dict) else 0

        # Calculate accuracy (1 if recall > 0, 0 otherwise)
        # For chemical entities, use recall_formula; for gene/protein entities, use recall_exact
        # For auto mode, check the detected entity type
        if entity_type == "auto":
            detected_type = entity_type_dict.get(species_id, "unknown") if entity_type_dict else "unknown"
            if detected_type == "chemical":
                accuracy = 1 if recall_formula > 0 else 0
            else:  # gene, protein, or unknown
                accuracy = 1 if recall_exact > 0 else 0
        elif entity_type == "chemical":
            accuracy = 1 if recall_formula > 0 else 0
        else:  # gene, protein, or other entity types
            accuracy = 1 if recall_exact > 0 else 0
        
        # Use display name from SBML if available
        display_name = display_names.get(species_id, '')
        
        # Get specific qualifier for this species
        # For existing annotations, show the qualifier used for the matching annotation
        # For new predictions, show 'is' as default
        if existing_ids and qualifier_annotations and species_id in qualifier_annotations:
            # Find qualifiers for existing annotations
            existing_qualifiers = []
            for ann_id in existing_ids:
                if ann_id in qualifier_annotations[species_id]:
                    existing_qualifiers.append(qualifier_annotations[species_id][ann_id])
            specific_qualifier = ', '.join(existing_qualifiers) if existing_qualifiers else 'is'
        else:
            specific_qualifier = 'is'  # Default for new predictions
            
        # Only include reason for the first species in the model to save space
        species_reason = reason if idx == 0 else ''
        
        # Get detected entity type if available
        detected_entity_type = entity_type_dict.get(species_id, entity_type) if entity_type_dict else entity_type

        # Create row in AMAS format
        row = {
            'model': model_name,
            'species_id': species_id,
            'display_name': display_name,
            'detected_entity_type': detected_entity_type, 
            'synonyms_LLM': llm_synonyms,
            'reason': species_reason,
            'exist_annotation_id': existing_ids,
            'exist_annotation_name': exist_annotation_name,
            'predictions': predictions,
            'predictions_names': prediction_names,
            'match_score': match_scores,
            'recall_formula': recall_formula,
            'precision_formula': precision_formula,
            'recall_exact': recall_exact,
            'precision_exact': precision_exact,
            'accuracy': accuracy,
            'total_time': total_time,
            'llm_time': llm_time,
            'query_time': search_time,
            'tax_id': tax_id,
            'tax_name': tax_name,
            'qualifier': specific_qualifier,
        }
        result_rows.append(row)
    
    return pd.DataFrame(result_rows)

def _save_llm_results(model_file: str, llm_model: str, output_dir: str, 
                     synonyms_dict: Dict[str, List[str]], reason: str, entity_type: str, 
                     save_llm_results_folder: str = None, entity_type_dict: Dict[str, str] = None):
    """
    Save LLM results to file.
    
    Args:
        model_file: Path to model file
        llm_model: LLM model used
        output_dir: Output directory
        synonyms_dict: LLM-generated synonyms
        reason: LLM reasoning
        entity_type: Type of entity being annotated
        save_llm_results_folder: Custom folder name for LLM results. If None, uses timestamp.
        entity_type_dict: Dictionary mapping species IDs to detected entity types (optional)
    """
    model_name = Path(model_file).stem
    if llm_model == "meta-llama/llama-3.3-70b-instruct:free":
        llm_name = "llama-3.3-70b-instruct"
    elif llm_model == "meta-llama/llama-3.3-70b-instruct":
        llm_name = "llama-3.3-70b-instruct"
    else:
        llm_name = llm_model

    # Use custom folder name or timestamp-based folder name
    if save_llm_results_folder:
        output_dir = output_dir + save_llm_results_folder
    else:
        # Generate timestamp-based folder name
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        output_dir = output_dir + f"{llm_name}/{entity_type}/{timestamp}"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    output_file = Path(output_dir) / f"{model_name}.txt"
    
    with open(output_file, 'w') as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"LLM: {llm_model}\n")
        f.write(f"Entity Type: {entity_type}\n")
        f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("Synonyms:\n")
        for species_id, synonyms in synonyms_dict.items():
            detected_type = entity_type_dict.get(species_id, entity_type) if entity_type_dict else entity_type
            f.write(f"{species_id} ({detected_type}): {synonyms}\n")
        f.write(f"\nReason: {reason}\n")
    print(f"LLM results saved to: {output_file}")

def print_evaluation_results(results_csv: str, ref_results_csv = None, bqbiol_qualifiers: List[str] = None, entity_types: List[str] = None):
    """
    Print evaluation results summary.
    
    Args:
        results_csv: Path to results CSV file
        ref_results_csv: Path to reference results CSV file to filter against
        bqbiol_qualifiers: List of bqbiol qualifiers to extract (e.g. ['is', 'isVersionOf', 'hasPart'])
        entity_types: List of entity types to filter for (e.g. ['chemical', 'protein'])
    """
    decimal_places = 3
    if not os.path.exists(results_csv):
        print(f"Results file not found: {results_csv}")
        return
    
    df = pd.read_csv(results_csv)
    
    if df.empty:
        print("No results to display")
        return
    
    # Filter by reference results if provided
    if ref_results_csv:
        ref_df = pd.read_csv(ref_results_csv)
        if not ref_df.empty:
            # Create a set of (model, species_id) pairs from reference
            ref_pairs = set(zip(ref_df['model'], ref_df['species_id']))
            # Filter current results to only include pairs that exist in reference
            mask = df.apply(lambda row: (row['model'], row['species_id']) in ref_pairs, axis=1)
            df = df[mask]
            
            if df.empty:
                print("No overlapping results found between current results and reference")
                return
            
            print(f"Filtered results to {len(df)} entries that exist in reference: {ref_results_csv}")
        else:
            print(f"Reference file is empty: {ref_results_csv}")
    else:
        print(f"Showing all results")
    
    # Filter results by qualifiers if provided
    if bqbiol_qualifiers:
        print(f"Filtering results by qualifiers: {bqbiol_qualifiers}")
        # Handle both comma- and space-separated, or single, or repeated qualifiers
        def all_qualifiers_in(row):
            qualifier_str = str(row['qualifier']).strip()
            # If no ',' present, split by whitespace to catch single or multi by space, else split by comma
            if ',' in qualifier_str:
                qualifiers = [q.strip() for q in qualifier_str.split(',')]
            else:
                qualifiers = [qualifier_str]
            return qualifiers and all(q in bqbiol_qualifiers for q in qualifiers)
        df = df[df.apply(all_qualifiers_in, axis=1)]
    
    # Filter results by entity types if provided
    if entity_types:
        print(f"Filtering results by entity types: {entity_types}")
        if 'detected_entity_type' in df.columns:
            df = df[df['detected_entity_type'].isin(entity_types)]
        else:
            print("Warning: No entity type column found in results")
        
        if df.empty:
            print(f"No results found for entity types: {entity_types}")
            return
    
    print("Number of models assessed: %d" % df['model'].nunique())
    print("Number of models with predictions: %d" % df[df['predictions'] != '[]']['model'].nunique())
    print("Number of annotations evaluated: %d" % len(df)) 

    # NA and UNK rate for synonyms_LLM
    n_NA = (df['synonyms_LLM']=='[]').sum()
    n_UNK = (df['synonyms_LLM'].str.contains('UNK')).sum()
    NA_rate = n_NA / len(df)
    UNK_rate = n_UNK / len(df)
    print(f"NA rate for synonyms_LLM: {round(NA_rate, decimal_places)}")
    print(f"UNK rate for synonyms_LLM: {round(UNK_rate, decimal_places)}")
    
    # Calculate per-model averages
    print("--------------------------------")
    print("Per-model statistics:")
    model_accuracy = df.groupby('model')['accuracy'].mean().mean()
    print(f"Average accuracy (per model): {round(model_accuracy, decimal_places)}")
    
    recall_formula = df.groupby('model')['recall_formula'].mean().mean()
    print(f"Ave. recall (formula): {round(recall_formula, decimal_places)}")
    
    precision_formula = df.groupby('model')['precision_formula'].mean().mean()
    print(f"Ave. precision (formula): {round(precision_formula, decimal_places)}")
    
    recall_exact = df.groupby('model')['recall_exact'].mean().mean()
    print(f"Ave. recall (exact): {round(recall_exact, decimal_places)}")
    
    precision_exact = df.groupby('model')['precision_exact'].mean().mean()
    print(f"Ave. precision (exact): {round(precision_exact, decimal_places)}")
    
    # Calculate per-species averages
    print("--------------------------------")
    print("Per-species statistics:")
    species_accuracy = df['accuracy'].mean()
    print(f"Average accuracy (per species): {round(species_accuracy, decimal_places)}")
    
    species_recall_formula = df['recall_formula'].mean()
    print(f"Ave. recall (formula, per species): {round(species_recall_formula, decimal_places)}")
    
    species_precision_formula = df['precision_formula'].mean()
    print(f"Ave. precision (formula, per species): {round(species_precision_formula, decimal_places)}")
    
    species_recall_exact = df['recall_exact'].mean()
    print(f"Ave. recall (exact, per species): {round(species_recall_exact, decimal_places)}")
    
    species_precision_exact = df['precision_exact'].mean()
    print(f"Ave. precision (exact, per species): {round(species_precision_exact, decimal_places)}")

    print("--------------------------------")
    print("Time:")
    # Total time
    mean_processing_time = df.groupby('model')['total_time'].first().mean()
    print(f"Ave. total time (per model): {round(mean_processing_time, decimal_places)}")
    
    # Total time per element
    num_elements = df.groupby('model').size().mean()
    mean_processing_time_per_element = mean_processing_time / num_elements if num_elements > 0 else 0
    print(f"Ave. total time (per element, per model): {round(mean_processing_time_per_element, decimal_places)}")

    # LLM time
    mean_llm_time = df.groupby('model')['llm_time'].first().mean()
    print(f"Ave. LLM time (per model): {round(mean_llm_time, decimal_places)}")
    
    mean_llm_time_per_element = mean_llm_time / num_elements if num_elements > 0 else 0
    print(f"Ave. LLM time (per element, per model): {round(mean_llm_time_per_element, decimal_places)}")
    
    # Average number of predictions per species
    def safe_eval_predictions(x):
        """Safely evaluate predictions string."""
        try:
            if isinstance(x, str) and x.startswith('[') and x.endswith(']'):
                return eval(x)
            elif isinstance(x, list):
                return x
            else:
                return []
        except Exception:
            return []
    
    df['parsed_predictions'] = df['predictions'].apply(safe_eval_predictions)
    df['num_predictions'] = df['parsed_predictions'].apply(len)
    average_predictions = df['num_predictions'].mean()
    print(f"Average number of predictions per species: {round(average_predictions, decimal_places)}")

def calculate_species_statistics(recommendations: List[Recommendation],
                                existing_annotations: Dict[str, List[str]]) -> Dict[str, Dict[str, float]]:
    """
    Calculate evaluation statistics for species recommendations.
    Simplified version of getSpeciesStatistics from AMAS.
    
    Args:
        recommendations: List of Recommendation objects
        existing_annotations: Dictionary of existing annotations
        
    Returns:
        Dictionary with recall and precision statistics
    """
    stats = {}
    
    for rec in recommendations:
        species_id = rec.id
        predicted_ids = rec.candidates
        existing_ids = existing_annotations.get(species_id, [])
        
        # Calculate simple recall and precision
        if existing_ids:
            # Recall: fraction of existing annotations that were predicted
            matches = set(predicted_ids) & set(existing_ids)
            recall = len(matches) / len(existing_ids) if existing_ids else 0
        else:
            recall = 0
        
        if predicted_ids:
            # Precision: fraction of predictions that match existing annotations
            matches = set(predicted_ids) & set(existing_ids)
            precision = len(matches) / len(predicted_ids) if predicted_ids else 0
        else:
            precision = 0
        
        stats[species_id] = {
            'recall_exact': recall,
            'precision_exact': precision,
            'recall_formula': 0,  # Not implemented
            'precision_formula': 0  # Not implemented
        }
    
    return stats

def process_saved_llm_responses(response_folder: str, 
                               model_dir: str, 
                               prev_results_csv: str, 
                               method: str = "direct",
                               top_k: int = 3,
                               n_return: int = None,
                               entity_type: str = "chemical",
                               database: Union[str, List[str]] = "chebi",
                               tax_id: str = None,
                               bqbiol_qualifiers: List[str] = None,
                               output_dir: str = './results/', 
                               output_file: str = 'reprocessed_results.csv',
                               model_type: str = "default",
                               verbose: bool = False,
                               validation: bool = False,
                               source_fidelity: bool = False,
                               max_completion_tokens: int = None,
                               message: str = "") -> pd.DataFrame:
    """
    Process saved LLM response files to generate species evaluation statistics,
    keeping the same species information as in the previous results.

    Saved synonym text does not include structured normalization, chemistry
    checks, source fidelity, or identity ranking. Those arguments must stay at
    their defaults; re-run evaluate_models_in_folder to apply them.
    
    Args:
        response_folder: Path to folder containing saved LLM response files
        model_dir: Path to directory containing the original model files
        prev_results_csv: Path to previous results CSV from evaluate_models
        method: Method to use for database search ("direct", "rag")
        top_k: Number of top candidates to retrieve per species
        entity_type: Type of entity being annotated ("chemical", "gene", "protein", "auto")
        database: Database being used (string for single database, list for auto mode)
        tax_id: Taxonomy ID for NCBI gene / UniProt search, list or string
        bqbiol_qualifiers: List of bqbiol qualifiers to extract (e.g. ['is', 'isVersionOf', 'hasPart'])
        output_dir: Path to directory where results should be saved
        output_file: Name of the output CSV file
        model_type: Type of embedding model ("default", "openai") for RAG method
        verbose: If True, show detailed logging. If False, minimize output.
        
    Returns:
        DataFrame with evaluation results
    """
    if validation or source_fidelity or message or max_completion_tokens is not None or (
        n_return is not None and n_return != top_k
    ):
        raise ValueError(
            "Saved synonym files cannot replay validation, source_fidelity, "
            "n_return ranking, or prompt changes. Re-run evaluate_models_in_folder."
        )

    # Configure verbosity
    _configure_verbosity(verbose)
    
    # Clear any existing ChromaDB clients to avoid conflicts
    clear_chromadb_cache()
    
    # Load previous results
    if not os.path.exists(prev_results_csv):
        raise FileNotFoundError(f"Previous results file not found: {prev_results_csv}")
    
    prev_df = pd.read_csv(prev_results_csv)
    
    # Group previous results by model
    model_data = {}
    for model_name in prev_df['model'].unique():
        model_df = prev_df[prev_df['model'] == model_name]
        model_data[model_name] = {
            'species_info': {},
            'llm_time': model_df['llm_time'].iloc[0] if 'llm_time' in model_df.columns else 0
        }
        
        # Store species-specific information
        for _, row in model_df.iterrows():
            species_id = row['species_id']
            model_data[model_name]['species_info'][species_id] = {
                'display_name': row['display_name'],
                'exist_annotation_id': row['exist_annotation_id'],
                'exist_annotation_name': row['exist_annotation_name']
            }
    
    # List to track models with parsing errors
    parse_errors = []
    
    # Process each LLM response file
    response_files = [f for f in os.listdir(response_folder) if f.endswith('.txt')]
    
    all_results = []
    
    for idx, response_file in enumerate(response_files):
        print(f"Processing {idx+1}/{len(response_files)}: {response_file}")
        
        # Extract model name from filename (remove .txt extension but keep .xml)
        model_name = response_file.replace('.txt', '.xml')

        # Read response file
        with open(os.path.join(response_folder, response_file), 'r') as f:
            content = f.read()
        
        # Find the model name as it appears in the previous results
        if model_name not in model_data:
            # try .sbml
            model_name = model_name.replace('.xml', '.sbml')
            if model_name not in model_data:
                print(f"Model {model_name} not found in previous results, skipping")
                parse_errors.append(f"{response_file}: Model not found in previous results")
                continue
        model_file = os.path.join(model_dir, model_name)

        # Parse the LLM response
        try:
            # synonyms_dict, entity_type_dict, reason = parse_llm_response(result)
            synonyms_dict, entity_type_dict, reason, component_dict = parse_llm_response(content, entity_type)
        except Exception as e:
            logger.error(f"Error parsing LLM response for {response_file}: {e}")
            parse_errors.append(f"{response_file}: Error parsing LLM response - {str(e)}")
            continue
        
        try:
            # Only evaluate species that exist in BOTH the previous results AND the LLM response
            species_from_prev_results = set(model_data[model_name]['species_info'].keys())
            species_with_llm_synonyms = set(synonyms_dict.keys())
            
            # Find intersection - only species that have both previous results AND LLM synonyms
            specs_to_evaluate = list(species_from_prev_results & species_with_llm_synonyms)
            
            if not specs_to_evaluate:
                print(f"No overlapping species between previous results and LLM response for {model_name}, skipping")
                parse_errors.append(f"{response_file}: No overlapping species found")
                continue
            
            # print(f"Evaluating {len(specs_to_evaluate)} species for {model_name} (intersection of {len(species_from_prev_results)} prev and {len(species_with_llm_synonyms)} LLM)")
            
            # Time the database search
            query_start_time = time.time()
            
            # Handle auto entity type detection
            if entity_type == "auto":
                # Convert database to list if it's a string
                allowed_databases = [database] if isinstance(database, str) else database
                
                # Group species by detected entity type
                species_by_type = {}
                for species_id in specs_to_evaluate:
                    detected_type = entity_type_dict.get(species_id, "unknown")
                    if detected_type not in species_by_type:
                        species_by_type[detected_type] = []
                    species_by_type[detected_type].append(species_id)
                
                if verbose:
                    logger.info(f"Detected entity types: {dict((k, len(v)) for k, v in species_by_type.items())}")
                
                # Process each entity type group
                all_recommendations = []
                for detected_type, species_list in species_by_type.items():
                    # For unknown entity types, create empty recommendations
                    if detected_type == "unknown":
                        if verbose:
                            logger.warning(f"There are {len(species_list)} species with unknown entity type")
                        for species_id in species_list:
                            empty_rec = Recommendation(
                                id=species_id,
                                synonyms=synonyms_dict.get(species_id, []),
                                candidates=[],
                                candidate_names=[],
                                match_score=[]
                            )
                            all_recommendations.append(empty_rec)
                        continue
                    
                    # Special handling for complexes: per-component DB, or all DBs if untyped
                    if detected_type == "complex":
                        if verbose:
                            logger.info(f"Searching {len(species_list)} complex entities in {allowed_databases}")
                        all_recommendations.extend(_eval_search_complexes(
                            species_list, synonyms_dict, allowed_databases, method, top_k,
                            tax_id, model_type, component_dict, verbose,
                        ))
                        continue
                    
                    # Get appropriate database for this entity type
                    target_database = _get_database_for_entity_type(detected_type, allowed_databases)
                    
                    if target_database is None:
                        if verbose:
                            logger.warning(f"No valid database found for entity type '{detected_type}' in {allowed_databases} for {len(species_list)} species")
                        # Create empty recommendations for species without valid database
                        for species_id in species_list:
                            empty_rec = Recommendation(
                                id=species_id,
                                synonyms=synonyms_dict.get(species_id, []),
                                candidates=[],
                                candidate_names=[],
                                match_score=[]
                            )
                            all_recommendations.append(empty_rec)
                        continue
                    
                    if verbose:
                        logger.info(f"Searching {target_database} for {len(species_list)} {detected_type} entities")
                    
                    # Search the appropriate database
                    with suppress_outputs(verbose):
                        if method == "direct":
                            if target_database == "chebi":
                                group_recommendations = get_species_recommendations_direct(species_list, synonyms_dict, database="chebi", top_k=top_k)
                            elif target_database == "ncbigene":
                                group_recommendations = get_species_recommendations_direct(species_list, synonyms_dict, database="ncbigene", tax_id=tax_id, top_k=top_k)
                            elif target_database == "uniprot":
                                group_recommendations = get_species_recommendations_direct(species_list, synonyms_dict, database="uniprot", tax_id=tax_id, top_k=top_k)
                            else:
                                if verbose:
                                    logger.error(f"Database {target_database} not supported")
                                continue
                        elif method == "rag":
                            if target_database == "chebi":
                                group_recommendations = get_species_recommendations_rag(species_list, synonyms_dict, database="chebi", top_k=top_k, model_type=model_type)
                            elif target_database == "ncbigene":
                                group_recommendations = get_species_recommendations_rag(species_list, synonyms_dict, database="ncbigene", tax_id=tax_id, top_k=top_k, model_type=model_type)
                            elif target_database == "uniprot":
                                group_recommendations = get_species_recommendations_rag(species_list, synonyms_dict, database="uniprot", tax_id=tax_id, top_k=top_k, model_type=model_type)
                            else:
                                if verbose:
                                    logger.error(f"Database {target_database} not supported")
                                continue
                        else:
                            if verbose:
                                logger.error(f"Invalid method: {method}")
                            continue
                    
                    if group_recommendations:
                        all_recommendations.extend(group_recommendations)
                
                recommendations = all_recommendations
            else:
                # Standard single entity type workflow
                with suppress_outputs(verbose):
                    if method == "direct":
                        if database == "chebi":
                            recommendations = get_species_recommendations_direct(specs_to_evaluate, synonyms_dict, database="chebi", top_k=top_k)
                        elif database == "ncbigene":
                            recommendations = get_species_recommendations_direct(specs_to_evaluate, synonyms_dict, database="ncbigene", tax_id=tax_id, top_k=top_k)
                        elif database == "uniprot":
                            recommendations = get_species_recommendations_direct(specs_to_evaluate, synonyms_dict, database="uniprot", tax_id=tax_id, top_k=top_k)
                        else:
                            print(f"Database {database} not supported")
                            return None
                    elif method == "rag":
                        if database == "chebi":
                            recommendations = get_species_recommendations_rag(specs_to_evaluate, synonyms_dict, database="chebi", top_k=top_k, model_type=model_type)
                        elif database == "ncbigene":
                            recommendations = get_species_recommendations_rag(specs_to_evaluate, synonyms_dict, database="ncbigene", tax_id=tax_id, top_k=top_k, model_type=model_type)
                        elif database == "uniprot":
                            recommendations = get_species_recommendations_rag(specs_to_evaluate, synonyms_dict, database="uniprot", tax_id=tax_id, top_k=top_k, model_type=model_type)
                        else:
                            print(f"Database {database} not supported")
                            return None
                    else:
                        print(f"Invalid method: {method}")
                        return None
            
            query_end_time = time.time()
            dict_search_time = query_end_time - query_start_time
            
            # Get existing annotations for statistics calculation
            existing_annotations = {}
            qualifier_annotations = {}
            
            if entity_type == "auto":
                # For auto mode, collect existing annotations from all specified databases
                allowed_databases = [database] if isinstance(database, str) else database
                
                for db in allowed_databases:
                    if db == "chebi":
                        chebi_annotations, chebi_qualifiers = find_species_with_formulas(model_file, bqbiol_qualifiers)
                        existing_annotations.update(chebi_annotations)
                        qualifier_annotations.update(chebi_qualifiers)
                    elif db == "ncbigene":
                        gene_annotations, gene_qualifiers = find_species_with_gene_annotations(model_file, bqbiol_qualifiers, tax_id)
                        existing_annotations.update(gene_annotations)
                        qualifier_annotations.update(gene_qualifiers)
                    elif db == "uniprot":
                        protein_annotations, protein_qualifiers = find_species_with_protein_annotations(model_file, bqbiol_qualifiers, tax_id)
                        existing_annotations.update(protein_annotations)
                        qualifier_annotations.update(protein_qualifiers)
            elif entity_type == "chemical" and database == "chebi":
                existing_annotations, qualifier_annotations = find_species_with_formulas(model_file, bqbiol_qualifiers)
            elif entity_type == "gene" and database == "ncbigene":
                existing_annotations, qualifier_annotations = find_species_with_gene_annotations(model_file, bqbiol_qualifiers, tax_id)
            elif entity_type == "protein" and database == "uniprot":
                existing_annotations, qualifier_annotations = find_species_with_protein_annotations(model_file, bqbiol_qualifiers, tax_id)
            
            # Filter existing_annotations to match the species we're actually evaluating
            existing_annotations = {species_id: existing_annotations[species_id] 
                                    for species_id in specs_to_evaluate 
                                    if species_id in existing_annotations}
            
            # Previous LLM time from original run
            previous_llm_time = model_data[model_name]['llm_time']
            
            # Convert to AMAS-compatible format
            result_df = _convert_format(
                recommendations, existing_annotations, model_name, 
                synonyms_dict, reason, previous_llm_time + dict_search_time, 
                previous_llm_time, dict_search_time, entity_type, database, tax_id, tax_name=None, model_file=model_file, bqbiol_qualifiers=bqbiol_qualifiers, qualifier_annotations=qualifier_annotations, entity_type_dict=entity_type_dict
            )
            
            if not result_df.empty:
                all_results.append(result_df)
        
        except Exception as e:
            print(f"Error processing {model_name}: {e}")
            parse_errors.append(f"{response_file}: Error during processing - {str(e)}")
            continue
    
    # Combine all results
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save DataFrame
        output_file_path = output_path / output_file
        combined_df.to_csv(output_file_path, index=False)
        print(f"Saved results to {output_file_path}")
        
        # Save parse errors to file
        error_file_path = output_path / 'parse_errors.txt'
        with open(error_file_path, 'w') as f:
            f.write(f"Total errors: {len(parse_errors)}\n\n")
            for error in parse_errors:
                f.write(f"{error}\n")
        print(f"Saved {len(parse_errors)} parse errors to {error_file_path}")
        
        return combined_df
    else:
        print("No results generated for any models")
        return pd.DataFrame()

def compare_results(*csv_paths: str, ref_results_csv: str = None) -> dict:
    """
    Compare results from multiple CSVs by filtering to only include common models and species.
    Prints detailed statistics for each CSV and a summary comparison table.
    
    Args:
        *csv_paths: Paths to result CSVs (must be at least 2)
        ref_results_csv: Path to reference results CSV file to filter against
        
    Returns:
        Dictionary mapping CSV path to filtered DataFrame (only common models/species)
    """
    if len(csv_paths) < 2:
        raise ValueError("At least two CSV paths must be provided.")

    # Helper to safely parse predictions
    def safe_eval_predictions(x):
        try:
            if isinstance(x, str) and x.startswith('[') and x.endswith(']'):
                return eval(x)
            elif isinstance(x, list):
                return x
            else:
                return []
        except Exception:
            return []

    # Load all DataFrames and check existence
    dfs = []
    for path in csv_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Results file not found: {path}")
        dfs.append(pd.read_csv(path))

    # Filter by reference results if provided
    if ref_results_csv:
        ref_df = pd.read_csv(ref_results_csv)
        if not ref_df.empty:
            # Create a set of (model, species_id) pairs from reference
            ref_pairs = set(zip(ref_df['model'], ref_df['species_id']))
            
            # Filter each DataFrame to only include pairs that exist in reference
            filtered_dfs = []
            for df in dfs:
                mask = df.apply(lambda row: (row['model'], row['species_id']) in ref_pairs, axis=1)
                filtered_df = df[mask]
                filtered_dfs.append(filtered_df)
            print(f"Filtered all results to only include entries that exist in reference: {ref_results_csv}")
        else:
            print(f"Reference file is empty: {ref_results_csv}")
    else:
        print(f"Showing all results, filtering to common models/species in models")

        # Find common models and species_ids across all DataFrames
        model_sets = [set(df['model'].unique()) for df in dfs]
        common_models = set.intersection(*model_sets)

        # For each model, find common species_ids across all DataFrames
        common_model_species = {}
        for model in common_models:
            species_sets = []
            for df in dfs:
                species_sets.append(set(df[df['model'] == model]['species_id'].unique()))
            common_species = set.intersection(*species_sets)
            if common_species:
                common_model_species[model] = common_species

        # Flatten to set of (model, species_id) pairs
        common_pairs = set()
        for model, species_set in common_model_species.items():
            for species_id in species_set:
                common_pairs.add((model, species_id))

        if not common_pairs:
            print("No common models and species found across all results.")
            return {}

        # Filter each DataFrame to only include common (model, species_id) pairs
        filtered_dfs = []
        for df in dfs:
            filtered = df[df.apply(lambda row: (row['model'], row['species_id']) in common_pairs, axis=1)].copy()
            filtered_dfs.append(filtered)

    # Print stats for each filtered DataFrame
    print("="*70)
    print("COMPARISON OF RESULTS")
    print("="*70)
    stats = []
    for csv_path, filtered_df in zip(csv_paths, filtered_dfs):
        print(f"\n{'='*60}\nRESULTS FOR: {csv_path}\n{'='*60}")
        n_models = filtered_df['model'].nunique()
        n_models_with_preds = filtered_df[filtered_df['predictions'] != '[]']['model'].nunique()
        print(f"Number of models assessed: {n_models}")
        print(f"Number of models with predictions: {n_models_with_preds}")
        print(f"Number of species tested: {filtered_df['species_id'].nunique()}")

        # Per-model averages
        if 'accuracy' in filtered_df.columns:
            model_accuracy = filtered_df.groupby('model')['accuracy'].mean().mean()
            print(f"Average accuracy (per model): {model_accuracy:.2f}")
            print(f"Average accuracy (per species): {filtered_df['accuracy'].mean():.2f}")
        else:
            model_accuracy = None
        if 'recall_formula' in filtered_df.columns:
            recall_formula = filtered_df.groupby('model')['recall_formula'].mean().mean()
            print(f"Average recall (formula) (per model): {recall_formula:.2f}")
        else:
            recall_formula = None
        if 'precision_formula' in filtered_df.columns:
            precision_formula = filtered_df.groupby('model')['precision_formula'].mean().mean()
            print(f"Average precision (formula) (per model): {precision_formula:.2f}")
        else:
            precision_formula = None
        if 'recall_exact' in filtered_df.columns:
            recall_exact = filtered_df.groupby('model')['recall_exact'].mean().mean()
            print(f"Average recall (exact) (per model): {recall_exact:.2f}")
            print(f"Average recall (exact) (per species): {filtered_df['recall_exact'].mean():.2f}")
        else:
            recall_exact = None
        if 'precision_exact' in filtered_df.columns:
            precision_exact = filtered_df.groupby('model')['precision_exact'].mean().mean()
            print(f"Average precision (exact) (per model): {precision_exact:.2f}")
            print(f"Average precision (exact) (per species): {filtered_df['precision_exact'].mean():.2f}")
        else:
            precision_exact = None
        # Total time
        if 'total_time' in filtered_df.columns:
            mean_processing_time = filtered_df.groupby('model')['total_time'].first().mean()
            print(f"Ave. total time (per model): {mean_processing_time:.2f}")
            num_elements = filtered_df.groupby('model').size().mean()
            mean_processing_time_per_element = mean_processing_time / num_elements if num_elements > 0 else 0
            print(f"Ave. total time (per element, per model): {mean_processing_time_per_element:.2f}")
        else:
            mean_processing_time = None
            mean_processing_time_per_element = None
            num_elements = None

        if 'llm_time' in filtered_df.columns:
            mean_llm_time = filtered_df.groupby('model')['llm_time'].first().mean()
            print(f"Ave. LLM time (per model): {mean_llm_time:.2f}")
            mean_llm_time_per_element = mean_llm_time / num_elements if num_elements and num_elements > 0 else 0
            print(f"Ave. LLM time (per element, per model): {mean_llm_time_per_element:.2f}")
        else:
            mean_llm_time = None
            mean_llm_time_per_element = None

        filtered_df['parsed_predictions'] = filtered_df['predictions'].apply(safe_eval_predictions)
        filtered_df['num_predictions'] = filtered_df['parsed_predictions'].apply(len)
        avg_preds_per_species = filtered_df['num_predictions'].mean()
        print(f"Average number of predictions per species: {avg_preds_per_species:.2f}")

        stats.append({
            'CSV': csv_path,
            'Models Assessed': n_models,
            'Average Accuracy': model_accuracy,
            'Average Recall (Formula)': recall_formula,
            'Average Precision (Formula)': precision_formula,
            'Average Total Time': mean_processing_time,
            'Average LLM Time': mean_llm_time,
            'Avg Predictions per Species': avg_preds_per_species
        })

    # Print summary table
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    summary_df = pd.DataFrame(stats)
    summary_df = summary_df.set_index('CSV')
    print(summary_df.to_string(float_format="%.2f"))

    # Return dictionary of filtered DataFrames
    # return {csv_path: filtered_df for csv_path, filtered_df in zip(csv_paths, filtered_dfs)}

def add_distance_columns_to_results(results_csv: str, output_csv: str = None, model_name: str = 'all-MiniLM-L6-v2'):
    """
    Add 'distance_string' and 'distance_embedding' columns to a results CSV.
    - distance_string: normalized character difference between exist_annotation_name and predictions_names
    - distance_embedding: cosine distance between the embedding of the existing annotation name and the prediction name
    Args:
        results_csv: Path to the input results CSV
        output_csv: Path to save the new CSV (optional)
        model_name: SentenceTransformer model to use for embeddings
    Returns:
        DataFrame with new columns
    """
    import ast
    df = pd.read_csv(results_csv)
    model = SentenceTransformer(model_name)
    def norm_char_dist(a, b):
        if not isinstance(a, str) or not isinstance(b, str) or not a or not b:
            return 1.0
        return abs(len(a) - len(b)) / max(len(a), len(b)) if max(len(a), len(b)) > 0 else 0.0
    def get_first(lst):
        if isinstance(lst, list) and lst:
            return lst[0]
        if isinstance(lst, str):
            try:
                val = ast.literal_eval(lst)
                if isinstance(val, list) and val:
                    return val[0]
            except Exception:
                return lst
        return ''
    distance_strings = []
    distance_embeddings = []
    for idx, row in df.iterrows():
        exist_name = get_first(row.get('exist_annotation_name', ''))
        pred_name = get_first(row.get('predictions_names', ''))
        # String distance
        dist_str = norm_char_dist(str(exist_name), str(pred_name))
        distance_strings.append(dist_str)
        # Embedding distance
        if exist_name and pred_name:
            try:
                emb_exist = model.encode(str(exist_name), convert_to_numpy=True)
                emb_pred = model.encode(str(pred_name), convert_to_numpy=True)
                dist_emb = float(cosine_distances([emb_exist], [emb_pred])[0][0])
            except Exception:
                dist_emb = 1.0
        else:
            dist_emb = 1.0
        distance_embeddings.append(dist_emb)
    df['distance_string'] = distance_strings
    df['distance_embedding'] = distance_embeddings
    if output_csv:
        df.to_csv(output_csv, index=False)
    return df

def filter_qualifiers_in_results(results_csv: str, 
                                 bqbiol_qualifiers: List[str],
                                 model_dir: str,
                                 output_csv: str = None,
                                 entity_type: str = "chemical",
                                 database: str = "chebi") -> pd.DataFrame:
    """
    Filter previously saved results to only include species that use the specified bqbiol qualifiers.
    
    Args:
        results_csv: Path to CSV file containing previously saved evaluation results
        bqbiol_qualifiers: List of bqbiol qualifiers to filter for (e.g. ['is', 'isVersionOf', 'hasPart'])
        model_dir: Directory containing the original SBML model files
        output_csv: Path to save filtered results (optional)
        entity_type: Type of entity ("chemical" or "gene" or "protein")
        database: Database being used ("chebi" or "ncbigene" or "uniprot")
        
    Returns:
        Filtered DataFrame containing only results for species with the specified qualifiers
    """
    if not os.path.exists(results_csv):
        raise FileNotFoundError(f"Results file not found: {results_csv}")
    
    # Load the results
    df = pd.read_csv(results_csv)
    
    if df.empty:
        return df
    
    # Get unique models from the results
    models_in_results = df['model'].unique()
    
    filtered_rows = []
    
    for model_name in models_in_results:
        # Find the corresponding model file
        model_file = None
        for ext in ['.xml', '.sbml']:
            potential_path = os.path.join(model_dir, model_name.replace('.xml', '').replace('.sbml', '') + ext)
            if os.path.exists(potential_path):
                model_file = potential_path
                break
        
        if not model_file:
            logger.warning(f"Model file not found for {model_name}, skipping")
            continue
        
        # Get annotations for this model using the specified qualifiers
        if entity_type == "chemical" and database == "chebi":
            qualified_annotations = find_species_with_chebi_annotations(model_file, bqbiol_qualifiers)
        elif entity_type == "gene" and database == "ncbigene":
            qualified_annotations = find_species_with_ncbigene_annotations(model_file, bqbiol_qualifiers)
        elif entity_type == "protein" and database == "uniprot":
            qualified_annotations = find_species_with_uniprot_annotations(model_file, bqbiol_qualifiers)
        else:
            logger.warning(f"Entity type {entity_type} with database {database} not supported")
            continue
        
        # Filter rows for this model to only include species with qualified annotations
        model_rows = df[df['model'] == model_name]
        for _, row in model_rows.iterrows():
            species_id = row['species_id']
            if species_id in qualified_annotations:
                filtered_rows.append(row)
    
    # Create filtered DataFrame
    if filtered_rows:
        filtered_df = pd.DataFrame(filtered_rows)
        filtered_df.reset_index(drop=True, inplace=True)
    else:
        filtered_df = pd.DataFrame()
    
    # Save if output path provided
    if output_csv:
        filtered_df.to_csv(output_csv, index=False)
        logger.info(f"Filtered results saved to: {output_csv}")
    
    return filtered_df

def analyze_bqbiol_qualifier_statistics(model_dir: str, 
                                       output_file: str = 'bqbiol_qualifier_statistics.csv',
                                       verbose: bool = True) -> pd.DataFrame:
    """
    Analyze how many species have annotations using each bqbiol qualifier across all models,
    broken down by ontology (chebi, uniprot, ncbigene, etc.).
    
    Args:
        model_dir: Directory containing SBML model files
        output_file: Path to save the statistics table
        verbose: If True, show detailed logging
        
    Returns:
        DataFrame with statistics table where rows are ontologies, columns are qualifiers,
        and cells show the number of species that contain that ontology term for that qualifier
    """
    from core.model_info import find_species_with_chebi_annotations, find_species_with_ncbigene_annotations, find_species_with_uniprot_annotations, detect_model_format
    from utils.constants import CHEBI_URI_PATTERNS, NCBIGENE_URI_PATTERNS, UNIPROT_URI_PATTERNS, ModelType
    
    # Common bqbiol qualifiers to check
    common_qualifiers = [
        'is', 'isVersionOf', 'hasVersion', 'isDescribedBy', 'hasPart','isPartOf',
        'hasProperty', 'isPropertyOf', 'isEncodedBy', 'encodes', 'isHomologTo',  
        'occursIn', 'hasTaxon', 'isRelatedTo'
    ]
    
    # Initialize statistics storage - start with main ontologies, others will be added dynamically
    ontology_stats = {
        'chebi': {qualifier: 0 for qualifier in common_qualifiers},
        'ncbigene': {qualifier: 0 for qualifier in common_qualifiers},
        'uniprot': {qualifier: 0 for qualifier in common_qualifiers},
    }
    
    models_with_qualifiers = {qualifier: set() for qualifier in common_qualifiers}
    model_type_counts = {'SBML': 0, 'SBML-qual': 0, 'SBML-fbc': 0}
    discovered_ontologies = set()
    
    # Get all model files
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.xml') or f.endswith('.sbml')]
    
    if verbose:
        print(f"Analyzing {len(model_files)} models for bqbiol qualifier statistics...")
    
    for model_file in model_files:
        model_path = os.path.join(model_dir, model_file)
        
        if verbose:
            print(f"Processing {model_file}...")
        
        # Detect model type
        try:
            model_type, format_info = detect_model_format(model_path)
            model_type_counts[model_type.value] += 1
        except Exception as e:
            if verbose:
                logger.warning(f"Error detecting model type for {model_file}: {e}")
            continue
        
        for qualifier in common_qualifiers:
            # Check for each ontology using specific qualifiers
            try:
                # ChEBI annotations
                chebi_annotations = find_species_with_chebi_annotations(model_path, [qualifier])
                if chebi_annotations:
                    ontology_stats['chebi'][qualifier] += len(chebi_annotations)
                    models_with_qualifiers[qualifier].add(model_file)
                
                # NCBI Gene annotations  
                ncbigene_annotations = find_species_with_ncbigene_annotations(model_path, [qualifier])
                if ncbigene_annotations:
                    ontology_stats['ncbigene'][qualifier] += len(ncbigene_annotations)
                    models_with_qualifiers[qualifier].add(model_file)
                
                # UniProt annotations
                uniprot_annotations = find_species_with_uniprot_annotations(model_path, [qualifier])
                if uniprot_annotations:
                    ontology_stats['uniprot'][qualifier] += len(uniprot_annotations)
                    models_with_qualifiers[qualifier].add(model_file)
                
                # Detect and count other ontologies
                other_ontologies = _detect_and_count_other_ontologies(model_path, qualifier, model_type)
                for ontology, count in other_ontologies.items():
                    if count > 0:
                        # Add new ontology if not seen before
                        if ontology not in ontology_stats:
                            ontology_stats[ontology] = {q: 0 for q in common_qualifiers}
                        
                        ontology_stats[ontology][qualifier] += count
                        models_with_qualifiers[qualifier].add(model_file)
                        discovered_ontologies.add(ontology)
                
            except Exception as e:
                if verbose:
                    logger.warning(f"Error processing {model_file} for qualifier {qualifier}: {e}")
                continue
    
    # Convert to DataFrame
    stats_df = pd.DataFrame(ontology_stats).T  # Transpose so ontologies are rows
    stats_df.index.name = 'Ontology'
    
    # Add a summary row showing number of models that contain each qualifier
    model_counts = {qualifier: len(models_with_qualifiers[qualifier]) for qualifier in common_qualifiers}
    summary_row = pd.DataFrame([model_counts], index=['Models_with_qualifier'])
    stats_df = pd.concat([stats_df, summary_row])
    
    # Save results
    stats_df.to_csv(output_file)
    if verbose:
        print(f"Statistics saved to: {output_file}")
        print(f"\nModel type distribution:")
        for model_type, count in model_type_counts.items():
            print(f"  {model_type}: {count} models")
        
        print(f"\nDiscovered ontologies: {sorted(discovered_ontologies)}")
        print("\nSummary:")
        print(stats_df)
        
        print(f"\nNumber of models that contain at least one annotation for each qualifier:")
        for qualifier in common_qualifiers:
            count = len(models_with_qualifiers[qualifier])
            print(f"  {qualifier}: {count} models")
    
    return stats_df

def _detect_and_count_other_ontologies(model_file: str, qualifier: str, model_type: 'ModelType') -> Dict[str, int]:
    """
    Enhanced helper function to detect and count annotations for various ontologies for a specific qualifier.
    Handles different model types (SBML, SBML_FBC, SBML_QUAL).
    Extracts ontology names from identifiers.org URLs automatically.
    """
    import libsbml
    from utils.constants import ModelType
    
    reader = libsbml.SBMLReader()
    document = reader.readSBML(model_file)
    model = document.getModel()
    
    if model is None:
        return {}
    
    ontology_counts = {}
    
    def extract_ontologies_from_qualifier_content(qualifier_content: str):
        """Extract ontology names from identifiers.org URLs in qualifier content."""
        # Pattern to match the term after 'identifiers.org/' and before any following / or :
        identifiers_pattern = r'http[s]?://identifiers\.org/([^/:]+)'
        miriam_pattern = r'urn:miriam:([^:\s<>"]+)'
        
        # Find all identifiers.org URLs
        identifiers_matches = re.findall(identifiers_pattern, qualifier_content)
        miriam_matches = re.findall(miriam_pattern, qualifier_content)
        
        # Combine and count ontologies
        all_ontologies = identifiers_matches + miriam_matches
        
        for ontology in all_ontologies:
            # Clean up ontology name (remove any trailing characters)
            ontology = ontology.strip()
            if ontology and ontology not in ['chebi', 'ncbigene', 'uniprot']:  # Skip main ones we handle separately
                if ontology not in ontology_counts:
                    ontology_counts[ontology] = 0
                ontology_counts[ontology] += 1
    
    def check_annotations_for_species(species_list):
        """Helper to check annotations for a list of species objects."""
        for species in species_list:
            if species.isSetAnnotation():
                annotation_str = species.getAnnotation().toXMLString()
                
                # Check if this species has the qualifier in bqbiol
                qualifier_match = re.search(
                    r'<bqbiol:{}[^>]*?>.*?</bqbiol:{}>'.format(
                        re.escape(qualifier), re.escape(qualifier)
                    ), 
                    annotation_str, 
                    flags=re.DOTALL
                )
                
                # Also check for bqmodel qualifier (incorrect usage)
                model_qualifier_match = re.search(
                    r'<bqmodel:{}[^>]*?>.*?</bqmodel:{}>'.format(
                        re.escape(qualifier), re.escape(qualifier)
                    ), 
                    annotation_str, 
                    flags=re.DOTALL
                )
                
                if qualifier_match:
                    qualifier_content = qualifier_match.group(0)
                    extract_ontologies_from_qualifier_content(qualifier_content)
                
                if model_qualifier_match:
                    qualifier_content = model_qualifier_match.group(0)
                    extract_ontologies_from_qualifier_content(qualifier_content)
    
    # Handle different model types
    if model_type == ModelType.SBML:
        # Regular SBML models - check species
        check_annotations_for_species(model.getListOfSpecies())
        
    elif model_type == ModelType.SBML_FBC:
        # SBML-FBC models - check species and gene products
        check_annotations_for_species(model.getListOfSpecies())
        
        fbc_plugin = model.getPlugin("fbc")
        if fbc_plugin:
            gene_products = []
            for gene_product in fbc_plugin.getListOfGeneProducts():
                gene_products.append(gene_product)
            check_annotations_for_species(gene_products)
    
    elif model_type == ModelType.SBML_QUAL:
        # SBML-qual models - check qualitative species
        qual_plugin = model.getPlugin("qual")
        if qual_plugin:
            qual_species = []
            for qual_spec in qual_plugin.getListOfQualitativeSpecies():
                qual_species.append(qual_spec)
            check_annotations_for_species(qual_species)
    
    return ontology_counts

def find_models_with_many_species(model_dir: str,
                                 threshold: int = 50,
                                 entity_type: str = "chemical",
                                 database: str = "chebi",
                                 tax_id: str = None,
                                 bqbiol_qualifiers: list = None,
                                 verbose: bool = True) -> pd.DataFrame:
    """
    Find models that have more than a specified threshold of species to evaluate.
    
    Args:
        model_dir: Directory containing SBML model files
        threshold: Minimum number of species to consider a "large" model (default: 50)
        entity_type: Type of entity ("chemical", "gene", "protein")
        database: Database to check ("chebi", "ncbigene", "uniprot")
        tax_id: Taxonomy ID for NCBI gene / UniProt search
        bqbiol_qualifiers: List of bqbiol qualifiers to extract (e.g. ['is', 'isVersionOf', 'hasPart'])
        verbose: If True, show detailed logging
        
    Returns:
        DataFrame with columns: model_file, num_species, model_name
    """
    # Configure verbosity
    _configure_verbosity(verbose)
    
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.xml') or f.endswith('.sbml')]
    model_files.sort()
    
    large_models = []
    
    for model_file in model_files:
        model_path = os.path.join(model_dir, model_file)
        
        try:
            # Get existing annotations to determine number of species
            if entity_type == "chemical" and database == "chebi":
                existing_annotations, _ = find_species_with_formulas(model_path, bqbiol_qualifiers)
            elif entity_type == "gene" and database == "ncbigene":
                existing_annotations, _ = find_species_with_gene_annotations(model_path, bqbiol_qualifiers, tax_id)
            elif entity_type == "protein" and database == "uniprot":
                existing_annotations, _ = find_species_with_protein_annotations(model_path, bqbiol_qualifiers, tax_id)
            else:
                continue
                
            num_species = len(existing_annotations)
            
            if num_species > threshold:
                large_models.append({
                    'model_file': model_file,
                    'num_species': num_species
                })
                if verbose:
                    logger.info(f"Found large model: {model_file} with {num_species} species")
                    
        except Exception as e:
            if verbose:
                logger.warning(f"Error processing {model_file}: {e}")
            continue
    
    df = pd.DataFrame(large_models)
    
    return df


if __name__ == "__main__":
    _self_check_evaluation_frame()
    print("evaluation frame self-check ok")
