"""
Database Search for AAAIM

Handles database searches for annotation candidates.
Currently supports ChEBI, extensible to other databases.
"""

from pathlib import Path
import sys

# Make repo root importable when this module is executed directly (e.g. via debugger)
# or when the working directory is not the repository root.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


import os
import re
import lzma
import pickle
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple
from dataclasses import dataclass
import logging
from collections import Counter, defaultdict
from itertools import product
import chromadb
from chromadb.utils import embedding_functions
from utils.constants import (
    REF_CHEBI2LABEL, 
    REF_NAMES2CHEBI, 
    REF_NCBIGENE2LABEL, 
    REF_NAMES2NCBIGENE, 
    REF_UNIPROT2LABEL, 
    REF_NAMES2UNIPROT,
    REF_CHEBI2KEGG_COMPOUND, 
    REF_KEGG_REACTION2NAME, 
    REF_KEGG2EC, 
    REF_KEGG_REACTION_FEATURES, 
    REF_KEGG_PARSED_REACTIONS
    ) # from utils.constants import SYNONYM_WORDS_TO_REMOVE
from core.data_types import Recommendation, ReactionRecommendation
from core.reaction.hierarchy_relaxation import (
    expand_chebi_with_metadata,
    iter_chebi_for_species,
    kegg_ids_for_chebi_term,
    merge_chebi_to_kegg_mapping,
)
from core.reaction.scoring import unified_reaction_objective
from core.reaction.classification import classify_reaction
from core.reaction.kegg_definition import extract_classifications


logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

# Global ChromaDB client cache to avoid conflicts
_CHROMADB_CLIENTS = {}

# Cache for loaded dictionaries
_CHEBI_CLEANNAMES_DICT: Optional[Dict[str, List[str]]] = None
_CHEBI_LABEL_DICT: Optional[Dict[str, str]] = None
_NCBIGENE_NAMES_DICT: Optional[Dict[str, List[str]]] = None
_NCBIGENE_LABEL_DICT: Optional[Dict[str, str]] = None
_UNIPROT_NAMES_DICT: Optional[Dict[str, List[str]]] = None
_UNIPROT_LABEL_DICT: Optional[Dict[str, str]] = None
_CHEBI2KEGG_DICT: Optional[Dict[str, str]] = None
_KEGG_REACTION2NAME_DICT: Optional[Dict[str, str]] = None
_KEGG2EC_DICT: Optional[Dict[str, Dict[str, List[str]]]] = None
_KEGG_REACTION_FEATURES_DICT: Optional[Dict[str, Dict[str, Any]]] = None
_KEGG_PARSED_REACTIONS_DICT: Optional[Dict[str, Dict[str, Any]]] = None

def get_data_dir() -> Path:
    """Get the path to the AAAIM data directory."""
    current_dir = Path(__file__).parent.parent
    return current_dir / "data" 

def load_chebi_cleannames_dict() -> Dict[str, List[str]]:
    """
    Load the ChEBI clean names to ChEBI ID dictionary.
    
    Returns:
        Dictionary mapping clean names to lists of ChEBI IDs
    """
    global _CHEBI_CLEANNAMES_DICT
    
    if _CHEBI_CLEANNAMES_DICT is None:
        data_file = get_data_dir() / "chebi" / REF_NAMES2CHEBI
        
        if not data_file.exists():
            raise FileNotFoundError(f"ChEBI cleannames data file not found: {data_file}")
        
        with lzma.open(data_file, 'rb') as f:
            _CHEBI_CLEANNAMES_DICT = pickle.load(f)
    
    return _CHEBI_CLEANNAMES_DICT

def load_chebi_label_dict() -> Dict[str, str]:
    """
    Load the ChEBI ID to label dictionary.
    
    Returns:
        Dictionary mapping ChEBI IDs to their labels
    """
    global _CHEBI_LABEL_DICT
    
    if _CHEBI_LABEL_DICT is None:
        data_file = get_data_dir() / "chebi" / REF_CHEBI2LABEL
        
        if not data_file.exists():
            raise FileNotFoundError(f"ChEBI label data file not found: {data_file}")
        
        with lzma.open(data_file, 'rb') as f:
            _CHEBI_LABEL_DICT = pickle.load(f)
    
    return _CHEBI_LABEL_DICT

def load_ncbigene_names_dict(tax_id: str = None) -> Dict[str, List[str]]:
    """
    Load the NCBI gene names to NCBI gene ID dictionary.
    
    Args:
        tax_id: If provided, loads organism-specific reference file.
                If None, tries to load the old combined file for backwards compatibility.
    
    Returns:
        Dictionary mapping clean names to lists of NCBI gene IDs
    """
    global _NCBIGENE_NAMES_DICT
    
    # Use a cache key that includes tax_id to handle multiple organisms
    cache_key = f"ncbigene_names_{tax_id or 'combined'}"
    
    # Check if we have this specific version cached
    if not hasattr(load_ncbigene_names_dict, '_cache'):
        load_ncbigene_names_dict._cache = {}
    
    if cache_key in load_ncbigene_names_dict._cache:
        return load_ncbigene_names_dict._cache[cache_key]
    
    if tax_id:
        # Load organism-specific file
        data_file = get_data_dir() / "ncbigene" / f"names2ncbigene_tax{tax_id}_protein-coding.lzma"
    else:
        # Try to load combined file
        data_file = get_data_dir() / "ncbigene" / REF_NAMES2NCBIGENE
    
    if not data_file.exists():
        if tax_id:
            raise FileNotFoundError(f"NCBI gene names data file not found for tax_id {tax_id}: {data_file}")
        else:
            raise FileNotFoundError(f"NCBI gene names data file not found: {data_file}")
    
    with lzma.open(data_file, 'rb') as f:
        names_dict = pickle.load(f)
    
    # Cache the result
    load_ncbigene_names_dict._cache[cache_key] = names_dict
    
    return names_dict

def load_ncbigene_label_dict() -> Dict[str, str]:
    """
    Load the NCBI gene ID to label dictionary.
    
    Returns:
        Dictionary mapping NCBI gene IDs to their labels
    """
    global _NCBIGENE_LABEL_DICT
    
    if _NCBIGENE_LABEL_DICT is None:
        data_file = get_data_dir() / "ncbigene" / REF_NCBIGENE2LABEL
        
        if not data_file.exists():
            raise FileNotFoundError(f"NCBI gene label data file not found: {data_file}")
        
        with lzma.open(data_file, 'rb') as f:
            _NCBIGENE_LABEL_DICT = pickle.load(f)
    
    return _NCBIGENE_LABEL_DICT

def load_uniprot_names_dict(tax_id: str = None) -> Dict[str, List[str]]:
    """
    Load the UniProt clean names to UniProt ID dictionary.
    
    Args:
        tax_id: If provided, loads organism-specific reference file.
                If None, tries to load the old combined file for backwards compatibility.
    
    Returns:
        Dictionary mapping clean names to lists of UniProt IDs
    """
    global _UNIPROT_NAMES_DICT
    
    # Use a cache key that includes tax_id to handle multiple organisms
    cache_key = f"uniprot_names_{tax_id or 'combined'}"
    
    # Check if we have this specific version cached
    if not hasattr(load_uniprot_names_dict, '_cache'):
        load_uniprot_names_dict._cache = {}
    
    if cache_key in load_uniprot_names_dict._cache:
        return load_uniprot_names_dict._cache[cache_key]
    
    if tax_id:
        # Load organism-specific file
        data_file = get_data_dir() / "uniprot" / f"names2uniprot_tax{tax_id}.lzma"
    else:
        # Try to load combined file
        data_file = get_data_dir() / "uniprot" / REF_NAMES2UNIPROT
    
    if not data_file.exists():
        if tax_id:
            raise FileNotFoundError(f"UniProt names data file not found for tax_id {tax_id}: {data_file}")
        else:
            raise FileNotFoundError(f"UniProt names data file not found: {data_file}")
    
    with lzma.open(data_file, 'rb') as f:
        names_dict = pickle.load(f)
    
    # Cache the result
    load_uniprot_names_dict._cache[cache_key] = names_dict
    
    return names_dict

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
    
    if _UNIPROT_LABEL_DICT is not None:
        return _UNIPROT_LABEL_DICT

    if tax_id:
        # Load organism-specific file
        data_file = get_data_dir() / "uniprot" / f"uniprot2label_tax{tax_id}.lzma"
    else:
        # Try to load combined file
        data_file = get_data_dir() / "uniprot" / REF_UNIPROT2LABEL
    
    if not data_file.exists():
        if tax_id:
            raise FileNotFoundError(f"UniProt label data file not found for tax_id {tax_id}: {data_file}")
        else:
            raise FileNotFoundError(f"UniProt label data file not found: {data_file}")
    
    with lzma.open(data_file, 'rb') as f:
        label_dict = pickle.load(f)
    
    return label_dict

def load_chebi2kegg_dict() -> Dict[str, str]:
    """
    Load the reference ChEBI → KEGG compound mapping (pickled).

    Values are typically a single KEGG compound id but may be lists where one
    ChEBI maps to several compounds. For graph/scoring code, normalize with
    ``hierarchy_relaxation.merge_chebi_to_kegg_mapping``. For expanding
    amendment tables row-wise, use ``utils.map_chebi_to_kegg``.

    Returns:
        Raw mapping as loaded from disk (ChEBI id → KEGG id(s)).
    """
    global _CHEBI2KEGG_DICT
    
    if _CHEBI2KEGG_DICT is None:
        data_file = get_data_dir() / "kegg" / REF_CHEBI2KEGG_COMPOUND
        
        if not data_file.exists():
            raise FileNotFoundError(f"ChEBI to KEGG compound mapping file not found: {data_file}")
        
        with lzma.open(data_file, 'rb') as f:
            _CHEBI2KEGG_DICT = pickle.load(f)
    
    return _CHEBI2KEGG_DICT

def load_kegg_label_dict(): 
    # For KEGG reaction annotation, we don't currently maintain a species/reaction-id → label
    # dictionary analogous to ChEBI/NCBIGene/UniProt. Returning an empty dict keeps downstream
    # code paths consistent (callers expect a mapping with ``.get``).
    return {}

def load_kegg_reaction2name_dict() -> Dict[str, str]:
    """
    Load the KEGG reaction ID to name dictionary.
    
    Returns:
        Dictionary mapping KEGG reaction IDs to their names
    """
    global _KEGG_REACTION2NAME_DICT
    
    if _KEGG_REACTION2NAME_DICT is None:
        data_file = get_data_dir() / "kegg" / REF_KEGG_REACTION2NAME
        
        if not data_file.exists():
            raise FileNotFoundError(f"KEGG reaction to name mapping file not found: {data_file}")
        
        with lzma.open(data_file, 'rb') as f:
            _KEGG_REACTION2NAME_DICT = pickle.load(f)
    
    return _KEGG_REACTION2NAME_DICT

def load_kegg2ec_dict() -> Dict[str, Dict[str, List[str]]]:
    """
    Load the KEGG ID to EC number mapping dictionary.
    
    Returns:
        Dictionary mapping KEGG IDs to EC numbers with additional metadata
    """
    global _KEGG2EC_DICT
    
    if _KEGG2EC_DICT is None:
        data_file = get_data_dir() / "kegg" / REF_KEGG2EC
        
        if not data_file.exists():
            raise FileNotFoundError(f"KEGG to EC mapping file not found: {data_file}")
        
        with lzma.open(data_file, 'rb') as f:
            _KEGG2EC_DICT = pickle.load(f)
    
    return _KEGG2EC_DICT

def load_kegg_reaction_features_dict() -> Dict[str, Dict[str, Any]]:
    """
    Load the parsed KEGG reactions dictionary containing detailed reaction features.
    
    The dictionary contains information about KEGG reactions including:
        {'R01600': {
            'ENTRY': 'R01600                      Reaction',
            'NAME': 'ATP:beta-D-glucose 6-phosphotransferase',
            'DEFINITION': 'ATP + beta-D-Glucose <=> ADP + beta-D-Glucose 6
            'EQUATION': 'C00002 + C00221 <=> C00008 + C01172',
            'RCLASS': 'RC00002  C00002_C00008\nRC00017  C00221_C01172',
            'ENZYME': '2.7.1.1         2.7.1.2',
            'PATHWAY': 'rn00010  Glycolysis / Gluconeogenesis\nrn01100 
            'BRITE': 'Enzymatic reactions [BR:br08201]\n2. Transferase 
            'ORTHOLOGY': 'K00844  hexokinase [EC:2.7.1.1]\nK00845  glucoki
            }
        }
    
    Returns:
        Dictionary mapping KEGG reaction IDs to their feature dictionaries
    """
    global _KEGG_REACTION_FEATURES_DICT
    
    if _KEGG_REACTION_FEATURES_DICT is None:
        data_file = get_data_dir() / "kegg" / REF_KEGG_REACTION_FEATURES
        
        if not data_file.exists():
            raise FileNotFoundError(f"KEGG reaction features data file not found: {data_file}")
        
        with lzma.open(data_file, 'rb') as f:
            _KEGG_REACTION_FEATURES_DICT = pickle.load(f)
    
    return _KEGG_REACTION_FEATURES_DICT


def load_kegg_parsed_reactions_dict() -> Dict[str, Dict[str, Any]]:
    """
    Load the list of dicts containing detailed reaction features.
    
    Each dictionary contains information about KEGG reactions including:
    - 'reaction_id': 'R00002',
    - 'name': 'reduced ferredoxin:dinitrogen oxidoreductase (ATP-hydrolysing)',
    - 'ec_numbers': ['1.18.6.1'],
    - 'direction': 'reversible',
    - 'substrates': ['C00002', 'C00001', 'C00138'],
    - 'products': ['C05359', 'C00009', 'C00008', 'C00139'],
    - 'pathways': [],
    - 'raw_equation': '16 C00002 + 16 C00001 + 8 C00138 <=> 8 C05359 + 16 C00009 + 16 C00008 + 8 C00139'}
    
    Returns:
        Dictionary mapping KEGG reaction IDs to their feature dictionaries
    """
    global _KEGG_PARSED_REACTIONS_DICT
    
    if _KEGG_PARSED_REACTIONS_DICT is None:
        data_file = get_data_dir() / "kegg" / REF_KEGG_PARSED_REACTIONS
        
        if not data_file.exists():
            raise FileNotFoundError(f"Parsed KEGG reactions data file not found: {data_file}")
        
        with lzma.open(data_file, 'rb') as f:
            _KEGG_PARSED_REACTIONS_DICT = pickle.load(f)
    
    return _KEGG_PARSED_REACTIONS_DICT


def remove_symbols(text: str) -> str:
    """
    Remove all characters except numbers and letters.
    
    Args:
        text: Input text to clean
        
    Returns:
        Text with only alphanumeric characters
    """
    return re.sub(r'[^a-zA-Z0-9]', '', text)


# def clean_synonym(synonym: str) -> str:
#     """
#     Clean a synonym by removing unwanted words and normalizing.
    
#     This function:
#     1. Converts to lowercase
#     # 2. Removes words from SYNONYM_WORDS_TO_REMOVE (case-insensitive, whole word match)
#     3. Removes symbols (keeps only alphanumeric characters)
    
#     Args:
#         synonym: The synonym string to clean
        
#     Returns:
#         Cleaned and normalized synonym
#     """
#     if not synonym or synonym == 'UNK':
#         return synonym
    
#     # Convert to lowercase
#     text = synonym.lower()
    
#     # # Remove unwanted words (whole word match, case-insensitive)
#     # # Sort by length descending to remove longer phrases first (e.g., "plasma membrane" before "membrane")
#     # words_to_remove = sorted(SYNONYM_WORDS_TO_REMOVE, key=len, reverse=True)
#     # for word in words_to_remove:
#     #     # Use word boundary to match whole words only
#     #     # Handle multi-word phrases and single words
#     #     pattern = r'\b' + re.escape(word.lower()) + r'\b'
#     #     text = re.sub(pattern, '', text)
    
#     # # Remove extra whitespace
#     # text = ' '.join(text.split())
    
#     # # Strip leading/trailing whitespace and common punctuation artifacts
#     # text = text.strip(' -_,;:')
    
#     return text


# def clean_synonyms(synonyms: List[str]) -> List[str]:
#     """
#     Clean a list of synonyms by removing unwanted words and normalizing.
    
#     If cleaning removes all content from a synonym, the normalized original 
#     (lowercase, symbols removed) is kept to avoid losing potentially useful terms.
    
#     Args:
#         synonyms: List of synonym strings to clean
        
#     Returns:
#         List of cleaned synonyms (duplicates are removed)
#     """
#     cleaned = []
#     seen = set()
    
#     for synonym in synonyms:
#         clean_syn = clean_synonym(synonym)
        
#         # If cleaning removed everything, use the normalized original
#         if not clean_syn:
#             clean_syn = synonym.lower().strip()
        
#         if clean_syn and clean_syn not in seen:
#             cleaned.append(clean_syn)
#             seen.add(clean_syn)
    
#     return cleaned if cleaned else synonyms  # Return original if all cleaned to empty


def get_species_recommendations_direct(species_ids: List[str], synonyms_dict, database: str = "chebi", tax_id: Any = None, top_k: int = 3) -> List[Recommendation]:
    """
    Find recommendations by directly matching against database synonyms.
    
    Parameters:
    - species_ids (list): List of species IDs to evaluate.
    - synonyms_dict (dict): Mapping of species IDs to synonyms.
    - database (str): Database to search ("chebi", "ncbigene", "uniprot")
    - tax_id (str/list): For ncbigene/uniprot database, the organism's tax_id for organism-specific lookup. If list, search all tax_ids for each species.
    - top_k (int): Number of top candidates to return per species based on hit_count.
    
    Returns:
    - list: List of Recommendation objects with candidates and names.
    """
    if database == "chebi":
        return _get_chebi_recommendations_direct(species_ids, synonyms_dict, top_k=top_k)
    elif database == "ncbigene":
        return _get_ncbigene_recommendations_direct(species_ids, synonyms_dict, tax_id=tax_id, top_k=top_k)
    elif database == "uniprot":
        return _get_uniprot_recommendations_direct(species_ids, synonyms_dict, tax_id=tax_id, top_k=top_k)
    elif database == "kegg":
        return _get_kegg_recommendations_direct(species_ids, synonyms_dict, top_k=top_k)
    else:
        logger.error(f"Database {database} not supported for direct search")
        return []

def _get_chebi_recommendations_direct(species_ids: List[str], synonyms_dict, top_k: int = 3) -> List[Recommendation]:
    """
    Find ChEBI recommendations by directly matching against ChEBI synonyms.
    """
    cleannames_dict = load_chebi_cleannames_dict()
    label_dict = load_chebi_label_dict()
    
    recommendations = []
    
    for spec_id in species_ids:
        # Get synonyms for this species ID
        if isinstance(synonyms_dict, dict):
            synonyms = synonyms_dict.get(spec_id, [spec_id])
        elif isinstance(synonyms_dict, tuple) and len(synonyms_dict) == 2:
            # If it's a tuple with two items (dict and reason)
            synonyms = synonyms_dict[0].get(spec_id, [spec_id])
        else:
            synonyms = [spec_id]
        
        # Skip if only 'UNK' synonym
        if synonyms == ['UNK'] or (len(synonyms) == 1 and synonyms[0] == 'UNK'):
            # Create empty recommendation for UNK
            recommendation = Recommendation(
                id=spec_id,
                synonyms=synonyms,
                candidates=[],
                candidate_names=[],
                match_score=[]
            )
            recommendations.append(recommendation)
            continue
        
        
        all_candidates = []
        all_candidate_names = []
        hit_count = {}  # Dictionary to track how many times each candidate appears
        
        # Query for each synonym
        for synonym in synonyms:
            norm_synonym = remove_symbols(synonym.lower())
            # Check all entries in cleannames dict for matches
            for ref_name, chebi_ids in cleannames_dict.items():
                if norm_synonym == ref_name.lower():
                    for chebi_id in chebi_ids:
                        chebi_name = label_dict.get(chebi_id, chebi_id)
                        
                        if chebi_id not in all_candidates:
                            all_candidates.append(chebi_id)
                            all_candidate_names.append(chebi_name)
                            hit_count[chebi_id] = 1
                        else:
                            hit_count[chebi_id] += 1
        
        # Sort candidates by hit_count (descending) and take top_k
        if all_candidates:
            # Create list of (candidate, name, hit_count) tuples
            candidate_tuples = [(candidate, name, hit_count[candidate]) 
                               for candidate, name in zip(all_candidates, all_candidate_names)]
            
            # Sort by hit_count descending
            candidate_tuples.sort(key=lambda x: x[2], reverse=True)
            
            # Take top_k candidates
            top_candidates = candidate_tuples[:top_k]
            
            # Extract sorted lists
            all_candidates = [candidate for candidate, _, _ in top_candidates]
            all_candidate_names = [name for _, name, _ in top_candidates]
        
        # Calculate normalized match scores (hit_count / number_of_synonyms)
        num_synonyms = len(synonyms)
        match_score_list = [hit_count.get(candidate, 0) / num_synonyms for candidate in all_candidates]
        
        # Create recommendation object
        recommendation = Recommendation(
            id=spec_id,
            synonyms=synonyms,
            candidates=all_candidates,
            candidate_names=all_candidate_names,
            match_score=match_score_list
        )
        recommendations.append(recommendation)
    
    return recommendations

def _get_ncbigene_recommendations_direct(species_ids: List[str], synonyms_dict, tax_id: Any = None, top_k: int = 3) -> List[Recommendation]:
    """
    Find NCBI gene recommendations by directly matching against NCBI gene synonyms.
    Args:
        species_ids: List of species IDs to evaluate
        synonyms_dict: Mapping of species IDs to synonyms
        tax_id: Organism's tax_id for each species (str, list). If list, search all tax_ids for each species.
        top_k: Number of top candidates to return per species based on hit_count.
    """
    label_dict = load_ncbigene_label_dict()
    recommendations = []
    for spec_id in species_ids:
        # Get synonyms for this species ID
        if isinstance(synonyms_dict, dict):
            synonyms = synonyms_dict.get(spec_id, [spec_id])
        elif isinstance(synonyms_dict, tuple) and len(synonyms_dict) == 2:
            # If it's a tuple with two items (dict and reason)
            synonyms = synonyms_dict[0].get(spec_id, [spec_id])
        else:
            synonyms = [spec_id]
        # Skip if only 'UNK' synonym
        if synonyms == ['UNK'] or (len(synonyms) == 1 and synonyms[0] == 'UNK'):
            # Create empty recommendation for UNK
            recommendation = Recommendation(
                id=spec_id,
                synonyms=synonyms,
                candidates=[],
                candidate_names=[],
                match_score=[]
            )
            recommendations.append(recommendation)
            continue
        
        # # Clean synonyms to remove unwanted words before search
        # cleaned_synonyms = clean_synonyms(synonyms)
        
        all_candidates = []
        all_candidate_names = []
        hit_count = {}
        # Determine which tax_ids to search
        if isinstance(tax_id, list):
            tax_ids_to_search = tax_id
        else:
            tax_ids_to_search = [tax_id]
        # Query for each synonym and each tax_id
        for synonym in synonyms:
            norm_synonym = remove_symbols(synonym.lower())
            for tid in tax_ids_to_search:
                try:
                    names_dict = load_ncbigene_names_dict(tax_id=tid)
                except Exception as e:
                    logger.warning(f"Error loading NCBI gene names for tax_id {tid}: {e}")
                    continue
                for ref_name, gene_ids in names_dict.items():
                    if norm_synonym == ref_name.lower():
                        for gene_id in gene_ids:
                            gene_name = label_dict.get(gene_id, gene_id)
                            if gene_id not in all_candidates:
                                all_candidates.append(gene_id)
                                all_candidate_names.append(gene_name)
                                hit_count[gene_id] = 1
                            else:
                                hit_count[gene_id] += 1
        
        # Sort candidates by hit_count (descending) and take top_k
        if all_candidates:
            # Create list of (candidate, name, hit_count) tuples
            candidate_tuples = [(candidate, name, hit_count[candidate]) 
                               for candidate, name in zip(all_candidates, all_candidate_names)]
            
            # Sort by hit_count descending
            candidate_tuples.sort(key=lambda x: x[2], reverse=True)
            
            # Take top_k candidates
            top_candidates = candidate_tuples[:top_k]
            
            # Extract sorted lists
            all_candidates = [candidate for candidate, _, _ in top_candidates]
            all_candidate_names = [name for _, name, _ in top_candidates]
        
        num_synonyms = len(synonyms)
        match_score_list = [hit_count.get(candidate, 0) / num_synonyms for candidate in all_candidates]
        
        # Create recommendation object
        recommendation = Recommendation(
            id=spec_id,
            synonyms=synonyms,
            candidates=all_candidates,
            candidate_names=all_candidate_names,
            match_score=match_score_list
        )
        recommendations.append(recommendation)
    return recommendations

def _get_uniprot_recommendations_direct(species_ids: List[str], synonyms_dict, tax_id: Any = None, top_k: int = 3) -> List[Recommendation]:
    """
    Find UniProt recommendations by directly matching against UniProt synonyms.
    Args:
        species_ids: List of species IDs to evaluate
        synonyms_dict: Mapping of species IDs to synonyms
        tax_id: Organism's tax_id for each species (str, list). If list, search all tax_ids for each species.
        top_k: Number of top candidates to return per species based on hit_count.
    """
    label_dict = load_uniprot_label_dict(tax_id=tax_id)
    recommendations = []
    for spec_id in species_ids:
        # Get synonyms for this species ID
        if isinstance(synonyms_dict, dict):
            synonyms = synonyms_dict.get(spec_id, [spec_id])
        elif isinstance(synonyms_dict, tuple) and len(synonyms_dict) == 2:
            # If it's a tuple with two items (dict and reason)
            synonyms = synonyms_dict[0].get(spec_id, [spec_id])
        else:
            synonyms = [spec_id]
        # Skip if only 'UNK' synonym
        if synonyms == ['UNK'] or (len(synonyms) == 1 and synonyms[0] == 'UNK'):
            # Create empty recommendation for UNK
            recommendation = Recommendation(
                id=spec_id,
                synonyms=synonyms,
                candidates=[],
                candidate_names=[],
                match_score=[]
            )
            recommendations.append(recommendation)
            continue
        
        # # Clean synonyms to remove unwanted words before search
        # cleaned_synonyms = clean_synonyms(synonyms)
        
        all_candidates = []
        all_candidate_names = []
        hit_count = {}
        # Determine which tax_ids to search
        if isinstance(tax_id, list):
            tax_ids_to_search = tax_id
        else:
            tax_ids_to_search = [tax_id]
        # Query for each cleaned synonym and each tax_id
        for synonym in synonyms:
            norm_synonym = remove_symbols(synonym.lower())
            # print(f"Norm synonym: {norm_synonym}, synonym: {synonym}")
            for tid in tax_ids_to_search:
                try:
                    names_dict = load_uniprot_names_dict(tax_id=tid)
                except Exception as e:
                    logger.warning(f"Error loading UniProt names for tax_id {tid}: {e}")
                    continue
                for ref_name, uniprot_ids in names_dict.items():
                    if norm_synonym == ref_name.lower():
                        for uniprot_id in uniprot_ids:
                            # print(f"Uniprot id: {uniprot_id}, ref_name: {ref_name}")
                            uniprot_name = label_dict.get(uniprot_id, uniprot_id)
                            if uniprot_id not in all_candidates:
                                all_candidates.append(uniprot_id)
                                all_candidate_names.append(uniprot_name)
                                hit_count[uniprot_id] = 1
                            else:
                                hit_count[uniprot_id] += 1
        
        # Sort candidates by hit_count (descending) and take top_k
        if all_candidates:
            # Create list of (candidate, name, hit_count) tuples
            candidate_tuples = [(candidate, name, hit_count[candidate]) 
                               for candidate, name in zip(all_candidates, all_candidate_names)]
            
            # Sort by hit_count descending
            candidate_tuples.sort(key=lambda x: x[2], reverse=True)
            
            # Take top_k candidates
            top_candidates = candidate_tuples[:top_k]
            
            # Extract sorted lists
            all_candidates = [candidate for candidate, _, _ in top_candidates]
            all_candidate_names = [name for _, name, _ in top_candidates]
        
        num_synonyms = len(synonyms)
        match_score_list = [hit_count.get(candidate, 0) / num_synonyms for candidate in all_candidates]
        
        # Create recommendation object
        recommendation = Recommendation(
            id=spec_id,
            synonyms=synonyms,
            candidates=all_candidates,
            candidate_names=all_candidate_names,
            match_score=match_score_list
        )
        recommendations.append(recommendation)
    return recommendations


def _get_kegg_recommendations_rulebased(
    normalized_reactions,
    cofactors_to_ignore: set = {},
    top_k: int = None,
    spectators: bool = False,
    *,
    relaxation_levels_by_entity: Optional[Mapping[str, int]] = None,
    penalty_lam: float = 0.0,
    max_relax_level: int = 1,
    species_to_chebi: Optional[Mapping[str, Any]] = None,
    parent_map: Optional[Mapping[str, set]] = None,
    child_map: Optional[Mapping[str, set]] = None,
    chebi_to_kegg: Optional[Mapping[str, Any]] = None,
    max_ancestor_depth: int = 2,
    max_descendant_depth: int = 1,
    max_species_relax_depth: Optional[int] = None,
    max_reaction_relax_depth: Optional[int] = None,
) -> List[Recommendation]:
    """
    Find KEGG reaction recommendations by matching model reactions to KEGG reactions.
    
    Args:
        species_ids: List of reaction IDs to evaluate
        cofactors_to_ignore: Set of KEGG IDs of cofactors to ignore
        top_k: Number of top candidates to return per reaction
        
    Returns:
        List of Recommendation objects with candidates and match scores
    """
    def split_recommendation(rec: ReactionRecommendation) -> List[ReactionRecommendation]:
        """Split one ReactionRecommendation into multiple per candidate."""
        new_recs = []
        for cand, cand_name, score in zip(rec.candidates, rec.candidate_names, rec.match_score):
            new_recs.append(
                ReactionRecommendation(
                    id=rec.id,
                    synonyms=rec.synonyms,
                    candidates=[cand],              # single candidate
                    candidate_names=[cand_name],    # single name
                    match_score=[score],            # single score
                    substrates=rec.substrates,
                    products=rec.products,
                    equation=rec.equation,
                    metadata=rec.metadata
                )
            )
        return new_recs
    
    # Helper to normalize a candidate entry into a single KEGG ID string
    def _normalize_cand_id(c):
        if isinstance(c, (list, tuple, set)):
            return next(iter(c)) if c else None
        if isinstance(c, dict):
            return c.get("kegg_id")
        return c

    def _candidate_meta(c) -> Dict[str, Any]:
        if isinstance(c, dict):
            return {
                "canonical_id": str(c.get("canonical_id", "")).strip(),
                "direction": str(c.get("direction", "exact")).strip(),
                "distance": int(c.get("distance", 0) or 0),
            }
        return {"canonical_id": "", "direction": "exact", "distance": 0}

    def _collect_side(side_block: Dict[str, Any], participant_relaxation: Dict[str, Dict[str, Any]]):
        side_keys = set()
        side_counter = Counter()
        for species_id, v in side_block.items():
            coeff = v.get('coeff', 1)
            for cand in v.get('candidates', []):
                cid = _normalize_cand_id(cand)
                if cid is None:
                    continue
                side_keys.add(cid)
                side_counter[cid] += coeff
                meta = _candidate_meta(cand)
                key = f"{species_id}|{cid}" if species_id else cid
                prev = participant_relaxation.get(key)
                if prev is None or meta["distance"] < prev["distance"]:
                    participant_relaxation[key] = {
                        "species_id": species_id,
                        "canonical_id": meta["canonical_id"],
                        "kegg_id": cid,
                        "direction": meta["direction"],
                        "distance": meta["distance"],
                    }
                elif meta["distance"] == prev["distance"]:
                    rank = {"exact": 2, "down": 1, "up": 0}
                    if rank.get(meta["direction"], -1) > rank.get(prev.get("direction"), -1):
                        participant_relaxation[key] = {
                            "species_id": species_id,
                            "canonical_id": meta["canonical_id"],
                            "kegg_id": cid,
                            "direction": meta["direction"],
                            "distance": meta["distance"],
                        }
        return side_keys, side_counter

    def _strict_filter(sub_keys: set, prod_keys: set):
        only_cofactors_subs = all(key in cofactors_to_ignore for key in sub_keys)
        only_cofactors_prods = all(key in cofactors_to_ignore for key in prod_keys)
        _fk = {}
        if chebi_to_kegg is not None and parent_map is not None:
            _fk = {
                "chebi_to_kegg": chebi_to_kegg,
                "parent_map": parent_map,
                "child_map": child_map,
                "reaction_ontology_max_up": max_ancestor_depth,
                "reaction_ontology_max_down": max_descendant_depth,
            }
        if only_cofactors_subs or only_cofactors_prods:
            candidates = filter_kegg_reactions(sub_keys, prod_keys, **_fk)
            filtered_species_local = set(sub_keys) | set(prod_keys)
        else:
            candidates = filter_kegg_reactions(sub_keys, prod_keys, **_fk) + filter_kegg_reactions(
                sub_keys, prod_keys, cofactors_to_ignore=cofactors_to_ignore, **_fk
            )
            candidates = set(candidates)
            filtered_species_local = {
                k for k in (set(sub_keys) | set(prod_keys)) if k not in cofactors_to_ignore
            }
        return candidates, filtered_species_local

    def _build_relaxed_block(
        side_block: Dict[str, Any],
        *,
        depth: int,
        direction_limit_up: int,
        direction_limit_down: int,
    ) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for species_id, v in side_block.items():
            coeff = v.get("coeff", 1)
            chebi_ids = (
                iter_chebi_for_species(species_to_chebi or {}, str(species_id))
                if species_to_chebi is not None
                else []
            )
            if not chebi_ids or parent_map is None or chebi_to_kegg is None:
                out[species_id] = {"species_id": species_id, "coeff": coeff, "candidates": v.get("candidates", [])}
                continue
            expanded_candidates: List[Dict[str, Any]] = []
            seen: Set[Tuple[str, str, str, int]] = set()
            for chebi_id in chebi_ids:
                expansion = expand_chebi_with_metadata(
                    chebi_id,
                    parent_map,
                    child_map=child_map,
                    max_up_depth=min(int(depth), int(direction_limit_up)),
                    max_down_depth=min(int(depth), int(direction_limit_down)),
                )
                for cid, meta in expansion.items():
                    keggs = kegg_ids_for_chebi_term(cid, chebi_to_kegg)
                    for kid in sorted(keggs):
                        direction = str(meta.get("direction", "exact"))
                        dist = int(meta.get("distance", 0))
                        key = (kid, cid, direction, dist)
                        if key in seen:
                            continue
                        seen.add(key)
                        expanded_candidates.append(
                            {
                                "kegg_id": kid,
                                "canonical_id": cid,
                                "direction": direction,
                                "distance": dist,
                            }
                        )
            out[species_id] = {
                "species_id": species_id,
                "coeff": coeff,
                "candidates": expanded_candidates,
            }
        return out

    def _species_ids_from_equation_side(side_str: str) -> set:
        out = set()
        side = str(side_str or "").strip()
        if not side:
            return out
        for term in side.split("+"):
            parts = term.strip().split()
            if not parts:
                continue
            if len(parts) == 1:
                met = parts[0]
            else:
                try:
                    float(parts[0])
                except ValueError:
                    met = term.strip()
                else:
                    met = parts[-1]
            met = met.lstrip("$").strip()
            if met:
                out.add(met)
        return out

    def _reaction_species_ids(reaction_equation: str) -> Tuple[set, set]:
        if "=>" in reaction_equation or "->" in reaction_equation:
            lhs, rhs = re.split(r"=>|->", reaction_equation, maxsplit=1)
            return _species_ids_from_equation_side(lhs), _species_ids_from_equation_side(rhs)
        return set(), set()

    def _expand_one_species(chebi_id: str, depth: int) -> List[Dict[str, Any]]:
        if not chebi_id or parent_map is None or chebi_to_kegg is None:
            return []
        expansion = expand_chebi_with_metadata(
            chebi_id,
            parent_map,
            child_map=child_map,
            max_up_depth=min(int(depth), int(max_ancestor_depth)),
            max_down_depth=min(int(depth), int(max_descendant_depth)),
        )
        out: List[Dict[str, Any]] = []
        for cid, meta in expansion.items():
            keggs = kegg_ids_for_chebi_term(cid, chebi_to_kegg)
            for kid in sorted(keggs):
                out.append(
                    {
                        "kegg_id": kid,
                        "canonical_id": cid,
                        "direction": str(meta.get("direction", "exact")),
                        "distance": int(meta.get("distance", 0)),
                    }
                )
        return out

    def _recover_species_kegg_candidates(
        side_block: Dict[str, Any],
        species_ids: set,
        *,
        species_depth_cap: int,
    ) -> None:
        """Fill ChEBI→KEGG expansion candidates for species that have none (Stage 1A)."""
        for sid in sorted(species_ids):
            entry = side_block.get(sid)
            has_candidates = bool(entry and entry.get("candidates"))
            if has_candidates:
                continue
            coeff = 1 if entry is None else entry.get("coeff", 1)
            chebi_ids = iter_chebi_for_species(species_to_chebi, sid)
            recovered: List[Dict[str, Any]] = []
            seen_r: Set[Tuple[str, str, str, int]] = set()
            for chebi_id in chebi_ids:
                for depth in range(1, species_depth_cap + 1):
                    chunk = _expand_one_species(chebi_id, depth)
                    for item in chunk:
                        key = (
                            item["kegg_id"],
                            item["canonical_id"],
                            item["direction"],
                            item["distance"],
                        )
                        if key not in seen_r:
                            seen_r.add(key)
                            recovered.append(item)
                    if chunk:
                        break
            side_block[sid] = {"species_id": sid, "coeff": coeff, "candidates": recovered}

    try:
        logger.info(f"Loading KEGG reaction data...")
        # Load KEGG reaction data
        kegg_parsed_reactions_dict = load_kegg_parsed_reactions_dict()
        kegg_reaction_features_dict = load_kegg_reaction_features_dict()
        logger.info(f"Loaded {len(kegg_parsed_reactions_dict)} parsed KEGG reactions")
        logger.info(f"Loaded {len(kegg_reaction_features_dict)} KEGG reaction features")
        
        recommendations = []
        
        for reaction_id in normalized_reactions:
            reaction_label = reaction_id.get('id')
            
            reaction_str = reaction_id.get('reaction_string')
            # Extract substrate and product mappings from normalized reaction
            # Note: map_metabolites_to_kegg returns either a dict or an empty list []
            model_subs = reaction_id.get('substrates', {})
            model_prods = reaction_id.get('products', {})

            # Handle case where map_metabolites_to_kegg returns empty list (no mappings found)
            if isinstance(model_subs, list):
                model_subs = {}
            if isinstance(model_prods, list):
                model_prods = {}

            reaction_relax_levels: Dict[str, int] = {}
            if relaxation_levels_by_entity and penalty_lam != 0:
                involved_met_ids: set = set()
                if isinstance(model_subs, dict):
                    involved_met_ids |= set(model_subs.keys())
                if isinstance(model_prods, dict):
                    involved_met_ids |= set(model_prods.keys())
                reaction_relax_levels = { # build out the relaxation levels set for this reaction
                    mid: int(relaxation_levels_by_entity.get(mid, 0) or 0)
                    for mid in involved_met_ids
                }

            species_depth_cap = (
                max(1, int(max_relax_level))
                if max_species_relax_depth is None
                else max(1, int(max_species_relax_depth))
            )
            reaction_depth_cap = (
                max(1, int(max_relax_level))
                if max_reaction_relax_depth is None
                else max(1, int(max_reaction_relax_depth))
            )

            active_subs = dict(model_subs)
            active_prods = dict(model_prods)

            # --- Stage 1A: species-level relaxation (independent trigger) ---
            # Trigger: species has no KEGG candidates (including dropped/unmapped species).
            lhs_species, rhs_species = _reaction_species_ids(reaction_str)

            if species_to_chebi is not None and parent_map is not None and chebi_to_kegg is not None:
                _recover_species_kegg_candidates(
                    active_subs, lhs_species, species_depth_cap=species_depth_cap
                )
                _recover_species_kegg_candidates(
                    active_prods, rhs_species, species_depth_cap=species_depth_cap
                )

            # --- Stage 1B: strict reaction filtering ---
            participant_relaxation: Dict[str, Dict[str, Any]] = {}
            model_sub_keys, sub_counter = _collect_side(active_subs, participant_relaxation)
            model_prod_keys, prod_counter = _collect_side(active_prods, participant_relaxation)
            filtered_reaction_list, filtered_species = _strict_filter(model_sub_keys, model_prod_keys)

            # --- Stage 2: reaction-level fallback (independent trigger) ---
            # Trigger: species have KEGG candidates but no matching KEGG reactions.
            if (
                len(filtered_reaction_list) == 0
                and species_to_chebi is not None
                and parent_map is not None
                and chebi_to_kegg is not None
            ):
                previous_nodes: set = set()
                up_cap = max(1, int(max_ancestor_depth))
                down_cap = max(1, int(max_descendant_depth))
                for depth in range(1, reaction_depth_cap + 1):
                    trial_subs = _build_relaxed_block(
                        active_subs,
                        depth=depth,
                        direction_limit_up=up_cap,
                        direction_limit_down=down_cap,
                    )
                    trial_prods = _build_relaxed_block(
                        active_prods,
                        depth=depth,
                        direction_limit_up=up_cap,
                        direction_limit_down=down_cap,
                    )
                    expanded_nodes = set()
                    for side in (trial_subs, trial_prods):
                        for v in side.values():
                            for c in v.get("candidates", []):
                                if isinstance(c, dict):
                                    expanded_nodes.add(str(c.get("canonical_id", "")).strip())
                    if expanded_nodes == previous_nodes and depth > 1:
                        break
                    previous_nodes = expanded_nodes

                    trial_participant_relaxation: Dict[str, Dict[str, Any]] = {}
                    trial_sub_keys, trial_sub_counter = _collect_side(trial_subs, trial_participant_relaxation)
                    trial_prod_keys, trial_prod_counter = _collect_side(trial_prods, trial_participant_relaxation)
                    trial_candidates, trial_filtered_species = _strict_filter(trial_sub_keys, trial_prod_keys)
                    if len(trial_candidates) > 0:
                        filtered_reaction_list = trial_candidates
                        filtered_species = trial_filtered_species
                        active_subs = trial_subs
                        active_prods = trial_prods
                        participant_relaxation = trial_participant_relaxation
                        model_sub_keys, sub_counter = trial_sub_keys, trial_sub_counter
                        model_prod_keys, prod_counter = trial_prod_keys, trial_prod_counter
                        break

            # Direction/distance remain in participant_relaxation for debugging; scores are
            # not penalized for hierarchy hops (multiple relaxed candidates may coexist).
            reaction_penalty = 0.0

            # Keep selected mapping (strict or relaxed) on recommendation payload.
            model_subs = active_subs
            model_prods = active_prods
            reaction_type = classify_reaction(
                reaction_str,
                filtered_species=filtered_species,
                candidates=filtered_reaction_list,
            )
            matches = []

            # Create a (substrates, products) pair in Counter form for similarity scoring
            cartesian_products = [(sub_counter, prod_counter)]
            # Compare with each KEGG reaction
            for kegg_id in filtered_reaction_list:
                for i in cartesian_products:
                    max_score, score_forward, score_reverse = score_model_against_kegg_reaction(
                        i[0],
                        i[1],
                        kegg_id,
                        kegg_parsed_reactions_dict=kegg_parsed_reactions_dict,
                        cofactors_to_ignore=cofactors_to_ignore,
                        spectators=spectators,
                    )
                    # Ranking / top-k: use unified objective only (raw similarity = max_score).
                    match_score = unified_reaction_objective(
                        max_score,
                        reaction_relax_levels if reaction_relax_levels else None,
                        lam=penalty_lam,
                        max_relax_level=max_relax_level,
                    )
                    adjusted_score = float(match_score)
                    matches.append(
                        {
                            "model_reaction_id": reaction_label,
                            "kegg_reaction_id": kegg_id,
                            "score_forward": score_forward,
                            "score_reverse": score_reverse,
                            "base_match_score": match_score,
                            "reaction_penalty": reaction_penalty,
                            "match_score": adjusted_score,
                        }
                    )
            
            # Sort matches by final score (descending)
            matches.sort(key=lambda x: x['match_score'], reverse=True)
            
            # Keep top_k matches
            if top_k:
                top_matches = matches[:top_k]
            else:
                top_matches = matches
            
            # Extract candidates and scores for recommendation
            candidates = [match['kegg_reaction_id'] for match in top_matches]
            match_scores = [match['match_score'] for match in top_matches]
            
            # Get reaction names from KEGG
            candidate_names = []
            for kegg_id in candidates:
                orthology = kegg_reaction_features_dict.get(kegg_id, kegg_id).get("ORTHOLOGY", "")
                candidate_names.append(extract_classifications(orthology, 'orthology'))

            # Create recommendation object
            recommendation = ReactionRecommendation(
                id=reaction_label,
                synonyms=[],
                equation=reaction_str, 
                substrates=model_subs,
                products=model_prods,
                candidates=candidates,
                candidate_names=candidate_names,
                match_score=match_scores,
                metadata={
                    "reaction_type": reaction_type,
                    "filtered_species_count": int(len(filtered_species)),
                    "candidate_count": int(len(filtered_reaction_list)),
                    "participant_relaxation": sorted(
                        participant_relaxation.values(),
                        key=lambda x: (x.get("species_id", ""), x.get("kegg_id", "")),
                    ),
                    "reaction_penalty": reaction_penalty,
                    "failed_default_score": 0.0,
                },
            )
            if reaction_type == "failed_mapping":
                # Keep one record so downstream aggregation can score failed-but-eligible reactions.
                recommendation.match_score = [0.0]
                recommendations.append(recommendation)
            elif reaction_type == "non_mappable":
                # Keep one record for coverage tracking; excluded by aggregator from scoring.
                recommendation.match_score = []
                recommendations.append(recommendation)
            else:
                recommendations.extend(split_recommendation(recommendation))
            
        return recommendations
        
    except Exception as e:
        logger.error(f"Error in KEGG recommendation: {e}")
        import traceback
        traceback.print_exc()
        return []


def _get_kegg_recommendations_direct(reaction_ids: List[str], synonyms_dict, top_k: int = 3) -> List[Recommendation]:
    """
    Find KEGG recommendations by directly matching against KEGG compound synonyms.
    Args:
        reaction_ids: List of species IDs to evaluate
        synonyms_dict: Mapping of species IDs to synonyms
        top_k: Number of top candidates to return per species based on hit_count.
    """
    # Load necessary KEGG dictionaries
    logger.info(f"Loading KEGG reaction data...")
    kegg_reaction_features_dict = load_kegg_reaction_features_dict()
    logger.info(f"Loaded {len(kegg_reaction_features_dict)} KEGG reactions")

    recommendations = []
    
    for reaction_id in reaction_ids:
        # Get synonyms for this species ID
        if isinstance(synonyms_dict, dict):
            synonyms = synonyms_dict.get(reaction_id, [reaction_id])
        elif isinstance(synonyms_dict, tuple) and len(synonyms_dict) == 2:
            # If it's a tuple with two items (dict and reason)
            synonyms = synonyms_dict[0].get(reaction_id, [reaction_id])
        else:
            synonyms = [reaction_id]
        
        # Skip if only 'UNK' synonym
        if synonyms == ['UNK'] or (len(synonyms) == 1 and synonyms[0] == 'UNK'):
            # Create empty recommendation for UNK
            recommendation = Recommendation(
                id=reaction_id,
                synonyms=synonyms,
                candidates=[],
                candidate_names=[],
                match_score=[]
            )
            recommendations.append(recommendation)
            continue
        
        # # Clean synonyms to remove unwanted words before search
        # cleaned_synonyms = clean_synonyms(synonyms)
        
        all_candidates = []
        all_candidate_names = []
        hit_count = {}
        
        # Query for each cleaned synonym
        for synonym in synonyms:
            norm_synonym = remove_symbols(synonym.lower())
            
            if norm_synonym.startswith('R') and len(norm_synonym)==5 and norm_synonym[-5:].isdigit():
                kegg_reaction_id = norm_synonym.upper()
                if kegg_reaction_id in kegg_reaction_features_dict:
                    kegg_name = kegg_reaction_features_dict.get(kegg_id, kegg_id).get("NAME", "")
                    
                    if kegg_id not in all_candidates:
                        all_candidates.append(kegg_id)
                        all_candidate_names.append(kegg_name)
                        hit_count[kegg_id] = 1
                    else:
                        hit_count[kegg_id] += 1
            
            # Then try direct name matching with KEGG reaction names
            for kegg_id in kegg_reaction_features_dict:
                name = kegg_reaction_features_dict.get(kegg_id, kegg_id).get("NAME", "")
                if norm_synonym == remove_symbols(name.lower()): # this could benefit from fuzzy matching
                    kegg_name = name
                    
                    if kegg_id not in all_candidates:
                        all_candidates.append(kegg_id)
                        all_candidate_names.append(kegg_name)
                        hit_count[kegg_id] = 1
                    else:
                        hit_count[kegg_id] += 1
            
            # Also check for partial matches in reaction orthology/names if no direct matches found
            # if not all_candidates:
            for kegg_id in kegg_reaction_features_dict:
                orthology = kegg_reaction_features_dict.get(kegg_id, kegg_id).get("ORTHOLOGY", "")
                clean_orthology = remove_symbols(extract_classifications(orthology, 'orthology').lower())
                name = kegg_reaction_features_dict.get(kegg_id, kegg_id).get("NAME", "")
                clean_name = remove_symbols(name.lower())

                if (norm_synonym in clean_orthology or clean_orthology in norm_synonym) and clean_orthology:
                    kegg_orthology = orthology
                    
                    if kegg_id not in all_candidates:
                        all_candidates.append(kegg_id)
                        all_candidate_names.append(kegg_orthology)
                        # Lower confidence for partial matches
                        hit_count[kegg_id] = 0.5
                    else:
                        hit_count[kegg_id] += 0.5

                elif (norm_synonym in clean_name or clean_name in norm_synonym) and clean_name:
                    kegg_name = name
                    
                    if kegg_id not in all_candidates:
                        all_candidates.append(kegg_id)
                        all_candidate_names.append(kegg_name)
                        # Lower confidence for partial matches
                        hit_count[kegg_id] = 0.5
                    else:
                        hit_count[kegg_id] += 0.5

        
        # Sort candidates by hit_count (descending) and take top_k
        if all_candidates:
            # Create list of (candidate, name, hit_count) tuples
            candidate_tuples = [(candidate, name, hit_count[candidate])
                               for candidate, name in zip(all_candidates, all_candidate_names)]
            
            # Sort by hit_count descending
            candidate_tuples.sort(key=lambda x: x[2], reverse=True)
            
            # Take top_k candidates
            if top_k:
                top_candidates = candidate_tuples[:top_k]
            else:
                top_candidates = candidate_tuples

            
            # Extract sorted lists
            all_candidates = [candidate for candidate, _, _ in top_candidates]
            all_candidate_names = [name for _, name, _ in top_candidates]
        
        num_synonyms = len(synonyms)
        match_score_list = [hit_count.get(candidate, 0) / num_synonyms for candidate in all_candidates]
        
        # Create recommendation object
        recommendation = Recommendation(
            id=reaction_id,
            synonyms=synonyms,
            candidates=all_candidates,
            candidate_names=all_candidate_names,
            match_score=match_score_list
        )
        recommendations.append(recommendation)
    
    return recommendations    


def _expand_kegg_compound_set_by_chebi_ontology(
    kegg_compound_ids: Set[str],
    *,
    chebi_to_kegg: Mapping[str, Any],
    parent_map: Mapping[str, Set[str]],
    child_map: Optional[Mapping[str, Set[str]]],
    kegg_to_chebis: Mapping[str, Set[str]],
    max_up_depth: int,
    max_down_depth: int,
) -> Set[str]:
    """
    Expand a set of KEGG compound IDs with other compounds linked to nearby ChEBI
    terms (is_a parents/children), for relaxed reaction-side subset checks.
    """
    if max_up_depth <= 0 and max_down_depth <= 0:
        return set(kegg_compound_ids)
    out: Set[str] = set(kegg_compound_ids)
    for kid in kegg_compound_ids:
        k = str(kid).strip()
        if not k:
            continue
        for chebi in kegg_to_chebis.get(k, ()):
            ch = str(chebi).strip()
            if not ch:
                continue
            for term in expand_chebi_with_metadata(
                ch,
                parent_map,
                child_map=child_map,
                max_up_depth=max_up_depth,
                max_down_depth=max_down_depth,
            ):
                out.update(kegg_ids_for_chebi_term(term, chebi_to_kegg))
    return out


def filter_kegg_reactions(
    model_sub_keys: List[Counter],
    model_prod_keys: List[Counter],
    cofactors_to_ignore={},
    *,
    chebi_to_kegg: Optional[Mapping[str, Any]] = None,
    parent_map: Optional[Mapping[str, Set[str]]] = None,
    child_map: Optional[Mapping[str, Set[str]]] = None,
    reaction_ontology_max_up: int = 2,
    reaction_ontology_max_down: int = 1,
) -> List[Dict[str, Any]]:
    """
    Filter KEGG reactions based on substrate and product matching.

    When ``chebi_to_kegg`` and ``parent_map`` are provided, each reaction's
    substrate and product KEGG compound sets are expanded with other compounds
    linked to nearby ChEBI ontology terms (``reaction_ontology_max_up`` /
    ``reaction_ontology_max_down`` hops along is_a), so subset matching can
    relate related metabolites across the ontology.

    Args:
        model_subs: List of Counter objects representing model substrates
        model_prods: List of Counter objects representing model products
        kegg_parsed_reactions_dict: Dictionary of KEGG reaction data
        cofactors_to_ignore: set of KEGG IDs of cofactors
        chebi_to_kegg: Optional ChEBI→KEGG map for reaction-side expansion
        parent_map: Optional ChEBI child→parents map (is_a)
        child_map: Optional ChEBI parent→children map (derived if omitted)
        reaction_ontology_max_up: Max is_a hops upward when expanding (0 disables)
        reaction_ontology_max_down: Max hops downward when expanding (0 disables)

    Returns:
        List of KEGG reactions that contain all model substrates and products
    """
    kegg_parsed_reactions_dict = load_kegg_parsed_reactions_dict() # load_kegg_reaction_features_dict
    # Get unique keys from the model substrates and products

    expand_reaction_side = (
        chebi_to_kegg is not None
        and parent_map is not None
        and (reaction_ontology_max_up > 0 or reaction_ontology_max_down > 0)
    )
    kegg_to_chebis: Optional[Dict[str, Set[str]]] = None
    if expand_reaction_side:
        kegg_to_chebis = defaultdict(set)
        for chebi, kids in merge_chebi_to_kegg_mapping(chebi_to_kegg).items():
            c = str(chebi).strip()
            if not c:
                continue
            for x in kids:
                k = str(x).strip()
                if k:
                    kegg_to_chebis[k].add(c)

    filtered_reactions = []
    partial_matches = []

    for kegg_id in kegg_parsed_reactions_dict:
        # Get sets of KEGG substrates and products
        kegg_subs_set = set(kegg_parsed_reactions_dict.get(kegg_id, kegg_id).get('substrates', []))
        a = len(kegg_subs_set)
        kegg_prods_set = set(kegg_parsed_reactions_dict.get(kegg_id, kegg_id).get('products', []))
        if expand_reaction_side and kegg_to_chebis is not None:
            kegg_subs_set = _expand_kegg_compound_set_by_chebi_ontology(
                kegg_subs_set,
                chebi_to_kegg=chebi_to_kegg,
                parent_map=parent_map,
                child_map=child_map,
                kegg_to_chebis=kegg_to_chebis,
                max_up_depth=reaction_ontology_max_up,
                max_down_depth=reaction_ontology_max_down,
            )
            b = len(kegg_subs_set)
            assert a <= b, "KEGG substrate set expanded by ChEBI ontology should not be smaller"
            kegg_prods_set = _expand_kegg_compound_set_by_chebi_ontology(
                kegg_prods_set,
                chebi_to_kegg=chebi_to_kegg,
                parent_map=parent_map,
                child_map=child_map,
                kegg_to_chebis=kegg_to_chebis,
                max_up_depth=reaction_ontology_max_up,
                max_down_depth=reaction_ontology_max_down,
            )

        # Check if all model metabolites are in KEGG reaction (ignore counts)
        subs_match = model_sub_keys.issubset(kegg_subs_set)
        prods_match = model_prod_keys.issubset(kegg_prods_set)
        
        if subs_match and prods_match:
            filtered_reactions.append(kegg_id)
        elif subs_match:
            partial_matches.append(kegg_id)
        elif prods_match:
            partial_matches.append(kegg_id)

    #if not filtered_reactions: 
    #    filtered_reactions=partial_matches
    
    return filtered_reactions

def cancel_spectators(model_subs: Counter, model_prods: Counter):
    """
    Cancel spectator metabolites (same metabolite and same stoichiometry)
    from substrates and products. Works directly with Counters.
    
    Parameters
    ----------
    model_subs : Counter
        Substrate stoichiometry, e.g., Counter({"ATP": 1, "Glucose": 1})
    model_prods : Counter
        Product stoichiometry, e.g., Counter({"ADP": 1, "Glucose": 1})
    
    Returns
    -------
    (Counter, Counter)
        New Counters after cancellation
    """
    # Find the intersection (min stoichiometry of each metabolite)
    common = model_subs & model_prods   # stoichiometry-aware AND

    # Subtract the common terms from both sides
    new_subs = model_subs - common
    new_prods = model_prods - common

    return new_subs, new_prods

def score_model_against_kegg_reaction(
    sub_counter: Counter,
    prod_counter: Counter,
    kegg_reaction_id: str,
    *,
    kegg_parsed_reactions_dict: Optional[Dict[str, Any]] = None,
    cofactors_to_ignore: Optional[set] = None,
    spectators: bool = False,
) -> float:
    """
    Same forward/reverse similarity as ``_get_kegg_recommendations_rulebased`` for one
    reference KEGG reaction (used by problematic-metabolite leave-one-out checks).
    """
    if cofactors_to_ignore is None:
        cofactors_to_ignore = set()
    if kegg_parsed_reactions_dict is None:
        kegg_parsed_reactions_dict = load_kegg_parsed_reactions_dict()

    entry = kegg_parsed_reactions_dict.get(kegg_reaction_id, kegg_reaction_id)
    if not isinstance(entry, dict):
        entry = {}

    model_sub_keys = set(sub_counter.keys())
    model_prod_keys = set(prod_counter.keys())
    only_cofactors_subs = all(key in cofactors_to_ignore for key in model_sub_keys)
    only_cofactors_prods = all(key in cofactors_to_ignore for key in model_prod_keys)

    kegg_subs = Counter(set(entry.get("substrates", [])))
    kegg_prods = Counter(set(entry.get("products", [])))

    if not spectators:
        kegg_subs, kegg_prods = cancel_spectators(kegg_subs, kegg_prods)

    if only_cofactors_subs or only_cofactors_prods:
        score_forward = compute_similarity(sub_counter, kegg_subs) + compute_similarity(
            prod_counter, kegg_prods
        )
        score_reverse = compute_similarity(sub_counter, kegg_prods) + compute_similarity(
            prod_counter, kegg_subs
        )
    else:
        score_forward = compute_similarity(sub_counter, kegg_subs, cofactors_to_ignore) + compute_similarity(
            prod_counter, kegg_prods, cofactors_to_ignore
        )
        score_reverse = compute_similarity(sub_counter, kegg_prods, cofactors_to_ignore) + compute_similarity(
            prod_counter, kegg_subs, cofactors_to_ignore
        )

    score_forward /= 2
    score_reverse /= 2
    return max(score_forward, score_reverse), score_forward, score_reverse


def compute_similarity(counter1: Counter, counter2: Counter, cofactors_to_ignore: set = {}) -> float:
    """
    Compute Jaccard-like similarity between two reaction sides with stoichiometry awareness.
    
    This function calculates a similarity score between two sets of metabolites,
    taking into account their stoichiometric coefficients and filtering out common cofactors.
    
    Args:
        counter1: Counter object for first reaction side (substrates or products)
        counter2: Counter object for second reaction side (substrates or products)
        cofactors_to_ignore: Set of cofactor IDs to ignore in the comparison
        
    Returns:
        Similarity score between 0.0 (no similarity) and 1.0 (identical)
    """
    # Filter out cofactors
    c1 = {k: v for k, v in counter1.items() if k not in cofactors_to_ignore}
    c2 = {k: v for k, v in counter2.items() if k not in cofactors_to_ignore}

    # Degenerate/low-information comparisons are not treated as matches.
    if not c1 and not c2:
        return 0.0
    if not c1 or not c2:
        return 0.0
    
    # Calculate stoichiometry-aware Jaccard similarity
    # Sum of minimum values (intersection) divided by sum of maximum values (union)
    intersection = sum(min(c1.get(k, 0), c2.get(k, 0)) for k in set(c1) | set(c2))
    union = sum(max(c1.get(k, 0), c2.get(k, 0)) for k in set(c1) | set(c2))
    
    if union == 0:
        return 0.0
        
    return intersection / union


def get_embedding_function(model_type: str = "default"):
    """
    Get the appropriate embedding function based on model type.
    
    Args:
        model_type: Type of embedding model ("default", "openai")
        
    Returns:
        ChromaDB embedding function
    """
    if model_type == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable is required for OpenAI embeddings")
        logger.info("Using OpenAI text-embedding-ada-002 model")
        return embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.environ.get("OPENAI_API_KEY"),
            model_name="text-embedding-ada-002",
        )
    else:  # default
        logger.info("Using sentence transformer all-MiniLM-L6-v2 model")
        return embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

def get_chromadb_client(persist_directory: str, collection_name: str, model_type: str = "default"):
    """
    Get or create a ChromaDB client and collection, handling conflicts properly.
    
    Args:
        persist_directory: Directory for ChromaDB storage
        collection_name: Name of the collection
        model_type: Type of embedding model
        
    Returns:
        Tuple of (client, collection)
    """
    client_key = f"{persist_directory}_{collection_name}_{model_type}"
    
    if client_key in _CHROMADB_CLIENTS:
        return _CHROMADB_CLIENTS[client_key]
    
    try:
        # Try to initialize ChromaDB client
        client = chromadb.PersistentClient(path=persist_directory)
        
        # Get embedding function
        embedding_function = get_embedding_function(model_type)
        
        # Get the collection
        collection = client.get_collection(
            name=collection_name,
            embedding_function=embedding_function
        )
        
        # Cache the client and collection
        _CHROMADB_CLIENTS[client_key] = (client, collection)
        
        logger.info(f"Using RAG embeddings from collection '{collection_name}' with {model_type} model")
        
        return client, collection
        
    except Exception as e:
        error_msg = str(e).lower()
        
        # Handle the specific "already exists" error
        if "already exists" in error_msg and "different settings" in error_msg:
            logger.warning(f"ChromaDB client conflict detected. Attempting to use in-memory client as fallback.")
            
            try:
                # Try using an in-memory client as fallback (this won't persist but will work for queries)
                client = chromadb.Client()
                
                # Try to load the collection from persistent storage manually
                # This is a workaround - the collection might not be available in memory
                raise ValueError(f"ChromaDB client conflict. Please restart Python session or check for other running processes using {persist_directory}")
                
            except Exception as fallback_error:
                logger.error(f"Fallback client also failed: {fallback_error}")
                raise ValueError(f"ChromaDB unavailable due to client conflict. Error: {e}")
        else:
            logger.error(f"Could not access ChromaDB collection '{collection_name}': {e}")
            raise ValueError(f"ChromaDB collection not available. Make sure embeddings have been created first. Error: {e}")

def force_clear_chromadb():
    """
    Force clear ChromaDB cache and try to cleanup any hanging clients.
    """
    global _CHROMADB_CLIENTS
    
    # Clear our cache
    _CHROMADB_CLIENTS.clear()
    
    # Try to garbage collect
    import gc
    gc.collect()
    
    logger.info("Forced ChromaDB cleanup completed")

def get_species_recommendations_rag(
    species_ids: List[str], 
    synonyms_dict, 
    model_type: str = "default",
    persist_directory: str = "chroma_storage",
    collection_name: str = None,
    top_k: int = 3,
    database: str = "chebi",
    tax_id: str = None,
    reaction_participants: List[str] = None
) -> List[Recommendation]:
    """
    Find recommendations using RAG embeddings.
    
    Parameters:
    - species_ids (list): List of species IDs to evaluate.
    - synonyms_dict (dict): Mapping of species IDs to synonyms.
    - collection_name (str): ChromaDB collection name. If None, will be set to default collection name.
    - model_type (str): Type of embedding model ("default", "openai").
    - persist_directory (str): ChromaDB storage directory.
    - top_k (int): Number of top candidates to retrieve per species.
    - database (str): Database to search ("chebi", "ncbigene").
    - tax_id (str/list): For ncbigene database, the organism's tax_id for organism-specific lookup. Use 9606 by default. If list, search all tax_ids for each species.
    
    Returns:
    - list: List of Recommendation objects with candidates and similarity scores.
    """
    persist_directory = os.path.join(get_data_dir(), persist_directory)
    recommendations = []

    # Helper to get collection for a given tax_id
    def get_collection_for_taxid(tid):
        if database == "ncbigene":
            cname = f"ncbigene_default_tax{tid}"
        elif database == "uniprot":
            cname = f"uniprot_default_tax{tid}"
        else:
            cname = f"{database}_default_tax{tid}"
        client, collection = get_chromadb_client(persist_directory, cname, model_type)
        return collection
    # If database is ncbigene/uniprot and tax_id is a list, aggregate results
    if database in ["ncbigene", "uniprot"] and isinstance(tax_id, list):
        for spec_id in species_ids:
            if isinstance(synonyms_dict, dict):
                synonyms = synonyms_dict.get(spec_id, [spec_id])
            elif isinstance(synonyms_dict, tuple) and len(synonyms_dict) == 2:
                synonyms = synonyms_dict[0].get(spec_id, [spec_id])
            else:
                synonyms = [spec_id]
            if synonyms == ['UNK'] or (len(synonyms) == 1 and synonyms[0] == 'UNK'):
                recommendation = Recommendation(
                    id=spec_id,
                    synonyms=synonyms,
                    candidates=[],
                    candidate_names=[],
                    match_score=[]
                )
                recommendations.append(recommendation)
                continue
            
            # # Clean synonyms for non-chemical databases
            # search_synonyms = clean_synonyms(synonyms)
            
            agg_candidates = {}
            agg_names = {}
            for tid in tax_id:
                try:
                    collection = get_collection_for_taxid(tid)
                except Exception as e:
                    logger.warning(f"Could not access {database.upper()} RAG collection for tax_id {tid}: {e}")
                    continue
                for synonym in synonyms:
                    try:
                        results = collection.query(
                            query_texts=[synonym],
                            n_results=top_k,
                            include=["metadatas", "distances"]
                        )
                        for metadata, distance in zip(results['metadatas'][0], results['distances'][0]):
                            db_id = metadata.get('ncbigene_id', 'Unknown')
                            db_name = metadata.get('name', 'Unknown')
                            similarity_score = round(1 - distance, 3)
                            if db_id not in agg_candidates or similarity_score > agg_candidates[db_id]:
                                agg_candidates[db_id] = similarity_score
                                agg_names[db_id] = db_name
                    except Exception as e:
                        logger.warning(f"Error querying synonym '{synonym}' for species '{spec_id}' in tax_id {tid}: {e}")
                        continue
            # Sort and select top_k
            sorted_candidates = sorted(agg_candidates.items(), key=lambda x: x[1], reverse=True)[:top_k]
            all_candidates = [db_id for db_id, _ in sorted_candidates]
            all_candidate_names = [agg_names[db_id] for db_id, _ in sorted_candidates]
            match_score_list = [agg_candidates[db_id] for db_id, _ in sorted_candidates]
            recommendation = Recommendation(
                id=spec_id,
                synonyms=synonyms,
                candidates=all_candidates,
                candidate_names=all_candidate_names,
                match_score=match_score_list
            )
            recommendations.append(recommendation)
        return recommendations
    # If database is ncbigene/uniprot and tax_id is a str or None (single organism)
    if database in ["ncbigene", "uniprot"]:
        if not tax_id:
            default_tax_id = 9606
            logger.warning(f"No tax_id provided for {database} RAG search. Using default tax_id {default_tax_id}.")
            tax_id = default_tax_id
        if collection_name is None and model_type == "default":
            collection_name = f"{database}_default_tax{tax_id}"
        elif collection_name is None and model_type == "openai":
            collection_name = f"{database}_openai_tax{tax_id}"
        try:
            client, collection = get_chromadb_client(persist_directory, collection_name, model_type)
        except Exception as e:
            logger.error(f"Could not access {database.upper()} RAG collection '{collection_name}': {e}")
            raise
    elif database == "chebi":
        if collection_name is None and model_type == "default":
            collection_name = "chebi_default_numonly"
        elif collection_name is None and model_type == "openai":
            collection_name = "chebi_openai_numonly"
        try:
            client, collection = get_chromadb_client(persist_directory, collection_name, model_type)
        except Exception as e:
            logger.error(f"Could not access ChEBI RAG collection '{collection_name}': {e}")
            raise
    elif database == "kegg":
        if collection_name is None and model_type == "default":
            collection_name = "kegg_reactions_default"
        try:
            client, collection = get_chromadb_client(persist_directory, collection_name, model_type)
        except Exception as e:
            logger.error(f"Could not access KEGG RAG collection '{collection_name}': {e}")
            raise
    else:
        logger.error(f"Database {database} not supported for RAG search")
        return []
    # Standard single-collection logic
    for spec_id in species_ids:
        if isinstance(synonyms_dict, dict):
            synonyms = synonyms_dict.get(spec_id, [spec_id])
        elif isinstance(synonyms_dict, tuple) and len(synonyms_dict) == 2:
            synonyms = synonyms_dict[0].get(spec_id, [spec_id])
        else:
            synonyms = [spec_id]
        if synonyms == ['UNK'] or (len(synonyms) == 1 and synonyms[0] == 'UNK'):
            recommendation = Recommendation(
                id=spec_id,
                synonyms=synonyms,
                candidates=[],
                candidate_names=[],
                match_score=[]
            )
            recommendations.append(recommendation)
            continue
        
        # # Clean synonyms for non-chemical databases
        # if database != "chebi":
        #     search_synonyms = clean_synonyms(synonyms)
        # else:
        #     search_synonyms = synonyms
        
        all_candidates = []
        all_candidate_names = []
        candidate_scores = {}
        candidate_names = {}  # Keep track of candidate names separately
        
        for synonym in synonyms:
            try:
                if database=='kegg':
                    #  model_info -> reactions
                    # split each string by ':' and then extract the participants
                    query = f"Reaction type: {synonym}\nParticipants:{reaction_participants}"
                    # cake
                    results = collection.query(
                        query_texts=[query],
                        n_results=top_k,
                        include=["metadatas", "distances"]
                    )
                else:
                    results = collection.query(
                        query_texts=[synonym],
                        n_results=top_k,
                        include=["metadatas", "distances"]
                    )
                for metadata, distance in zip(results['metadatas'][0], results['distances'][0]):
                    if database == "chebi":
                        db_id = metadata.get('chebi_id', 'Unknown')
                    elif database == "ncbigene":
                        db_id = metadata.get('ncbigene_id', 'Unknown')
                    elif database == "uniprot":
                        db_id = metadata.get('uniprot_id', 'Unknown')
                    elif database == "kegg":
                        db_id = metadata.get('kegg_id', 'Unknown')
                    else:
                        db_id = metadata.get('id', 'Unknown')
                    db_name = metadata.get('name', 'Unknown')
                    similarity_score = round(1 - distance, 3)
                    if db_id not in candidate_scores:
                        all_candidates.append(db_id)
                        all_candidate_names.append(db_name)
                        candidate_scores[db_id] = similarity_score
                        candidate_names[db_id] = db_name  # Store name mapping
                    else:
                        candidate_scores[db_id] = max(candidate_scores[db_id], similarity_score)
                        # Keep the name from first occurrence or update if needed
                        if db_id not in candidate_names:
                            candidate_names[db_id] = db_name
                # Only keep the top_k candidates
                if len(candidate_scores) > top_k:
                    sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
                    all_candidates = [db_id for db_id, _ in sorted_candidates]
                    all_candidate_names = [candidate_names[db_id] for db_id, _ in sorted_candidates]
                    candidate_scores = dict(sorted_candidates)
            except Exception as e:
                logger.warning(f"Error querying synonym '{synonym}' for species '{spec_id}': {e}")
                continue
        match_score_list = [candidate_scores.get(candidate, 0.0) for candidate in all_candidates]
        recommendation = Recommendation(
            id=spec_id,
            synonyms=synonyms,
            candidates=all_candidates,
            candidate_names=all_candidate_names,
            match_score=match_score_list
        )
        recommendations.append(recommendation)
    return recommendations

def search_database(entity_name: str, 
                   entity_type: str, 
                   database: str = "chebi",
                   max_candidates: int = 10,
                   tax_id: str = None) -> List[Tuple[str, float, str]]:
    """
    Search for annotation candidates in specified database.
    Currently supports ChEBI, NCBI gene, and UniProt, extensible to other databases.
    
    Args:
        entity_name: Name of entity to search for
        entity_type: Type of entity (chemical, gene, protein)
        database: Database to search in ("chebi", "ncbigene", "uniprot")
        max_candidates: Maximum number of candidates to return
        tax_id: For ncbigene/uniprot database, the organism's tax_id for organism-specific lookup
        
    Returns:
        List of tuples (database_id, confidence, description)
    """
    if database.lower() == "chebi":
        return _search_chebi(entity_name, max_candidates)
    elif database.lower() == "ncbigene":
        return _search_ncbigene(entity_name, max_candidates, tax_id=tax_id)
    elif database.lower() == "uniprot":
        return _search_uniprot(entity_name, max_candidates, tax_id=tax_id)
    else:
        logger.warning(f"Database {database} not yet supported")
        return []

def _search_chebi(entity_name: str, max_candidates: int = 10) -> List[Tuple[str, float, str]]:
    """
    Search ChEBI database for entity matches.
    
    Args:
        entity_name: Name to search for
        max_candidates: Maximum number of candidates
        
    Returns:
        List of tuples (chebi_id, confidence, description)
    """
    try:
        cleannames_dict = load_chebi_cleannames_dict()
        label_dict = load_chebi_label_dict()
        
        # Normalize entity name
        norm_name = remove_symbols(entity_name.lower())
        
        candidates = []
        
        # Direct match search
        for ref_name, chebi_ids in cleannames_dict.items():
            if norm_name == ref_name.lower():
                for chebi_id in chebi_ids:
                    chebi_name = label_dict.get(chebi_id, chebi_id)
                    confidence = 1.0  # Direct match gets highest confidence
                    candidates.append((chebi_id, confidence, chebi_name))
        
        # Partial match search if no direct matches
        if not candidates:
            for ref_name, chebi_ids in cleannames_dict.items():
                if norm_name in ref_name.lower() or ref_name.lower() in norm_name:
                    for chebi_id in chebi_ids:
                        chebi_name = label_dict.get(chebi_id, chebi_id)
                        # Calculate confidence based on string similarity
                        confidence = min(len(norm_name), len(ref_name.lower())) / max(len(norm_name), len(ref_name.lower()))
                        candidates.append((chebi_id, confidence, chebi_name))
        
        # Sort by confidence and limit results
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:max_candidates]
        
    except Exception as e:
        logger.error(f"ChEBI search failed for {entity_name}: {e}")
        return []

def _search_ncbigene(entity_name: str, max_candidates: int = 10, tax_id: str = None) -> List[Tuple[str, float, str]]:
    """
    Search NCBI gene database for entity matches.
    
    Args:
        entity_name: Name to search for
        max_candidates: Maximum number of candidates
        tax_id: Organism's tax_id for organism-specific gene lookup
        
    Returns:
        List of tuples (ncbigene_id, confidence, description)
    """
    try:
        names_dict = load_ncbigene_names_dict(tax_id=tax_id)
        label_dict = load_ncbigene_label_dict()
        
        # Normalize entity name
        norm_name = remove_symbols(entity_name.lower())
        
        candidates = []
        
        # Direct match search
        for ref_name, gene_ids in names_dict.items():
            if norm_name == ref_name.lower():
                for gene_id in gene_ids:
                    gene_name = label_dict.get(gene_id, gene_id)
                    confidence = 1.0  # Direct match gets highest confidence
                    candidates.append((gene_id, confidence, gene_name))
        
        # Partial match search if no direct matches
        if not candidates:
            for ref_name, gene_ids in names_dict.items():
                if norm_name in ref_name.lower() or ref_name.lower() in norm_name:
                    for gene_id in gene_ids:
                        gene_name = label_dict.get(gene_id, gene_id)
                        # Calculate confidence based on string similarity
                        confidence = min(len(norm_name), len(ref_name.lower())) / max(len(norm_name), len(ref_name.lower()))
                        candidates.append((gene_id, confidence, gene_name))
        
        # Sort by confidence and limit results
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:max_candidates]
        
    except Exception as e:
        logger.error(f"NCBI gene search failed for {entity_name}: {e}")
        return []

def _search_uniprot(entity_name: str, max_candidates: int = 10, tax_id: str = None) -> List[Tuple[str, float, str]]:
    """
    Search UniProt database for entity matches.
    
    Args:
        entity_name: Name to search for
        max_candidates: Maximum number of candidates
        tax_id: Organism's tax_id for organism-specific UniProt lookup
        
    Returns:
        List of tuples (uniprot_id, confidence, description)
    """
    try:
        names_dict = load_uniprot_names_dict(tax_id=tax_id)
        label_dict = load_uniprot_label_dict(tax_id=tax_id)
        
        # Normalize entity name
        norm_name = remove_symbols(entity_name.lower())
        
        candidates = []
        
        # Direct match search
        for ref_name, uniprot_ids in names_dict.items():
            if norm_name == ref_name.lower():
                for uniprot_id in uniprot_ids:
                    uniprot_name = label_dict.get(uniprot_id, uniprot_id)
                    confidence = 1.0  # Direct match gets highest confidence
                    candidates.append((uniprot_id, confidence, uniprot_name))
        
        # Partial match search if no direct matches
        if not candidates:
            for ref_name, uniprot_ids in names_dict.items():
                if norm_name in ref_name.lower() or ref_name.lower() in norm_name:
                    for uniprot_id in uniprot_ids:
                        uniprot_name = label_dict.get(uniprot_id, uniprot_id)
                        # Calculate confidence based on string similarity
                        confidence = min(len(norm_name), len(ref_name.lower())) / max(len(norm_name), len(ref_name.lower()))
                        candidates.append((uniprot_id, confidence, uniprot_name))
        
        # Sort by confidence and limit results
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:max_candidates]
        
    except Exception as e:
        logger.error(f"UniProt search failed for {entity_name}: {e}")
        return []

def is_database_available(database: str) -> bool:
    """
    Check if a database is available for searching.
    
    Args:
        database: Database name to check
        
    Returns:
        True if database is available
    """
    if database.lower() == "chebi":
        try:
            data_dir = get_data_dir()
            cleannames_file = data_dir / "chebi" / REF_NAMES2CHEBI
            labels_file = data_dir / "chebi" / REF_CHEBI2LABEL
            return cleannames_file.exists() and labels_file.exists()
        except Exception:
            return False
    elif database.lower() == "ncbigene":
        try:
            data_dir = get_data_dir()
            names_file = data_dir / "ncbigene" / REF_NAMES2NCBIGENE
            labels_file = data_dir / "ncbigene" / REF_NCBIGENE2LABEL
            return names_file.exists() and labels_file.exists()
        except Exception:
            return False
    elif database.lower() == "uniprot":
        try:
            data_dir = get_data_dir()
            # Check for organism-specific files (common tax_ids: 9606 for human, 10090 for mouse)
            common_tax_ids = ["9606", "10090"]
            for tax_id in common_tax_ids:
                names_file = data_dir / "uniprot" / f"names2uniprot_tax{tax_id}.lzma"
                labels_file = data_dir / "uniprot" / f"uniprot2label_tax{tax_id}.lzma"
                if names_file.exists() and labels_file.exists():
                    return True
            # Also check for old combined files for backwards compatibility
            names_file = data_dir / "uniprot" / REF_NAMES2UNIPROT
            labels_file = data_dir / "uniprot" / REF_UNIPROT2LABEL
            return names_file.exists() and labels_file.exists()
        except Exception:
            return False
    elif database.lower() == "kegg":
        try:
            data_dir = get_data_dir()
            chebi_to_kegg_map_file = data_dir / "kegg" / REF_CHEBI2KEGG_COMPOUND
            names_file = data_dir / "kegg" / REF_KEGG_REACTION2NAME
            ec_file = data_dir / "kegg" / REF_KEGG2EC
            reactions_file = data_dir / "kegg" / "parsed_kegg_reactions.lzma"
            return (chebi_to_kegg_map_file.exists() and 
                   names_file.exists() and 
                   ec_file.exists() and 
                   reactions_file.exists())
        except Exception:
            return False
    
    return False

def get_available_databases() -> List[str]:
    """
    Get list of available databases.
    
    Returns:
        List of available database names
    """
    available = []
    
    if is_database_available("chebi"):
        available.append("chebi")
    
    if is_database_available("ncbigene"):
        available.append("ncbigene")
    
    if is_database_available("uniprot"):
        available.append("uniprot")

    if is_database_available("kegg"):
        available.append("kegg")
    
    # Future databases can be added here
    # if is_database_available("go"):
    #     available.append("go")
    
    return available

def clear_chromadb_cache():
    """Clear the ChromaDB client cache."""
    global _CHROMADB_CLIENTS
    for client in _CHROMADB_CLIENTS.values():
        try:
            client.reset()
        except Exception:
            pass
    _CHROMADB_CLIENTS.clear()
    logger.info("ChromaDB cache cleared")

def list_available_organisms(data_dir=None):
    """
    List available organism-specific NCBI gene reference files.
    
    Args:
        data_dir: Directory containing reference files (default: auto-detect)
        
    Returns:
        list: List of available tax_ids
    """
    if data_dir is None:
        data_dir = get_data_dir() / "ncbigene"
    else:
        data_dir = Path(data_dir)
    
    # Look for organism-specific files
    pattern = "names2ncbigene_tax*_protein-coding.lzma"
    files = list(data_dir.glob(pattern))
    
    tax_ids = []
    for f in files:
        # Extract tax_id from filename: names2ncbigene_tax{tax_id}_protein_coding.lzma
        parts = f.stem.split('_')
        if len(parts) >= 2 and parts[1].startswith('tax'):
            tax_id = parts[1][3:]  # Remove 'tax' prefix
            tax_ids.append(tax_id)
    
    tax_ids.sort()
    return tax_ids

def get_organism_files_info(data_dir=None):
    """
    Get information about available organism-specific files.
    
    Args:
        data_dir: Directory containing reference files (default: auto-detect)
        
    Returns:
        dict: Information about available files per organism
    """
    if data_dir is None:
        data_dir = get_data_dir() / "ncbigene"
    else:
        data_dir = Path(data_dir)
    
    tax_ids = list_available_organisms(data_dir)
    
    organism_info = {}
    for tax_id in tax_ids:
        names2gene_file = data_dir / f"names2ncbigene_tax{tax_id}_protein-coding.lzma"
        gene2names_file = data_dir / f"ncbigene2names_tax{tax_id}_protein-coding.lzma"
        
        organism_info[tax_id] = {
            'has_names2gene': names2gene_file.exists(),
            'has_gene2names': gene2names_file.exists(),
            'names2gene_file': str(names2gene_file) if names2gene_file.exists() else None,
            'gene2names_file': str(gene2names_file) if gene2names_file.exists() else None,
            'complete': names2gene_file.exists() and gene2names_file.exists()
        }
    
    return organism_info 