"""Shared data types"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from collections import Counter

@dataclass
class Recommendation:
    """
    Recommendation dataclass for database search results.
    """
    id: str  # ID for the species
    synonyms: list  # List of synonyms predicted by LLM
    candidates: list  # List of database IDs (ChEBI IDs, NCBI gene IDs, etc.)
    candidate_names: list  # List of names of the predicted candidates
    match_score: list  # Match scores (normalized hit count for direct search, cosine similarity for RAG)
    # Optional candidate-level provenance.  The parallel lists are kept here so
    # existing callers that construct Recommendation with the original five
    # arguments remain compatible.
    candidate_taxa: List[Optional[str]] = field(default_factory=list)
    candidate_identities: List[str] = field(default_factory=list)
    candidate_identity_ranks: List[int] = field(default_factory=list)
    component_ids: List[str] = field(default_factory=list)
    component_names: List[str] = field(default_factory=list)
    component_types: List[str] = field(default_factory=list)
    candidate_ranks: List[int] = field(default_factory=list)
    unmatched_components: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class ReactionRecommendation(Recommendation):
    """
    Extended Recommendation class specifically for reaction annotations.
    Includes additional fields for reaction-specific information.
    """
    substrates: List[Counter] = field(default_factory=list)  # Substrate Counters
    products: List[Counter] = field(default_factory=list)  # Product Counters
    equation: str = ""  # Original reaction equation string
    metadata: Optional[Dict[str, Any]] = None  # Additional metadata about the reaction 
