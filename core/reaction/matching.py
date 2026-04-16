"""Equation parsing and metabolite-to-KEGG mapping."""

from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

from ..database_search import cancel_spectators

logger = logging.getLogger(__name__)


def normalize_reactions(model_reactions):
    """
    Prepare KEGG-mapped reaction dicts for comparison: drop cofactors and keep
    substrate/product multisets (Counters).

    **Not** ``hierarchy_relaxation.normalize_reaction``: that function maps
    ChEBI terms to (relaxed) KEGG compound ID sets using the ontology. This
    function assumes ``model_reactions`` already carry KEGG IDs in substrate /
    product lists.

    Args:
        model_reactions: List of reaction dictionaries
        
    Returns:
        List of normalized reaction dictionaries

    See Also:
        ``hierarchy_relaxation.normalize_reaction`` — ChEBI → KEGG compound sets.
    """
    normalized_reactions = []
    
    for rxn in model_reactions:
        subs = count_metabolites(rxn.get('substrates', []))
        prods = count_metabolites(rxn.get('products', []))
        
        normalized_reactions.append({
            'reaction_name_in_model': rxn.get('id', 'Unknown'),
            'substrate_counter': subs,
            'product_counter': prods,
        })
                
    return normalized_reactions
            
def count_metabolites(kegg_list):

    counter = Counter()

    for kegg_id in kegg_list:
        if kegg_id is None:
            continue  # skip unmapped
        if kegg_id:
            counter[kegg_id] += 1  # track stoichiometry
    return counter


def parse_metabolites(side: str) -> Counter:
    side = side.strip()
    if not side:
        return Counter()
    result = Counter()
    for term in side.split("+"):
        parts = term.strip().split()
        if len(parts) == 1:
            coeff = 1
            met = parts[0]
        else:
            try:
                coeff = float(parts[0])
            except ValueError:
                coeff = 1
                met = term.strip()
            else:
                met = parts[-1]
        met = met.lstrip("$")
        result[met] += coeff
    return result


def parse_reaction_equation(rxn_str: str) -> Tuple[Counter, Counter]:
    """
    Parse a reaction equation string into reactant and product metabolite Counters.

    Same rules as ``map_reactions_to_kegg`` (``+`` terms, optional stoichiometry).
    """
    if "=>" in rxn_str or "->" in rxn_str:
        lhs, rhs = re.split(r"=>|->", rxn_str)
    else:
        return Counter(), Counter()

    reactants = parse_metabolites(lhs)
    products = parse_metabolites(rhs)
    return reactants, products


def collect_species_ids_from_rxn_list(rxn_list: List[str], spectators: bool = False) -> Set[str]:
    """All metabolite species IDs appearing in ``rxn_list`` after optional spectator cancellation."""
    out: Set[str] = set()
    for rxn in rxn_list:
        if ":" in rxn:
            _, rxn_str = rxn.split(":", 1)
        else:
            rxn_str = rxn
        reactants, products = parse_reaction_equation(rxn_str)
        if not spectators:
            reactants, products = cancel_spectators(reactants, products)
        out.update(reactants.keys())
        out.update(products.keys())
    return out


def map_reactions_to_kegg(rxn_list: List[str], reaction_ids: List[str], id_df: pd.DataFrame, spectators=False) -> List[Dict[str, Any]]:
    """
    Map reaction strings to KEGG reaction identifiers.
    
    This function processes a list of reaction strings and maps the metabolites
    in each reaction to their corresponding KEGG IDs using the provided mapping DataFrame.
    
    Args:
        rxn_list: List of reaction strings in the format "id: reactants -> products"
        id_df: DataFrame with columns 'id' and 'KEGG_ID' mapping metabolite IDs to KEGG IDs
        
    Returns:
        List of dictionaries containing mapped reaction information:
        - id: Reaction identifier
        - reaction_string: Original reaction string
        - substrates: List of Counter objects with mapped substrate KEGG IDs and stoichiometry
        - products: List of Counter objects with mapped product KEGG IDs and stoichiometry
    """
    
    # Keep full rows so we can propagate relaxation metadata when available.
    id_lookup = id_df.copy()
    id_grouped = dict(list(id_lookup.groupby("id")))

    # Process each reaction
    output = []
       
    for idx, rxn in enumerate(rxn_list):
        # Extract reaction string (remove ID prefix if present)
        if ":" in rxn:
            _, rxn_str = rxn.split(":", 1)
        else:
            rxn_str = rxn

        # Parse reaction equation into reactants and products
        reactants, products = parse_reaction_equation(rxn_str)

        if not spectators: 
            # Stoichiometric cancellation -- eliminate specatators
            reactants, products = cancel_spectators(reactants, products)

        # Map metabolite CHEBI IDs to KEGG IDs
        substrates_mapped = map_metabolites_to_kegg(reactants, id_lookup, _grouped=id_grouped)
        products_mapped = map_metabolites_to_kegg(products, id_lookup, _grouped=id_grouped)

        # Store mapped reaction
        output.append({
            "id": reaction_ids[idx],
            "reaction_string": rxn_str,
            "substrates": substrates_mapped,
            "products": products_mapped
        })
    
    return output

def map_metabolites_to_kegg(
        counter: Counter,
        mapping_df: pd.DataFrame,
        *,
        _grouped: Optional[Dict[str, pd.DataFrame]] = None,
) -> List[Counter]:
        """
        Map metabolite IDs to KEGG IDs while preserving stoichiometry.
        
        For each metabolite in the counter, finds all possible KEGG IDs and
        generates all possible combinations of mappings.
        
        Args:
            counter: Counter mapping metabolite IDs to stoichiometric coefficients
            mapping_df: DataFrame mapping metabolite IDs to KEGG IDs (+ optional metadata)
            _grouped: Pre-computed ``mapping_df.groupby("id")`` dict for hot-path callers.
            
        Returns:
            List of Counter objects representing all possible KEGG ID mappings
        """
        has_relax_cols = "direction" in mapping_df.columns and "distance" in mapping_df.columns
        grouped = _grouped if _grouped is not None else dict(list(mapping_df.groupby("id")))

        id_choices = dict()
        for met, coeff in counter.items():
            try:
                met_rows = grouped.get(met)
                if met_rows is None or met_rows.empty:
                    raise KeyError(met)

                if has_relax_cols:
                    choices = []
                    for tup in met_rows.itertuples():
                        kid = tup.KEGG_ID
                        if pd.isna(kid) or str(kid).strip() == "":
                            continue
                        choices.append(
                            {
                                "kegg_id": str(kid).strip(),
                                "canonical_id": str(getattr(tup, "canonical_id", "")).strip(),
                                "direction": str(getattr(tup, "direction", "exact")).strip(),
                                "distance": int(getattr(tup, "distance", 0) or 0),
                            }
                        )
                else:
                    choices = [str(kid).strip() for kid in met_rows["KEGG_ID"].tolist() if str(kid).strip()]

                id_choices[met] = {
                    'species_id': met,
                    'coeff': coeff,
                    'candidates': choices
                }
            except (KeyError, IndexError):
                # Keep unmapped metabolites so downstream logic can:
                # - preserve stoichiometry (coeff)
                # - perform ChEBI-based recovery/relaxation to find candidates later
                logger.debug(f"No KEGG mapping found for metabolite: {met}")
                id_choices[met] = {
                    "species_id": met,
                    "coeff": coeff,
                    "candidates": [],
                }
        
        if not id_choices:
            return []
                       
        return id_choices
