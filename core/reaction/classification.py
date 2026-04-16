"""Reaction mappability classification (no dependency on database_search)."""

from __future__ import annotations

from typing import Iterable

from .amendment_config import CofactorConfig


def classify_reaction(
    reaction,
    filtered_species: Iterable[str],
    candidates: Iterable[str],
) -> str:
    """
    Classify a reaction by score eligibility.

    ``filtered_species`` is the set of KEGG compound IDs still in play after
    cofactor filtering (non-ignored metabolites on this reaction).

    Returns:
        - "ambiguous_mapping": ``filtered_species`` is empty (vacuous ``all()``),
          or every ID in it is a configured cofactor KEGG id
          (see :attr:`CofactorConfig.kegg_ids`). The transformation is
          underdetermined for a unique KEGG reaction mapping.
        - "non_mappable": the equation string has no ``->`` or ``=>``, an empty
          LHS/RHS after splitting, or there are zero candidate KEGG reactions
          after database filtering.
        - "mappable": equation parses, at least one candidate KEGG reaction id,
          and not classified as ambiguous_mapping above.
    """
    filtered = list(filtered_species) if filtered_species is not None else []
    cand = list(candidates) if candidates is not None else []
    if all(species in CofactorConfig().kegg_ids for species in filtered):
        return "ambiguous_mapping"
    if "=>" in reaction:
        lhs, rhs = (s.strip() for s in reaction.split("=>", 1))
    elif "->" in reaction:
        lhs, rhs = (s.strip() for s in reaction.split("->", 1))
    else:
        return "non_mappable"
    if not lhs or not rhs:
        return "non_mappable"
    if len(cand) == 0:
        return "non_mappable"

    return "mappable"
