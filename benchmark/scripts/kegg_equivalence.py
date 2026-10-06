"""KEGG reaction equivalence classes for equivalence-aware benchmark scoring.

Exact-id matching understates performance: KEGG often carries several reaction entries
for the same underlying biochemistry, so predicting a sibling entry is a materially
different kind of error from predicting an unrelated reaction. These helpers build
equivalence groups from the KEGG reaction feature table:

``ec``
    Shared EC number (``ENZYME``). This is the grouping that KEGG's BRITE
    ``Enzymatic reactions [BR:br08201]`` hierarchy encodes as its leaf level.
``ko``
    Shared KEGG Orthology id (``ORTHOLOGY``), i.e. the same ortholog group.
``brite_orthology``
    Shared EC *or* KO. This is the headline equivalence-aware criterion.
``rclass``
    Shared reaction class (``RCLASS``), i.e. the same chemical transformation
    pattern on a substrate pair. Reported separately as it is the loosest grouping.
"""

from __future__ import annotations

import functools
import re
import sys
from pathlib import Path
from typing import Dict, FrozenSet, Iterable, Mapping, Set

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

KO_RE = re.compile(r"\bK\d{5}\b")
EC_RE = re.compile(r"\b\d+\.[\d\-]+\.[\d\-]+\.[\d\-]+\b")
RCLASS_RE = re.compile(r"\bRC\d{5}\b")

EQUIVALENCE_KINDS = ("ec", "ko", "brite_orthology", "rclass")


@functools.lru_cache(maxsize=1)
def _features() -> Mapping[str, Mapping[str, str]]:
    from core.database_search import load_kegg_reaction_features_dict

    return load_kegg_reaction_features_dict()


@functools.lru_cache(maxsize=1)
def equivalence_index() -> Dict[str, Dict[str, FrozenSet[str]]]:
    """Map each KEGG reaction id to its group memberships per equivalence kind."""
    index: Dict[str, Dict[str, FrozenSet[str]]] = {}
    for kegg_id, feats in _features().items():
        enzyme = str(feats.get("ENZYME", "") or "")
        orthology = str(feats.get("ORTHOLOGY", "") or "")
        rclass = str(feats.get("RCLASS", "") or "")

        ec = frozenset(f"EC:{m}" for m in EC_RE.findall(enzyme))
        # KO descriptions embed "[EC:...]"; take EC only from the ENZYME field so the
        # two groupings stay independent.
        ko = frozenset(f"KO:{m}" for m in KO_RE.findall(orthology))
        rc = frozenset(RCLASS_RE.findall(rclass))

        index[str(kegg_id)] = {
            "ec": ec,
            "ko": ko,
            "brite_orthology": ec | ko,
            "rclass": rc,
        }
    return index


def groups_for(kegg_id: str, kind: str) -> FrozenSet[str]:
    return equivalence_index().get(str(kegg_id), {}).get(kind, frozenset())


def is_equivalent(candidate: str, truth_ids: Iterable[str], kind: str) -> bool:
    """True if ``candidate`` shares at least one group of ``kind`` with any truth id.

    Reactions with no annotation for a kind have an empty group set and therefore never
    match, so unannotated entries cannot inflate equivalence-aware scores.
    """
    cand_groups = groups_for(candidate, kind)
    if not cand_groups:
        return False
    for truth in truth_ids:
        if cand_groups & groups_for(truth, kind):
            return True
    return False


def matches_exact(candidate: str, truth_ids: Set[str]) -> bool:
    return str(candidate) in truth_ids


def match_kinds(candidate: str, truth_ids: Set[str]) -> Dict[str, bool]:
    """Evaluate a candidate under every criterion at once.

    Equivalence criteria are inclusive of exact matches: an exact hit is by definition
    also equivalent, even when the entry carries no EC or KO annotation.
    """
    exact = matches_exact(candidate, truth_ids)
    out = {"exact": exact}
    for kind in EQUIVALENCE_KINDS:
        out[kind] = exact or is_equivalent(candidate, truth_ids, kind)
    return out


def coverage_stats() -> Dict[str, int]:
    """How many KEGG reactions carry each kind of grouping (for reporting caveats)."""
    idx = equivalence_index()
    stats = {"kegg_reactions": len(idx)}
    for kind in EQUIVALENCE_KINDS:
        stats[f"with_{kind}"] = sum(1 for v in idx.values() if v[kind])
    return stats
