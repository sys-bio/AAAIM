"""
ChEBI ontology traversal and ChEBI → KEGG compound normalization with optional
relaxation along is_a parents.

Used by the KEGG reaction–mapping pipeline to widen mappings only when a
metabolite is unmatched or drives a weak reaction match.

ChEBI → KEGG in this package
----------------------------
The **same reference table** from ``load_chebi2kegg_dict`` (pickled
ChEBI→KEGG compound mapping) underlies:

- **This module** — ``merge_chebi_to_kegg_mapping`` turns the raw dict into
  ChEBI → set(KEGG). ``normalize_chebi`` / ``normalize_reaction`` add *optional*
  ontology relaxation (walking ChEBI parents) so more KEGG compounds can match
  during reaction scoring.

- **utils.map_chebi_to_kegg** — uses that table *without*
  relaxation: it expands recommendation rows so each ChEBI maps to one or more
  KEGG compound columns for downstream amendment logic. It does not walk the
  ChEBI hierarchy.

For table-only lookups (no relaxation), use ``kegg_ids_for_chebi_term`` or
``merge_chebi_to_kegg_mapping``; for reaction matching with optional relaxation,
use ``normalize_chebi`` / ``normalize_reaction``.
"""

from __future__ import annotations

import gzip
import json
import logging
from collections import defaultdict
from collections import deque
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
    Union,
)

import pandas as pd

from .kegg_compound_ids import parse_kegg_compound_id

# Baseline vs leave-one-ChEBI-out scores for ``detect_problematic_metabolites``.
# Call with ``exclude_chebi=None`` for full reaction; with a ChEBI id to drop
# that term's contribution on both sides of the fingerprint.
ReactionScoreMatcher = Callable[[Optional[str]], float]

logger = logging.getLogger(__name__)

ParentMap = Mapping[str, Set[str]]
ChebiToKegg = Mapping[str, Any]
ChildMap = Mapping[str, Set[str]]
_DIRECTION_PRIORITY: Dict[str, int] = {"up": 0, "down": 1, "exact": 2}

# Species id → one ChEBI string or multiple candidates (same id, multiple rows in recommendations).
SpeciesToChebi = Mapping[str, Any]


def iter_chebi_for_species(species_to_chebi: SpeciesToChebi, sid: str) -> List[str]:
    """
    Return ordered distinct ChEBI annotation strings for a model species id.

    Values may be a single string (backward compatible) or a sequence of strings
    when multiple recommendation rows exist for the same species.
    """
    v = species_to_chebi.get(sid)
    if v is None:
        return []
    if isinstance(v, str):
        s = v.strip()
        return [s] if s else []
    if isinstance(v, (list, tuple)):
        out: List[str] = []
        seen: Set[str] = set()
        for x in v:
            c = str(x).strip()
            if c and c not in seen:
                seen.add(c)
                out.append(c)
        return out
    s = str(v).strip()
    return [s] if s else []


def parse_chebi_obo(obo_path: Union[str, Path]) -> Dict[str, Set[str]]:
    """
    Parse a ChEBI OBO file and return directed is_a edges: child → parents.

    Returns:
        Mapping from ChEBI ID (e.g. CHEBI:17234) to a set of parent ChEBI IDs.
    """
    parent_map: Dict[str, Set[str]] = defaultdict(set)
    current_id: Optional[str] = None
    path = Path(obo_path)

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line == "[Term]":
                current_id = None
                continue
            if line.startswith("id:"):
                current_id = line.split("id:", 1)[1].strip()
            elif line.startswith("is_a:") and current_id:
                parent_id = line.split("is_a:", 1)[1].split("!")[0].strip()
                if parent_id:
                    parent_map[current_id].add(parent_id)

    return {k: set(v) for k, v in parent_map.items()}


def get_ancestors(
    chebi_id: str,
    parent_map: ParentMap,
    depth: Optional[int] = None,
) -> Set[str]:
    """
    Ancestors of chebi_id along is_a, excluding chebi_id itself.

    Args:
        chebi_id: ChEBI identifier.
        parent_map: Child → parent sets from parse_chebi_obo (or equivalent).
        depth: If None, full transitive closure. If a non-negative int, include
            only nodes within that many edges from chebi_id (parents at 1 hop,
            grandparents at 2, ...).
    """
    visited: Set[str] = set()
    frontier: List[Tuple[str, int]] = [(chebi_id, 0)]

    while frontier:
        current, d = frontier.pop()
        if current in visited:
            continue
        visited.add(current)

        if depth is not None and d >= depth:
            continue

        for parent in parent_map.get(current, ()):
            frontier.append((parent, d + 1))

    visited.discard(chebi_id)
    return visited


def expand_chebi_with_metadata(
    seed_chebi_id: str,
    parent_map: ParentMap,
    *,
    child_map: Optional[ChildMap] = None,
    max_up_depth: int = 0,
    max_down_depth: int = 0,
) -> Dict[str, Dict[str, Any]]:
    """
    BFS expansion returning per-node relaxation metadata.

    Returns mapping:
        chebi_id -> {"direction": "exact" | "up" | "down", "distance": int}

    Tie-breaking for nodes reachable by multiple paths:
    1) smaller distance
    2) direction preference exact > down > up
    """
    def _should_update(node: str, direction: str, distance: int) -> bool:
        prev = best.get(node)
        if prev is None:
            return True
        prev_dist = int(prev.get("distance", 10**9))
        if distance < prev_dist:
            return True
        if distance > prev_dist:
            return False
        prev_dir = str(prev.get("direction", "up"))
        return _DIRECTION_PRIORITY.get(direction, -1) > _DIRECTION_PRIORITY.get(prev_dir, -1)

    seed = str(seed_chebi_id).strip()
    if not seed:
        return {}

    if child_map is None:
        derived: Dict[str, Set[str]] = defaultdict(set)
        for child, parents in parent_map.items():
            for parent in parents:
                derived[str(parent)].add(str(child))
        child_map = {k: set(v) for k, v in derived.items()}

    best: Dict[str, Dict[str, Any]] = {seed: {"direction": "exact", "distance": 0}}
    q = deque([(seed, "exact", 0)])

    while q:
        node, direction, distance = q.popleft()

        if direction in ("exact", "up") and distance < int(max_up_depth):
            for parent in parent_map.get(node, ()):
                p = str(parent).strip()
                if not p:
                    continue
                cand_dir = "up"
                cand_dist = distance + 1
                if _should_update(p, cand_dir, cand_dist):
                    best[p] = {"direction": cand_dir, "distance": cand_dist}
                    q.append((p, cand_dir, cand_dist))

        if direction in ("exact", "down") and distance < int(max_down_depth):
            for child in child_map.get(node, ()):
                c = str(child).strip()
                if not c:
                    continue
                cand_dir = "down"
                cand_dist = distance + 1
                if _should_update(c, cand_dir, cand_dist):
                    best[c] = {"direction": cand_dir, "distance": cand_dist}
                    q.append((c, cand_dir, cand_dist))

    return best


def _coerce_kegg_values(raw: Any) -> Set[str]:
    """Normalize pickle/CSV values to a set of KEGG compound strings."""
    if raw is None:
        return set()
    if isinstance(raw, float) and str(raw) == "nan":
        return set()
    if isinstance(raw, str):
        s = raw.strip()
        return {s} if s else set()
    if isinstance(raw, (set, frozenset)):
        return {str(x).strip() for x in raw if str(x).strip()}
    if isinstance(raw, (list, tuple)):
        out: Set[str] = set()
        for x in raw:
            out.update(_coerce_kegg_values(x))
        return out
    return {str(raw).strip()} if str(raw).strip() else set()


def merge_chebi_to_kegg_mapping(raw: Mapping[str, Any]) -> Dict[str, Set[str]]:
    """
    Build ChEBI → set(KEGG) from a flat ChEBI→KEGG map (string or list values).

    This is the canonical way to consume ``load_chebi2kegg_dict()`` for graph
    and scoring code. For pandas row expansion without relaxation, see
    ``utils.map_chebi_to_kegg``.
    """
    merged: Dict[str, Set[str]] = defaultdict(set)
    for chebi, val in raw.items():
        if chebi is None:
            continue
        cid = str(chebi).strip()
        if not cid:
            continue
        merged[cid].update(_coerce_kegg_values(val))
    return {k: set(v) for k, v in merged.items()}


def _chebi_lookup_keys(chebi_id: str) -> Iterator[str]:
    """Try common key variants for cross-reference dicts."""
    c = chebi_id.strip()
    if not c:
        return
    yield c
    upper = c.upper()
    if upper.startswith("CHEBI:"):
        num = c.split(":", 1)[-1]
        if num.isdigit():
            yield num
            yield f"CHEBI:{num}"
    elif c.isdigit():
        yield f"CHEBI:{c}"


def kegg_ids_for_chebi_term(
    chebi_id: str,
    chebi_to_kegg: ChebiToKegg,
) -> Set[str]:
    """Direct KEGG compounds for this ChEBI term only (no ontology walk)."""
    for key in _chebi_lookup_keys(chebi_id):
        if key in chebi_to_kegg:
            return _coerce_kegg_values(chebi_to_kegg[key])
    return set()


def chebi_best_kegg_meta_with_ontology_fallback(
    seed_chebi_id: str,
    chebi_to_kegg: ChebiToKegg,
    parent_map: ParentMap,
    *,
    child_map: Optional[ChildMap] = None,
    max_ancestor_depth: int = 2,
    max_descendant_depth: int = 1,
) -> Dict[str, Dict[str, Any]]:
    """
    Map one ChEBI term to the best (direction, distance) KEGG compounds.

    Strategy:
    - Strict direct hits first (no ontology walk): each strict KEGG gets
      ``direction="exact"`` and ``distance=0``.
    - If strict hits are empty, walk ancestors (bounded by ``max_ancestor_depth``).
    - If ancestor-walk produced no KEGG hits, walk descendants instead
      (bounded by ``max_descendant_depth``).

    The return value is a dict keyed by KEGG compound id; values contain
    per-KEGG tie-breaking metadata when multiple expanded ChEBI terms map
    to the same KEGG compound.
    """
    seed = str(seed_chebi_id).strip()
    if not seed:
        return {}

    # Species annotated with bare KEGG compound ids (instead of ChEBI): identity map.
    kc = parse_kegg_compound_id(seed)
    if kc:
        return {kc: {"direction": "exact", "distance": 0}}

    # Strict mapping first (no ontology walk).
    direct_keggs = kegg_ids_for_chebi_term(seed, chebi_to_kegg)
    if direct_keggs:
        return {k: {"direction": "exact", "distance": 0} for k in direct_keggs}

    best_meta_by_kegg: Dict[str, Dict[str, Any]] = {}

    def _maybe_take(meta: Dict[str, Any], kegg_id: str) -> None:
        prev = best_meta_by_kegg.get(kegg_id)
        cand_dist = int(meta.get("distance", 0) or 0)
        cand_dir = str(meta.get("direction", "exact")).strip()
        if prev is None:
            best_meta_by_kegg[kegg_id] = {"direction": cand_dir, "distance": cand_dist}
            return

        prev_dist = int(prev.get("distance", 0) or 0)
        prev_dir = str(prev.get("direction", "exact")).strip()
        if cand_dist < prev_dist:
            best_meta_by_kegg[kegg_id] = {"direction": cand_dir, "distance": cand_dist}
        elif cand_dist == prev_dist:
            # Prefer exact > down > up (matches expand_chebi_with_metadata
            # tie-breaking).
            if _DIRECTION_PRIORITY.get(cand_dir, -1) > _DIRECTION_PRIORITY.get(prev_dir, -1):
                best_meta_by_kegg[kegg_id] = {"direction": cand_dir, "distance": cand_dist}

    # 1) Try ancestors (up) up to max_ancestor_depth.
    up_expanded = expand_chebi_with_metadata(
        seed,
        parent_map,
        child_map=child_map,
        max_up_depth=max_ancestor_depth,
        max_down_depth=0,
    )
    for expanded_term, meta in up_expanded.items():
        expanded_term = str(expanded_term).strip()
        if not expanded_term:
            continue
        for k in kegg_ids_for_chebi_term(expanded_term, chebi_to_kegg):
            _maybe_take(meta, k)

    # 2) If up produced nothing, then try descendants (down).
    if not best_meta_by_kegg and max_descendant_depth > 0:
        down_expanded = expand_chebi_with_metadata(
            seed,
            parent_map,
            child_map=child_map,
            max_up_depth=0,
            max_down_depth=max_descendant_depth,
        )
        for expanded_term, meta in down_expanded.items():
            expanded_term = str(expanded_term).strip()
            if not expanded_term:
                continue
            for k in kegg_ids_for_chebi_term(expanded_term, chebi_to_kegg):
                _maybe_take(meta, k)

    return best_meta_by_kegg


def chebi_best_kegg_ids_with_ontology_fallback(
    seed_chebi_id: str,
    chebi_to_kegg: ChebiToKegg,
    parent_map: ParentMap,
    *,
    child_map: Optional[ChildMap] = None,
    max_ancestor_depth: int = 2,
    max_descendant_depth: int = 1,
) -> List[str]:
    """Convenience wrapper returning just the sorted KEGG id list."""
    best_meta_by_kegg = chebi_best_kegg_meta_with_ontology_fallback(
        seed_chebi_id,
        chebi_to_kegg,
        parent_map,
        child_map=child_map,
        max_ancestor_depth=max_ancestor_depth,
        max_descendant_depth=max_descendant_depth,
    )
    return sorted(best_meta_by_kegg.keys())


def normalize_chebi(
    chebi_id: str,
    chebi_to_kegg: ChebiToKegg,
    parent_map: ParentMap,
    level: int = 0,
    max_depth: int = 2,
) -> Set[str]:
    """
    KEGG compound IDs reachable from a ChEBI term at a given relaxation level.

    Level 0: only direct ChEBI→KEGG hits.
    Level L≥1: union of KEGG hits over this term and is_a ancestors within
    min(L, max_depth) hops.

    **Not** ``annotation_workflow.normalize_reactions`` (KEGG cofactor filtering
    for whole reactions). For multi-metabolite ChEBI lists use
    ``normalize_reaction``.
    """
    kc_id = parse_kegg_compound_id(chebi_id)
    if kc_id:
        return {kc_id}

    if level <= 0:
        return kegg_ids_for_chebi_term(chebi_id, chebi_to_kegg)

    hop_cap = max(0, min(int(level), int(max_depth)))
    ancestors = get_ancestors(chebi_id, parent_map, depth=hop_cap)
    expanded = {chebi_id} | ancestors

    kegg_ids: Set[str] = set()
    for cid in expanded:
        kegg_ids.update(kegg_ids_for_chebi_term(cid, chebi_to_kegg))
    return kegg_ids


def compute_relaxation_penalty(
    direction: str,
    distance: int,
    *,
    lambda_down: float = 0.2,
    lambda_up: float = 1.2,
) -> float:
    """
    Penalty for one species-level relaxation hop.

    exact -> 0
    down  -> lambda_down * distance
    up    -> lambda_up * distance
    """

    if lambda_up <= lambda_down:
        raise ValueError("lambda_up must be greater than lambda_down")

    d = max(0, int(distance))
    dir_norm = str(direction).strip().lower()
    if dir_norm == "exact":
        return 0.0
    if dir_norm == "down":
        return float(lambda_down) * d
    if dir_norm == "up":
        return float(lambda_up) * d
    return float(lambda_up) * d


def normalize_reaction(
    chebi_metabolites: Iterable[str],
    chebi_to_kegg: ChebiToKegg,
    parent_map: ParentMap,
    level: int = 0,
    max_depth: int = 2,
    *,
    as_union: bool = False,
) -> Union[Set[str], Dict[str, Set[str]]]:
    """
    Map each ChEBI in chebi_metabolites to relaxed KEGG compound sets.

    **Not** related to ``annotation_workflow.normalize_reactions``, which strips
    cofactors from *already-KEGG-mapped* reaction lists for similarity scoring.

    Args:
        chebi_metabolites: ChEBI IDs participating in one reaction (any order).
        as_union: If True, return the union of all per-metabolite sets; if False,
            return a dict ChEBI → set(KEGG).

    Returns:
        Dict mapping each distinct input ChEBI string to KEGG IDs, or a single
        set if as_union is True.

    See Also:
        ``normalize_chebi`` — per-term ChEBI→KEGG with optional ontology walk.
        ``annotation_workflow.normalize_reactions`` — KEGG multiset / cofactor
        filtering for reaction comparison.
    """
    per: Dict[str, Set[str]] = {}
    for chebi_id in chebi_metabolites:
        c = str(chebi_id).strip()
        if not c:
            continue
        per[c] = normalize_chebi(c, chebi_to_kegg, parent_map, level=level, max_depth=max_depth)
    if as_union:
        out: Set[str] = set()
        for s in per.values():
            out.update(s)
        return out
    return per


def progressive_normalization(
    chebi_id: str,
    chebi_to_kegg: ChebiToKegg,
    parent_map: ParentMap,
    max_level: int = 2,
    max_depth: int = 2,
) -> Tuple[Set[str], int]:
    """
    Try relaxation levels 0 … max_level until at least one KEGG ID is found.

    Returns:
        (kegg_ids, level_used). If nothing matches at any level, returns
        (empty set, max_level).
    """
    for level in range(max_level + 1):
        kegg_ids = normalize_chebi(
            chebi_id, chebi_to_kegg, parent_map, level=level, max_depth=max_depth
        )
        if kegg_ids:
            return kegg_ids, level
    return set(), max_level


_CACHED_PARENT_MAP: Optional[Dict[str, Set[str]]] = None
_CACHED_PARENT_MAP_SOURCE: Optional[str] = None
_CACHED_CHILD_MAP: Optional[Dict[str, Set[str]]] = None
_CACHED_CHILD_MAP_SOURCE: Optional[str] = None


def load_chebi_parent_map(
    *,
    obo_path: Optional[Union[str, Path]] = None,
    gz_path: Optional[Union[str, Path]] = None,
    data_dir: Optional[Union[str, Path]] = None,
    use_cache: bool = True,
) -> Dict[str, Set[str]]:
    """
    Load ChEBI is_a parent map from gzipped JSON (list values) or from OBO.

    Search order: explicit gz_path → data_dir/chebi_parent_map.json.gz →
    explicit obo_path → data_dir/chebi.obo.
    """
    global _CACHED_PARENT_MAP, _CACHED_PARENT_MAP_SOURCE

    base = Path(data_dir) if data_dir else Path(__file__).resolve().parent.parent.parent / "data" / "chebi"
    gz = Path(gz_path) if gz_path else base / "chebi_parent_map.json.gz"
    obo = Path(obo_path) if obo_path else base / "chebi.obo"

    chosen: Optional[Path] = None
    if gz_path:
        chosen = Path(gz_path)
    elif gz.exists():
        chosen = gz
    elif obo_path:
        chosen = Path(obo_path)
    elif obo.exists():
        chosen = obo

    if chosen is None or not chosen.exists():
        raise FileNotFoundError(
            f"No ChEBI parent source found. Tried gz={gz} and obo={obo}. "
            "Place chebi.obo or chebi_parent_map.json.gz under data/chebi/."
        )

    src_key = str(chosen.resolve())
    if use_cache and _CACHED_PARENT_MAP is not None and _CACHED_PARENT_MAP_SOURCE == src_key:
        return _CACHED_PARENT_MAP

    if chosen.suffix == ".gz" or str(chosen).endswith(".json.gz"):
        with gzip.open(chosen, "rt", encoding="utf-8") as f:
            raw = json.load(f)
        def _coerce_parent_values(val) -> Set[str]:
            """
            Normalize parent map values from gz JSON into a set[str].

            The cached JSON may store parents either as:
            - ["CHEBI:...","CHEBI:..."]
            - [["is_a","CHEBI:..."], ["is_a","CHEBI:..."], ...]
            - a single scalar
            """
            out: Set[str] = set()
            if val is None:
                return out

            # Make it iterable in a uniform way.
            items = val if isinstance(val, (list, tuple, set)) else [val]
            for item in items:
                if item is None:
                    continue
                if isinstance(item, str):
                    s = item.strip()
                    if s:
                        out.add(s)
                    continue
                if isinstance(item, (list, tuple, set)):
                    # Common form: ["is_a", "CHEBI:133004"]
                    for sub in item:
                        if isinstance(sub, str):
                            s = sub.strip()
                            if s.startswith("CHEBI:"):
                                out.add(s)
                    continue

                # Fallback: best-effort stringification for unexpected scalars.
                s = str(item).strip()
                if s:
                    out.add(s)
            return out

        parent_map = {str(k): _coerce_parent_values(v) for k, v in raw.items()}
    else:
        parent_map = dict(parse_chebi_obo(chosen))

    if use_cache:
        _CACHED_PARENT_MAP = parent_map
        _CACHED_PARENT_MAP_SOURCE = src_key

    logger.info("Loaded ChEBI parent map (%d terms) from %s", len(parent_map), chosen)
    return parent_map


def load_chebi_child_map(
    *,
    parent_map: Optional[ParentMap] = None,
    gz_path: Optional[Union[str, Path]] = None,
    data_dir: Optional[Union[str, Path]] = None,
    use_cache: bool = True,
) -> Dict[str, Set[str]]:
    """
    Load or derive parent->children map for downward expansion.
    """

    global _CACHED_CHILD_MAP, _CACHED_CHILD_MAP_SOURCE

    base = Path(data_dir) if data_dir else Path(__file__).resolve().parent.parent.parent / "data" / "chebi"
    gz = Path(gz_path) if gz_path else base / "chebi_child_map.json.gz"
    src_key = str(gz.resolve()) if gz.exists() else "__derived__"

    if use_cache and _CACHED_CHILD_MAP is not None and _CACHED_CHILD_MAP_SOURCE == src_key:
        return _CACHED_CHILD_MAP

    child_map: Dict[str, Set[str]] = defaultdict(set)

    if gz.exists():
        with gzip.open(gz, "rt", encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, dict):
            for parent, children in raw.items():
                p = str(parent).strip()
                if not p:
                    continue
                vals = children if isinstance(children, (list, set, tuple)) else [children]
                for child in vals:
                    c = str(child).strip()
                    if c:
                        child_map[p].add(c)

    if not child_map:
        pm = parent_map if parent_map is not None else load_chebi_parent_map(data_dir=data_dir)
        for child, parents in pm.items():
            c = str(child).strip()
            if not c:
                continue
            for parent in parents:
                p = str(parent).strip()
                if p:
                    child_map[p].add(c)
        src_key = "__derived__"

    out = {k: set(v) for k, v in child_map.items()}
    if use_cache:
        _CACHED_CHILD_MAP = out
        _CACHED_CHILD_MAP_SOURCE = src_key
    return out


def build_kegg_mapping_dataframe(
    *args: Any,
    max_ancestor_depth: int = 2,
    child_map: Optional[ChildMap] = None,
    max_descendant_depth: int = 1,
) -> Any:
    """
    Long-form DataFrame columns [id, KEGG_ID] for map_reactions_to_kegg.

    One row per (species, KEGG) pair; duplicate indices are OK for the
    existing lookup logic in map_metabolites_to_kegg.
    """
    # Backward-compatible call patterns:
    # - (species_to_chebi, relax_level, chebi_to_kegg, parent_map)
    # - (species_ids, species_to_chebi, relax_level, chebi_to_kegg, parent_map)
    species_ids: Optional[Iterable[str]] = None
    if len(args) == 4:
        species_to_chebi, relax_level, chebi_to_kegg, parent_map = args
    elif len(args) == 5:
        species_ids, species_to_chebi, relax_level, chebi_to_kegg, parent_map = args
    else:
        raise TypeError(
            "build_kegg_mapping_dataframe expected 4 positional args "
            "(species_to_chebi, relax_level, chebi_to_kegg, parent_map) "
            "or 5 positional args (species_ids, species_to_chebi, relax_level, "
            "chebi_to_kegg, parent_map)"
        )

    rows: List[Dict[str, Any]] = []
    species_iter = species_ids if species_ids is not None else species_to_chebi.keys()
    for sid in species_iter:
        chebi_list = iter_chebi_for_species(species_to_chebi, str(sid))
        if not chebi_list:
            continue
        for chebi in chebi_list:
            seed = str(chebi).strip()
            if not seed:
                continue

            # Strict direct hits first; then walk ancestors; then descendants.
            best_meta_by_kegg = chebi_best_kegg_meta_with_ontology_fallback(
                seed,
                chebi_to_kegg,
                parent_map,
                child_map=child_map,
                max_ancestor_depth=max_ancestor_depth,
                max_descendant_depth=max_descendant_depth,
            )

            for k in sorted(best_meta_by_kegg.keys()):
                meta = best_meta_by_kegg[k]
                rows.append(
                    {
                        "id": sid,
                        "KEGG_ID": k,
                        "canonical_id": seed,  # keep original seed term
                        "direction": meta.get("direction", "exact"),
                        "distance": int(meta.get("distance", 0) or 0),
                    }
                )
    if not rows:
        return pd.DataFrame(columns=["id", "KEGG_ID", "canonical_id", "direction", "distance"])
    return pd.DataFrame(rows)


def detect_unmapped_species_ids(
    species_ids: Iterable[str],
    species_to_chebi: SpeciesToChebi,
    relax_level: Mapping[str, int],
    chebi_to_kegg: ChebiToKegg,
    parent_map: ParentMap,
    max_ancestor_depth: int = 2,
) -> Set[str]:
    """
    Species (model metabolite IDs) whose ChEBI annotation yields no KEGG compounds
    at that species' current relaxation level.
    """
    unmapped: Set[str] = set()
    for sid in species_ids:
        chebi_list = iter_chebi_for_species(species_to_chebi, str(sid))
        if not chebi_list:
            continue
        lvl = int(relax_level.get(sid, 0))
        if not any(
            normalize_chebi(
                ch, chebi_to_kegg, parent_map, level=lvl, max_depth=max_ancestor_depth
            )
            for ch in chebi_list
        ):
            unmapped.add(sid)
    return unmapped


def detect_unmapped_metabolites(
    chebi_metabolites: Iterable[str],
    chebi_to_kegg: ChebiToKegg,
    parent_map: ParentMap,
    level: int = 0,
    max_depth: int = 2,
) -> Set[str]:
    """
    ChEBI terms that have no KEGG compound mapping at the given relaxation level.

    Uses ``normalize_chebi`` so semantics match the main normalization pipeline.
    """
    unmapped: Set[str] = set()
    seen: Set[str] = set()
    for chebi_id in chebi_metabolites:
        c = str(chebi_id).strip()
        if not c or c in seen:
            continue
        seen.add(c)
        if not normalize_chebi(c, chebi_to_kegg, parent_map, level=level, max_depth=max_depth):
            unmapped.add(c)
    return unmapped


def detect_problematic_metabolites(
    chebi_metabolites: Iterable[str],
    chebi_to_kegg: ChebiToKegg,
    parent_map: ParentMap,
    reaction_matcher: ReactionScoreMatcher,
    level: int = 0,
    max_depth: int = 2,
    threshold: float = 0.0,
) -> Set[str]:
    """
    ChEBI terms whose removal **increases** the reaction match score vs a fixed
    reference (leave-one-out on the ChEBI annotation).

    ``reaction_matcher`` must be provided by the integration layer. The
    contract is:

    - ``reaction_matcher(None)``: score for the full reaction fingerprint.
    - ``reaction_matcher(chebi_id)``: score when species carrying this ChEBI
      are omitted from substrate/product KEGG counters.

    ``chebi_to_kegg`` and ``parent_map`` are included for API symmetry with
    ``detect_unmapped_metabolites``; the matcher typically closes over the
    current mapping state so these may be unused here.

    Args:
        threshold: Minimum score gain (vs baseline) required to flag a term.
        level / max_depth: API symmetry with ``normalize_chebi`` / callers; the
            matcher closure from the pipeline encodes the active level.
        chebi_to_kegg / parent_map: API symmetry with ``detect_unmapped_metabolites``.
    """
    _ = (chebi_to_kegg, parent_map, level, max_depth)
    baseline = reaction_matcher(None)
    problematic: Set[str] = set()
    seen: Set[str] = set()
    for chebi_id in chebi_metabolites:
        c = str(chebi_id).strip()
        if not c or c in seen:
            continue
        seen.add(c)
        if reaction_matcher(c) > baseline + threshold:
            problematic.add(c)
    return problematic


def select_metabolites_to_relax(
    chebi_metabolites: Iterable[str],
    chebi_to_kegg: ChebiToKegg,
    parent_map: ParentMap,
    reaction_matcher: Optional[ReactionScoreMatcher] = None,
    *,
    level: int = 0,
    max_depth: int = 2,
    score_threshold: float = 0.0,
    participant_species: Optional[Iterable[str]] = None,
    species_to_chebi: Optional[SpeciesToChebi] = None,
    relax_level: Optional[Mapping[str, int]] = None,
) -> Set[str]:
    """
    Union of ChEBI annotation terms to relax: unmapped and/or score-sensitive.

    Unmapped detection uses ``detect_unmapped_species_ids`` when
    ``participant_species``, ``species_to_chebi``, and ``relax_level`` are
    provided (per-species relaxation levels). Otherwise uses
    ``detect_unmapped_metabolites`` with a single ``level`` for all ChEBI ids.

    Score-sensitive detection runs only if ``reaction_matcher`` is not None
    (see ``detect_problematic_metabolites``).
    """
    if (
        participant_species is not None
        and species_to_chebi is not None
        and relax_level is not None
    ):
        um_s = detect_unmapped_species_ids(
            participant_species,
            species_to_chebi,
            relax_level,
            chebi_to_kegg,
            parent_map,
            max_ancestor_depth=max_depth,
        )
        unmapped: Set[str] = set()
        for s in um_s:
            for ch in iter_chebi_for_species(species_to_chebi, str(s)):
                c = str(ch).strip()
                if not c:
                    continue
                lvl = int(relax_level.get(s, 0))
                if not normalize_chebi(
                    c, chebi_to_kegg, parent_map, level=lvl, max_depth=max_depth
                ):
                    unmapped.add(c)
    else:
        unmapped = detect_unmapped_metabolites(
            chebi_metabolites, chebi_to_kegg, parent_map, level=level, max_depth=max_depth
        )

    if reaction_matcher is None:
        return unmapped

    score_sensitive = detect_problematic_metabolites(
        chebi_metabolites,
        chebi_to_kegg,
        parent_map,
        reaction_matcher,
        level=level,
        max_depth=max_depth,
        threshold=score_threshold,
    )
    return unmapped | score_sensitive


def select_relaxations_by_global_improvement(
    candidate_species: Iterable[str],
    relaxation_levels: Mapping[str, int],
    compute_global_score: Callable[[Mapping[str, int]], float],
    *,
    max_relax_level: int,
    delta_threshold: float = 0.0,
):
    """
    Select species to relax only when the global objective improves.

    Args:
        candidate_species: Species IDs eligible for one-step trial relaxation.
        relaxation_levels: Current species -> level mapping.
        compute_global_score: Callable that evaluates the full-model objective for a
            given relaxation-level mapping.
        max_relax_level: Upper bound for any species relaxation level.
        delta_threshold: Minimum strictly-positive improvement required.
    """
    to_relax: List[str] = []

    current_score = compute_global_score(relaxation_levels)

    for s in candidate_species:
        if int(relaxation_levels.get(s, 0)) >= int(max_relax_level):
            continue

        trial_levels = relaxation_levels.copy()
        trial_levels[s] = int(trial_levels.get(s, 0)) + 1

        new_score = compute_global_score(trial_levels)

        if float(new_score) - float(current_score) > float(delta_threshold):
            to_relax.append(s)

    return to_relax
