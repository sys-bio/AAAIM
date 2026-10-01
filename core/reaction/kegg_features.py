"""Load and query KEGG reaction feature payloads (definitions, equations)."""

from __future__ import annotations

import logging
import lzma
import pickle
import re
from pathlib import Path
from typing import Dict, FrozenSet, List, Tuple

from utils.constants import REF_KEGG_REACTION_FEATURES

from .kegg_definition import extract_classifications

logger = logging.getLogger(__name__)

# A KEGG reaction id like ``R01600``.
_R_NUMBER_RE = re.compile(r"\bR\d{5}\b")

# BRITE section header marker. Examples we've seen in the data:
#   ``Enzymatic reactions [BR:br08201]``
#   ``IUBMB reaction hierarchy [BR:br08202]``
#   ``Overall reaction [br08210.html]``
_BRITE_SECTION_RE = re.compile(r"\[(?:BR:)?br\d+(?:\.html)?\]", re.IGNORECASE)

# IUBMB-block tag we care about for hierarchical grouping.
_IUBMB_TAG = "br08202"


def _normalize_kegg_reaction_id(annotation) -> str:
    """Resolve KEGG reaction id (R#####) from table/URI values like ``KEGG:R01600``."""
    if annotation is None:
        return ""
    if isinstance(annotation, float) and annotation != annotation:
        return ""
    s = str(annotation).strip()
    if not s or s.lower() == "nan":
        return ""
    if "KEGG:" in s.upper():
        s = s.split("KEGG:")[-1].strip()
    m = re.search(r"\b(R\d{5})\b", s, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()
    if re.fullmatch(r"R\d{5}", s, flags=re.IGNORECASE):
        return s.upper()
    return ""


class KEGGReactionFeatures:
    """Encapsulates KEGG reaction feature data and operations."""

    def __init__(self, features_dict: Dict):
        self._features = features_dict
        self._cache: Dict[str, str] = {}

    def get_participants(self, annotation: str) -> str:
        key = ("participants", annotation)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        kegg_id = _normalize_kegg_reaction_id(annotation)
        if not kegg_id:
            self._cache[key] = ""
            return ""
        definition = self._features.get(kegg_id, {}).get("DEFINITION", "")
        result = extract_classifications(definition, "definition")
        self._cache[key] = result
        return result

    def get_participant_ids(self, annotation: str) -> str:
        key = ("participant_ids", annotation)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        kegg_id = _normalize_kegg_reaction_id(annotation)
        if not kegg_id:
            self._cache[key] = ""
            return ""
        definition = self._features.get(kegg_id, {}).get("EQUATION", "")
        result = extract_classifications(definition, "definition")
        self._cache[key] = result
        return result

    def get_definition(self, annotation: str) -> str:
        """KEGG ``DEFINITION`` (human-readable reaction string) for the reaction."""
        key = ("definition", annotation)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        kegg_id = _normalize_kegg_reaction_id(annotation)
        if not kegg_id:
            self._cache[key] = ""
            return ""
        raw = (self._features.get(kegg_id, {}) or {}).get("DEFINITION", "") or ""
        result = " ".join(ln.strip() for ln in str(raw).splitlines() if ln.strip())
        self._cache[key] = result
        return result

    def get_iubmb_chains(self, annotation: str) -> Tuple[Tuple[str, ...], ...]:
        """Per-EC ancestor chains for a reaction from its IUBMB BRITE hierarchy.

        Parses the ``IUBMB reaction hierarchy [BR:br08202]`` block of the
        reaction's BRITE field. Within that block KEGG groups R-numbers by EC
        sub-section (e.g. ``2.7.1.1`` followed by an ordered list of R-numbers
        that walks the IUBMB tree from root to self).

        Returns a tuple of chains; each chain is an ordered tuple of bare
        KEGG reaction ids. The same reaction may appear in multiple chains
        (one per EC under which it's classified). Reactions without a
        ``br08202`` block return an empty tuple.

        Examples (against KEGG snapshot bundled in ``data/kegg/``)::

            R02848 -> ((R02848,),)
            R00299 -> ((R02848, R00299), (R00299,))
            R01786 -> ((R02848, R01786), (R00299, R01786))
            R01068 -> ((R01068,),)
            R01070 -> ((R01068, R01070),)
            R10049 -> ()
        """
        key = ("iubmb_chains", annotation)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        kegg_id = _normalize_kegg_reaction_id(annotation)
        if not kegg_id:
            self._cache[key] = ()
            return ()

        brite = (self._features.get(kegg_id, {}) or {}).get("BRITE", "") or ""
        if not brite:
            self._cache[key] = ()
            return ()

        in_iubmb = False
        chains: List[Tuple[str, ...]] = []
        current: List[str] = []

        def _flush() -> None:
            if current:
                chains.append(tuple(current))
                current.clear()

        for raw_line in str(brite).splitlines():
            line = raw_line.strip()
            if _BRITE_SECTION_RE.search(raw_line):
                # Entering a new section ends the current chain regardless of
                # whether we're leaving or staying inside br08202.
                _flush()
                in_iubmb = _IUBMB_TAG in raw_line.lower()
                continue
            if not in_iubmb or not line:
                continue
            # EC / hierarchy headers like "4.1.2.13" or "2. Transferase reactions"
            # delimit chains within the IUBMB block.
            if line[0].isdigit():
                _flush()
                continue
            r_ids = _R_NUMBER_RE.findall(raw_line)
            if r_ids:
                # Each KEGG line lists at most one R-number; take the first.
                current.append(r_ids[0])

        _flush()
        result: Tuple[Tuple[str, ...], ...] = tuple(chains)
        self._cache[key] = result
        return result

    def get_iubmb_ancestors(self, annotation: str) -> FrozenSet[str]:
        """Set of R-numbers above this reaction in its IUBMB BRITE hierarchy.

        Equal to the union of every chain returned by
        :meth:`get_iubmb_chains` minus the reaction's own id.
        """
        key = ("iubmb_ancestors", annotation)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        kegg_id = _normalize_kegg_reaction_id(annotation)
        if not kegg_id:
            self._cache[key] = frozenset()
            return frozenset()
        ancestors: set = set()
        for chain in self.get_iubmb_chains(annotation):
            ancestors.update(chain)
        ancestors.discard(kegg_id)
        result = frozenset(ancestors)
        self._cache[key] = result
        return result

    @classmethod
    def load_from_file(cls, data_path: str) -> "KEGGReactionFeatures":
        # Allow callers to pass a bare filename (historical default). If the file
        # isn't found relative to the current working directory, try common
        # project-relative locations (e.g. `data/kegg/`).
        candidate_paths: list[Path] = []
        if data_path:
            p = Path(data_path)
            candidate_paths.append(p)
            # If a bare filename (or relative path that doesn't exist), try
            # resolving under the repository's `data/kegg/` folder.
            repo_root = Path(__file__).resolve().parents[2]
            candidate_paths.append(repo_root / "data" / "kegg" / p.name)
            # Also allow callers that pass "kegg/<file>" (used by `data/load_data.py`)
            candidate_paths.append(repo_root / "data" / p)

        resolved_path = next((p for p in candidate_paths if p.exists()), Path(data_path))
        try:
            with lzma.open(resolved_path, "rb") as f:
                features_dict = pickle.load(f)
            logger.info("Loaded KEGG reaction features from %s", resolved_path)
            return cls(features_dict)
        except (FileNotFoundError, lzma.LZMAError) as e:
            if candidate_paths:
                logger.error(
                    "Error loading KEGG reaction features: %s (tried: %s)",
                    e,
                    ", ".join(str(p) for p in candidate_paths),
                )
            else:
                logger.error("Error loading KEGG reaction features: %s", e)
            return cls({})
