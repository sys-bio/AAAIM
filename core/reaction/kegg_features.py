"""Load and query KEGG reaction feature payloads (definitions, equations)."""

from __future__ import annotations

import logging
import lzma
import pickle
import re
from pathlib import Path
from typing import Dict, FrozenSet

from utils.constants import REF_KEGG_REACTION_FEATURES

from .kegg_definition import extract_classifications

logger = logging.getLogger(__name__)

_K_NUMBER_RE = re.compile(r"\bK\d{5,}\b")


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

    def get_orthology_k_numbers(self, annotation: str) -> FrozenSet[str]:
        """KEGG Orthology K-numbers parsed from the reaction's ORTHOLOGY field.

        Reactions sharing any K-number belong to the same KEGG-Orthology
        (BRITE) group, used for collapsing near-duplicate candidates before
        LLM ranking.
        """
        key = ("orthology_k_numbers", annotation)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        kegg_id = _normalize_kegg_reaction_id(annotation)
        if not kegg_id:
            self._cache[key] = frozenset()
            return frozenset()
        raw = (self._features.get(kegg_id, {}) or {}).get("ORTHOLOGY", "") or ""
        result = frozenset(_K_NUMBER_RE.findall(str(raw)))
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
