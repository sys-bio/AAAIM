"""Helpers for bare KEGG compound identifiers (``C#####``) in species annotations.

Some curated models annotate species with KEGG compound IDs instead of ChEBI.
Downstream code historically assumed ``annotation`` was always ChEBI; these
helpers detect when a string is already a KEGG compound id so it can be passed
through without ChEBI→KEGG table lookup or ontology walking.
"""

from __future__ import annotations

import re
from typing import Optional

# KEGG compound ids are conventionally ``C`` + five digits (e.g. ``C00002``).
_KEGG_COMPOUND_ID_RE = re.compile(r"^C\d{5}$", re.IGNORECASE)


def parse_kegg_compound_id(annotation: object) -> Optional[str]:
    """Return canonical ``C#####`` if ``annotation`` is a plain KEGG compound id.

    Accepts strings like ``c00002`` or ``C00002``. Returns ``None`` for ChEBI
    terms, free text, or empty values.
    """
    if annotation is None:
        return None
    s = str(annotation).strip()
    if not s:
        return None
    if _KEGG_COMPOUND_ID_RE.fullmatch(s):
        return s.upper()
    return None
