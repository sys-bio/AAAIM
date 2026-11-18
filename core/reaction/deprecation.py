"""Reaction method default, grounded resource guidance and the rule-based deprecation notice."""
from __future__ import annotations

import os
import warnings

DEFAULT_REACTION_METHOD = "rrf_grounded"
REACTION_METHODS = ("rrf_grounded", "rulebased")

# Warnings are attributed to the first caller frame outside the ``core`` package,
# so nested public wrappers all point at user code.
_CORE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + os.sep

RESOURCE_HELP = (
    "Install the verified encoder archive with "
    "core.reaction.retrieval.install_resources(archive, encoder_dir) and pass "
    "reaction_resources=RetrievalResources(catalog_path='data/kegg/kegg_reaction_features.lzma', "
    "encoder_dir=encoder_dir); see docs/rrf_grounded_api.md. Rule-based matching is never "
    "used as a fallback; pass reaction_method='rulebased' to opt in to the deprecated legacy method."
)


def rulebased_deprecation_message(via_method_alias: bool = False) -> str:
    alias = (
        " method='rulebased' is a legacy alias for reaction_method='rulebased'; species "
        "retrieval is selected separately with method='direct' or 'rag'."
        if via_method_alias else ""
    )
    return (
        "Rule-based KEGG reaction annotation (reaction_method='rulebased') is deprecated "
        "and is no longer the default reaction method." + alias +
        " Omit reaction_method, or pass reaction_method='rrf_grounded', to use grounded "
        "full-catalog RRF retrieval. It requires explicit local resources: "
        "reaction_resources=RetrievalResources(catalog_path=..., encoder_dir=...) after "
        "installing the verified encoder with core.reaction.retrieval.install_resources; "
        "see docs/rrf_grounded_api.md. The rule-based path remains available on explicit request."
    )


def warn_rulebased_deprecated(via_method_alias: bool = False) -> None:
    warnings.warn(
        rulebased_deprecation_message(via_method_alias),
        DeprecationWarning,
        skip_file_prefixes=(_CORE_DIR,),
    )
