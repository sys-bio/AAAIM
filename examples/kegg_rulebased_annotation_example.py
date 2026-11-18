#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Legacy rule-based KEGG reaction annotation example.

Rule-based reaction annotation is deprecated and is no longer the default; it
is selected here explicitly with ``reaction_method="rulebased"`` and emits a
``DeprecationWarning``. The default grounded method is shown in
``examples/grounded_reactions.py``.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.append(str(Path(__file__).parent.parent))
from core import annotate_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

load_dotenv()

model_file = "tests/test_models/glycolysis_part1.xml"
species_csv = "./examples/glycolysis_part1-recommendations.csv"
llm_model = "gpt-4o-mini"
TOP_K = 10


def main() -> None:
    logger.info("AAAIM KEGG Reaction Annotation Example")

    # Reactions only, using a precomputed species recommendation CSV
    result = annotate_model(
        model_file=model_file,
        llm_model=llm_model,
        annotate="reactions",
        reaction_method="rulebased",
        n_return=TOP_K,
        species_recommendations_df=species_csv,
    )

    # Combined species + reactions (uncomment to run):
    # result = annotate_model(
    #     model_file=model_file,
    #     llm_model=llm_model,
    #     annotate="both",
    #     reaction_method="rulebased",
    #     entity_type="chemical",
    #     database="chebi",
    #     top_k=TOP_K,
    #     n_return=3,
    # )
    # print(result.species_recommendations_df.head())
    # print(result.reaction_recommendations_df.head())

    df = result.reaction_recommendations_df
    if df.empty:
        print("No reaction recommendations generated.")
        if "error" in result.metrics:
            print(f"Error: {result.metrics['error']}")
        return df

    print("Annotation Results:")
    print(f"Total entities in model: {result.metrics['total_entities']}")
    print(f"Entities with predictions: {result.metrics['entities_with_predictions']}")
    print(f"Annotation rate: {result.metrics['annotation_rate']:.1%}")
    if result.metrics.get("accuracy") == result.metrics.get("accuracy"):
        print(f"Accuracy: {result.metrics['accuracy']:.1%}")
    else:
        print("Accuracy: N/A (no existing annotations to compare against)")
    print(f"Total time: {result.metrics['total_time']:.2f}s")
    return df


if __name__ == "__main__":
    main()
