#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""KEGG Reaction Annotation Example

Demonstrates the workflow for annotating reactions in SBML models using KEGG
references: ChEBI→KEGG mapping, rule-based reaction matching, initial
likelihoods, and iterative participant updates (see core.reaction.amendment).
"""

from __future__ import annotations

import logging
import sys

from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).parent.parent))
from core import annotate_model

from core.reaction.annotation_workflow import rank_kegg_annotations_with_llm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

load_dotenv()

model_file = "tests/test_models/glycolysis_part1.xml"
kegg_features_file = "data/kegg/kegg_reaction_features.lzma"
llm_model = "Llama-3.3-70B-Instruct"


# first annotate model using ChEBI to get a list of ChEBI recommendations
# In this example, a list of recommended ChEBI annotation is provided.
recommendations_df = pd.read_csv("./examples/glycolysis_part1-recommendations.csv")
TOP_K = 10

# If False (default), run generation-only (no scoring / participant updates).
# If True, run the full evaluation workflow (scoring + EM-like participant updates).
RUN_EVALUATION = False


def main() -> pd.DataFrame:
    logger.info("AAAIM KEGG Reaction Annotation Example")
    logger.info("=" * 50)

    _annotation_result, _metrics = annotate_model(
        model_file=model_file,
        llm_model=llm_model,
        method="rulebased",
        entity_type="reaction",
        database="kegg",
        top_k=TOP_K,
        species_recommendations_df=recommendations_df,
        evaluate_candidates=RUN_EVALUATION,
    )

    csv_path = Path(f"{Path(model_file).name}_recommendations.csv")
    result_df = pd.read_csv(csv_path)

    ranked_df = rank_kegg_annotations_with_llm(
        model_file=model_file,
        recommendations_df=result_df,
        llm_model=llm_model,
        kegg_features_file=kegg_features_file,
        top_k=TOP_K,
        csv_path=str(csv_path),
    )

    # Display annotation results
    if not ranked_df.empty:
        print("Annotation Results:")
        print(f"Total entities in model: {_metrics['total_entities']}")
        print(f"Entities with predictions: {_metrics['entities_with_predictions']}")
        print(f"Annotation rate: {_metrics['annotation_rate']:.1%}")
        
        if not pd.isna(_metrics['accuracy']):
            print(f"Accuracy (where existing annotations available): {_metrics['accuracy']:.1%}")
        else:
            print("Accuracy: N/A (no existing annotations to compare against)")
        
        print(f"Total time: {_metrics['total_time']:.2f}s")
        print()
        
    return ranked_df


if __name__ == "__main__":
    main()
