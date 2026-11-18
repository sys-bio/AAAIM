"""No-call public grounded reaction example on an arbitrary local SBML model.

python examples/grounded_reactions.py model.xml --encoder /local/selected-encoder
This example always uses dry_run=True, including with --both.
"""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import RetrievalResources, annotate_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--catalog", default="data/kegg/kegg_reaction_features.lzma")
    parser.add_argument("--species", help="Optional species recommendation CSV")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--both", action="store_true", help="Preview handoff only; dry-run skips species inference")
    parser.add_argument("--out", default="_rrf_grounded_work/example")
    args = parser.parse_args()
    result = annotate_model(args.model, annotate="both" if args.both else "reactions",
        reaction_method="rrf_grounded", llm_model="gpt-6.1-terra",
        reaction_resources=RetrievalResources(args.catalog, args.encoder, device=args.device),
        species_recommendations_df=args.species, reaction_reasoning_effort="low",
        max_completion_tokens=2048, dry_run=True, save_to=args.out)
    print(result.reaction_recommendations_df[["id", "status", "annotation"]].to_string(index=False))
    print(result.metrics["coverage"])


if __name__ == "__main__":
    main()
