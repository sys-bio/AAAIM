"""Run the selected BioModels use cases with the manuscript configuration."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.annotation_workflow import annotate_model


HERE = Path(__file__).resolve().parent
OUTPUT_ROOT = HERE / "results"
LLM_MODEL = "gpt-5-mini-2025-08-07"
RETRIEVAL_TOP_K = 3
FINAL_N_RETURN = 1

MODEL_CONFIGS = {
    "MODEL2503190002": {
        "entity_type": "auto",
        "database": ["chebi", "uniprot"],
        "tax_id": "9606",
        "max_completion_tokens": 10_000,
    },
    "MODEL2506050001": {
        "entity_type": "gene",
        "database": ["ncbigene"],
        "tax_id": ["9606", "10090", "10116"],
        "max_completion_tokens": 10_000,
    },
    "MODEL2507280001": {
        "entity_type": "chemical",
        "database": ["chebi"],
        "tax_id": None,
        "max_completion_tokens": 10_000,
    },
}


def _json_safe(value):
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def run(model_id: str, condition: str, *, validation: bool = False) -> None:
    config = MODEL_CONFIGS[model_id]
    output_dir = OUTPUT_ROOT / model_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_name = f"{condition}_validation" if validation else condition
    output_prefix = output_dir / output_name
    max_entities = 5 if condition == "smoke" else None

    result = annotate_model(
        model_file=str(HERE / f"{model_id}.xml"),
        llm_model=LLM_MODEL,
        method="direct",
        top_k=RETRIEVAL_TOP_K,
        n_return=FINAL_N_RETURN,
        max_entities=max_entities,
        entity_type=config["entity_type"],
        database=config["database"],
        tax_id=config["tax_id"],
        chunk_size=50,
        annotate="species",
        save_to=str(output_prefix),
        verbose=True,
        max_completion_tokens=(
            min(4_000, config["max_completion_tokens"])
            if condition == "smoke"
            else config["max_completion_tokens"]
        ),
        validation=validation,
    )
    metadata = {
        "model": model_id,
        "condition": output_name,
        "configuration": {
            "llm_model": LLM_MODEL,
            "method": "direct",
            "top_k": RETRIEVAL_TOP_K,
            "n_return": FINAL_N_RETURN,
            "max_entities": max_entities,
            "entity_type": config["entity_type"],
            "database": config["database"],
            "tax_id": config["tax_id"],
            "chunk_size": 50,
            "validation": validation,
        },
        "metrics": result.metrics,
    }
    metrics_path = Path(f"{output_prefix}_metrics.json")
    metrics_path.write_text(
        json.dumps(_json_safe(metadata), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Saved metrics to {metrics_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_id", choices=MODEL_CONFIGS)
    parser.add_argument("condition", choices=("smoke", "full"))
    parser.add_argument(
        "--validation",
        action="store_true",
        help="Enable structured output and deterministic validation guards.",
    )
    args = parser.parse_args()
    run(args.model_id, args.condition, validation=args.validation)


if __name__ == "__main__":
    main()
