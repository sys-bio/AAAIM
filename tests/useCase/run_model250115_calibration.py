"""Reproducible SBML-only and Table-S3 calibration runs for MODEL2501150001."""

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
MODEL_FILE = HERE / "MODEL2501150001.xml"
TABLE_S3_FILE = HERE / "context" / "MODEL2501150001_TableS3.txt"
OUTPUT_DIR = HERE / "results" / "MODEL2501150001"
LLM_MODEL = "gpt-5-mini-2025-08-07"
RETRIEVAL_TOP_K = 3
FINAL_N_RETURN = 1


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


def run_condition(
    name: str,
    *,
    max_entities=None,
    message: str = "",
    token_cap: int,
    validation: bool = False,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_name = f"{name}_validation" if validation else name
    output_prefix = OUTPUT_DIR / output_name
    result = annotate_model(
        model_file=str(MODEL_FILE),
        llm_model=LLM_MODEL,
        method="direct",
        top_k=RETRIEVAL_TOP_K,
        n_return=FINAL_N_RETURN,
        max_entities=max_entities,
        entity_type="auto",
        database=["chebi", "uniprot"],
        tax_id="9606",
        chunk_size=50,
        annotate="species",
        save_to=str(output_prefix),
        verbose=True,
        message=message,
        max_completion_tokens=token_cap,
        validation=validation,
    )
    metadata = {
        "model": "MODEL2501150001",
        "condition": output_name,
        "configuration": {
            "llm_model": LLM_MODEL,
            "method": "direct",
            "top_k": RETRIEVAL_TOP_K,
            "n_return": FINAL_N_RETURN,
            "max_entities": max_entities,
            "entity_type": "auto",
            "database": ["chebi", "uniprot"],
            "tax_id": "9606",
            "chunk_size": 50,
            "max_completion_tokens": token_cap,
            "supplementary_context": str(TABLE_S3_FILE) if message else None,
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
    parser.add_argument(
        "condition",
        choices=("smoke", "sbml_only", "table_s3", "calibration"),
        help="Run one condition or both full calibration conditions.",
    )
    parser.add_argument(
        "--validation",
        action="store_true",
        help="Enable structured output and deterministic validation guards.",
    )
    args = parser.parse_args()

    if args.condition == "smoke":
        run_condition(
            "smoke", max_entities=5, token_cap=4_000, validation=args.validation
        )
        return
    if args.condition in ("sbml_only", "calibration"):
        run_condition("sbml_only", token_cap=12_000, validation=args.validation)
    if args.condition in ("table_s3", "calibration"):
        run_condition(
            "table_s3",
            message=TABLE_S3_FILE.read_text(encoding="utf-8"),
            token_cap=12_000,
            validation=args.validation,
        )


if __name__ == "__main__":
    main()
