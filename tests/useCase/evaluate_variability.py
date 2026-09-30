"""Compare three independent source-fidelity runs of the variable use cases."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from evaluate_biomodel_usecases import (
    HERE, RESULTS, _model250115_identity, _prediction_key,
    evaluate_model2501150001, evaluate_model2503190002,
    evaluate_model2506050001,
)


CONDITIONS = (
    ("MODEL2501150001", "sbml_only"),
    ("MODEL2501150001", "table_s3"),
    ("MODEL2503190002", "full"),
    ("MODEL2506050001", "full"),
)
REPEATS = (1, 2, 3)


def _suffix(repeat: int) -> str:
    return "_source_fidelity" if repeat == 1 else f"_source_fidelity_repeat{repeat:02d}"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _identity_sets(df: pd.DataFrame, model_id: str) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for species_id, group in df.groupby("id", sort=False):
        values = set()
        for _, row in group[group["annotation"] != ""].iterrows():
            identity = _prediction_key(row)
            if model_id == "MODEL2501150001":
                identity = _model250115_identity(identity)
            values.add(identity)
        result[str(species_id)] = sorted(values)
    return result


def _accession_sets(df: pd.DataFrame) -> dict[str, list[str]]:
    return {
        str(species_id): sorted(set(group.loc[group["annotation"] != "", "annotation"]))
        for species_id, group in df.groupby("id", sort=False)
    }


def _unstable_by_species(per_run: list[dict[str, list[str]]]) -> list[str]:
    species_ids = set().union(*(set(run) for run in per_run))
    return sorted(
        sid for sid in species_ids
        if len({tuple(run.get(sid, [])) for run in per_run}) > 1
    )


def _normalization_signature(item: dict) -> str:
    components = [
        (str(component[0]).lower(), tuple(sorted(str(name).lower() for name in component[1])))
        for component in item.get("components", [])
    ]
    return json.dumps({
        "type": item.get("entity_type", ""),
        "names": sorted(str(name).lower() for name in item.get("names", [])),
        "components": sorted(components),
    }, sort_keys=True)


def _retrieval_sets(rows: list[dict]) -> dict[str, list[str]]:
    result: dict[str, set[str]] = {}
    for row in rows:
        sid = str(row.get("id", ""))
        annotation = str(row.get("annotation", ""))
        if sid and annotation:
            result.setdefault(sid, set()).add(annotation)
    return {sid: sorted(values) for sid, values in result.items()}


def _stage_comparison(model_id: str, condition: str) -> dict:
    traces = []
    for repeat in (2, 3):
        prefix = RESULTS / model_id / f"{condition}{_suffix(repeat)}"
        traces.append(json.loads(Path(f"{prefix}_trace.json").read_text()))
    normalized = [
        {sid: _normalization_signature(item) for sid, item in trace["normalization"].items()}
        for trace in traces
    ]
    retrieval = [_retrieval_sets(trace["retrieval_rows"]) for trace in traces]
    norm_ids = sorted(set(normalized[0]) | set(normalized[1]))
    return {
        "compared_repeats": [2, 3],
        "normalization_changed_species": [
            sid for sid in norm_ids if normalized[0].get(sid) != normalized[1].get(sid)
        ],
        "retrieval_pool_changed_species": _unstable_by_species(retrieval),
        "interpretation": (
            "Normalization and retrieved-pool comparisons use repeats 2 and 3 only; "
            "the saved first run predates optional trace capture."
        ),
    }


def _summary_for(model_id: str, condition: str, evaluation: dict) -> dict:
    if model_id == "MODEL2501150001":
        item = evaluation[condition]
        return {
            "candidate_species": item["species_with_candidates"],
            "expected_components_recovered": item["expected_identities_recovered"],
            "expected_components_total": item["expected_identities_total"],
            "species_with_all_expected": item["reference_species_complete"],
            "reference_species_total": item["reference_species_total"],
            "extra_component_predictions": len(item["wrong_predictions"]),
            "missing_components": len(item["missing_identities"]),
        }
    if model_id == "MODEL2503190002":
        return {
            "candidate_species": evaluation["species_with_candidates"],
            "exact_complete_species": evaluation["molecular_species_complete"],
            "reference_species_total": evaluation["molecular_species_total"],
            "extra_identities": len(evaluation["wrong_predictions"]),
            "missing_identities": len(evaluation["missing_identities"]),
        }
    return {
        "candidate_species": evaluation["species_with_candidates"],
        "exact_deposited_entrez_hits": evaluation["source_exact_entrez_hits"],
        "deposited_entrez_total": evaluation["source_reference_entrez_occurrences"],
        "supported_entrez_predictions": evaluation["reviewed_supported_predictions"],
        "predicted_entrez_on_mapped_nodes": evaluation["predicted_entrez_occurrences_on_reference_nodes"],
        "wrong_entrez_predictions": len(evaluation["wrong_predictions"]),
    }


def _error_frequencies(model_id: str, condition: str, evaluations: list[dict]) -> list[dict]:
    frequencies: Counter[tuple[str, str, str]] = Counter()
    for evaluation in evaluations:
        section = evaluation[condition] if model_id == "MODEL2501150001" else evaluation
        seen: set[tuple[str, str, str]] = set()
        for row in section.get("wrong_predictions", []):
            key = (
                str(row.get("id", "")),
                str(row.get("name", "")),
                str(row.get("predicted_identity", row.get("identity", ""))),
            )
            seen.add(key)
        for key in seen:
            frequencies[key] += 1
    return [
        {"species_id": sid, "name": name, "extra_or_wrong_identity": identity,
         "runs_observed": count}
        for (sid, name, identity), count in sorted(frequencies.items())
    ]


def _missing_frequencies(model_id: str, condition: str, evaluations: list[dict]) -> list[dict]:
    if model_id == "MODEL2506050001":
        return []  # Its reference unit is a taxon-specific Entrez occurrence.
    frequencies: Counter[tuple[str, str]] = Counter()
    for evaluation in evaluations:
        section = evaluation[condition] if model_id == "MODEL2501150001" else evaluation
        seen = {
            (str(row["id"]), str(row["missing_identity"]))
            for row in section["missing_identities"]
        }
        frequencies.update(seen)
    return [
        {"species_id": sid, "missing_identity": identity, "runs_observed": count}
        for (sid, identity), count in sorted(frequencies.items())
    ]


def main() -> None:
    evaluators = {
        "MODEL2501150001": evaluate_model2501150001,
        "MODEL2503190002": evaluate_model2503190002,
        "MODEL2506050001": evaluate_model2506050001,
    }
    for model_id, evaluator in evaluators.items():
        for repeat in (2, 3):
            evaluator(_suffix(repeat))

    conditions = []
    for model_id, condition in CONDITIONS:
        evaluations = []
        identity_sets = []
        accession_sets = []
        configs = []
        per_run = []
        for repeat in REPEATS:
            prefix = RESULTS / model_id / f"{condition}{_suffix(repeat)}"
            evaluation = json.loads(
                (RESULTS / model_id / f"evaluation{_suffix(repeat)}.json").read_text()
            )
            metadata = json.loads(Path(f"{prefix}_metrics.json").read_text())
            df = pd.read_csv(f"{prefix}_species.csv", dtype=str).fillna("")
            expected_species = metadata["metrics"]["total_entities"]
            trace_path = Path(f"{prefix}_trace.json")
            trace = json.loads(trace_path.read_text()) if trace_path.exists() else {}
            if repeat == 1:
                expected_ids = set(df["id"])
                if len(expected_ids) != expected_species:
                    raise ValueError(f"First run lacks rows without a trace: {prefix}")
            else:
                expected_ids = set(trace.get("normalization", {}))
                retrieved_ids = {row["id"] for row in trace.get("retrieval_rows", [])}
                if len(expected_ids) != expected_species or expected_ids != retrieved_ids:
                    raise ValueError(f"Incomplete normalization/retrieval repeat: {prefix}")
            omitted_ids = sorted(expected_ids - set(df["id"]))
            config = dict(metadata["configuration"])
            config.pop("repeat_id", None)
            configs.append(config)
            evaluations.append(evaluation)
            identity_sets.append(_identity_sets(df, model_id))
            accession_sets.append(_accession_sets(df))
            per_run.append({"repeat": repeat, **_summary_for(model_id, condition, evaluation),
                            "ranking_abstention_rows_omitted": omitted_ids})
        if configs[1:] != configs[:-1]:
            raise ValueError(f"Configuration mismatch for {model_id}/{condition}")
        conditions.append({
            "model_id": model_id,
            "condition": condition,
            "configuration": configs[0],
            "runs": per_run,
            "unstable_identity_species": _unstable_by_species(identity_sets),
            "unstable_accession_species": _unstable_by_species(accession_sets),
            "wrong_identity_frequencies": _error_frequencies(model_id, condition, evaluations),
            "missing_identity_frequencies": _missing_frequencies(model_id, condition, evaluations),
            "stage_comparison": _stage_comparison(model_id, condition),
        })

    result = {
        "design": "Three complete independent runs per condition; repeat 1 is the saved source-fidelity run.",
        "snapshot": "gpt-5-mini-2025-08-07",
        "input_sha256": {
            model_id: _sha256(HERE / f"{model_id}.xml")
            for model_id in sorted({item[0] for item in CONDITIONS})
        } | {"MODEL2501150001_TableS3": _sha256(HERE / "context" / "MODEL2501150001_TableS3.txt")},
        "conditions": conditions,
    }
    output = RESULTS / "variability_source_fidelity.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
