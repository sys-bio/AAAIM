"""Regression tests for the Phase-1 reaction-annotation benchmark builder.

These lock down the behaviours that silently broke the historical evaluation:

1. KEGG reaction URIs appear in both ``kegg.reaction:R00024`` (colon) and
   ``kegg.reaction/R00024`` (slash) forms. Only matching one form yields zero
   ground truth from real BioModels files.
2. A reaction may carry several valid KEGG identifiers; all must be preserved,
   not just the first encountered.
3. Source/sink/exchange (SSX) reactions must be retained as records and marked
   excluded, so reaction counts drop while model counts do not.
4. Repeated builds over unchanged inputs must produce byte-identical tables.

Run with::

    python -m pytest tests/test_benchmark_build.py -v
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.model_info import find_reactions_with_kegg_annotations  # noqa: E402


def _load_build_module():
    """Import the builder by path (``benchmark/`` is not a Python package)."""
    script = REPO_ROOT / "benchmark" / "scripts" / "build_benchmark.py"
    spec = importlib.util.spec_from_file_location("build_benchmark", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


build_benchmark = _load_build_module()


# ---------------------------------------------------------------------------
# Synthetic SBML construction
# ---------------------------------------------------------------------------

def _rdf_annotation(meta_id: str, uris: List[str]) -> str:
    lis = "\n".join(f'<rdf:li rdf:resource="{u}"/>' for u in uris)
    return f"""<annotation>
  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
           xmlns:bqbiol="http://biomodels.net/biology-qualifiers/">
    <rdf:Description rdf:about="#{meta_id}">
      <bqbiol:is>
        <rdf:Bag>
{lis}
        </rdf:Bag>
      </bqbiol:is>
    </rdf:Description>
  </rdf:RDF>
</annotation>"""


def _reaction_xml(
    rxn_id: str,
    reactants: List[str],
    products: List[str],
    kegg_uris: List[str],
) -> str:
    meta = f"meta_{rxn_id}"
    ann = _rdf_annotation(meta, kegg_uris) if kegg_uris else ""
    r_block = "".join(
        f'<speciesReference species="{s}" stoichiometry="1"/>' for s in reactants
    )
    p_block = "".join(
        f'<speciesReference species="{s}" stoichiometry="1"/>' for s in products
    )
    reactant_list = f"<listOfReactants>{r_block}</listOfReactants>" if reactants else ""
    product_list = f"<listOfProducts>{p_block}</listOfProducts>" if products else ""
    return f"""<reaction id="{rxn_id}" metaid="{meta}" reversible="false">
      {ann}
      {reactant_list}
      {product_list}
    </reaction>"""


def write_model(
    path: Path,
    reactions: List[Dict[str, object]],
    species: Optional[List[str]] = None,
    model_name: str = "SyntheticTestModel",
) -> Path:
    """Write a minimal but valid SBML L2V4 model with the given reactions."""
    species = species or sorted(
        {
            s
            for r in reactions
            for s in list(r.get("reactants", [])) + list(r.get("products", []))
        }
    )
    species_xml = "".join(
        f'<species id="{s}" compartment="c" initialConcentration="1"/>' for s in species
    )
    reactions_xml = "\n".join(
        _reaction_xml(
            str(r["id"]),
            list(r.get("reactants", [])),
            list(r.get("products", [])),
            list(r.get("kegg_uris", [])),
        )
        for r in reactions
    )
    sbml = f"""<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level2/version4" level="2" version="4">
  <model id="synthetic" name="{model_name}">
    <listOfCompartments>
      <compartment id="c" size="1"/>
    </listOfCompartments>
    <listOfSpecies>{species_xml}</listOfSpecies>
    <listOfReactions>
{reactions_xml}
    </listOfReactions>
  </model>
</sbml>
"""
    path.write_text(sbml, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# 1. URI form regression
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "uri,expected",
    [
        ("http://identifiers.org/kegg.reaction/R00024", "R00024"),
        ("https://identifiers.org/kegg.reaction/R00024", "R00024"),
        ("http://identifiers.org/kegg.reaction:R00024", "R00024"),
        ("https://identifiers.org/kegg.reaction:R00024", "R00024"),
        ("urn:miriam:kegg.reaction:R00024", "R00024"),
    ],
)
def test_kegg_uri_forms_are_all_extracted(tmp_path, uri, expected):
    """Slash, colon, and URN forms must all yield ground truth.

    BioModels overwhelmingly uses the slash form; a colon-only pattern silently
    produced zero ground-truth reactions.
    """
    model = write_model(
        tmp_path / "m.xml",
        [{"id": "R1", "reactants": ["A"], "products": ["B"], "kegg_uris": [uri]}],
    )
    ground_truth, _ = find_reactions_with_kegg_annotations(str(model))
    assert "R1" in ground_truth, f"no ground truth extracted from URI form: {uri}"
    assert ground_truth["R1"] == [expected]


def test_mixed_uri_forms_in_one_model(tmp_path):
    model = write_model(
        tmp_path / "m.xml",
        [
            {
                "id": "R_slash",
                "reactants": ["A"],
                "products": ["B"],
                "kegg_uris": ["http://identifiers.org/kegg.reaction/R00024"],
            },
            {
                "id": "R_colon",
                "reactants": ["B"],
                "products": ["C"],
                "kegg_uris": ["http://identifiers.org/kegg.reaction:R01512"],
            },
        ],
    )
    ground_truth, _ = find_reactions_with_kegg_annotations(str(model))
    assert ground_truth == {"R_slash": ["R00024"], "R_colon": ["R01512"]}


# ---------------------------------------------------------------------------
# 2. Multiple ground-truth IDs
# ---------------------------------------------------------------------------

def test_multiple_kegg_ids_on_one_reaction_are_all_preserved(tmp_path):
    """All valid IDs are kept; the historical code kept only the first."""
    model = write_model(
        tmp_path / "m.xml",
        [
            {
                "id": "R1",
                "reactants": ["A"],
                "products": ["B"],
                "kegg_uris": [
                    "http://identifiers.org/kegg.reaction/R00024",
                    "http://identifiers.org/kegg.reaction/R01512",
                    "http://identifiers.org/kegg.reaction/R01063",
                ],
            }
        ],
    )
    ground_truth, _ = find_reactions_with_kegg_annotations(str(model))
    assert sorted(ground_truth["R1"]) == ["R00024", "R01063", "R01512"]

    res = build_benchmark.process_model(
        "SYN0001", model, {}, with_candidates=False, work_dir=tmp_path / "w"
    )
    row = res.reaction_rows[0]
    assert row["num_ground_truth_ids"] == 3
    assert row["ground_truth_kegg_all"] == "R00024;R01063;R01512"
    assert row["ground_truth_kegg_primary"] == "R00024"
    assert row["included_in_eval"] is True


def test_invalid_kegg_id_is_excluded_at_reaction_level(tmp_path):
    """Malformed identifiers are excluded as data problems, with the raw value kept."""
    model = write_model(
        tmp_path / "m.xml",
        [
            {
                "id": "R_bad",
                "reactants": ["A"],
                "products": ["B"],
                "kegg_uris": ["http://identifiers.org/kegg.reaction/R123456789"],
            }
        ],
    )
    res = build_benchmark.process_model(
        "SYN0002", model, {}, with_candidates=False, work_dir=tmp_path / "w"
    )
    row = res.reaction_rows[0]
    assert row["included_in_eval"] is False
    assert row["exclusion_reason"] == "invalid_ground_truth_id"
    assert "R123456789" in row["invalid_ground_truth_ids"]


# ---------------------------------------------------------------------------
# 3. SSX handling
# ---------------------------------------------------------------------------

def test_ssx_reactions_retained_but_marked_excluded(tmp_path):
    """SSX reactions stay as records; they reduce reaction counts, not model counts."""
    model = write_model(
        tmp_path / "m.xml",
        [
            {
                "id": "R_normal",
                "reactants": ["A"],
                "products": ["B"],
                "kegg_uris": ["http://identifiers.org/kegg.reaction/R00024"],
            },
            {
                "id": "R_sink",
                "reactants": ["B"],
                "products": [],
                "kegg_uris": ["http://identifiers.org/kegg.reaction/R01512"],
            },
            {
                "id": "R_source",
                "reactants": [],
                "products": ["A"],
                "kegg_uris": ["http://identifiers.org/kegg.reaction/R01063"],
            },
        ],
    )
    res = build_benchmark.process_model(
        "SYN0003", model, {}, with_candidates=False, work_dir=tmp_path / "w"
    )

    # The model itself is still included.
    assert res.status == "included"

    by_id = {r["reaction_id"]: r for r in res.reaction_rows}
    assert len(by_id) == 3, "SSX reactions must be retained as records, not dropped"

    assert by_id["R_normal"]["included_in_eval"] is True
    assert by_id["R_normal"]["is_exchange_ssx"] is False

    for ssx_id in ("R_sink", "R_source"):
        assert by_id[ssx_id]["is_exchange_ssx"] is True
        assert by_id[ssx_id]["included_in_eval"] is False
        assert by_id[ssx_id]["exclusion_reason"] == "exchange_ssx"

    assert res.summary["num_ground_truth_reactions"] == 3
    assert res.summary["num_eval_reactions"] == 1
    assert res.summary["num_ssx_excluded"] == 2

    reasons = {(e["reaction_id"], e["reason"]) for e in res.exclusions}
    assert ("R_sink", "exchange_ssx") in reasons
    assert ("R_source", "exchange_ssx") in reasons


def test_model_with_only_ssx_reactions_stays_included(tmp_path):
    """All-SSX models contribute zero evaluable reactions but are not dropped."""
    model = write_model(
        tmp_path / "m.xml",
        [
            {
                "id": "R_sink",
                "reactants": ["A"],
                "products": [],
                "kegg_uris": ["http://identifiers.org/kegg.reaction/R00024"],
            }
        ],
    )
    res = build_benchmark.process_model(
        "SYN0004", model, {}, with_candidates=False, work_dir=tmp_path / "w"
    )
    assert res.status == "included"
    assert res.summary["num_eval_reactions"] == 0
    assert res.summary["zero_eval_reactions"] is True


# ---------------------------------------------------------------------------
# 4. Model-level exclusion vs pipeline failure
# ---------------------------------------------------------------------------

def test_model_without_kegg_annotations_is_scientific_exclusion(tmp_path):
    model = write_model(
        tmp_path / "m.xml",
        [{"id": "R1", "reactants": ["A"], "products": ["B"], "kegg_uris": []}],
    )
    res = build_benchmark.process_model(
        "SYN0005", model, {}, with_candidates=False, work_dir=tmp_path / "w"
    )
    assert res.status == "excluded_no_ground_truth"
    assert res.exclusions[0]["reason"] == "no_kegg_reaction_annotations"
    assert not res.pipeline_failures


def test_missing_file_is_pipeline_failure_not_exclusion(tmp_path):
    res = build_benchmark.process_model(
        "SYN0006",
        tmp_path / "does_not_exist.xml",
        {},
        with_candidates=False,
        work_dir=tmp_path / "w",
    )
    assert res.status == "pipeline_failure"
    assert res.pipeline_failures[0]["failure_type"] == "file_missing"
    assert res.exclusions == [], "a missing download is not a scientific exclusion"


def test_unparseable_file_is_pipeline_failure(tmp_path):
    bad = tmp_path / "bad.xml"
    bad.write_text("this is not SBML at all", encoding="utf-8")
    res = build_benchmark.process_model(
        "SYN0007", bad, {}, with_candidates=False, work_dir=tmp_path / "w"
    )
    assert res.status == "pipeline_failure"
    assert res.pipeline_failures[0]["failure_type"] == "parse_error"
    assert res.exclusions == []


def test_parser_discrepancy_is_flagged(tmp_path):
    """A raw kegg.reaction mention we cannot extract must be surfaced, not ignored."""
    model = write_model(
        tmp_path / "m.xml",
        [
            {
                "id": "R1",
                "reactants": ["A"],
                "products": ["B"],
                # Deliberately malformed URI: mentions kegg.reaction, no extractable ID.
                "kegg_uris": ["http://identifiers.org/kegg.reaction/NOT_AN_ID"],
            }
        ],
    )
    res = build_benchmark.process_model(
        "SYN0008", model, {}, with_candidates=False, work_dir=tmp_path / "w"
    )
    assert res.diagnostics is not None
    assert res.diagnostics["reactions_with_raw_kegg_mention"] == 1
    assert res.diagnostics["reactions_with_extracted_kegg"] == 0
    assert res.diagnostics["parser_discrepancy"] is True


# ---------------------------------------------------------------------------
# 5. Determinism
# ---------------------------------------------------------------------------

def _run_build(models_dir: Path, out_dir: Path, manifest: Path) -> Dict[str, str]:
    argv = [
        "build_benchmark.py",
        "--manifest", str(manifest),
        "--models-dir", str(models_dir),
        "--registry", str(models_dir / "missing_registry.json"),
        "--output-dir", str(out_dir),
    ]
    old = sys.argv
    try:
        sys.argv = argv
        build_benchmark.main()
    finally:
        sys.argv = old

    hashes = {}
    for name in ("reactions.csv", "model_summary.csv", "exclusions.csv", "model_clusters.csv"):
        p = out_dir / name
        if p.exists():
            hashes[name] = hashlib.sha256(p.read_bytes()).hexdigest()
    return hashes


def test_repeated_builds_are_byte_identical(tmp_path):
    models_dir = tmp_path / "models"
    models_dir.mkdir()

    write_model(
        models_dir / "SYN0100.xml",
        [
            {
                "id": "R2",
                "reactants": ["B"],
                "products": ["C"],
                "kegg_uris": ["http://identifiers.org/kegg.reaction/R01512"],
            },
            {
                "id": "R1",
                "reactants": ["A"],
                "products": ["B"],
                "kegg_uris": [
                    "http://identifiers.org/kegg.reaction/R00024",
                    "http://identifiers.org/kegg.reaction:R01063",
                ],
            },
            {
                "id": "R_sink",
                "reactants": ["C"],
                "products": [],
                "kegg_uris": ["http://identifiers.org/kegg.reaction/R01070"],
            },
        ],
    )
    write_model(
        models_dir / "SYN0101.xml",
        [{"id": "R1", "reactants": ["A"], "products": ["B"], "kegg_uris": []}],
    )

    manifest = tmp_path / "manifest.txt"
    manifest.write_text("# synthetic\nSYN0100\nSYN0101\nSYN0102\n", encoding="utf-8")

    first = _run_build(models_dir, tmp_path / "out1", manifest)
    second = _run_build(models_dir, tmp_path / "out2", manifest)

    assert first, "build produced no artifacts"
    assert first == second, f"non-deterministic build:\n{first}\n{second}"


def test_artifacts_use_lf_and_digests_match_bytes(tmp_path):
    """Artifacts must be LF-terminated and VERSION.json digests must match.

    ``Path.write_text`` silently converts ``\\n`` to ``\\r\\n`` on Windows, which
    made the recorded digests platform-dependent and stopped them matching the
    LF-normalised blobs git stores.
    """
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    write_model(
        models_dir / "SYN0300.xml",
        [
            {
                "id": "R1",
                "reactants": ["A"],
                "products": ["B"],
                "kegg_uris": ["http://identifiers.org/kegg.reaction/R00024"],
            }
        ],
    )
    manifest = tmp_path / "manifest.txt"
    manifest.write_text("SYN0300\n", encoding="utf-8")

    out = tmp_path / "out"
    _run_build(models_dir, out, manifest)

    version = json.loads((out / "VERSION.json").read_text(encoding="utf-8"))
    for name, expected in version["artifact_sha256"].items():
        raw = (out / name).read_bytes()
        assert b"\r\n" not in raw, f"{name} contains CRLF; digests would not survive checkout"
        actual = hashlib.sha256(raw).hexdigest()
        assert actual == expected, f"{name} digest does not match its bytes"


def test_build_reconciles_and_separates_failures(tmp_path):
    """Included + excluded + pipeline failures must equal the manifest exactly."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()

    write_model(
        models_dir / "SYN0200.xml",
        [
            {
                "id": "R1",
                "reactants": ["A"],
                "products": ["B"],
                "kegg_uris": ["http://identifiers.org/kegg.reaction/R00024"],
            },
            {
                "id": "R_sink",
                "reactants": ["B"],
                "products": [],
                "kegg_uris": ["http://identifiers.org/kegg.reaction/R01512"],
            },
        ],
    )
    write_model(
        models_dir / "SYN0201.xml",
        [{"id": "R1", "reactants": ["A"], "products": ["B"], "kegg_uris": []}],
    )
    (models_dir / "SYN0202.xml").write_text("not sbml", encoding="utf-8")

    manifest = tmp_path / "manifest.txt"
    manifest.write_text("SYN0200\nSYN0201\nSYN0202\nSYN0203\n", encoding="utf-8")

    out = tmp_path / "out"
    _run_build(models_dir, out, manifest)

    summary = json.loads((out / "benchmark_summary.json").read_text(encoding="utf-8"))
    assert summary["manifest_models"] == 4
    assert summary["models_included"] == 1
    assert summary["models_excluded_no_ground_truth"] == 1
    assert summary["models_pipeline_failure"] == 2  # unparseable + missing
    assert summary["total_ground_truth_reactions"] == 2
    assert summary["evaluable_reactions"] == 1
    assert summary["reactions_excluded_ssx"] == 1

    invariants = json.loads((out / "invariants.json").read_text(encoding="utf-8"))
    by_name = {c["check"]: c for c in invariants["checks"]}
    assert by_name["model_records_reconcile"]["passed"] is True
    assert by_name["reaction_records_reconcile"]["passed"] is True

    version = json.loads((out / "VERSION.json").read_text(encoding="utf-8"))
    assert "reactions.csv" in version["artifact_sha256"]


def test_cluster_ids_are_stable_and_group_related_models():
    shared = {f"R{i:05d}" for i in range(1, 21)}
    gt_sets = {
        "BIOMD0000000472": set(shared),
        "BIOMD0000000471": set(shared) | {"R99999"},
        "BIOMD0000000013": {"R00100", "R00101"},
    }
    cluster_of, groups = build_benchmark.cluster_models(gt_sets, 0.9)

    # The two near-identical variants share a cluster; the unrelated model does not.
    assert cluster_of["BIOMD0000000471"] == cluster_of["BIOMD0000000472"]
    assert cluster_of["BIOMD0000000013"] != cluster_of["BIOMD0000000471"]

    # Cluster ID derives from the smallest member accession, so it is stable.
    assert cluster_of["BIOMD0000000471"] == "CLU_BIOMD0000000471"

    # Every model receives a cluster ID, singletons included.
    assert set(cluster_of) == set(gt_sets)
    assert len(groups) == 1
    assert groups.iloc[0]["group_size"] == 2

    # Re-running yields identical assignments.
    again, _ = build_benchmark.cluster_models(gt_sets, 0.9)
    assert again == cluster_of


def test_containment_linkage_catches_extended_variants():
    """A model that extends another must share its cluster.

    Mirrors the real Smallbone2013 yeast variants: 189 shared reactions, but
    one model adds 23 more, giving Jaccard 0.86 — under the 0.9 threshold —
    while containment is 0.96. Without containment linkage these related
    models would land in different train/test partitions.
    """
    base = {f"R{i:05d}" for i in range(1, 198)}
    extended = set(list(base)[:189]) | {f"R9{i:04d}" for i in range(23)}
    gt_sets = {"BIOMD0000000471": base, "BIOMD0000000472": set(base), "BIOMD0000000473": extended}

    shared = len(base & extended)
    jaccard = shared / len(base | extended)
    containment = shared / min(len(base), len(extended))
    assert jaccard < 0.9, "fixture should reproduce the sub-threshold Jaccard"
    assert containment >= 0.9

    cluster_of, groups = build_benchmark.cluster_models(gt_sets, 0.9, 0.9)
    assert (
        cluster_of["BIOMD0000000473"] == cluster_of["BIOMD0000000471"]
    ), "extended variant must not be split from its base model"
    assert len(groups) == 1
    assert groups.iloc[0]["group_size"] == 3
    assert "containment" in groups.iloc[0]["linkage_rules"]


def test_containment_linkage_ignores_trivially_contained_small_models():
    """A tiny model contained in a genome-scale one must not be merged.

    Containment is 1.0 here, but the models are unrelated in scope. Merging on
    containment alone chains unrelated models into one cluster through the large
    models, which would collapse the partitioning entirely.
    """
    big = {f"R{i:05d}" for i in range(1, 501)}
    tiny = {"R00001", "R00002"}
    cluster_of, groups = build_benchmark.cluster_models(
        {"BIOMD0000000001": big, "BIOMD0000000002": tiny}, 0.9, 0.9
    )
    assert cluster_of["BIOMD0000000002"] != cluster_of["BIOMD0000000001"]
    assert groups.empty


def test_clustering_does_not_collapse_the_corpus():
    """Guard against over-merging: a chain of tiny models must stay separate."""
    gt_sets = {"BIOMD00000000%02d" % i: {f"R1{i:04d}", "R00001"} for i in range(1, 11)}
    gt_sets["BIOMD0000009999"] = {f"R{i:05d}" for i in range(1, 2001)}
    cluster_of, _ = build_benchmark.cluster_models(gt_sets, 0.9, 0.9)
    assert len(set(cluster_of.values())) >= 10, "clustering collapsed unrelated models"


def test_containment_needs_meaningful_absolute_overlap():
    """One shared identifier is coincidence, not shared lineage."""
    a = {"R00001"}
    b = {"R00001", "R00002"}
    cluster_of, groups = build_benchmark.cluster_models(
        {"BIOMD0000000001": a, "BIOMD0000000002": b}, 0.9, 0.9
    )
    assert cluster_of["BIOMD0000000001"] != cluster_of["BIOMD0000000002"]
    assert groups.empty
