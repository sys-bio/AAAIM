# Reaction Annotation Benchmark (Phase 1)

Frozen, checksum-verified benchmark for evaluating KEGG reaction annotation under
imperfect metabolite identifiers.

Current version: **`phase1-v1`** — 74 models, 5,838 ground-truth reactions,
5,816 evaluable. See [`data/RECONCILIATION.md`](data/RECONCILIATION.md) for the
observed counts and why they differ from the historical 68 / 4,379 report.

## Layout

```
benchmark/
  manifest/
    models.txt            # 75 BioModels accessions (canonical manifest)
    model_paths.txt       # Repo-relative paths for evaluation scripts
    model_registry.json   # Generated: URLs, upstream + local SHA-256, revisions
  models/                 # Downloaded SBML (gitignored, reproducible)
  data/                   # Frozen benchmark tables (generated)
  scripts/
    download_biomodels.py
    build_benchmark.py
```

## Quick start

```bash
python benchmark/scripts/download_biomodels.py   # resolve, download, verify
python benchmark/scripts/build_benchmark.py      # build frozen tables
python -m pytest tests/test_benchmark_build.py   # regression tests
```

Add `--with-candidates` to attach rule-based candidates and heuristic scores.
This is slower and requires the KEGG reference data under `data/kegg/`.

## Provenance

Main SBML filenames are **not** predictable from the accession
(`BIOMD0000000013_url.xml` vs `iJN1463.xml`), so the downloader resolves each
model's file list from the BioModels API rather than guessing. Every file is
verified against the upstream SHA-256 published by the API; `model_registry.json`
records the download URL, upstream and local digests, SBML format version,
curation status, and the latest upstream revision.

## Pipeline failures are not scientific exclusions

This distinction is enforced in code and in the outputs.

**Pipeline failures** (`pipeline_failures.csv`) indicate a defect in our tooling
or in the snapshot. They must be resolved or explicitly justified, never folded
into exclusion statistics.

| Failure type | Meaning |
|---|---|
| `file_missing` | Download absent — a tooling failure, not a property of the data |
| `parse_error` | libSBML cannot parse the file; examined individually |
| `checksum_unverified` | Local bytes disagree with the upstream digest |
| `provenance_missing` | Registry has no entry for an accession |
| `registry_missing` | No registry at all, so nothing is provenance-verified |

**Scientific exclusions** (`exclusions.csv`) are properties of the data.

| Level | Reason | Effect |
|---|---|---|
| model | `no_kegg_reaction_annotations` | Model carries no reaction-level ground truth |
| reaction | `exchange_ssx` | Empty reactant or product side; record kept, marked not evaluable |
| reaction | `invalid_ground_truth_id` | Identifier fails `R#####` validation |

SSX exclusions reduce **reaction** counts, never model counts. A model whose
reactions are all SSX stays in the benchmark and contributes zero evaluable
reactions; that case is reported as `models_with_zero_eval_reactions`.

## Parser problems are surfaced, not absorbed

`parser_diagnostics.csv` compares, per model, the number of reactions whose raw
annotation mentions `kegg.reaction` against the number from which an identifier
was actually extracted. Any shortfall sets `parser_discrepancy` — the signature
of a URI-form or parser defect rather than a data property.

It also counts species carrying `kegg.reaction` annotations. The manifest was
assembled by grepping raw file bytes, so models are selected as "KEGG-annotated"
even when the only mentions are misplaced species annotations
(`misplaced_annotations_only`). `BIOMD0000000579` is exactly this case.

## Ground truth policy

**All** valid KEGG reaction IDs are preserved in `ground_truth_kegg_all`;
`ground_truth_kegg_primary` is the first for backward-compatible scoring. 91
reactions carry more than one valid ID, one of them 10. Scoring against a single
arbitrary ID systematically under-credits correct predictions.

## Train/test partitioning

Split on `cluster_id` from `model_clusters.csv`, never on individual reactions.
Clusters group near-duplicate model variants by ground-truth overlap (Jaccard, or
containment gated on comparable size and a minimum shared-identifier count) so
closely related models always land in the same partition.

Cluster IDs are `CLU_<smallest member accession>` and stay stable across rebuilds
while membership is unchanged. Every included model has a cluster ID, singletons
included: 74 models across 54 clusters, 9 clusters holding more than one model.

## Determinism and versioning

Tables are sorted on stable keys and written with a fixed line terminator; no
timestamps enter the data tables. Two consecutive builds produce byte-identical
output across all 11 artifacts. `VERSION.json` freezes the version with SHA-256
digests of every artifact plus the manifest and registry, and records whether all
invariants passed.

## Dataset invariants

`invariants.json` records ten checks, all currently passing:

1. Manifest holds 75 unique accessions
2. Every registry entry carries a download URL and local SHA-256
3. Every downloaded file matches the upstream SHA-256
4. Every included model parses
5. Every evaluable reaction has at least one valid ground-truth ID
6. All included ground-truth IDs are well formed
7. Reaction records reconcile: total = evaluable + reaction-level exclusions
8. Model records reconcile: manifest = included + excluded + pipeline failures
9. Every included model has a cluster ID
10. `(model_id, reaction_id)` keys are unique

## Outputs

| File | Contents |
|---|---|
| `reactions.csv` | One row per (model, reaction) with ground truth and flags |
| `model_context.csv` | Model title and notes; join on `model_id` |
| `model_summary.csv` | Per-model status, counts, cluster assignment |
| `model_clusters.csv` | `model_id` → `cluster_id` for partitioning |
| `exclusions.csv` | Scientific exclusions only |
| `pipeline_failures.csv` | Tooling and provenance failures |
| `duplicate_groups.csv` | Multi-member clusters with linkage rules |
| `species_annotations.csv` | Species-level annotation inventory |
| `parser_diagnostics.csv` | Raw-vs-extracted KEGG mention comparison |
| `benchmark_summary.json` | Observed counts |
| `invariants.json` | Invariant check results |
| `VERSION.json` | Frozen version with artifact digests |
| `RECONCILIATION.md` | Discrepancy analysis vs the historical report |

## Branch integration

Phase 1 integrates evaluation code from `upstream/test/no-rule-evals`, including
the KEGG URI slash-form fix that `main` lacks, KEGG-compound species support, SSX
detection, and the ontology/cofactor ablation toggles.
