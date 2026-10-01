# Reconciliation: observed counts vs the historical 68 models / 4,379 reactions

Benchmark version `phase1-v1`. Built from a checksum-verified snapshot of all 75
manifest accessions. Every number below is read from the generated artifacts,
not assumed.

## Observed counts

| Quantity | Observed |
|---|---|
| Manifest accessions (unique) | 75 |
| Models downloaded and SHA-256 verified | 75 / 75 |
| Models parsed by libSBML | 75 / 75 |
| Pipeline failures | 0 |
| Models included in benchmark | 74 |
| Models scientifically excluded | 1 |
| Ground-truth reactions (all records) | 5,838 |
| Evaluable reactions | 5,816 |
| Reactions excluded, exchange/SSX | 19 |
| Reactions excluded, malformed ground-truth ID | 3 |
| Reactions carrying more than one KEGG ID | 91 |
| Distinct KEGG reaction identifiers | 1,906 |
| Duplicate clusters with more than one member | 9 (29 models) |
| Distinct clusters (partition units) | 54 |

`reactions.csv` SHA-256: `f11ebc6f4d0734deb14c054caf277b81695b22668017d70af9c43cd6146004f1`

All ten dataset invariants pass; see `invariants.json`.

## The single scientific exclusion

**BIOMD0000000579** carries no reaction-level KEGG annotation. It was
investigated individually rather than accepted at face value, because a
"KEGG-annotated" model yielding zero ground truth is exactly the signature of a
parser defect.

Finding: the model has 175 reactions, **all** of which carry annotations, but
none reference `kegg.reaction`. The 16 raw `kegg.reaction` mentions in the file
belong to **13 species**, for example species `s308` annotated with
`kegg.reaction/R00289`. Annotating a metabolite with a reaction identifier is a
curation error in the source model.

This is a genuine data-level exclusion, not a tooling failure. It is recorded in
`parser_diagnostics.csv` under `misplaced_annotations_only`, and it exposes a
manifest-construction weakness: the original `find_kegg_in_biomodels.py` selected
models by grepping raw file bytes for `kegg.reaction`, which cannot distinguish a
reaction annotation from a misplaced species annotation. The manifest therefore
over-selects.

## Why the counts differ from 68 / 4,379

The historical figures are treated as a clue. Exclusions were **not** adjusted to
reproduce them. Four candidate explanations were tested.

### 1. Upstream file drift — ruled out

Every one of the 75 models has its latest upstream revision dated **before** the
2025-11-06 snapshot date, so no model has changed at BioModels since the
historical run. The corpus we built is the same set of model versions the
historical run would have seen.

One caveat worth recording: the copy of `BIOMD0000000013` previously vendored in
`tests/test_models/` is 99,166 bytes, whereas the verified upstream main file is
96,608 bytes. The historical `tests/BioModels_251106/` directory is absent from
the repository and from this machine, so its byte-level contents cannot be
compared. Local snapshots obtained by bulk download were evidently not identical
to the API's `main` SBML file, which is precisely why this benchmark now pins
upstream checksums.

### 2. A colon-only KEGG URI pattern — a real defect, and the likely reason this
was hard to reproduce

`main` matched only `kegg.reaction:R00024`. Real BioModels files overwhelmingly
use the slash form `kegg.reaction/R00024`, so `main`'s pattern extracts **zero**
ground-truth reactions from this corpus. The unmerged evaluation branch carried
the fix; `main` did not. The fix is now on this branch and locked down by
parametrised regression tests over slash, colon, and URN forms.

This means the historical numbers could not have been produced by `main`'s
extraction code, and any attempt to reproduce them from `main` would have yielded
nothing at all.

### 3. Ground-truth policy — affects scoring, not counts

The historical helper kept only the first KEGG ID per reaction. That changes
measured accuracy but not the number of reaction records, so it cannot account
for the 1,459-reaction gap. It does matter scientifically: **91 reactions** carry
multiple valid IDs, one of them (`BIOMD0000000015`, reaction `den`) carrying 10.
Scoring against a single arbitrary ID under-credits correct predictions on those
reactions.

### 4. Which models were actually evaluated — the most plausible cause

The corpus is extremely top-heavy. Seven genome-scale models contribute **4,556
of 5,838** ground-truth reactions (78%); the other 67 models contribute 1,282.
Median reactions per model is 10; the largest single model contributes 1,047.
Any difference in how the largest models were handled swings the total enormously:

| Scenario | Models | Reactions |
|---|---|---|
| This build (all included) | 74 | 5,838 |
| Excluding the 2 genome-scale models with no species annotations | 72 | 4,418 |
| Excluding all 7 genome-scale models | 67 | 1,282 |
| Historical report | 68 | 4,379 |

No subset rule tested reproduces 68 / 4,379 exactly. The nearest simple rule —
dropping the two genome-scale models that have no usable species annotations
(`BIOMD0000001090`, `BIOMD0000001091`, 1,420 reactions between them) — yields
4,418 reactions across 72 models, still not a match.

Two concrete pieces of evidence point at the model list rather than the rules:

- The committed evaluation script defaulted to
  `tests/kegg_annotated_files-test.txt`, a file that **does not exist** in the
  repository, rather than the 75-entry `kegg_annotated_files.txt`. The published
  numbers therefore likely came from a different, probably truncated, list.
- The historical outputs (`per_reaction_results.csv`, `results_summary.csv`) were
  never committed, so the excluded set cannot be recovered directly.

**Conclusion.** The difference is attributable to which models were fed to the
historical run, not to the corpus having changed and not to the exclusion rules
applied here. 68 / 4,379 is not reproducible from the 75-model manifest under any
defensible rule, and it is not adopted as an acceptance criterion. The current
counts are fully reconciled and reproducible: 75 = 74 included + 1 excluded + 0
pipeline failures, and 5,838 = 5,816 evaluable + 22 reaction-level exclusions.

## Reaction-level exclusions in detail

**Exchange/SSX (19 reactions across 9 models).** Reactions with an empty
reactant or product side, such as `PRPP =>` in `BIOMD0000000015`. The rule-based
matcher generates no candidates for these when `include_exchange_reactions=False`,
so they are retained as records and marked `included_in_eval=False`. They reduce
reaction counts only; no model is excluded on this basis, and
`models_with_zero_eval_reactions` is 0.

**Malformed ground-truth ID (3 reactions).** Reaction `gluconeogenesis_ser` in
`BIOMD0000000268`, `BIOMD0000000450`, and `BIOMD0000000674` is annotated
`R0006565`, which has seven digits where KEGG uses five. A curation typo in three
related models, all of which fall in cluster `CLU_BIOMD0000000268`.

## Duplicate clusters and partitioning

Clustering links models by ground-truth overlap using two rules, both recorded
per cluster in `duplicate_groups.csv`:

- **Jaccard** ≥ 0.9 catches same-scope variants, such as
  `Yamada2003_JAK_STAT_pathway` and its SOCS1 knockout.
- **Containment** ≥ 0.9 catches models that extend another, gated on comparable
  set size (ratio ≥ 0.5) and at least 5 shared identifiers. Without containment
  the three Smallbone2013 yeast variants split apart: `BIOMD0000000473` shares
  189 reactions with `BIOMD0000000471`/`472` yet scores only 0.86 by Jaccard
  because it adds 23 more. Without the size and overlap gates, clustering
  collapses — a two-reaction model is trivially contained in a genome-scale one,
  which chained unrelated models into one cluster and cut distinct clusters from
  65 to 29. Both failure modes are covered by regression tests.

Cluster IDs are `CLU_<lexicographically smallest member>`, so they are stable
across rebuilds for unchanged membership. Every included model has a cluster ID,
singletons included, so train/test splits partition on `cluster_id` and never on
individual reactions.

One judgment call is worth flagging. `CLU_BIOMD0000000042` groups seven yeast
glycolysis models from seven different papers (Nielsen1998, Hynne2001,
Galazzo1990, Teusink2000, Bakker2001, Albert2005, Ralser2007). They are not
variants of one another in provenance terms, but they annotate substantially the
same KEGG reactions, so separating them across a split would leak labels. The
grouping is deliberately conservative; thresholds are exposed as
`--duplicate-threshold`, `--containment-threshold` if a stricter provenance-based
definition is preferred later.

## Reproducing this build

```bash
python benchmark/scripts/download_biomodels.py   # 75/75 checksum-verified
python benchmark/scripts/build_benchmark.py      # deterministic artifacts
python -m pytest tests/test_benchmark_build.py   # 21 regression tests
```

Two consecutive builds produced byte-identical output across all 11 artifacts.
