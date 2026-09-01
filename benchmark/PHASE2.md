# Phase 2: candidate retrieval and baseline performance

Phase 2 establishes what the rule-based generator can retrieve, and how well simple
rankers order those candidates, before any training happens. It builds on the frozen
Phase 1 benchmark (tag `benchmark-phase1-v1`, commit `2129bc5`) and never modifies it.

Candidates are stored in their own tables, keyed by model, reaction and candidate. Phase
1's `reactions.csv` remains candidate-free.

## Status

The pipeline is written and unit-tested end to end. **The full generation pass has not
been run**: it takes roughly three days of wall-clock time, so it is meant to be run
locally rather than inside an agent session. Everything below is ready to execute.

## Pipeline

| Step | Script | Output | Cost |
| --- | --- | --- | --- |
| 0 | `build_species_evidence.py` | `species_evidence.csv` | ~20 s |
| 1 | `build_reaction_strata.py` | `reaction_strata.csv` | ~10 s |
| 2 | `build_reaction_text.py` | `reaction_text.csv` | ~10 s |
| 3 | `generate_candidates.py` | `candidates.csv`, `candidate_status.csv`, `candidate_generation_failures.csv` | **~3 days** |
| 4 | `analyze_retrieval.py` | `retrieval_ceiling.json`, `reaction_retrieval.csv`, `retrieval_ceiling_by_stratum.csv` | ~1 min |
| 5 | `rank_baselines.py` | `baseline_table.csv`, `baseline_rankings.csv`, `failure_stratification.csv` | minutes (LLM: longer) |

### Running it locally

```powershell
# Steps 0-2: fast, run once.
python benchmark/scripts/build_species_evidence.py
python benchmark/scripts/build_reaction_strata.py
python benchmark/scripts/build_reaction_text.py

# Smoke test first: the 3 smallest pending models, one worker.
python benchmark/scripts/generate_candidates.py --limit 3 --workers 1

# Full pass. Resumable: each model is cached under benchmark/data/_candidates_cache/
# and skipped on re-run, so this can be stopped and restarted freely.
$env:PYTHONHASHSEED = "0"
python benchmark/scripts/generate_candidates.py --workers 4

# Rebuild the frozen tables from the cache without regenerating.
python benchmark/scripts/generate_candidates.py --assemble-only

# Steps 4-5.
python benchmark/scripts/analyze_retrieval.py
python benchmark/scripts/rank_baselines.py --rankers heuristic lexical embedding random oracle
```

Notes on the long step:

* **Resume** is automatic and keyed on `config_id`, a hash of `GENERATION_CONFIG`. Changing
  any generation parameter invalidates every cached model on purpose.
* **Workers**: memory, not CPU, is the constraint (~350 MB–1.1 GB per worker for the ChEBI
  and KEGG reference maps). On a 16 GB machine 4 workers is comfortable.
* **`--scope`** controls how much work is done. `evaluable` (default) generates only for
  reactions carrying ground truth. `all` reproduces production behaviour, where the
  generator sees every reaction in the model, and is substantially slower — many models
  pass far more reactions to the generator than the benchmark scores (`BIOMD0000000244`:
  50 vs 2). The scope is recorded in `config_id`, so the two are never mixed.
* Set `PYTHONHASHSEED=0`. Candidate ranks are re-derived deterministically, but pinning the
  seed keeps upstream set iteration stable too.

## Candidate table schema

`candidates.csv`, one row per (reaction, candidate):

| Column | Meaning |
| --- | --- |
| `model_id` | BioModels accession |
| `reaction_id` | SBML reaction id |
| `candidate_kegg` | Candidate KEGG reaction id |
| `raw_rank` | 1-based rank in the generator's own ordering |
| `heuristic_score` | AAAIM rule-based score for the candidate |
| `relaxation_level` | Max ChEBI hierarchy hops used for the reaction (0 = exact) |
| `relaxation_direction` | `exact`, or `up`/`down`/`down+up` when relaxed |
| `config_id` | Hash of the generation configuration that produced the row |

`raw_rank` is re-derived as score descending, KEGG id ascending. The upstream ordering is a
stable sort over set-derived iteration order, which is not reproducible across processes,
so ranks taken directly from it would not be either.

## Statuses and failures

Every evaluable reaction gets exactly one row in `candidate_status.csv`. The statuses
separate scientific limitations from pipeline failures:

| Status | Meaning | Kind |
| --- | --- | --- |
| `ok` | Constrained candidate set with ≥1 candidate | success |
| `no_candidates` | Constraints applied, nothing matched | retrieval failure |
| `unconstrained_candidate_set` | No mapped participants, so the "candidate set" is all of KEGG | retrieval failure |
| `no_species_evidence` | No usable species annotations, generator could not run | retrieval failure |
| `exchange_skipped` | SSX exchange reaction, excluded in Phase 1 | excluded |
| `generation_failed` | An exception was raised | **pipeline failure** |
| `absent_from_generator_output` | Reaction in the frozen table but not returned | **pipeline failure** |

Only the last two appear in `candidate_generation_failures.csv`. A candidate-generation
exception is never silently converted into an empty candidate list:
`_get_kegg_recommendations_rulebased` is called with `strict_errors=True` so it raises, and
if a model raises the harness retries it reaction by reaction to attribute the failure
precisely rather than losing the whole model.

## Bugs found while building Phase 2

Four upstream defects were found that each corrupted results silently. All are fixed, and
each is pinned by a test in `tests/test_phase2_candidates.py`.

### 1. Reaction ids were misattributed (most serious)

`map_reactions_to_kegg` labels the i-th reaction string with `reaction_ids[i]`, but the
reaction list comes from `extract_reactions_from_sbml`, which **filters out** reactions
mentioning none of the mapped species, while the id list came from `get_all_reaction_ids`,
which does not. From the first omission onward, every candidate was attached to the wrong
reaction.

This was not rare: `BIOMD0000001090` yields 529 ids but 528 reactions, and many models
filter heavily (`BIOMD0000000093`: 46 → 3). Fixed by adding
`extract_reactions_with_ids_from_sbml`, which returns ids and reaction strings from the
same pass; `extract_reactions_from_sbml` now delegates to it and is unchanged for callers.

### 2. KEGG compound URIs only matched the colon form

`KEGG_COMPOUND_URI_PATTERNS` matched `kegg.compound:C00022` but BioModels writes
`kegg.compound/C00022`. This is the same bug class fixed for KEGG *reaction* URIs in Phase
1; the compound patterns were missed.

Consequence: three models looked like they had no species annotations at all —
`BIOMD0000000725`, `BIOMD0000001090`, `BIOMD0000001091`, i.e. **1,451 reactions, 25% of the
benchmark**. They in fact annotate species with KEGG compound ids directly, which is the
most directly usable evidence available since it needs no ChEBI→KEGG mapping. Phase 1's
`species_source = "none"` for those reactions is an artefact of this bug, not a property of
the models.

`species_evidence.csv` now unions ChEBI and direct KEGG compound annotations, and no
evaluable reaction is left without species evidence:

| Evidence class | Reactions |
| --- | --- |
| `chebi+kegg_compound` | 4,260 |
| `kegg_compound` only | 1,482 |
| `chebi` only | 67 |
| none | 7 |

Phase 1 artefacts are left untouched; this is a Phase 2 input table.

### 3. Unconstrained reactions "matched" the entire KEGG database

`filter_kegg_reactions` keeps a KEGG reaction when `model_keys.issubset(kegg_keys)`. The
empty set is a subset of everything, so a reaction whose participants all failed to map
matched **all 12,312 KEGG reactions** with score 0.

On `BIOMD0000000725`, 17 of 31 reactions hit this, which would have reported a trivially
perfect "recall at any rank" while ranking the answer outside the top 10. Such reactions
are now detected (`filtered_species_count == 0`, a perfect 1:1 signal in testing), recorded
as `unconstrained_candidate_set`, and stored with **no** candidate rows. The absence of
retrieval is not a candidate set.

The remaining 14 constrained reactions on that model produced 1–2 candidates each with the
correct answer at rank 1 — retrieval on this generator tends to either nail it or produce
nothing usable, which is exactly what the ceiling analysis needs to quantify.

### 4. A batch exception discarded the whole model

`_get_kegg_recommendations_rulebased` wrapped its entire per-reaction loop in
`except Exception: return []`. A failure on reaction 900 of 1,047 silently discarded the
899 successful reactions before it. Fixed with an opt-in `strict_errors` flag; the
permissive default is preserved for existing callers.

## Metrics

### Retrieval ceiling (`analyze_retrieval.py`)

Recall at any rank, Recall@1/3/5/10, mean candidate-set size, and the percentage with zero
candidates. Reported over all evaluable reactions (zero-candidate reactions charged as
failures — the headline) and, separately, conditional on a non-empty candidate set.

### Exact vs equivalence-aware matching

Exact id matching understates performance, since KEGG carries several entries for the same
biochemistry. `kegg_equivalence.py` builds groups from the KEGG feature table:

| Criterion | Basis | KEGG reactions covered |
| --- | --- | --- |
| `exact` | Identical reaction id | 12,312 |
| `ec` | Shared EC number (`ENZYME`), the leaf level of BRITE `br08201` | 10,874 |
| `ko` | Shared KEGG Orthology id (`ORTHOLOGY`) | 7,181 |
| `brite_orthology` | Shared EC **or** KO — the headline equivalence criterion | — |
| `rclass` | Shared reaction class (`RCLASS`), the loosest grouping | 10,663 |

Exact hits always count as equivalent. Entries with no annotation for a criterion have an
empty group set and can never match, so unannotated reactions cannot inflate the scores.

### Three averaging modes

Seven genome-scale models hold 78.3% of evaluable reactions, so every metric is reported as
`reaction_micro` (mean over reactions), `model_macro` (mean over models of each model's
reaction mean), and `cluster_macro` (mean over Phase 1 clusters of each cluster's reaction
mean).

### Baseline rankers (`rank_baselines.py`)

All rankers only reorder the frozen candidate set, which bounds them by the retrieval
ceiling and splits every error into exactly one of:

* **retrieval failure** — the answer is absent from the candidate set; no reranker can fix it
* **reranking failure** — the answer is present but not placed first

| Ranker | Method | Requirements |
| --- | --- | --- |
| `heuristic` | Existing AAAIM rule-based score (shipped behaviour) | none |
| `lexical` | TF-IDF cosine, reaction text vs KEGG text | scikit-learn |
| `embedding` | MiniLM cosine over the same text | chromadb + onnxruntime |
| `llm` | LLM reranking of the top 15 candidates | `OPENAI_API_KEY` or `OPENROUTER_API_KEY` |
| `random` | Seeded shuffle (lower reference point) | none |
| `oracle` | Best possible reordering (upper reference point) | none |

`random` and `oracle` bracket what reranking can achieve on a given candidate set, making
the reranking headroom explicit.

LLM reranking is opt-in and was **not run**: no API key is present in this environment. It
also costs one call per reaction, so prefer a seeded stratified subsample:

```powershell
python benchmark/scripts/rank_baselines.py --rankers llm --llm-sample 400 --append
```

Failure stratification covers species annotation source, missing annotations, whether
ontology relaxation was required, reaction complexity, genome-scale vs smaller models, and
the retrieval-vs-reranking split.

## Testing

```powershell
python -m pytest tests/test_phase2_candidates.py tests/test_benchmark_build.py -q
```

48 tests, ~3 s. The Phase 2 tests pin all four bugs above plus deterministic tie-breaking,
multiple ground-truth ids, equivalence-group parsing, and the three averaging modes.

## Phase 1 housekeeping completed here

* Tagged `benchmark-phase1-v1` at commit `2129bc5`.
* SBML inputs archived: `benchmark/dist/aaaim-benchmark-phase1-v1-sbml-inputs.zip`
  (75 files, 63.3 MB → 3.67 MB). Digest and upload instructions in
  `benchmark/dist/RELEASE_phase1-v1.md`. `gh` is not installed, so the upload is manual.
* `verify_snapshot.py` verifies the working copy or a restored archive against
  `model_registry.json`. Both currently pass on all 75 files.

## Not yet done

* Run the generation pass and populate the candidate tables (the 3-day step).
* Freeze a Phase 2 version file with artifact digests, once real outputs exist.
* LLM reranking (needs an API key).
* Cluster sensitivity analysis (conservative functional vs provenance-based clusters),
  which you deferred; `CLU_BIOMD0000000042` remains grouped as a conservative
  pathway-family cluster for the primary split.
