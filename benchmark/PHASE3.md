# Phase 3: retrieval-first recovery

Phase 3 starts from tag **`benchmark-phase2-v1`** (commit `1958057`). It does not
modify Phase 1 or Phase 2 artifacts and does not regenerate candidates.

Phase 2 showed that retrieval, not ranking, is the bottleneck:

| Quantity | Value | Population |
| --- | ---: | --- |
| Evaluable reactions | 5,816 | all evaluable |
| Zero candidates | 3,457 (59.44%) | all evaluable |
| Nonempty set, exact answer absent | 340 | all evaluable |
| Exact answer retrievable | 2,019 | all evaluable |
| Heuristic exact Top-1 | 1,934 | all evaluable |
| Retrievable but not rank-1 | 85 | all evaluable |
| Overall exact retrieval failure | **65.29%** | all evaluable |
| Max gain from perfect reranking | **+1.46 pp** | all evaluable |
| Share of errors that are retrieval failures | **97.8%** | all failures |

A reranker-only Phase 3 is capped at +1.46 pp. This phase therefore studies **open-set
recovery** and **learned full-database retrieval**. Closed-set reranking is a control,
not the programme.

This commit is design and offline scaffolding only. No API calls, no model downloads,
no training.

## Research questions

1. Can a model recover the correct KEGG reaction when AAAIM’s rule-based retriever
   returns no useful answer?
2. How much apparent recovery is parametric LLM knowledge versus evidence-backed
   database/tool retrieval?
3. Can a learned full-database retriever outperform strict subset-containment retrieval?
4. Which compact forms of biological context improve recovery enough to justify their
   token cost?
5. Can the system recognize insufficient evidence and abstain?
6. When should the system use deterministic ranking, learned retrieval, LLM inference,
   tool-assisted recovery, or abstention?

These are different tasks. Mixing them inflates recovery:

| Task | What it measures | What it is not |
| --- | --- | --- |
| Closed-set reranking | Ordering a frozen Phase 2 candidate set | Recovery of missing answers |
| Open-set direct LLM | Parametric identification with no catalog search | Database retrieval |
| Learned full-database retrieval | Ranking all ~12,312 KEGG reactions from a query encoder | Reranking Phase 2 candidates |
| Tool-assisted recovery | Answering from recorded search evidence | An unsupported guess that happens to be right |
| End-to-end routed accuracy | A future gate choosing among the above, plus abstention | Any single mode’s headline |

A correct direct-LLM guess is **parametric identification**. It is reported separately
from evidence-backed tool recovery. The model’s self-report (`basis`) is metadata, not
proof.

## Layout

Artifacts live in `benchmark/phase3/`, not in the frozen `benchmark/data/` tree.

| Script | Output |
| --- | --- |
| `build_phase3_strata.py` | `strata.csv`, `strata_summary.json` |
| `build_phase3_splits.py` | `splits.csv`, `split_summary.json`, `target_overlap.json` |
| `build_phase3_lookups.py` | `species_names.csv`, `kegg_catalog_ids.json` |
| `sample_phase3_pilot.py` | `pilot_sample.csv` (no ground truth), `pilot_answer_key.csv`, `pilot_summary.json` |
| `phase3_prompts.py --write` | `pilot_prompts.jsonl` |
| `phase3_cost.py` | `cost_estimate.json` |
| `phase3_modes.py` | schemas, mocks, cache (library) |
| `phase3_eval.py` | offline scoring (library) |

```powershell
$env:PYTHONHASHSEED = "0"
python benchmark/scripts/build_phase3_strata.py
python benchmark/scripts/build_phase3_splits.py
python benchmark/scripts/build_phase3_lookups.py
python benchmark/scripts/sample_phase3_pilot.py
python benchmark/scripts/phase3_prompts.py --write
python benchmark/scripts/phase3_cost.py
```

`benchmark/phase3/_*/` is gitignored (live response cache). Pricing is read from a file;
`pricing.example.json` is labelled EXAMPLE ONLY.

## Outcome strata

Every evaluable reaction gets exactly one stratum from **exact** Phase 2 matching:

| Stratum | Definition | Corpus |
| --- | --- | ---: |
| `unconstrained` | status `unconstrained_candidate_set` | 811 |
| `empty_constrained` | status `no_candidates` | 2,646 |
| `nonempty_answer_absent` | nonempty set, exact answer absent | 340 |
| `retrievable_rerank_failure` | exact answer present, heuristic not Top-1 | 85 |
| `retrievable_top1_success` | heuristic exact Top-1 | 1,934 |
| **Total** | mutually exclusive, exhaustive | **5,816** |

Equivalence-aware hit columns are stored beside the stratum. They never assign it.

## Leakage-resistant splits

Partition units are the **54 Phase 1 clusters**. No cluster appears in more than one
split. `CLU_BIOMD0000000042` (seven yeast models) is already one cluster and stays
together; it landed in **validation**.

Algorithm `cluster_greedy_v1`, seed **20260902**. Target shares 60/15/25
train/validation/test by reaction count. Clusters are assigned largest-first to the
split that least increases a weighted squared-error loss, then locally improved.
Loss weights up-weight the rare `retrievable_rerank_failure` stratum and genome-scale
mass. Phase 1 membership is never rewritten to make the numbers prettier.

Observed assignment:

| Split | Reactions | Models | Clusters | Genome-scale rxn | Share |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 3,497 | 29 | 21 | 2,542 | 0.601 |
| validation | 969 | 18 | 12 | 792 | 0.167 |
| test | 1,350 | 27 | 21 | 1,220 | 0.232 |
| **total** | **5,816** | **74** | **54** | 4,554 | 1.000 |

Test stratum counts (the **final** held-out set; not used for the exploratory pilot):

| Stratum | Test | Train | Val |
| --- | ---: | ---: | ---: |
| unconstrained | 36 | 690 | 85 |
| empty_constrained | 708 | 1,519 | 419 |
| nonempty_answer_absent | 174 | 144 | 22 |
| retrievable_rerank_failure | 22 | 47 | 16 |
| retrievable_top1_success | 410 | 1,097 | 427 |

### Why unconstrained is unbalanced

`CLU_BIOMD0000001091` alone holds **653** of the 811 unconstrained reactions. That
cluster cannot be fractionally split. Putting it in test would flood the held-out
set with one genome-scale model. It therefore stays in train, and the test split
has only 36 unconstrained reactions. The exploratory pilot uses **validation**
(85 unconstrained), so this cluster constraint does not starve the pilot. The
split is not rewritten to manufacture a round 200.

### KEGG target overlap

The catalog is closed (~12,312 KEGG reactions). Seeing a target id in training does
not leak the *query*, but it can inflate parametric memorization. Overlap is allowed
and must be reported.

| Quantity | Value |
| --- | ---: |
| Distinct test targets | 877 |
| Distinct train targets | 1,596 |
| Distinct validation targets | 658 |
| Targets in both train and test | 645 |
| Test targets seen in train or validation | 658 (75.03%) |
| Test targets never seen in train or validation | 219 (24.97%) |

Future results must be split into **seen** vs **unseen** targets relative to the
data actually used to **fit** the model: train-only if the retriever is trained
on train (method selection on validation), or train+validation if the final
retriever is refit before the test run. Frequencies are in `target_overlap.json`.

## Open-set pilot sample

Drawn **only from validation**. Test is reserved for one frozen-method run after
the pilot chooses context variant, prompt, provider/model, abstention, and tool
strategy. Sequence: train fits the retriever; validation runs the exploratory
LLM pilot; test runs the frozen method once.

Seed **20260902**. Default quotas 50/50/50/25/25. Within a stratum, clusters are
visited round-robin after a seeded within-cluster shuffle. Train and test are
never used as backfill.

| Stratum | Quota | Eligible in validation | Selected | Shortfall |
| --- | ---: | ---: | ---: | ---: |
| unconstrained | 50 | 85 | 50 | 0 |
| empty_constrained | 50 | 419 | 50 | 0 |
| nonempty_answer_absent | 50 | 22 | 22 | 28 (took all) |
| retrievable_rerank_failure | 25 | 16 | 16 | 9 (took all) |
| retrievable_top1_success | 25 | 427 | 25 | 0 |
| **total** | 200 | | **163** | |

163 reactions, 17 models, 12 clusters. The shortfalls are accepted; the cluster split is not rewritten
to reach 200.

`pilot_sample.csv` is what a model may see: no ground-truth KEGG ids.
`pilot_answer_key.csv` holds the labels. Prompts are scanned for `R#####` and
`kegg.reaction` URIs and fail loudly on a hit.

This size is large enough to see whether unconstrained / empty / answer-absent
recovery is even possible, and small enough that example mid-band chat prices stay
in the low single-digit dollars for three context variants. It is **not** large
enough for significance theatre; evaluation reports cluster-aware intervals and
their limits.

## Context variants

Provider-independent payloads, template `phase3-open-set-v2`.

| Variant | Contents | Bound |
| --- | --- | --- |
| `target_only` | Equation, participant names, ChEBI / KEGG *compound* ids, direction | no model chain |
| `target_plus_model` | Plus title and a 280-character redacted description | no model chain |
| `target_plus_neighborhood` | Plus up to **k=4** other reactions in the same model that share participants | k is configurable; selection is (−shared, reaction_id) |

Participant display names come from a `(model_id, species_id) → species_name`
table extracted from SBML (`species_names.csv`). Names are **not** inferred by
zipping unique equation ids with semicolon-separated `substrate_names` /
`product_names`. That positional join attaches the wrong name when a species
appears on both sides (for example NFAT vs calcineurin in
`BIOMD0000000122/R1`) or when a name itself contains a semicolon. Missing ids
fall back to the species id, never to a shifted name list.

| Variant | Contents | Bound |
| --- | --- | --- |
| `target_only` | Equation, participant names, ChEBI / KEGG *compound* ids, direction | no model chain |
| `target_plus_model` | Plus title and a 280-character redacted description | no model chain |
| `target_plus_neighborhood` | Plus up to **k=4** other reactions in the same model that share participants | k is configurable; selection is (−shared, reaction_id) |

The entire model reaction list is never included. Neighbor order is deterministic.
KEGG *reaction* ids are redacted from notes, from KEGG-shaped SBML reaction ids
in the model-visible payload, and then forbidden anywhere in the prompt,
including filenames and URLs. Join keys in `pilot_prompts.jsonl` may still be
KEGG-shaped when a BioModels file uses the KEGG id as the SBML reaction id;
those keys are not shown to the model. Compound ids (`C#####`, ChEBI) are
allowed: they are species evidence, not the answer.

The open-set instruction asks for JSON with up to three ordered `R#####` ids,
confidence, explicit abstention, a short rationale, and a self-reported
`basis` ∈ {`recalled_knowledge`, `supplied_evidence`, `mixed`}. Self-report is not
treated as evidence.

## Experiment modes

Common `ModeResult` schema in `phase3_modes.py`. Live HTTP is `LiveCallBlocked`
until a budgeted run is explicitly approved.

1. **Direct open-set LLM** — no candidates, no tools. Parametric identification.
2. **Tool-assisted recovery** — queries, hits, source ids/URLs, and snippets are
   recorded. A prediction is `evidence_backed` only when **that predicted
   identifier** appears in the recorded evidence identifiers. Per prediction:
   `prediction_supported_by_evidence`, `supporting_evidence_ids`. Outcomes:
   correct and evidence-supported; correct but unsupported; incorrect despite
   evidence; abstained after retrieval.
3. **Closed-set control** — frozen Phase 2 candidates only. Inapplicable when the
   set is empty. Not open-set recovery.
4. **Learned full-database retrieval** — query against all KEGG reactions. Schema
   is ready; **training is not started**.

Responses are cached by SHA-256 of `(mode, provider, model, prompt)`. Re-running a
cached item is a no-op. Tests use `MemoryCache` and `MockProvider`.

## Evaluation

`phase3_eval.py` scores mocked or future cached outputs:

- Exact and BRITE/orthology Top-1 / Top-3
- Identifier class against the frozen Phase 2 KEGG catalog: malformed,
  well-formed but absent, or in-catalog. Syntax `R#####` is not existence.
- Abstention rate; accuracy among answered reactions
- Accuracy-versus-coverage curve
- By stratum, context variant, mode, and seen vs unseen **fit** targets
- Cluster-macro where the subset is large enough
- Evidence-linked outcomes for tool mode
- Recall@1/3/5/10 and MRR for retrieval-shaped outputs
- Parser/compliance: confidence outside [0, 1], duplicate predicted ids,
  `abstain=false` with no predictions

Abstention is **not** a hallucinated identifier. It is incorrect for full-coverage
accuracy and is the uncovered case in selective accuracy.

Uncertainty: cluster bootstrap (1,000 resamples) with an explicit limitation string.
The 163-reaction validation pilot is too small to support strong significance claims.

## Token and cost estimate (no API)

489 calls (163 validation reactions × 3 variants). Scaffolding estimate: 4
characters ≈ 1 token (`chars_div_4_scaffold`). This method **must not** gate a
paid run; replace it with the chosen model's tokenizer or a conservative
provider-specific bound before any live call. Planned max output: 400 tokens/call.

| Variant | Calls | Mean input tokens | Total input |
| --- | ---: | ---: | ---: |
| target_only | 163 | 288 | 46,894 |
| target_plus_model | 163 | 359 | 58,533 |
| target_plus_neighborhood | 163 | 456 | 74,362 |
| **bounded total** | **489** | | **179,789** |
| Whole-model-context counterfactual | 163 | | **907,270** |

Repeating every reaction in the model for every target uses **5.05×** more input
tokens on this sample. That is the quadratic anti-pattern this design avoids.

Example prices from `pricing.example.json` (date 2026-09-02, **not a quote**,
replace before any paid run):

| Placeholder band | Expected USD | Worst-case USD (2× output) |
| --- | ---: | ---: |
| small chat (~$0.15/$0.60 per 1M) | 0.14 | 0.26 |
| mid chat (~$3/$15 per 1M) | 3.47 | 6.41 |
| large chat (~$15/$75 per 1M) | 17.37 | 32.04 |

No live run proceeds until the sample is frozen, leakage tests pass, this estimate
is printed, caching is on, a real tokenizer is wired, and the user approves
**provider, model, sample size, and budget**.

Proposed options, **not a choice**: a small/cheap chat model for the first pilot
(sensitivity to parametric knowledge vs noise); a mid-size general model; a
science-tuned model if one is already licensed. Tool-assisted mode additionally
needs an explicit KEGG/BioModels search backend. None of these is selected here.

## Learned full-database retriever (design only)

This is a **retriever over all ~12,312 KEGG reactions**, not a reranker of Phase 2
sets. Phase 2’s heuristic is already within 3.6 pp of the conditional oracle; the
missing 65% are reactions whose candidate set is empty or omits the answer.
Reranking cannot touch those. A bi-encoder can.

### Architecture

- Small bi-encoder (scientific or general text checkpoint, to be chosen at training
  time; **not downloaded now**).
- Contrastive training on (reaction query, KEGG document) pairs.
- Precompute KEGG document embeddings once; retrieve with exact or ANN search.
- Evaluate on the cluster-separated splits above, reporting seen vs unseen targets.

### Query fields

SBML equation; participant names; available ChEBI and KEGG *compound* ids; optional
bounded model title / k-neighborhood from the same prompt builder. Never the
ground-truth reaction id, never `kegg.reaction` URIs.

### Document fields

KEGG equation, definition/name, EC, KO, RCLASS. The document id is the retrieval
key, not a string that should appear in the query.

### Hard negatives

- Phase 2 candidates for the same reaction (when they exist and are wrong)
- KEGG reactions sharing compounds
- Lexically similar KEGG text
- Retrieved false positives from an initial encoder (ANCE-style)
- Equivalence-aware near matches: **not** automatic negatives. Label as
  equivalent / adjacent / unrelated. Treating BRITE siblings as hard negatives
  would punish chemically correct retrieval.

### Training

- Objective: InfoNCE / MultipleNegativesRanking, in-batch negatives plus explicit
  hard negatives.
- Batch: 32–64 queries, 1 positive + 7–15 hard negatives, plus in-batch.
- Max length: 256 query / 256 document tokens as a starting point.
- Validation: cluster-macro Recall@10 and MRR on the validation split, stratified
  by Phase 2 stratum. Early stop on validation Recall@10.
- Model selection: primary = Recall@10 on *unseen* validation targets; secondary =
  unconstrained and empty_constrained Recall@10 (the Phase 2 hole).
- Hardware: a single 12–24 GB GPU should fine-tune a MiniLM/SciBERT-scale encoder
  on 3.5k training reactions in well under a day; CPU-only is possible but slower.
  Exact runtime is deferred until a checkpoint is chosen.
- Reproducibility: seed 20260902, frozen splits, logged config, hashed query/document
  text, checkpoints under `benchmark/phase3/_checkpoints/` (gitignored).
- Tracking: a local JSONL run log is enough for the first experiment; no external
  SaaS is required.

### Optional cross-encoder

A small cross-encoder reranker may be trained later **only** on retrievable
reactions (the 2,019). It is bounded by the Phase 2 oracle (+1.46 pp overall) and
is secondary.

## Routed system (not implemented)

A future gate, without access to ground truth at inference:

| Signal | Action |
| --- | --- |
| Phase 2 set small, high heuristic margin | Keep the heuristic |
| Phase 2 set nonempty but large/flat scores | Learned or LLM rerank (secondary) |
| Empty, unconstrained, or low-confidence set | Full-database retrieval and/or tool-assisted recovery |
| Low evidence (few mapped participants, low retriever score) | Abstain |

**Answer absence is not observable.** Proxies to learn or threshold without labels:

- `filtered_species_count` / unconstrained flag (already in Phase 2 status)
- Candidate-set size and score entropy
- Retriever margin between rank-1 and rank-2
- A binary calibrator trained on the train split to predict
  `stratum ∈ {unconstrained, empty_constrained, nonempty_answer_absent}`

Those proxies will be wrong sometimes; the evaluation of a router is end-to-end
selective accuracy, not oracle routing.

## Stop point

This branch stops at tested offline scaffolding. Remaining decisions that need
approval before any paid or GPU experiment:

1. Provider and model for the 163-reaction × 3-variant open-set **validation**
   pilot, a real tokenizer for that model, and a budget cap.
2. Accept the validation shortfalls (answer-absent 22 vs 50, rerank-failure 16
   vs 25). Do not restructure the cluster split to reach 200.
3. Tool backend for mode 2 (local KEGG files vs a network API).
4. Which encoder checkpoint to fine-tune, once downloading weights is allowed.
5. Whether the final retriever is refit on train+validation; that choice defines
   “seen target” for the test evaluation.

Do not start API calls or training until those are explicit.
