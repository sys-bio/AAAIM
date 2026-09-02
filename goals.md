# AAAIM research project: goals and roadmap

**Status updated:** 2026-09-03  
**Target:** begin manuscript drafting in October 2026 without waiting for every stretch experiment

## Project objective

AAAIM is already a biological annotation tool. This project is therefore not simply to build an annotation tool, but to turn AAAIM into a reproducible research system for studying metabolic reaction annotation when metabolite evidence and database retrieval are incomplete.

The central research question is:

> Can learned retrieval, compact-context LLM inference, and tool-assisted recovery improve metabolic reaction annotation while recognizing when available evidence is insufficient?

A current working title is:

> **Metabolic Reaction Annotation Under Incomplete Biological Evidence: Benchmarking Rule-Based Retrieval, Learned Models, and Tool-Augmented LLMs**

The desired final artifact is:

- A frozen, reproducible benchmark with leakage-resistant model clusters
- Credible rule-based, lexical, embedding, learned, and LLM baselines
- At least one small neural model trained for full-database retrieval or ranking
- A rigorous separation of retrieval failure, reranking failure, and open-set recovery
- Calibration, abstention, and selective-accuracy evaluation
- Robustness experiments under degraded metabolite annotations
- A public experimental pipeline and research-oriented README
- A preprint or technical report

## What the repository investigation established

Substantial evaluation work existed on the unmerged upstream branch `sys-bio/AAAIM:test/no-rule-evals`, but it was stranded and not reproducible from `main`. It contained a 75-model manifest, ground-truth extraction, ChEBI-to-KEGG candidate generation, rule-based and LLM reranking, Top-k evaluation, and several ablations.

The historical claim of 68 models and 4,379 reactions could not be reproduced. The historical script referenced an uncommitted input list, and the relevant outputs were never committed. The new benchmark therefore records the observed corpus transparently instead of changing exclusions to match the historical numbers.

The reconstruction also uncovered several silent-corruption bugs, including URI-pattern mismatches, reaction-ID misalignment after filtering, empty-set candidate explosions, and model-wide loss after a single reaction error. These are now fixed and pinned by tests.

## Phase 1: Frozen benchmark — complete

Phase 1 recovered and verified all 75 BioModels accessions, pinned source files by checksum, documented exclusions, preserved multiple valid KEGG identifiers, detected related-model clusters, and produced deterministic benchmark artifacts.

Observed benchmark:

| Quantity | Result |
|---|---:|
| Manifest models | 75 |
| Included models | 74 |
| Scientifically excluded models | 1 |
| Ground-truth reactions | 5,838 |
| Evaluable reactions | 5,816 |
| Exchange/SSX exclusions | 19 |
| Malformed-ID exclusions | 3 |
| Reactions with multiple valid KEGG IDs | 91 |
| Leakage-control model clusters | 54 |

The Phase 1 snapshot is tagged `benchmark-phase1-v1`.

## Phase 2: Candidate retrieval and baseline evaluation — complete and frozen

Phase 2 froze the rule-based candidate-generation outputs and separated candidate retrieval from candidate reranking. The complete run assembled all 74 models with zero pipeline failures.

Observed candidate-generation results:

| Status | Reactions |
|---|---:|
| Nonempty candidate set (`ok`) | 2,359 |
| Empty constrained search (`no_candidates`) | 2,646 |
| No mapped participants (`unconstrained_candidate_set`) | 811 |
| Total evaluable | 5,816 |
| Candidate rows | 91,802 |

Headline retrieval results:

| Metric | Exact | BRITE/orthology-aware |
|---|---:|---:|
| Recall at any rank, reaction micro | 34.7% | 35.1% |
| Recall@1, reaction micro | 33.3% | 34.3% |
| Recall at any rank, model macro | 43.8% | 44.9% |
| Recall@1, model macro | 35.5% | 40.0% |
| Recall at any rank, cluster macro | 42.4% | 43.5% |
| Recall@1, cluster macro | 35.9% | 38.9% |

The zero-candidate rate is 59.4%. Overall exact retrieval failure is therefore approximately 65.3%, including both zero-candidate reactions and reactions whose nonempty set lacks the correct answer.

Conditional on the 2,359 reactions with nonempty candidate sets:

| Ranker | Exact Top-1 |
|---|---:|
| Existing heuristic | 82.0% |
| Lexical | 81.6% |
| Random | 78.1% |
| Oracle | 85.6% |

The key conclusion is that **retrieval, not reranking, is the dominant bottleneck**. The existing heuristic is only 3.6 percentage points below the oracle among reactions with candidates. Across the complete benchmark, the gap between exact candidate ceiling and existing Top-1 performance is only about 1.4 percentage points.

The Phase 2 reporting and reproducibility issues have been resolved. Overall retrieval failure is 65.29% of all 5,816 evaluable reactions; conditional retrieval failure is 14.41% among the 2,359 reactions with nonempty candidate sets. `BIOMD0000001063` was confirmed to contain genuine, extremely large candidate sets caused mainly by reactions with only one mapped participant, not duplicate rows or a pipeline defect. Per-reaction metrics weight each reaction once.

All Phase 2 invariants pass, aggregate artifacts and per-model caches are restorable, and verification is read-only. The snapshot is tagged `benchmark-phase2-v1`.

## New upstream integration work — immediate priority

Upstream `main` now integrates species annotation and reaction annotation into one public workflow. Commit `1611464db16a2f55c0fc02a983275cdaa21cab26` introduced the documented `annotate="species"`, `annotate="reactions"`, and `annotate="both"` modes; subsequent upstream commits adjusted ranking and messages.

This integration is strategically important because it turns the benchmarked reaction code into an end-user workflow. It must be validated before new modeling work is allowed to obscure basic product correctness.

### Required integration validation

- Reconcile current upstream `main` with the benchmark branches without losing the Phase 1–3 fixes. Do this deliberately; do not assume a clean merge because the reaction code evolved on both lines.
- Build an end-to-end test matrix for:
  - species-only annotation;
  - reaction-only annotation from ChEBI annotations already in SBML;
  - reaction-only annotation from a species recommendation DataFrame;
  - reaction-only annotation from a species recommendation CSV;
  - combined species-then-reaction annotation;
  - empty or unusable species results;
  - direct and RAG species retrieval;
  - EM enabled and disabled;
  - mocked LLM failures and malformed responses.
- Assert output schemas, species-to-reaction handoff, reaction/species IDs, saved files, metrics, and `AnnotationResult` attributes—not merely that the call does not raise.
- Add at least one complete SBML integration test with mocked LLM responses. Existing tests around commit `1611464` mainly cover helper behavior.
- Resolve the return-type inconsistency in error paths: public documentation says annotation calls return `AnnotationResult`, while some reaction failures still return a raw `(DataFrame, metrics)` tuple.
- Confirm whether direct KEGG-compound species annotations are supported through the public combined/reaction wrapper. The documentation currently says “ChEBI (or KEGG compound),” but `_load_species_recommendations()` filters inputs to ChEBI rows. Either implement and test KEGG-compound inputs or narrow the documentation.

### Documentation review for commit `1611464`

The new documentation is a useful foundation and correctly exposes the combined workflow. Before considering it final:

- Verify every example against the current public API and run examples in CI where practical.
- Document `top_k` and `n_return` separately and make clear that reaction retrieval does not use `top_k` in the same way as species retrieval.
- Put the EM runtime/default behavior in the main reaction-annotation section rather than only in advanced parameters.
- Document failure behavior, required reference files, output filenames, and the exact provenance accepted for species annotations.
- Correct small source/documentation hygiene issues found during review, including duplicated argument/assignment lines, while avoiding unrelated refactors.

## Retrieval modernization for species annotation

The current method named `rag` is a dense vector-retrieval pipeline built with ChromaDB. ChEBI, NCBI Gene, and UniProt index entries are mostly individual names/synonyms embedded with `all-MiniLM-L6-v2`; an older optional path uses `text-embedding-ada-002`. This is functional, but it should not be treated as the only modern retrieval baseline.

LunaStarr's “BM26” comment almost certainly refers to **BM25**, a lexical ranking method. BM25 is especially plausible for ontology entity linking because exact names, abbreviations, identifiers, and rare tokens can be more informative than generic semantic similarity.

### Feasible retrieval experiment

Do not replace the existing dense RAG path immediately. Implement a common retriever interface and compare:

1. Current direct dictionary/synonym matching.
2. BM25 lexical retrieval over canonical names plus synonyms.
3. Current dense embedding retrieval.
4. A simple hybrid candidate union or rank fusion of BM25 and dense retrieval.

Evaluate candidate Recall@1/3/10, latency, index size, and downstream LLM-selection accuracy separately for ChEBI, NCBI Gene, and UniProt. Use the same frozen inputs and ground truth for all methods. Only promote a new default after the comparison. Updating the dense encoder is a later option, not a prerequisite for the BM25 baseline.

## Expectation-maximization decision

The EM-style participant/reaction update loop remains active in the reaction pipeline. It defaults to five iterations but can already be disabled with `em_max_iterations=0`. Because it is expensive, the immediate goal is to determine whether it earns its runtime—not to delete it first.

### EM ablation and decision rule

- Benchmark `em_max_iterations=0`, `1`, `2`, and `5` on a small but representative set spanning model sizes and Phase 2 retrieval strata.
- Measure wall-clock time, candidate recall, Top-1 accuracy, number of recovered reactions, and any newly introduced false candidates.
- If zero or one iteration preserves essentially all useful accuracy while materially reducing runtime, make the faster setting the default and retain the iterative method as an explicitly experimental option.
- If EM provides no reproducible gain, deprecate it in the public path before considering code removal.
- Run a full-corpus EM comparison only if the small ablation shows a meaningful benefit; do not commit to another multi-day run without that evidence.

## Uncurated BioModels case studies

The new uncurated BioModels examples are external-use cases, not a replacement for the curated benchmark. Because they lack reaction-level ground truth, do not report their output as accuracy.

For each selected model:

- Record the accession/revision, selection rationale, model size, existing annotation coverage, and whether it was used during development.
- Run the frozen workflow and capture runtime, token cost, candidate coverage, abstentions, and proposed species/reaction annotations.
- Have a domain expert manually review a predefined sample, with evidence links and an explicit rubric.
- Report these as qualitative case studies or prospective annotation demonstrations.
- Keep test models chosen after method decisions separate from examples used to debug the workflow.

## Phase 3: Recover answers missing from rule-based retrieval

Phase 3 should address the observed bottleneck directly. It should not begin as a large reranking project confined to the existing candidate sets.

### Phase 3A: Open-set recovery pilot — scaffold complete, live run pending

Construct a leakage-resistant, stratified evaluation sample containing:

- `unconstrained_candidate_set` reactions
- `no_candidates` reactions
- reactions with nonempty candidate sets that omit the correct answer
- matched successful controls

Compare three modes using compact, reaction-local context:

1. **Direct LLM identification:** ask for a KEGG reaction without supplying candidates. This measures parametric knowledge or memorized biological associations.
2. **Tool-assisted recovery:** allow database search or retrieval, require evidence, and allow abstention.
3. **Existing closed-set selection:** retain the current candidate-constrained approach as a control where applicable.

The cluster-separated split and validation-only pilot have been created on `benchmark/phase-3`. The split contains 3,497 train, 969 validation, and 1,350 test reactions. The exploratory pilot contains 163 validation reactions across five retrieval strata and three bounded context variants (489 prompts). The held-out test set has not been used for method selection.

Ground-truth leakage from KEGG-shaped SBML reaction IDs was detected and fixed. The current `phase3-open-set-v3` prompts contain zero KEGG reaction IDs under the embedded-ID detector. Direct mode explicitly permits internal model knowledge while prohibiting external tools and candidate lists. Tool evidence is linked per prediction, and evidence outcomes evaluate top-1 support.

Next live step: implement and inspect a capped OpenAI runner, then run a nine-call operational smoke test using three validation reactions and all three context variants. The smoke test validates authentication, structured output, caching, token accounting, restart behavior, and budget enforcement; it is not an accuracy estimate. The current starting model is `gpt-5.6-terra`, subject to exact-version availability at run time.

### Phase 3B: Train a learned full-database retriever — important but schedule-dependent

Train a small scientific text encoder or bi-encoder to retrieve from the complete KEGG reaction database, rather than only reorder candidates generated by the existing chemical rules.

Each query may combine:

- The local SBML reaction equation
- Participant names and available identifiers
- A compact model title or pathway description
- Limited neighboring-reaction context, tested through ablation

Candidate documents may combine the KEGG equation, definition, enzyme/orthology fields, and reaction-class information. Remove or guard against KEGG-ID leakage from query text.

Use model-cluster-separated train/validation/test splits. Use chemically or textually similar KEGG reactions as hard negatives. Evaluate Recall@1/3/5/10, MRR, calibration, and results by evidence/failure stratum.

This learned retriever satisfies the goal of training a modern neural model while targeting the failure mode that Phase 2 actually revealed. It remains valuable for the career/research artifact, but it must not block the October manuscript start. If integration, EM, and retrieval validation consume September, specify the experiment and begin it in parallel with manuscript drafting rather than delaying writing.

### Phase 3C: Limited reranking study

Reranking remains useful but secondary. Evaluate it only where a correct answer is available to rank, emphasizing ambiguous or large candidate sets. Compare the existing heuristic, lexical similarity, embeddings, a trained cross-encoder, and an LLM reranker. Report both conditional improvements and their much smaller effect on overall benchmark accuracy.

## Phase 4: Routed hybrid system, uncertainty, and abstention — stretch for first manuscript

Build a system that chooses an appropriate path:

- Small, high-confidence candidate set: inexpensive deterministic or learned ranking
- Ambiguous or very large candidate set: stronger learned or LLM reranking
- Empty or answer-absent candidate set: learned full-database retrieval or tool-assisted recovery
- Insufficient evidence: abstain

Evaluate calibration, correctness detection, answer-absence detection, accuracy-versus-coverage curves, confidence thresholds, and end-to-end selective accuracy. Controlled answer-absent cases may be created by removing the correct candidate from otherwise valid candidate sets.

## Phase 5: Robustness to degraded biological information — stretch for first manuscript

Systematically test the dependence on metabolite annotations:

- Randomly remove 10%, 25%, 50%, and 75% of species annotations.
- Remove annotations from one side of a reaction.
- Replace specific ChEBI terms with ontology ancestors.
- Introduce plausible but incorrect metabolite mappings.
- Compare rule-based retrieval, ontology relaxation, learned retrieval, direct LLM recovery, and tool-assisted recovery.

## Phase 6: Package the research artifact

After the experimental design stabilizes:

- Move remaining benchmark logic out of ad hoc notebooks.
- Add fixed configurations, seeds, environment documentation, and CI.
- Cache candidate sets and LLM responses without committing secrets.
- Generate manuscript figures and tables directly from frozen outputs.
- Publish a research-oriented README, reproducible CLI, and documented release.
- Write the preprint or technical report.

## Token-efficiency principle

Do not send the entire model reaction chain separately for every reaction. That repeats model context approximately quadratically as models grow.

Use compact reaction-local prompts and route expensive calls selectively:

- Include the target reaction and normalized participants.
- Add only a small, fixed amount of model or neighboring-reaction context when an ablation shows that it helps.
- Rerank only genuinely ambiguous candidate sets.
- Use open-set or tool-assisted calls only for sampled retrieval failures initially.
- Cache every response and track input/output tokens and cost.

This preserves the ability to measure an LLM's open-set biological knowledge without making the complete evaluation unnecessarily expensive.

## Scope for an October manuscript start

Writing should begin in October even if every stretch experiment is not finished. The first manuscript needs a coherent minimum story, not every possible extension.

### Must-have before or during early October

- Verified species→reaction combined workflow with real end-to-end tests.
- Reviewed and corrected public documentation.
- Frozen Phase 1 and Phase 2 benchmark results.
- EM timing/accuracy ablation and a documented default decision.
- At least one fair species-retrieval comparison including BM25.
- OpenAI Phase 3 validation pilot, if funding and the live runner are ready.
- A fixed protocol and initial outputs for the uncurated-model case studies.
- Manuscript outline, figure/table list, methods skeleton, and ownership assignments.

### Valuable but allowed to continue during writing

- Cross-provider LLM replication.
- Learned full-catalog bi-encoder.
- Tool-assisted recovery.
- Full held-out test evaluation after all method choices are frozen.
- Expanded robustness experiments.

### Explicitly out of scope unless early results demand it

- Rebuilding every subsystem simultaneously.
- Running all model/context/tool combinations over all 5,816 reactions.
- A full-corpus EM run before the subset ablation demonstrates value.
- Treating uncurated models as quantitative ground truth.

## Feasible schedule

Assuming focused work begins now and LunaStarr can review asynchronously:

| Window | Deliverable |
|---|---|
| Sep 3–9 | Reconcile branches; review commit `1611464`; build and run combined-workflow integration tests; identify documentation corrections |
| Sep 10–16 | Run EM subset ablation; choose default; implement BM25 baseline behind a common interface |
| Sep 17–23 | Compare direct/BM25/dense/hybrid retrieval; run the capped OpenAI operational smoke test and validation pilot if approved |
| Sep 24–30 | Run initial uncurated-model case studies; freeze September results; generate core tables/figures; agree on manuscript outline |
| October | Begin writing immediately; run learned-retriever, cross-provider, and final frozen evaluations in parallel only where they strengthen the agreed paper story |

This is feasible for an October **writing start**, not necessarily an October submission. A realistic first complete draft is late October to November if collaboration and compute are available. A polished submission is more plausibly November to December. The schedule should be revisited after the integration tests and EM subset ablation, because those are the largest near-term uncertainty reducers.

## Immediate next actions

1. Ask LunaStarr for the exact uncurated BioModels accession list, her expected outputs, and what she meant by BM25/hybrid retrieval.
2. Create a short-lived integration branch from current upstream `main`; do not merge the benchmark branch blindly.
3. Turn the species→reaction handoff requirements into the end-to-end test matrix above.
4. Review documentation against observed behavior and open narrowly scoped fixes.
5. Design the EM subset and decision threshold before running it.
6. Keep the Phase 3 test split sealed while integration work proceeds.
7. Start a manuscript outline no later than the final week of September, even if some experiments remain pending.
