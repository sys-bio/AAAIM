# AAAIM research project: goals and roadmap

**Status updated:** 2026-11-07
**Current priority:** review the consolidated canonical phase branches and Luna handoff; preserve Phase 5's paid-run pause. Luna's 541-request grounded plan prefers 2,048 total output tokens under a proposed $16 cap, with no execution authorization.

## Current status and agreed sequence

This update preserves historical benchmark results and revises the roadmap using reported project progress and the user's explicit scope decisions. Current repository reports remain authoritative for exact configurations and metrics. This document does not certify a new repository verification run.

| Workstream | Status |
|---|---|
| Phase 1 frozen benchmark | Complete |
| Phase 2 rule-based retrieval baseline | Complete and frozen |
| Phase 3A direct/open-set LLM pilot | Direct validation experiments completed; the broader original multi-mode proposal is not automatically fulfilled |
| Phase 3B learned full-catalog retrieval | Bi-encoder trained; lexical, dense, and fusion comparisons performed; verify exact experiment coverage in repository reports |
| Phase 3C grounded-fusion validation pilot | Complete under its implemented scope |
| Phase 3C human audit | Complete as a scoped single-reviewer, two-pass descriptive audit |
| Phase 4 routing, uncertainty, and abstention | Bounded cached study complete; grounded method retained |
| Final method and evaluation freeze | Draft; finalize after Phase 5 and operational gates |
| Held-out test evaluation | Remains sealed; later one-time evaluation after the final freeze and run authorization |
| Phase 5 robustness experiments | Local retrieval study and runner preparation complete; matched paid comparison paused, no funds allocated or execution authorized |
| LunaStarr MODEL2507280001 request | Selected RRF retrieval verified for all 541 reactions; exact grounded plans prepared; predictions and biological review pending |
| Phase 6 packaging and manuscript | Remaining work; may proceed alongside Phase 4 |

**Agreed order:** consolidate and review the phase/task branches and Luna handoff → execute Luna's grounded plan only after separate explicit authorization → revisit Phase 5 only through a fresh budget decision → final method/evaluation freeze → separately authorized held-out evaluation → Phase 6 packaging, scope closure, and manuscript.

Phases 4 and 5 were originally stretch work; the user chose to pursue both. Phase 4 is complete under its bounded protocol. This does not make every older proposed experiment a requirement. Earlier calendar targets below are historical, not current deadlines.

### Other workstreams requiring a repository status check

Upstream integration and public API tests, documentation fixes, EM ablation/default selection, species-side retrieval comparisons, and uncurated-model case studies remain in this roadmap. Their completion is not established by closing reaction-side Phase 3C. Review evidence and assign ownership before treating them as complete or as blockers for the manuscript.

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

## Upstream integration work — completion status to verify

Upstream `main` now integrates species annotation and reaction annotation into one public workflow. Commit `1611464db16a2f55c0fc02a983275cdaa21cab26` introduced the documented `annotate="species"`, `annotate="reactions"`, and `annotate="both"` modes; subsequent upstream commits adjusted ranking and messages.

This integration is strategically important because it turns the benchmarked reaction code into an end-user workflow. Its validation remains a separate product-readiness workstream. Reconcile its current status with repository evidence; do not infer completion from the reaction benchmark experiments.

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

## Retrieval modernization across annotation types

The current method named `rag` is a dense vector-retrieval pipeline built with ChromaDB. ChEBI, NCBI Gene, and UniProt index entries are mostly individual names/synonyms embedded with `all-MiniLM-L6-v2`; an older optional path uses `text-embedding-ada-002`. This is functional, but it should not be treated as the only modern retrieval baseline. Retrieval modernization applies to species annotations across ChEBI, NCBI Gene, and UniProt and to full-catalog KEGG reaction retrieval, although each database must be evaluated separately.

LunaStarr confirmed that she meant **BM25**, a lexical ranking method, and that her concern applies to all supported annotation types. BM25 is especially plausible for ontology entity linking because exact names, abbreviations, identifiers, and rare tokens can be more informative than generic semantic similarity. The particular paper or implementation she had in mind can be collected later; it is not a prerequisite for implementing the standard baseline.

### Feasible retrieval experiment

Do not replace the existing dense RAG path immediately. Implement a common retriever interface and compare:

1. Current direct dictionary/synonym matching.
2. BM25 lexical retrieval over canonical names plus synonyms.
3. Current dense embedding retrieval.
4. A simple hybrid of BM25 and dense retrieval using reciprocal rank fusion (RRF), with candidate union retained as an auditable intermediate.

Evaluate candidate Recall@1/3/10, latency, index size, and downstream LLM-selection accuracy separately for ChEBI, NCBI Gene, UniProt, and KEGG reactions wherever labeled evaluation data are available. Use the same frozen inputs and ground truth for all methods. Only promote a new default after the comparison. Updating the dense encoder is a later option, not a prerequisite for the BM25 baseline.

BM25 remains a strong, interpretable baseline rather than an obsolete method. More elaborate retrievers such as SPLADE or ColBERT are out of the immediate scope unless BM25, dense retrieval, and their hybrid expose a clear failure that justifies the additional complexity.

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

### Phase 3A: Direct/open-set LLM validation — implemented experiments completed

The cluster-separated split contains 3,497 train, 969 validation, and 1,350 test reactions. The exploratory validation pilot contains 163 reactions across five retrieval strata. Historical scaffolding considered three bounded context variants (489 prompts); do not infer that all scaffolded conditions were executed.

Direct LLM identification was evaluated as a research probe without retrieved candidate lists. Grounded retrieval and selection were subsequently evaluated in Phase 3C. Direct, tool-assisted, and closed-set conditions must be distinguished in the paper; the original proposal does not establish that every mode or context variant was completed.

KEGG-shaped reaction-ID leakage was detected and addressed during development. Preserve the implemented leakage checks, cached responses, runner accounting, and exact experiment provenance. Refer to frozen repository reports for executed model IDs, budgets, configurations, and results rather than the older planned starting model or credit allocation.

The held-out test has not been used for method selection and must remain sealed.

### Phase 3B: Learned full-catalog retrieval — implemented study completed

A small pretrained bi-encoder was fine-tuned for full-catalog KEGG reaction retrieval. The implemented comparisons include BM25 lexical retrieval, off-the-shelf dense retrieval, learned retrieval, and reciprocal-rank fusion. The selected BM25–fine-tuned-bi-encoder fusion supplies the grounded Phase 3C candidate evidence.

Preserve model-cluster separation, query leakage guards, frozen catalog records, training/checkpoint provenance, negative-sampling details, and retrieval outputs. Use repository reports for exact encoder/checkpoint IDs, Recall@k, MRR, latency, and uncertainty estimates.

Do not assume that every proposed context ablation, rule-based candidate union, calibration experiment, or species-side retrieval comparison was performed. Identify these separately as implemented, deferred, or unverified.

### Phase 3C: Grounded-fusion validation study — complete under implemented scope

The implemented study used retrieved KEGG candidates and reaction-local evidence for constrained LLM selection or abstention on the 163-reaction validation pilot. Preserve its exact method, outputs, mechanical metrics, and reports as the Phase 4 baseline.

**Scope decision:** accept the implemented grounded-fusion validation pilot and scoped single-reviewer audit as completed Phase 3C work. The original broader comparison of heuristic, lexical, embedding, trained cross-encoder, and LLM rerankers is **not declared fulfilled**. Omitted comparisons remain documented scope limitations or future work; they are not automatically reopened.

#### Completed single-reviewer audit

The reviewer completed 60 stratified cases in each of two sequential passes:

- Pass 1: blinded assessment of source labels, without system outputs.
- Pass 2: evidence-informed assessment of system predictions and abstentions, retaining the original Pass 1 judgments.

The revised exports validated with the expected frozen IDs, digests, required fields, and identical embedded Pass 1 judgments. Two assessment inconsistencies were clarified; no other answers changed. Original exports and previous analyses were preserved.

**Single-reviewer decision:** independent second review and inter-reviewer adjudication are deferred. This explicitly supersedes the original two-reviewer plan for the present scoped audit. The two passes are not independent reviews. Do not claim inter-reviewer agreement or independently adjudicated labels.

The final analysis is recorded locally under:

`benchmark/phase3/error_audit/review_work/analysis_20261105_single_reviewer_final/`

Consult `AUDIT_SCOPE.md`, `FINAL_ASSESSMENT.md`, and `PHASE3C_CHECKLIST.md` for the decision and evidence. The reported 22 flagged cases remain findings or limitations, rather than automatic completion blockers; an insufficient-information case remains unresolved.

Audit counts describe a stratified sample. Do not generalize unweighted sample rates to the full validation set or benchmark. Do not convert reviewer judgments into corrected reference labels or corrected benchmark accuracy automatically. Disclose single-reviewer limitations, potential Pass 2 anchoring, and reviewer involvement in development where applicable.

Personal review exports and private case-level analyses remain ignored by Git. Frozen labels, predictions, and original benchmark accuracy remain unchanged.

## Phase 4: Routed hybrid system, uncertainty, and abstention — bounded study complete

Objective: reduce LLM use while retaining accuracy and coverage. The cached comparison found 93/163 correct for fusion Top-1, versus 102/163 correct and 124/163 answered for grounded selection. The broadest routing gate preserved the latter counts while simulating 142 instead of 163 LLM calls: a 12.9% reduction, below the prespecified 20% criterion. Confidence filtering lost one correct answer without saving calls. The selected method remains fusion followed by grounded selection or abstention on every reaction.

These are cached-replay findings, not a newly executed end-to-end system or a measured speedup. The development protocol below is retained for traceability, not a requirement to repeat Phase 4.

### Objective and protocol

Determine whether a simple routing policy can improve the accuracy–coverage–resource tradeoff over the frozen Phase 3C method. Write the protocol before running experiments, including hypotheses, eligible data, candidate policies, confidence signals, threshold selection, baselines, metrics, resource limits, and stopping rules.

Start with the smallest useful, interpretable routing study. Potential routes are:

- Inexpensive deterministic or learned selection for high-confidence inputs.
- Grounded LLM selection for ambiguous inputs.
- Full-catalog retrieval when rule-based evidence is inadequate.
- Abstention when available evidence is insufficient.

Only include routes supported by the implementation and available evidence. Routing may use prediction-time features; it must never use ground-truth membership, known correctness, audit flags, or labels unavailable at deployment.

### Data and development restrictions

- Keep the held-out test sealed, including test-level diagnostics.
- Fit models/calibration using training data and select methods using validation data.
- Disclose that the validation pilot and human audit have already influenced development.
- Do not use the human-reviewed sample as corrected training labels or tune specifically to its judgments.
- Reuse cached outputs where scientifically valid. Label cached policy simulations distinctly from newly executed end-to-end evaluations.
- Missing route outputs cannot be replaced by fabricated counterfactual predictions.
- Begin with local implementation, tests, cached analysis, and bounded local experiments. Paid API calls or substantial new training/inference require an explicit run scope and budget decision; prepare runners in dry-run mode first.

### Baselines and outcomes

Compare the strongest existing retriever's Top-1 output, frozen Phase 3C grounded selection, a simple confidence-threshold policy, and the proposed routing policy on aligned eligible inputs.

Evaluate where supported:

- Overall accuracy, explicitly accounting for abstentions.
- Coverage, selective accuracy, and risk–coverage curves.
- Area under the risk–coverage curve, with its definition.
- Correctness detection and reference-answer-absence detection.
- Calibration when data and sample size support it.
- Performance by retrieval stratum and model cluster.
- LLM-call frequency, token usage, latency, and cost where measured.

Keep reference-based mechanical evaluation separate from biological human judgments. Raw retrieval scores, margins, and LLM confidence are not calibrated probabilities by default.

Controlled answer-absent cases may be constructed by removing **all** accepted reference IDs from candidate evidence. Mark them synthetic and report them separately. Absence of a reference answer is not proof that no biologically defensible candidate exists.

Use cluster-aware uncertainty estimates where feasible. Limit threshold search and report validation-selection optimism. A negative result is acceptable: retain the stronger prior method if Phase 4 does not improve the stated tradeoff.

### Completion criteria

Phase 4 is complete when the scoped implementation, reproducible validation comparisons, uncertainty/resource analysis, limitations, and method-selection rationale are documented and verified. A more complex router is not required merely to claim a new component.

### Final method and evaluation freeze — after Phase 5

Finalize the method freeze only after Phase 5 and any resulting method decisions are complete. Preserve Phase 3C as an unchanged comparison baseline throughout development.

The freeze package must specify:

- Exact selected methods and baselines.
- Model/checkpoint identities, prompts, catalog/retrieval settings, routing rules, and thresholds.
- Code revision, configurations, seeds, environments, and artifact digests.
- Preprocessing, leakage protections, and split definitions.
- Primary/secondary metrics and cluster-aware comparison/uncertainty procedures.
- Failure, retry, caching, abstention, and exclusion rules.
- Hard budget controls and cost/token accounting.
- The held-out command and dry-run checks.
- A prohibition on test-driven method changes and selective reruns.

If required outputs or decisions are missing, label the package a **draft freeze** and identify the blockers. Do not claim a final freeze prematurely.

### Later held-out evaluation

Execute the planned held-out evaluation only after the final freeze and separate authorization for the concrete run and any cost. Apply the frozen protocol without tuning on test outcomes. Document operational failures and deviations transparently; rerun only under predeclared failure/retry rules.

## Phase 5: Robustness to degraded biological information — retrieval complete; grounded comparison pending

Objective: measure dependence on metabolite identifiers while preserving names, equation text, and stoichiometry. The local study covered 163 validation reactions across 19 conditions (3,097 aligned query conditions), recomputing strict-rule, BM25, trained-encoder, and fusion retrieval.

At nominal 75% identifier-bundle removal, mean Recall@10 changed from 73.6% to 55.2% for BM25, 74.2% to 74.4% for the trained encoder, and 81.6% to 73.0% for fusion. This stability concerns identifier loss with names retained, not loss of all biological evidence. Report realized doses, unchanged cases, uncertainty, and the development-pilot sampling limitation.

The selected end-to-end method is unchanged. Retrieval findings alone do not establish grounded accuracy or abstention robustness.

The request-control finding is resolved by a matched `phase5-grounded-no-provenance-v2` design: accurate operational provenance is retained in sidecars, and both arms are rebuilt from the same frozen retrieval artifact and serializer. Historical clean requests are not a compatible baseline; all 326 matched requests are new calls. The historical GPT-5.6 plan/results are preserved, not reused. By user direction, the active plan is GPT-6.1 Terra in both arms, with a new model-specific cache namespace and plan digest. Public API/schema support is documented; account-level availability remains unverified.

The local review verified perturbation reconstruction and saved retrieval metrics; GPU inference was not independently rerun. Phase 5 remains incomplete pending a valid grounded comparison and analysis.

The roadmap perturbation families are:

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

## Additional docket: Luna's MODEL2507280001 request

- [x] Inspect and reconcile the historical case-study branch and original 541-reaction SBML.
- [x] Keep the public ChEBI-to-KEGG rule-based diagnostic separately named; its 222 candidate-bearing and 319 no-candidate outcomes do not exclude cases from RRF.
- [x] Verify the selected full-catalog BM25 + trained epoch-1 bi-encoder RRF rankings and all 541 reconstructed query texts; retain Top-10 evidence for every reaction.
- [x] Prepare exact GPT-6.1 Terra grounded requests with accurate sidecar provenance and unit-specific cache identities.
- [x] Assess 512 versus 2,048 total output tokens; prefer 2,048 because GPT-5.6 observations do not validate reducing GPT-6.1's output budget. Proposed 541-attempt reservation is USD 15.010828750 under a USD 16 cap, not authorized; current credit of USD 19.97 is user-reported and independently unverified.
- [x] Prepare the local collaborator handoff and keep it unsent.
- [ ] Produce grounded predictions only after explicit authorization, then review selections, abstentions and failures separately. Prior support judgments are descriptive, not independent ground truth or benchmark accuracy.

The canonical task branch is `feature/luna-model2507280001`. Physical worktree paths and ignored request/case artifacts are preserved. The held-out benchmark remains sealed; Phase 5 receives no allocation. See `benchmark/BRANCHES.md` for branch ownership and archival recovery.

## Model-version decision before further paid runs

The selected model for the next matched clean/degraded validation comparison is `gpt-6.1-terra`, by user decision on 2026-11-06. The active dry-run package is `benchmark/phase5/validation/persistent_runner_gpt61_v1/`, plan digest `c1c8cbec4df627983efc235d52c2a0e248689cf068ebc153327e0fdf7e3e95b0`; the matched request plan is `benchmark/phase5/validation/matched_prompt_gpt61_v1/`. It covers 163 pairs / 326 calls and 317 distinct bodies. GPT-5.6 plans and results remain preserved as model-specific historical artifacts; their responses and authorization do not transfer.

Budget proposal for the new plan: public Standard estimates are $1.74 at estimated input plus historical mean output, $2.12 assuming all default 30-minute cache writes at that output scenario, or $5.22 with all cache writes and maximum 2,048-token outputs. The persistent runner reserves $4.640720 per dispatch against maximum documented input/output exposure; the all-326 full-reservation scenario is $1,512.87. Proposed hard cap: $40; currently authorized: $0. The cap can stop early and is not a guarantee all calls complete. Recheck pricing and obtain separate authorization for this exact digest before any paid execution.

Official short-context Standard prices checked 2026-11-06: GPT-5.6 Terra $2 input/$12 output per million tokens; GPT-6.1 Terra $1 input/$5 output per million tokens. Actual task cost also depends on token usage, caching, tier and endpoint modifiers.

- [x] Prepare and dry-run the bounded GPT-6.1 Terra comparison with identical matched-arm settings/schema and fresh clean/degraded requests. Do not execute until separate authorization for this exact plan and budget.
- [ ] A model switch changes the scientific method. Preserve old results as model-specific historical findings.
- [ ] Retain unchanged benchmark, rule-based/BM25/encoder/fusion results and completed human audit; these need not be repeated merely because the generator changes.
- [ ] Evaluate the selected new generator on the full clean 163-case validation pilot and the matched degraded condition for Phase 5. Both paired conditions must use the same selected model and serialization.
- [ ] Recompute or explicitly retire model-dependent routing/confidence conclusions if claiming them for the new model. Do not transfer old LLM confidence/results by assumption.
- [ ] Revalidate schema, API settings, cache/ledger keys, identity policy, cost reservations, request-plan digest and final freeze. Old-model authorization and responses cannot silently authorize or supply a new-model run.
- [ ] Keep held-out evaluation sealed until final model selection and freeze. A model change before unsealing does not require rerunning an already-unexecuted test.

## Project completion after Phase 5

**Finishing Phase 5 completes the currently selected development experiments, not the entire project.** No further exploratory phase is automatically required. The remaining work is final evaluation, reproducibility, scope closure, and writing. Negative routing results and robustness limitations can be reported without starting another optimization cycle.

### Gate 1: Close development and reporting

- [x] Resolve Phase 5 prompt controls and prepare the valid matched clean/degraded request design.
- [x] Integrate and verify the persistent validation budget/cache runner using dry runs first.
- [ ] Obtain authorization for the concrete validation request plan and hard budget, then execute and analyze it.
- [ ] Report paired grounded correctness, coverage, selective accuracy, and reference-based abstention with clear denominators and uncertainty.
- [ ] Finalize Phase 5 results, limitations, and experiment-coverage checklist.
- [ ] Add the Phase 3C support-metric clarification: direct-mode in-catalog guesses without retrieved evidence and grounded out-of-evidence predictions have different definitions. The “130 → 0” count does not establish elimination of 130 biological errors or hallucinations. Preserve the independently measured 102/163 versus 93/163 accuracy result.
- [ ] Preserve frozen historical artifacts; publish corrections in versioned amendments or addenda.

The single-reviewer audit is already complete under its agreed scope. Independent second review and further adjudication are deferred, not completion blockers.

### Gate 2: Final method and evaluation freeze

- [ ] Pin selected methods/baselines, checkpoint and tokenizer, provider identity/drift policy, prompts, serialization, retrieval settings, thresholds if any, configurations, and code/artifact digests.
- [ ] Integrate a working test executor with label separation, durable caches, persistent attempt/cost accounting, and fail-closed preflight checks.
- [ ] Freeze primary/secondary metrics, cluster-aware uncertainty, eligibility/exclusions, failure/compliance handling, abstentions, and restart/retry rules.
- [ ] Verify resources, availability, and pricing; obtain authorization for the concrete held-out run and budget.
- [ ] Pass synthetic and permitted validation dry-run checks before test execution.

Keep the freeze explicitly draft until its gates pass. Do not inspect held-out reactions, labels, predictions, or diagnostics during method development.

### Gate 3: Held-out evaluation

- [ ] Execute the planned one-time evaluation after the final freeze and run authorization.
- [ ] Freeze predictions before joining labels and computing final results.
- [ ] Report all eligible cases, failures, abstentions, resource use, and protocol deviations.
- [ ] Compare against frozen baselines with prespecified metrics and uncertainty.
- [ ] Make no test-driven method/prompt/threshold changes or selective reruns; operational retries follow the frozen rules.

The stratified 163-reaction development pilot does not replace the independent final evaluation or establish population-wide accuracy.

### Gate 4: Resolve broader scope and ownership

For each older workstream, record an evidence path and owner, or an explicit deferral and its implication for paper claims. Do not automatically launch more experiments.

| Workstream | Required closure |
| --- | --- |
| Combined species-to-reaction workflow and API tests | Verify handoff/error behavior or limit product integration claims |
| Public documentation and examples | Confirm consistency with the supported interface |
| EM ablation and default decision | Verify evidence or explicitly exclude this comparison/default optimization |
| Species/gene/protein retrieval comparisons | Agree with LunaStarr on ownership, available results, and combined-paper scope |
| Uncurated BioModels case studies | Include as qualitative demonstrations or defer; do not claim accuracy without ground truth |
| Broad reranker comparison and other extensions | Document implemented coverage and deferred proposals |

Scope closure may mean documenting a justified exclusion rather than running an experiment. Agree on reaction-side versus combined AAAIM manuscript scope with LunaStarr.

### Gate 5: Phase 6 packaging and manuscript

- [ ] Provide reproducible commands, configurations, seeds, environment/dependency documentation, provenance, and artifact verification.
- [ ] Document checkpoint/tokenizer restoration, catalog access, and cache requirements while protecting credentials and private review exports.
- [ ] Complete meaningful supported-pipeline tests and integration fixes needed for the actual release claims.
- [ ] Generate final figures/tables from frozen outputs and reconcile every quantitative claim with its source.
- [ ] Complete methods, results, discussion, limitations, availability statements, author contributions, and collaborator review.
- [ ] Prepare the research README and release/archive materials; actual publication/release actions require separate authorization.
- [ ] Finish the agreed preprint or technical report. Journal acceptance is not a prerequisite for finishing the development and writing deliverables.

**Project completion criterion:** the scoped development study/audit are documented; final held-out evaluation is executed and reported; broader workstreams are resolved or explicitly deferred; reproducible artifacts are ready; and the agreed manuscript/report is complete. Publication and deployment are separately chosen milestones.

## Token-efficiency principle

Do not send the entire model reaction chain separately for every reaction. That repeats model context approximately quadratically as models grow.

Use compact reaction-local prompts and route expensive calls selectively:

- Include the target reaction and normalized participants.
- Add only a small, fixed amount of model or neighboring-reaction context when an ablation shows that it helps.
- Rerank only genuinely ambiguous candidate sets.
- Use open-set or tool-assisted calls only for sampled retrieval failures initially.
- Cache every response and track input/output tokens and cost.

This preserves the ability to measure an LLM's open-set biological knowledge without making the complete evaluation unnecessarily expensive.

## Historical manuscript scope — September plan, status to reconcile

The following priorities preserve the earlier plan for traceability. They are not evidence of current completion or current deadlines. The first manuscript needs a coherent minimum story; reconcile these items with repository reports and collaborator ownership while Phase 4 proceeds.

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

## Historical schedule — superseded

Assuming focused work begins now and LunaStarr can review asynchronously:

| Window | Deliverable |
|---|---|
| Sep 3–9 | Reconcile branches; review commit `1611464`; build and run combined-workflow integration tests; identify documentation corrections |
| Sep 10–16 | Run EM subset ablation; choose default; implement BM25 and RRF behind a common retriever interface |
| Sep 17–23 | Compare direct/BM25/dense/RRF retrieval; run the capped OpenAI operational smoke test and validation pilot once credits arrive |
| Sep 24–30 | Run initial uncurated-model case studies; finalize the KEGG bi-encoder protocol; freeze September results; generate core tables/figures; agree on manuscript outline |
| October | Begin writing immediately; train/evaluate the KEGG bi-encoder and run selected final evaluations in parallel where they strengthen the agreed paper story |

This is feasible for an October **writing start**, not necessarily an October submission. A realistic first complete draft is late October to November if collaboration and compute are available. A polished submission is more plausibly November to December. The schedule should be revisited after the integration tests and EM subset ablation, because those are the largest near-term uncertainty reducers.

## Immediate next actions — current

1. Review the consolidated canonical phase branches and the separate Luna task branch using `benchmark/BRANCHES.md`.
2. Keep Luna's 541-request GPT-6.1 Terra plan at 2,048 total output tokens; its proposed USD 16 cap requires separate authorization before any paid execution.
3. Keep Phase 5 paid execution paused and preserve every model-specific plan and completed result.
4. Resolve broader scope/ownership, final freeze, later authorized held-out evaluation, and Phase 6 packaging/writing without launching new experiments during branch cleanup.
5. Keep the held-out benchmark sealed and the collaborator handoff unsent unless separately authorized.

### LunaStarr MODEL2507280001 request — local status, updated 2026-11-07

- The existing 541-reaction Phase 3B report is a separate research benchmark review, not a run of the current public `annotate="reactions"` workflow. Its historical source model parses identically to the current use-case SBML (593 species, 541 reaction IDs); the raw worktree hash differs from the Git blob because of CRLF conversion. No BioModels revision label is recorded.
- The current public reaction workflow has been exercised through `annotate_model()` with the actual SBML and cached 552-row species recommendation CSV, while local candidate retrieval and the provider were mocked. Output/result alignment, CSV serialization, and `max_completion_tokens` forwarding passed; this is plumbing verification, not a prediction run.
- Full local candidate generation is complete in the isolated current-upstream worktree. Runtime was 150.451 seconds, including 138.691 seconds in repeated catalog/ontology filtering (1,797 filter calls; 21,321 unique side expansions). A run-scoped cache preserved candidate signatures across empty, ambiguous, and large fixtures. A positional reaction-ID alignment defect exposed by extraction omissions was fixed. `max_completion_tokens` reaches both ranking stages.
- Of 541 reactions, 518 were extracted; 23 had no mapped participant species. The output recommendation table has 3,789 rows, including 296 blank placeholders. Actual candidates total 3,493 across 222 reactions; 319 reactions have zero candidates (296 extracted without a candidate ID plus 23 not extracted). Candidate-count distribution: 0:319, 1:146, 2:32, 3:2, 4:1, 5:3, 6:1, 7:4, 8:3, 9:6, 10:2, 11:3, 12:1, 13:3, 15:5, 18:1, 19:2, 28:3, 112:1, 443:1, 1,136:2. These counts do not make biological judgments.
- No-call construction through the real ranker generated 223 requests in each mutually exclusive branch: one meaningfulness call plus 222 reaction ranking calls. Complete payloads and hashes, input-size bounds, and cost estimates are in ignored artifacts at `benchmark/integration_worktrees/luna_public/tests/useCase/results/MODEL2507280001/public_reaction_audit/`. The approximately $20 API credit was not verified. Existing five-retry policy has a pessimistic TRUE-context bound of $46.13 uncached / $110.19 at 12-hour cache-write prices; even one attempt is up to $18.37 under that cache-write bound. No paid calls or held-out/external biological access occurred.
- The historical human-review summary remains descriptive rubric evidence (e.g. 182/264 strict Top-1 support for selected fusion), not independent gold-standard accuracy. Do not describe mocked output as a biological prediction or alter frozen artifacts.

## Revision record

- 2026-09-03: Original roadmap status date; historical experiments and calendar targets retained where useful.
- 2026-11-05: Updated implemented Phase 3 progress; accepted scoped Phase 3C completion; documented the single-reviewer audit decision; selected Phase 4 as next work; placed final method/evaluation freeze after Phase 4 and before the later held-out run.

- 2026-11-06: Recorded completed Phase 4, local Phase 5 retrieval results and pending controls, and explicit completion gates after Phase 5. This roadmap edit authorizes no new paid runs or test access.
- 2026-11-06: Selected GPT-6.1 Terra for the Phase 5 matched run. Preserved GPT-5.6 plans/results; prepared and dry-run verified the new 326-call paired plan and model-specific persistent budget/cache package. Proposed USD 40 cap; USD 0 authorized. No paid or held-out execution occurred.
- 2026-11-06: Recorded LunaStarr request assessment. Existing research review is reusable descriptive evidence, but the current public reaction workflow has only had mocked integration coverage; full candidate generation and any actual public reaction ranking remain outstanding. No paid calls or held-out access.
- 2026-11-07: Completed the full local public reaction candidate stage, profiled and optimized repeated ontology expansion, corrected reaction-ID alignment, and froze complete no-call GPT-6.1 Terra request branches with pessimistic cost accounting. Targeted tests: 53 passed. Actual inference and biological review remain pending separate authorization and a verified spend guard; this work does not complete LunaStarr's full prediction/review request.
- 2026-11-07: Corrected the LunaStarr MODEL2507280001 research analysis to use the selected full-catalog BM25 + trained epoch-1 bi-encoder RRF Top-10 pipeline, separately from public rule-based diagnostics. Reused archived retrieval outputs after validating the inference archive/checkpoint, catalog, SBML identity, all 541 rebuilt query texts, component ranks, RRF Top-100, and Top-10 evidence. Prepared 541 exact GPT-6.1 Terra grounded requests; no calls or held-out access. Estimated pessimistic reservation USD 10.855272500, proposed USD 12 cap conditional on verified available balance, not authorized. Candidate retrieval and request preparation are complete; actual grounded predictions and biological review remain pending.
- 2026-11-07: Reassessed Luna's output budget: retain the established 2,048-token total output limit because no GPT-6.1 grounded usage validates reducing it. Preserved the 512-token plan and prepared both options from identical frozen inputs. Preferred 541-attempt plan reserves USD 15.010828750 under a proposed USD 16 cap (not authorized); current USD 19.97 credit is user-reported, independently unverified, leaving USD 3.97 unallocated at that cap. Phase 5 stays paused with no allocation. Exact plans, usage assessment, failure accounting and execution proposal are under the ignored isolated-worktree `selected_rrf_output_budget_20261107/` directory; actual grounded predictions and review remain pending.

- 2026-11-07: Consolidated duplicate local integration branches into the canonical phase histories and renamed the Luna branch to `feature/luna-model2507280001`. Preserved all original tips in archive refs plus a verified local bundle, all frozen experiment artifacts, and ignored personal reports. Upstream review branches remain untouched. This cleanup authorizes no paid execution or held-out access.
