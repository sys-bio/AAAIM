# AAAIM research project: goals and roadmap

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

## Phase 2: Candidate retrieval and baseline evaluation — generation complete

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

Two Phase 2 reporting issues must be resolved before freezing the release:

1. `retrieval_fail=14.4%` in the baseline output is conditional on having a nonempty candidate set. It must be labeled separately from the approximately 65.3% overall retrieval-failure rate.
2. `BIOMD0000001063` produced 77,073 of 91,802 candidate rows. Candidate-set size and degeneracy must be characterized so a few enormous sets do not quietly dominate storage, runtime, or future LLM costs.

## Immediate milestone: audit and freeze Phase 2

Before starting new modeling work:

- Audit all generated artifacts and invariants.
- Add explicit overall-versus-conditional retrieval-failure metrics and labels.
- Analyze candidate-set size distributions and extreme candidate explosions.
- Confirm that per-reaction metrics are not distorted by the large candidate sets.
- Run the embedding baseline if its environment can be installed reproducibly.
- Freeze the complete Phase 2 artifacts with checksums and a version tag.
- Write a concise Phase 2 results report that becomes the empirical justification for Phase 3.

## Phase 3: Recover answers missing from rule-based retrieval

Phase 3 should address the observed bottleneck directly. It should not begin as a large reranking project confined to the existing candidate sets.

### Phase 3A: Open-set recovery pilot

Construct a leakage-resistant, stratified evaluation sample containing:

- `unconstrained_candidate_set` reactions
- `no_candidates` reactions
- reactions with nonempty candidate sets that omit the correct answer
- matched successful controls

Compare three modes using compact, reaction-local context:

1. **Direct LLM identification:** ask for a KEGG reaction without supplying candidates. This measures parametric knowledge or memorized biological associations.
2. **Tool-assisted recovery:** allow database search or retrieval, require evidence, and allow abstention.
3. **Existing closed-set selection:** retain the current candidate-constrained approach as a control where applicable.

The pilot should use a seeded stratified sample before any full-dataset API run. It must record prompts, responses, token usage, cost, latency, evidence, and abstention. Direct LLM guesses must be reported separately from evidence-backed tool recovery.

### Phase 3B: Train a learned full-database retriever

Train a small scientific text encoder or bi-encoder to retrieve from the complete KEGG reaction database, rather than only reorder candidates generated by the existing chemical rules.

Each query may combine:

- The local SBML reaction equation
- Participant names and available identifiers
- A compact model title or pathway description
- Limited neighboring-reaction context, tested through ablation

Candidate documents may combine the KEGG equation, definition, enzyme/orthology fields, and reaction-class information. Remove or guard against KEGG-ID leakage from query text.

Use model-cluster-separated train/validation/test splits. Use chemically or textually similar KEGG reactions as hard negatives. Evaluate Recall@1/3/5/10, MRR, calibration, and results by evidence/failure stratum.

This learned retriever satisfies the goal of training a modern neural model while targeting the failure mode that Phase 2 actually revealed.

### Phase 3C: Limited reranking study

Reranking remains useful but secondary. Evaluate it only where a correct answer is available to rank, emphasizing ambiguous or large candidate sets. Compare the existing heuristic, lexical similarity, embeddings, a trained cross-encoder, and an LLM reranker. Report both conditional improvements and their much smaller effect on overall benchmark accuracy.

## Phase 4: Routed hybrid system, uncertainty, and abstention

Build a system that chooses an appropriate path:

- Small, high-confidence candidate set: inexpensive deterministic or learned ranking
- Ambiguous or very large candidate set: stronger learned or LLM reranking
- Empty or answer-absent candidate set: learned full-database retrieval or tool-assisted recovery
- Insufficient evidence: abstain

Evaluate calibration, correctness detection, answer-absence detection, accuracy-versus-coverage curves, confidence thresholds, and end-to-end selective accuracy. Controlled answer-absent cases may be created by removing the correct candidate from otherwise valid candidate sets.

## Phase 5: Robustness to degraded biological information

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
