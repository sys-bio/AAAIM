Yes — the conference was **ISMB/ECCB 2027**, the flagship meeting of the International Society for Computational Biology. It will be held July 18–22, 2027, in Copenhagen. The proceedings call has not yet been posted, so we should treat it as the target while preserving *Bioinformatics* as the natural journal alternative. [Official ISMB/ECCB 2027 page](https://www.iscb.org/ismbeccb2027/home)

After reviewing the current `benchmark/phase-3` branch, I think the paper should focus specifically on **metabolic reaction annotation under incomplete evidence**, rather than attempting to cover every species-annotation capability in AAAIM. The strongest story is:

> Existing biochemical rules rank reactions well when they retrieve the correct answer, but they frequently fail to retrieve any usable answer. Full-catalog retrieval and selectively deployed LLMs can address this missing-candidate bottleneck, although unconstrained LLM predictions remain overconfident and unreliable.

That is a real computational-biology finding, not merely a software description or prompt-comparison paper.

## Recommended working title

**Metabolic Reaction Annotation Under Incomplete Biological Evidence: Benchmarking Rule-Based Retrieval, Learned Models, and Large Language Models**

I would avoid “uncertainty-aware” in the title for now. The current direct-LLM confidence values are badly calibrated, so uncertainty awareness is still a goal of the final routed system rather than an established result.

## Proposed paper outline

### Abstract

Use a five-part structure:

1. **Problem:** Biosimulation models frequently lack standardized reaction annotations, while existing annotation systems depend on incomplete or inconsistent metabolite identifiers.
2. **Approach:** Introduce AAAIM and a frozen BioModels-to-KEGG benchmark that decomposes reaction annotation into candidate retrieval, candidate ranking, and open-set recovery.
3. **Principal baseline result:** The benchmark contains 5,816 evaluable reactions from 74 models. Existing rule-based retrieval fails to contain the exact answer for 65.3% of reactions, whereas its heuristic chooses the correct answer for 95.8% of reactions when that answer is retrievable.
4. **New-method result:** Summarize the final BM25, dense, hybrid, trained bi-encoder, and selective-LLM results once available. The current direct-LLM validation pilot recovers roughly 26%–28% of true retrieval failures, but also produces many confident incorrect identifiers.
5. **Conclusion:** Reliable reaction annotation requires retrieval expansion and evidence-aware routing, not simply a stronger reranker.

### 1. Introduction

#### 1.1 Biological motivation

* Computational models depend on standardized ontology annotations for reuse, comparison, search, and integration.
* SBML reaction annotations are difficult because reaction names and species identifiers are frequently incomplete, model-specific, or noncanonical.
* Metabolite annotations may be absent, overly broad, or incompatible across ChEBI and KEGG.

#### 1.2 Technical gap

* Traditional rule-based systems depend on recognized participants and database mappings.
* Candidate-constrained LLM reranking cannot recover a correct reaction that candidate generation never retrieved.
* Fully open-ended LLM annotation can recover missing answers, but may hallucinate plausible-looking database identifiers.

#### 1.3 Study objective

State the central research question from the roadmap:

> Can learned retrieval, compact-context LLM inference, and tool-assisted recovery improve metabolic reaction annotation while recognizing when available evidence is insufficient?

#### 1.4 Contributions

I would claim four primary contributions:

1. A frozen, reproducible benchmark of 5,816 KEGG-annotated reactions from 74 BioModels, including cluster-based leakage controls.
2. A failure-aware evaluation framework separating:

   * unconstrained retrieval,
   * empty constrained retrieval,
   * nonempty sets missing the answer,
   * reranking failures,
   * successful retrieval and ranking.
3. A systematic comparison of deterministic, lexical, dense, learned, and LLM-based reaction-annotation methods.
4. An evidence-aware hybrid system that routes cases between efficient closed-set annotation, full-catalog retrieval, selective LLM recovery, and abstention.

The fourth contribution is the intended final system; its exact wording should depend on what the remaining experiments support.

### 2. Related Work

#### 2.1 Semantic annotation of systems-biology models

* SBML annotations and MIRIAM-style identifiers.
* Existing annotation and model-curation tools.
* Species versus reaction annotation.

#### 2.2 Biomedical entity linking and ontology retrieval

* Exact synonym matching.
* BM25 and other lexical retrieval.
* Dense bi-encoder retrieval.
* Cross-encoder reranking.
* Hybrid candidate fusion.

#### 2.3 Language models for biological annotation

* LLM-based ontology normalization and database identification.
* Retrieval-augmented versus parametric inference.
* Hallucinated identifiers, evidence attribution, and abstention.

End this section by distinguishing AAAIM: it evaluates the **entire retrieval–ranking–recovery chain under incomplete biochemical evidence**.

### 3. AAAIM System

#### 3.1 Input and output

* Input: SBML model, reaction equation, participant names, and available ChEBI or KEGG compound identifiers.
* Output: ranked KEGG reaction identifiers, scores or confidence, supporting evidence, and an abstention option.

#### 3.2 Species-to-reaction workflow

Briefly describe:

* extraction and normalization of SBML species;
* species grounding against ChEBI, NCBI Gene, or UniProt;
* conversion of metabolite annotations into reaction evidence;
* KEGG reaction candidate generation;
* ranking, recovery, and optional model updating.

The public repository already exposes species-only, reaction-only, and combined workflows. [AAAIM README](https://github.com/janisshin/AAAIM)

#### 3.3 Reaction-annotation architecture

Organize the method into three conceptually distinct layers:

1. **Candidate retrieval**
2. **Candidate ranking**
3. **Open-set recovery or abstention**

That decomposition should be visible throughout the paper.

#### 3.4 Failure taxonomy

Define the five mutually exclusive strata currently used:

* `unconstrained`
* `empty_constrained`
* `nonempty_answer_absent`
* `retrievable_rerank_failure`
* `retrievable_top1_success`

### 4. Benchmark Construction

#### 4.1 BioModels corpus

* Initial manifest: 75 models.
* Included: 74.
* Excluded for lacking usable ground truth: 1.
* Ground-truth reactions: 5,838.
* Evaluable reactions after scientific exclusions: 5,816.
* Reactions with multiple valid KEGG identifiers: 91.

#### 4.2 Ground-truth extraction and exclusions

Explain:

* KEGG URI parsing;
* preservation of multiple valid annotations;
* exchange/SSX exclusions;
* malformed-ID exclusions;
* corrections for URI-pattern and reaction-alignment bugs.

#### 4.3 Leakage-resistant partitioning

* Detect related-model clusters.
* Partition clusters—not individual reactions—into train, validation, and test.
* Current split:

  * train: 3,497 reactions;
  * validation: 969;
  * test: 1,350.
* Keep the test partition untouched until method selection is complete.

#### 4.4 Reproducibility

Describe checksums, frozen artifacts, deterministic scripts, versioned KEGG catalog, fixed seeds, and benchmark release tags. [Research roadmap](https://github.com/janisshin/AAAIM/blob/benchmark/phase-3/goals.md)

### 5. Annotation Methods

#### 5.1 Rule-based candidate generation

Describe participant mapping, chemical similarity, direction handling, cofactors/spectators, ontology relaxation, and the existing heuristic score.

#### 5.2 Closed-set ranking baselines

* Random ranking.
* Lexical similarity.
* Existing AAAIM heuristic.
* Oracle ranking, representing the candidate ceiling.
* LLM or trained cross-encoder reranking, if retained in the final experiment.

#### 5.3 Full-catalog retrieval

This will be the most important new methodological section:

* BM25 over KEGG equations, definitions, enzymes, orthology, and reaction classes.
* Off-the-shelf dense retrieval.
* BM25+dense reciprocal-rank fusion.
* AAAIM-trained KEGG bi-encoder.
* Union or fusion with rule-generated candidates.

#### 5.4 Direct open-set LLM inference

* Compact reaction-local prompt.
* No supplied candidate identifiers.
* Explicit ability to abstain.
* Three evaluated contexts:

  * target only;
  * target plus model context;
  * target plus neighboring reactions.
* Leakage controls removing KEGG-shaped identifiers.

#### 5.5 Tool-assisted recovery

Once implemented:

* LLM searches or retrieves from the frozen KEGG catalog.
* Prediction must be supported by catalog evidence.
* Unsupported or insufficient-evidence cases should be abstained from.

#### 5.6 Routed hybrid system

Specify the intended decision logic:

* reliable small candidate set → inexpensive closed-set ranker;
* missing or inadequate candidates → full-catalog retriever;
* genuinely ambiguous cases → stronger reranker or tool-assisted LLM;
* insufficient evidence → abstain.

### 6. Experimental Design

#### 6.1 Research questions

I would organize the experiments around these questions:

* **RQ1:** How often does existing candidate generation retrieve the correct reaction?
* **RQ2:** When the correct answer is retrieved, how much improvement is possible through reranking?
* **RQ3:** Can full-catalog lexical or learned retrieval recover answers missed by biochemical rules?
* **RQ4:** Can direct or tool-assisted LLMs recover additional failures?
* **RQ5:** Does model or neighborhood context improve LLM inference?
* **RQ6:** Can a routed system improve accuracy while maintaining useful coverage?

#### 6.2 Metrics

* Exact Recall@1/3/5/10.
* Equivalence-aware recall using BRITE/orthology, EC, KO, and RCLASS.
* MRR.
* Accuracy conditional on nonempty candidates.
* Overall end-to-end accuracy.
* Recovery rate within each failure stratum.
* Coverage and selective accuracy.
* Calibration and correctness-detection metrics.
* Runtime, token usage, and monetary cost.

Report reaction-micro, model-macro, and cluster-macro results.

### 7. Results

#### 7.1 The frozen benchmark exposes widespread retrieval failure

Present the corpus and strata counts.

#### 7.2 Candidate retrieval—not reranking—is the principal bottleneck

Current headline results:

* 59.4% receive no usable candidate set.
* 65.3% do not have the exact answer anywhere in the candidate set.
* Among nonempty sets, heuristic Top-1 accuracy is 82.0%.
* Among reactions where the answer is retrievable, the heuristic fails on only 4.2%.
* The overall heuristic-to-oracle gap is only about 1.5 percentage points.

This is probably the paper’s most important existing result. [Phase 2 results](https://github.com/janisshin/AAAIM/blob/benchmark/phase-3/benchmark/PHASE2_RESULTS.md)

#### 7.3 Full-catalog retrieval comparison

Insert the eventual BM25, dense, RRF, bi-encoder, and rule-plus-learned union results here.

This section should become the paper’s main positive-method contribution.

#### 7.4 Direct LLMs recover some missing candidates but remain unreliable

Current validation result:

* True retrieval-failure recovery is approximately 26%–28%, depending on context.
* Exact Top-1 over the complete 163-reaction pilot is approximately 31%.
* Many incorrect predictions are valid-looking in-catalog identifiers.
* Self-reported confidence is extremely high for both correct and incorrect predictions.

#### 7.5 Added context does not reliably improve direct inference

* Model and neighborhood context each improved raw exact accuracy by only about 0.6 percentage points after rescue.
* Cluster-bootstrap intervals cross zero.
* Both context types help some reactions and harm others.
* Target-only context is therefore the current working default because it is cheaper and not demonstrably worse. [Phase 3 validation results](https://github.com/janisshin/AAAIM/tree/benchmark/phase-3/benchmark/phase3/validation_rescue_2048/sensitivity)

#### 7.6 Tool assistance and routed recovery

Compare:

* rules alone;
* best full-catalog retriever;
* direct LLM;
* tool-assisted LLM;
* final routed system.

#### 7.7 Ablations

Include whichever are completed:

* EM iterations: 0, 1, 2, 5.
* Query fields and KEGG document fields.
* Hard-negative strategy.
* Model versus neighborhood context.
* Candidate fusion.
* Degraded species annotations.

### 8. Discussion

#### 8.1 Main interpretation

Reaction annotation is not principally a “choose among close candidates” problem. It is often a **candidate-discovery problem caused by incomplete biological grounding**.

#### 8.2 Why direct LLM inference is insufficient

* It demonstrates useful biological knowledge.
* It can recover answers excluded by deterministic rules.
* But it confidently returns nearby or plausible incorrect KEGG identifiers.
* Therefore, direct inference is best treated as a selective recovery component—not the default annotator.

#### 8.3 Implications for annotation-system design

Discuss modular evaluation, evidence-aware retrieval, routing, abstention, and the value of diagnosing where an annotation failed.

#### 8.4 Practical implications

Explain how the system can assist curators without presenting uncertain recommendations as authoritative annotations.

### 9. Limitations

* Ground truth is derived from existing BioModels annotations and may itself contain errors.
* KEGG reaction identifiers are not always one-to-one with modeled biological events.
* Some mechanistic SBML events have no appropriate KEGG reaction.
* Model clusters reduce but do not eliminate biological overlap.
* The current LLM evaluation uses one model and a limited validation sample.
* Self-reported LLM confidence is not calibrated.
* The direct-LLM pilot is validation-only and must not be represented as final held-out performance.
* Species-annotation errors propagate into reaction annotation.
* Database versions and commercial LLM behavior can change.

### 10. Conclusion

Conclude with three points:

1. AAAIM provides a reproducible benchmark and system for reaction annotation under incomplete evidence.
2. Existing chemical rules are effective rankers but incomplete retrievers.
3. The most promising direction is a hybrid system combining deterministic biochemical evidence, full-catalog learned retrieval, selectively invoked language models, and abstention.

## Proposed main figures and tables

| Item     | Content                                                                                                       |
| -------- | ------------------------------------------------------------------------------------------------------------- |
| Figure 1 | AAAIM workflow and retrieval–ranking–recovery decomposition                                                   |
| Figure 2 | Benchmark construction, exclusions, model clustering, and train/validation/test split                         |
| Figure 3 | Failure waterfall: 5,816 evaluable → 2,359 with candidates → 2,019 answer-retrievable → 1,934 heuristic Top-1 |
| Figure 4 | Recall@k for rules, BM25, dense, RRF, and trained bi-encoder                                                  |
| Figure 5 | Recovery rate by failure stratum for retrieval and LLM methods                                                |
| Figure 6 | Accuracy–coverage curve for the final routed system                                                           |
| Table 1  | Benchmark composition                                                                                         |
| Table 2  | Phase 2 candidate ceiling and ranker comparison                                                               |
| Table 3  | Full-catalog retrieval comparison                                                                             |
| Table 4  | Direct and tool-assisted LLM results                                                                          |
| Table 5  | Ablations, runtime, and cost                                                                                  |

The key organizational choice is to structure the paper around **research questions and failure modes**, not “Phase 1, Phase 2, Phase 3.” Those phases are useful project management, but the publication’s narrative should be: benchmark → diagnose bottleneck → address bottleneck → route or abstain.
