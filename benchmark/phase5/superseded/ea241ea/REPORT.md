# Phase 5 robustness results and experiment coverage

Status: local retrieval study complete; Phase 5 overall remains blocked on the prespecified degraded grounded-LLM validation comparison. Selected method is unchanged: BM25 + epoch-1 trained bi-encoder RRF Top-10 followed by grounded selection/abstention for every reaction. No retraining, adaptation or policy retuning occurred.

## Population and executed work

Exactly 163 frozen validation reactions, 19 aligned query conditions (3,097 total): clean, four random removal fractions with three seeds each, reactant-side removal, product-side removal, one-hop ancestor replacement, and three sibling-corruption seeds. This is not all 969 validation reactions or a held-out evaluation. The 609 annotated participant bundles are query-level occurrences, not 609 unique model species; there are 1,306 distinct identifier links across these occurrences.
All variants were actually rebuilt and reranked for strict rules, BM25, the trained encoder and the selected fusion. Only byte-identical query/mapping inputs were memoized; 1,573 unique query texts were encoded. One fixed catalog encoding was shared because the catalog did not change. No clean rankings were assigned to changed inputs.

| Method | Clean R@1 | Clean R@10 | Clean empty rate |
| --- | ---: | ---: | ---: |
| rule_strict | 0.1534 | 0.1656 | 0.6564 |
| bm25 | 0.5215 | 0.7362 | 0.0000 |
| trained | 0.6012 | 0.7423 | 0.0000 |
| fusion | 0.5706 | 0.8160 | 0.0000 |

The strict-rule control uses existing core matching/scoring with direct mappings and no iterative ontology relaxation. It is not the historical ontology-relaxed Phase 2 configuration. Its clean baseline is explicitly recomputed here and all its deltas are paired against that baseline.
BM25 reproduces all 163 historical Top-100 lists. The encoder reproduces all historical Top-1 and Top-10 lists; one Top-100 tail differs due to the disclosed numerical inference path. Fusion Top-10 historical agreement is recorded in historical_clean_comparison.json. Comparisons below use recomputed clean results throughout, so no historical population or precision mix is hidden.

## Paired robustness findings

| Method | Perturbation family | Replicates | R@10 | Change vs clean | 95% interval for change |
| --- | --- | ---: | ---: | ---: | --- |
| rule_strict | remove_0.1 | 3 | 0.1820 | +0.0164 | [+0.0000, +0.0438] |
| rule_strict | remove_0.25 | 3 | 0.2249 | +0.0593 | [+0.0289, +0.1228] |
| rule_strict | remove_0.5 | 3 | 0.2679 | +0.1022 | [+0.0413, +0.1580] |
| rule_strict | remove_0.75 | 3 | 0.1759 | +0.0102 | [-0.0901, +0.0864] |
| rule_strict | reactants_1 | 1 | 0.2515 | +0.0859 | [+0.0413, +0.1397] |
| rule_strict | products_1 | 1 | 0.2761 | +0.1104 | [+0.0058, +0.2661] |
| rule_strict | ancestor_1 | 1 | 0.0736 | -0.0920 | [-0.1932, -0.0309] |
| rule_strict | wrong_0.25 | 3 | 0.0573 | -0.1084 | [-0.2477, -0.0411] |
| bm25 | remove_0.1 | 3 | 0.7342 | -0.0020 | [-0.0121, +0.0000] |
| bm25 | remove_0.25 | 3 | 0.6933 | -0.0429 | [-0.0870, -0.0157] |
| bm25 | remove_0.5 | 3 | 0.6299 | -0.1063 | [-0.2629, -0.0285] |
| bm25 | remove_0.75 | 3 | 0.5521 | -0.1840 | [-0.3723, -0.0746] |
| bm25 | reactants_1 | 1 | 0.6687 | -0.0675 | [-0.1556, -0.0242] |
| bm25 | products_1 | 1 | 0.6442 | -0.0920 | [-0.2614, -0.0187] |
| bm25 | ancestor_1 | 1 | 0.7362 | +0.0000 | [+0.0000, +0.0000] |
| bm25 | wrong_0.25 | 3 | 0.7157 | -0.0204 | [-0.0569, +0.0073] |
| trained | remove_0.1 | 3 | 0.7485 | +0.0061 | [+0.0000, +0.0122] |
| trained | remove_0.25 | 3 | 0.7485 | +0.0061 | [+0.0000, +0.0122] |
| trained | remove_0.5 | 3 | 0.7464 | +0.0041 | [-0.0114, +0.0117] |
| trained | remove_0.75 | 3 | 0.7444 | +0.0020 | [-0.0144, +0.0113] |
| trained | reactants_1 | 1 | 0.7362 | -0.0061 | [-0.0280, +0.0000] |
| trained | products_1 | 1 | 0.7485 | +0.0061 | [+0.0000, +0.0120] |
| trained | ancestor_1 | 1 | 0.7423 | +0.0000 | [+0.0000, +0.0000] |
| trained | wrong_0.25 | 3 | 0.7423 | +0.0000 | [+0.0000, +0.0000] |
| fusion | remove_0.1 | 3 | 0.8160 | +0.0000 | [+0.0000, +0.0000] |
| fusion | remove_0.25 | 3 | 0.8078 | -0.0082 | [-0.0705, +0.0390] |
| fusion | remove_0.5 | 3 | 0.7485 | -0.0675 | [-0.2585, +0.0246] |
| fusion | remove_0.75 | 3 | 0.7301 | -0.0859 | [-0.2920, +0.0153] |
| fusion | reactants_1 | 1 | 0.7914 | -0.0245 | [-0.0943, +0.0147] |
| fusion | products_1 | 1 | 0.7485 | -0.0675 | [-0.2825, +0.0294] |
| fusion | ancestor_1 | 1 | 0.8282 | +0.0123 | [+0.0000, +0.0377] |
| fusion | wrong_0.25 | 3 | 0.7935 | -0.0225 | [-0.0571, -0.0023] |

The trained encoder is descriptively most stable under identifier removal (R@10 near its clean 0.7423 across all four levels). BM25 loses substantial R@10 at high removal. Fusion starts from higher clean R@10 and degrades less than BM25 in point estimates, but its intervals are wide and often include zero. paired_method_robustness.csv provides paired differences in changes between methods; these distinguish stability from absolute retrieval performance. Do not infer universal superiority from one curve or overlapping intervals. These are reused-development-pilot results, not an independent final test.
Removing constraints can improve strict-rule candidate recall by admitting more reactions; this is not proof that less biological evidence improves annotation. Candidate counts, emptiness, R@1/3/5/10/100 and MRR@10/100 must be read together. Full-catalog candidate recall is query-independent catalog coverage, not success at a useful rank.
Names, equation text, stoichiometry and other participant evidence remain present. Neural stability here does not establish robustness to loss of all biological information. Ancestor replacement retains direct KEGG IDs and many changes are consequently masked. Sibling corruptions are synthetic graph-constrained perturbations, not independently verified biological misannotations.

## Realized doses and unchanged cases

| Condition | Changed bundles / eligible | Removed identifier links / original | Unchanged queries / 163 | Selected but unchanged bundles |
| --- | ---: | ---: | ---: | ---: |
| clean | 0/609 | 0/1306 | 163/163 | 0 |
| remove_0.1_s20261106 | 49/609 | 104/1306 | 114/163 | 0 |
| remove_0.1_s20261107 | 49/609 | 87/1306 | 114/163 | 0 |
| remove_0.1_s20261108 | 49/609 | 102/1306 | 114/163 | 0 |
| remove_0.25_s20261106 | 168/609 | 373/1306 | 10/163 | 0 |
| remove_0.25_s20261107 | 168/609 | 335/1306 | 10/163 | 0 |
| remove_0.25_s20261108 | 168/609 | 357/1306 | 10/163 | 0 |
| remove_0.5_s20261106 | 345/609 | 769/1306 | 0/163 | 0 |
| remove_0.5_s20261107 | 345/609 | 722/1306 | 0/163 | 0 |
| remove_0.5_s20261108 | 345/609 | 762/1306 | 0/163 | 0 |
| remove_0.75_s20261106 | 477/609 | 1027/1306 | 0/163 | 0 |
| remove_0.75_s20261107 | 477/609 | 1009/1306 | 0/163 | 0 |
| remove_0.75_s20261108 | 477/609 | 1037/1306 | 0/163 | 0 |
| reactants | 299/609 | 643/1306 | 8/163 | 0 |
| products | 310/609 | 663/1306 | 2/163 | 0 |
| ancestor | 324/609 | 435/1306 | 68/163 | 285 |
| wrong_0.25_s20261106 | 78/609 | 270/1306 | 91/163 | 90 |
| wrong_0.25_s20261107 | 84/609 | 240/1306 | 85/163 | 84 |
| wrong_0.25_s20261108 | 82/609 | 257/1306 | 88/163 | 86 |

Random removal uses per-query half-up rounding, so nominal 10/25/50/75% corresponds to 8.05/27.59/56.65/78.33% of annotated bundles in this pilot. Each seed selects the same count but can remove different numbers of identifier links. One-sided removal has 8 reactant-side and 2 product-side unchanged/ineligible queries, retained in denominators. No shared-both-sides participant was selected in this pilot; the implementation/test handles that case. Ancestor replacement changes 324/609 bundles with 68 unchanged queries; 285 changed bundles retain direct KEGG evidence. Sibling replacement changes 78/84/82 bundles of 609 (12.81/13.79/13.46%), with 90/84/86 selected attempts unable to change; no outcome-driven refill is performed.
The full event-level perturbation_manifest.jsonl records old/new IDs and one-hop distances or no-replacement outcomes. No species names, stoichiometry, sides or equations were altered. SBML and ontology/mapping source bytes remain unchanged.

## Uncertainty and limitations

Two thousand bootstrap draws, seed 20261106, pair clean and degraded query outcomes while resampling model clusters and stochastic perturbation replicate indices. The same 163 query denominator is used across conditions; three seeds are not 489 independent reactions. Deterministic side/ancestor conditions have only cluster variation. There are 12 clusters, three stochastic seeds and a development-stratified pilot; intervals are descriptive, not population-wide error bars or multiplicity-corrected confirmatory evidence. Individual-seed minima/maxima and stratum/cluster denominators are exported. Corruption generation never reads reference membership, audit flags or known correctness.

## Missing end-to-end comparison and concrete resource decision

The written primary end-to-end condition is remove_0.5_s20261106 on the same 163 cases. The dry plan contains 163 new requests and 0 canonically identical clean-response reuses. No paid calls were made and no LLM predictions were fabricated for changed requests. Retrieval recall is not grounded accuracy or abstention robustness.
At current public Standard short-context prices for the unchanged model, the input-estimate/historical-mean-output scenario is $2.60; the scenario allowing all 2,048 output tokens is $6.33. Those estimates are not guarantees. A 32,768-input-unit pessimistic reservation is $14.69 for this plan; proposed hard cap $15, one attempt per missing request, zero currently authorized. No cache discount or regional modifier is assumed. [Official OpenAI API pricing](https://developers.openai.com/api/docs/pricing) was fetched on 2026-11-06; rates are recorded separately without changing old snapshots.
Expected value of the proposed run: measure paired grounded correctness, coverage, selective accuracy and reference-based abstention under 50% identifier removal. The decision required is authorization of this exact validation-only request plan and $15 maximum, with the selected model/settings unchanged. The CLI is dry-run-only; execute remains disabled. A persistent approved provider/cache/budget adapter must be integrated before live work; no new held-out executor is enabled.
Iterative ontology relaxation, direct LLM perturbation runs, other grounded perturbation cells, all-969 expansion and retraining are deferred optional extensions. Required comparisons are not silently replaced by clean cached outputs.

## Completion and next step

See COMPLETION_CHECKLIST.md and freeze/evaluation_freeze.json. Phase 5 retrieval work is complete, overall Phase 5 is not complete while the required degraded grounded comparison is blocked. Final method selection remains the Phase 4 grounded-fusion method; no adaptation is justified by this retrieval-only study. Resolve the bounded validation run before finalizing the method/evaluation freeze, then integrate/authorize a separate later one-time held-out evaluation. The held-out test stays sealed.
