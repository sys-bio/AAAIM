# Formal 60-reaction audit sampling report

## Freeze provenance

The formal sample was built from Prompt 1 commit `46844578de89df7db4d0deca258be4f33ccd2ead` using `phase3-formal-audit-sampling-v1`. The Prompt 1 manifest digest at that commit was `28d4f25aa1cd705f83ed6cc1c395fd48171956292e113903c37080ef88787c73`; all individual Prompt 1 artifact digests are frozen in `sampling_config.json`. Sampling seed: `20260910`. Blinded-order seed: `20260911`, derived as sampling seed plus one.

All Phase 1, Phase 2, Phase 3A, retrieval-baseline, Phase 3B, Phase 3C, and Prompt 1 digest gates passed before sampling. No held-out test labels or rows were loaded.

## Prespecified algorithm

Random controls were selected first from all 163 validation reactions using only stable identity, cluster, model, and split fields. Clusters are visited round-robin by their lowest current representation; seeded SHA-256 breaks ties. Within a cluster, the least represented model is preferred, followed by a seeded audit-ID tie-break. Outcome, correctness, abstention, disagreement, suspicion, and catalog fields cannot affect random-control selection. Within each Phase 3C failure priority pool, corrected stratum and seen/unseen representation are additional tie-break balances after cluster and model.

Primary assignment order was: `random_control`, `manual_nomination_or_documented_edge_case`, `phase3c_failure_harm`, `cross_method_disagreement`, `mechanically_suspicious`. A first assignment always wins. Later categories exclude selected audit IDs and refill deterministically. Phase 3C cases exhaust the seven priority pools in the configured order. Disagreement and suspicion categories take one balanced case per available subtype/rule before a cluster-balanced refill. No balance constraint required relaxation.

## Eligibility before selection

- Validation population: 163.
- Raw Phase 3C non-exact eligibility: 61.
- Raw cross-method disagreement eligibility: 151.
- Raw mechanical-suspicion eligibility: 50.
- Previously documented unique validation edge cases: 12.
- Failure-type availability: `{"evidence_compliance_case":2,"fusion_top1_correct_grounded_abstained":6,"fusion_top1_correct_grounded_incorrect":1,"other_grounded_nonexact":17,"truth_absent_fusion_top10_grounded_incorrect_in_evidence":12,"truth_fusion_ranks2_10_grounded_abstained":15,"truth_fusion_ranks2_10_grounded_incorrect":9}`.
- Disagreement-subtype availability: `{"bm25_correct_trained_incorrect":16,"exact_mismatch_brite_orthology_match":43,"fusion_correct_components_differ":24,"fusion_correct_grounded_harms":7,"fusion_incorrect_grounded_recovers":16,"grounded_correct_phase3a_incorrect":67,"large_ground_truth_rank_change":134,"one_method_abstains_another_correct":20,"phase3a_correct_grounded_incorrect":14,"trained_correct_bm25_incorrect":29}`.
- Mechanical-rule availability: `{"exact_mismatch_resolved_by_frozen_brite_orthology":43,"ground_truth_id_absent_from_frozen_catalog":5,"multiple_ground_truth_kegg_ids":6}`.

Stage eligibility after first-assignment deduplication:

- `random_control`: 163 raw, 0 already assigned, 163 available, 20 selected.
- `manual_nomination_or_documented_edge_case`: 12 raw, 0 already assigned, 12 available, 5 selected.
- `phase3c_failure_harm`: 61 raw, 11 already assigned, 50 available, 15 selected.
- `cross_method_disagreement`: 151 raw, 37 already assigned, 114 available, 10 selected.
- `mechanically_suspicious`: 50 raw, 20 already assigned, 30 available, 10 selected.

Failure subtype availability at its selection stage: `{"evidence_compliance_case":2,"fusion_top1_correct_grounded_abstained":5,"fusion_top1_correct_grounded_incorrect":1,"other_grounded_nonexact":15,"truth_absent_fusion_top10_grounded_incorrect_in_evidence":10,"truth_fusion_ranks2_10_grounded_abstained":11,"truth_fusion_ranks2_10_grounded_incorrect":7}`. Disagreement subtype availability at its selection stage: `{"bm25_correct_trained_incorrect":13,"exact_mismatch_brite_orthology_match":25,"fusion_correct_components_differ":13,"fusion_correct_grounded_harms":0,"fusion_incorrect_grounded_recovers":15,"grounded_correct_phase3a_incorrect":58,"large_ground_truth_rank_change":104,"one_method_abstains_another_correct":12,"phase3a_correct_grounded_incorrect":6,"trained_correct_bm25_incorrect":16}`. Mechanical-rule availability at its selection stage: `{"exact_mismatch_resolved_by_frozen_brite_orthology":24,"ground_truth_id_absent_from_frozen_catalog":5,"multiple_ground_truth_kegg_ids":3}`.

## Manual nominations and documented fills

Accepted nominations: 0. Rejected nominations: 0. Unresolved nominations: 2.

- `model_id,reaction_id`: unresolved_malformed — required fields missing or repeated CSV header row
- `BIOMD0000000013,E12`: unresolved_not_validation — exact key belongs only to non-validation split(s): train

Documented edge-case fills (5):

- `P3EA0052` from category:cross_method_disagreement, category:mechanically_suspicious, category:phase3c_failure_harm, disagreement:exact_mismatch_brite_orthology_match, disagreement:large_ground_truth_rank_change, documented_edge_case:benchmark/phase3/phase3b_fusion/qualitative_examples.json:fusion_loses_trained_hit_at_10, documented_edge_case:benchmark/phase3/phase3b_fusion/qualitative_examples.json:trained_biencoder_only_at_10, failure_type:truth_absent_fusion_top10_grounded_incorrect_in_evidence, suspicion:exact_mismatch_resolved_by_frozen_brite_orthology
- `P3EA0001` from category:cross_method_disagreement, category:mechanically_suspicious, disagreement:bm25_correct_trained_incorrect, disagreement:exact_mismatch_brite_orthology_match, disagreement:fusion_correct_components_differ, disagreement:grounded_correct_phase3a_incorrect, disagreement:large_ground_truth_rank_change, documented_edge_case:benchmark/phase3/phase3b_fusion/qualitative_examples.json:correct_for_both_at_10, documented_edge_case:benchmark/phase3/retrieval_baselines/qualitative_examples.json:bm25_only, documented_edge_case:benchmark/phase3/retrieval_baselines/qualitative_examples.json:fusion_harms_rank, documented_edge_case:benchmark/phase3/retrieval_baselines/qualitative_examples.json:unseen_target_behavior, suspicion:exact_mismatch_resolved_by_frozen_brite_orthology, suspicion:multiple_ground_truth_kegg_ids
- `P3EA0027` from category:cross_method_disagreement, category:phase3c_failure_harm, disagreement:large_ground_truth_rank_change, documented_edge_case:benchmark/phase3/phase3c_validation/qualitative_examples.json:answer_absent_appropriate_abstention, failure_type:other_grounded_nonexact
- `P3EA0016` from category:cross_method_disagreement, category:mechanically_suspicious, category:phase3c_failure_harm, disagreement:exact_mismatch_brite_orthology_match, disagreement:large_ground_truth_rank_change, disagreement:one_method_abstains_another_correct, disagreement:trained_correct_bm25_incorrect, documented_edge_case:benchmark/phase3/phase3b_fusion/qualitative_examples.json:fusion_recovers_beyond_bm25_at_10, failure_type:truth_fusion_ranks2_10_grounded_abstained, suspicion:exact_mismatch_resolved_by_frozen_brite_orthology
- `P3EA0002` from category:cross_method_disagreement, category:mechanically_suspicious, disagreement:fusion_correct_components_differ, disagreement:large_ground_truth_rank_change, disagreement:trained_correct_bm25_incorrect, documented_edge_case:benchmark/phase3/retrieval_baselines/qualitative_examples.json:seen_target_behavior, suspicion:multiple_ground_truth_kegg_ids

No corrected KEGG label was inferred, and nomination was not treated as evidence that a frozen label is wrong.

## Exact sample accounting

| Primary category | Quota | Selected |
| --- | ---: | ---: |
| `random_control` | 20 | 20 |
| `phase3c_failure_harm` | 15 | 15 |
| `cross_method_disagreement` | 10 | 10 |
| `mechanically_suspicious` | 10 | 10 |
| `manual_nomination_or_documented_edge_case` | 5 | 5 |

Unique reactions: 60. Failure-type representation: `{"fusion_top1_correct_grounded_abstained":5,"fusion_top1_correct_grounded_incorrect":1,"truth_fusion_ranks2_10_grounded_abstained":2,"truth_fusion_ranks2_10_grounded_incorrect":7}`. Disagreement subtype representation (counting every subtype attached to the 10 selected disagreement cases): `{"bm25_correct_trained_incorrect":2,"exact_mismatch_brite_orthology_match":1,"fusion_correct_components_differ":3,"fusion_correct_grounded_harms":0,"fusion_incorrect_grounded_recovers":3,"grounded_correct_phase3a_incorrect":5,"large_ground_truth_rank_change":8,"one_method_abstains_another_correct":3,"phase3a_correct_grounded_incorrect":1,"trained_correct_bm25_incorrect":1}`. Mechanical-rule representation: `{"exact_mismatch_resolved_by_frozen_brite_orthology":9,"ground_truth_id_absent_from_frozen_catalog":1,"multiple_ground_truth_kegg_ids":1}`. Full secondary overlaps are in `sampling_distribution.json` and the private crosswalk.

## Distribution

- Clusters (12 represented; maximum 11 cases): `{"CLU_BIOMD0000000017":6,"CLU_BIOMD0000000023":5,"CLU_BIOMD0000000042":11,"CLU_BIOMD0000000068":2,"CLU_BIOMD0000000088":5,"CLU_BIOMD0000000171":4,"CLU_BIOMD0000000190":5,"CLU_BIOMD0000000231":4,"CLU_BIOMD0000000245":3,"CLU_BIOMD0000000602":5,"CLU_BIOMD0000001061":4,"CLU_BIOMD0000001090":6}`.
- Models (17 represented; maximum 6 cases): `{"BIOMD0000000017":6,"BIOMD0000000023":5,"BIOMD0000000042":2,"BIOMD0000000061":1,"BIOMD0000000064":2,"BIOMD0000000068":2,"BIOMD0000000071":3,"BIOMD0000000088":5,"BIOMD0000000171":4,"BIOMD0000000190":5,"BIOMD0000000211":1,"BIOMD0000000231":4,"BIOMD0000000245":3,"BIOMD0000000247":2,"BIOMD0000000602":5,"BIOMD0000001061":4,"BIOMD0000001090":6}`.
- Corrected strata: `{"empty_constrained":20,"nonempty_answer_absent":16,"retrievable_rerank_failure":9,"retrievable_top1_success":10,"unconstrained":5}`.
- Seen/unseen: `{"seen":40,"unseen":20}`.
- Retrieval states: `{"truth_absent_from_fusion_top10":12,"truth_at_fusion_rank1":28,"truth_at_fusion_ranks2_10":20}`.
- Cases with at least one secondary eligibility reason: 57.

## Blinding preparation

`formal_audit_blinded_order.csv` contains only blinded order, stable audit ID, sample ID, model ID, and reaction ID. It excludes category, suspicion, failure, correctness, method outcome, priority, and random/error-enriched status. Its deterministic seed differs from the sampling seed, and its order differs from category presentation and selection order. Prompt 3 may use this skeleton to build Pass 1 materials; no reviewer forms or evidence packets were created here.

## Limitations

The full 60-case sample deliberately contains 40 error-enriched or edge-enriched cases, so it cannot estimate population label-error prevalence. Only the 20 outcome-independent random controls support an approximately unbiased prevalence estimate, subject to finite-sample uncertainty and the prespecified cluster/model balancing design. Mechanical flags, system failures, disagreements, and nominations are review triggers—not biological verdicts. No biological adjudication, reviewer verdict, API call, new inference, label change, or corrected-label sensitivity analysis was performed.
