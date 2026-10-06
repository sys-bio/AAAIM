# Phase 3B full bi-encoder training

One prespecified BGE-small configuration was trained for exactly three epochs. All checkpoint selection and analysis are validation-only: each 969-query, 12,312-document ranking was frozen before validation labels were joined, and no held-out test row or label was read.

## Epoch learning curve

| Epoch | Train loss | Updates | Train min | Rank min | Exact R@1 | R@3 | R@5 | R@10 | MRR@10 | Unseen R@10 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | — | 0 | 0.00 | 0.25 | 0.047472 | 0.120743 | 0.159959 | 0.236326 | 0.098146 | 0.172131 |
| 1 | 0.308874 | 434 | 1.12 | 0.26 | 0.840041 | 0.897833 | 0.911249 | 0.921569 | 0.871584 | 0.647541 |
| 2 | 0.111885 | 434 | 1.13 | 0.26 | 0.836945 | 0.897833 | 0.912281 | 0.921569 | 0.869384 | 0.631148 |
| 3 | 0.060443 | 434 | 1.13 | 0.26 | 0.839009 | 0.898865 | 0.911249 | 0.918473 | 0.870787 | 0.631148 |

## Validation-selected checkpoint

Epoch **1** was selected by the prespecified hierarchy: exact reaction-micro R@1, then R@10, then unseen R@10, then earlier epoch. Its retained reference is `benchmark/phase3/phase3b_full/_checkpoints/best.pt`.

## Frozen-baseline comparison

| Method | Exact R@1 | Exact R@10 | Delta R@1 vs selected | Delta R@10 vs selected |
|---|---:|---:|---:|---:|
| epoch0_bge_small | 0.047472 | 0.236326 | +0.792569 | +0.685243 |
| phase2_rule_based | 0.441692 | 0.459236 | +0.398349 | +0.462333 |
| bm25 | 0.780186 | 0.919505 | +0.059855 | +0.002064 |
| bge_m3_dense | 0.431373 | 0.785346 | +0.408668 | +0.136223 |
| bm25_bge_m3_rrf | 0.736842 | 0.924665 | +0.103199 | -0.003096 |

## Paired cluster bootstrap

10,000 percentile replicates use seed 20260902 and the 12 frozen validation clusters. Intervals containing zero are not evidence of superiority.

- Selected vs epoch0_bge_small, recall_at_1: +0.792570, 95% CI [+0.386076, +0.848934]; includes zero: false.
- Selected vs epoch0_bge_small, recall_at_10: +0.685243, 95% CI [+0.446429, +0.780919]; includes zero: false.
- Selected vs bge_m3_dense, recall_at_1: +0.408669, 95% CI [+0.203593, +0.479248]; includes zero: false.
- Selected vs bge_m3_dense, recall_at_10: +0.136223, 95% CI [+0.089286, +0.276923]; includes zero: false.
- Selected vs bm25, recall_at_1: +0.059856, 95% CI [-0.006143, +0.233803]; includes zero: true.
- Selected vs bm25, recall_at_10: +0.002064, 95% CI [-0.143141, +0.134021]; includes zero: true.
- Selected vs bm25_bge_m3_rrf, recall_at_1: +0.103199, 95% CI [+0.047050, +0.210884]; includes zero: false.
- Selected vs bm25_bge_m3_rrf, recall_at_10: -0.003096, 95% CI [-0.200743, +0.060092]; includes zero: true.

## Selected-checkpoint strata

Seen R@10: 0.961039; unseen R@10: 0.647541; true-retrieval-failure R@10: 0.875954; rerank-failure R@10: 0.882353.

## What fine-tuning changed

- incorrect_epoch0_to_correct_selected: 770 (seen 710, unseen 60, multi-positive 7).
- correct_epoch0_to_incorrect_selected: 2 (seen 1, unseen 1, multi-positive 0).
- improved_rank_without_top1: 103 (seen 82, unseen 21, multi-positive 5).
- worsened_rank: 3 (seen 1, unseen 2, multi-positive 1).
- recovered_within_top10: 664 (seen 606, unseen 58, multi-positive 8).
- lost_from_top10: 0 (seen 0, unseen 0, multi-positive 0).

Mechanically selected examples (lexicographically first eligible):

- successful_biochemical_retrieval_learned: `BIOMD0000000017/R2`; epoch 0 rank 3, selected rank 1, BM25 rank 2.
- harmed_by_fine_tuning: `BIOMD0000001061/R_CPC6MT`; epoch 0 rank 1, selected rank 3, BM25 rank 1.
- unseen_target_improvement: `BIOMD0000000023/v6`; epoch 0 rank None, selected rank 1, BM25 rank 1.
- unseen_target_regression: `BIOMD0000000017/R11`; epoch 0 rank 3, selected rank 4, BM25 rank 1.
- bm25_success_biencoder_misses: `BIOMD0000000017/R11`; epoch 0 rank 3, selected rank 4, BM25 rank 1.
- biencoder_success_bm25_misses: `BIOMD0000000017/R2`; epoch 0 rank 3, selected rank 1, BM25 rank 2.

## Runtime, storage, and recommendation

Training took 3.37 minutes and validation ranking took 1.03 minutes. Peak allocated/reserved VRAM was 2.09/2.22 GiB. Three completed epoch checkpoints occupy 1.12 GiB; including the retained best and latest copies, checkpoint storage is 1.86 GiB. All are ignored by Git.

Validation performance peaked before epoch 3 or showed degradation; extending this run is not currently justified.

The selected inference weights should be archived outside Git according to `archive_plan.json`; no upload was performed.
