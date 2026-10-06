# Phase 3 full-catalog retrieval baselines

Validation reactions: 969; models: 18; clusters: 12. Rankings were frozen before train/validation labels were joined; no held-out test query or label was loaded.

## Exact reaction-micro metrics

| Method | R@1 | R@3 | R@5 | R@10 | MRR@10 |
|---|---:|---:|---:|---:|---:|
| phase2_rule_based | 0.441692 | 0.459236 | 0.459236 | 0.459236 | 0.44926 |
| bm25 | 0.780186 | 0.884417 | 0.901961 | 0.919505 | 0.832932 |
| bge_m3_dense | 0.431373 | 0.626419 | 0.722394 | 0.785346 | 0.550519 |
| bm25_bge_m3_rrf | 0.736842 | 0.854489 | 0.897833 | 0.924665 | 0.802582 |

## BRITE/orthology-aware reaction-micro metrics

| Method | R@1 | R@3 | R@5 | R@10 | MRR@10 |
|---|---:|---:|---:|---:|---:|
| phase2_rule_based | 0.462332 | 0.464396 | 0.464396 | 0.464396 | 0.46302 |
| bm25 | 0.803922 | 0.904025 | 0.917441 | 0.931889 | 0.853961 |
| bge_m3_dense | 0.510836 | 0.684211 | 0.762642 | 0.814241 | 0.614642 |
| bm25_bge_m3_rrf | 0.780186 | 0.877193 | 0.913313 | 0.93808 | 0.833869 |

## Averaging, seen/unseen, and recovery

| Method | R@10 model-macro | R@10 cluster-macro | Seen R@10 (n=847) | Unseen R@10 (n=122) | True RF R@10 (n=524) |
|---|---:|---:|---:|---:|---:|
| phase2_rule_based | 0.396495 | 0.370843 | 0.460449 | 0.45082 | 0.0 |
| bm25 | 0.722061 | 0.708131 | 0.939787 | 0.778689 | 0.898855 |
| bge_m3_dense | 0.530871 | 0.562368 | 0.821724 | 0.532787 | 0.734733 |
| bm25_bge_m3_rrf | 0.771005 | 0.778151 | 0.939787 | 0.819672 | 0.90458 |

True retrieval failure is corrected to unconstrained (85), empty constrained (419), and nonempty-answer-absent (20); the 17 retrievable rerank failures are reported separately. Two frozen multi-label stratum discrepancies are preserved in `frozen_stratum_discrepancies.csv`.

## Recall@10 overlap

BM25 only: 157; dense only: 27; both: 734; neither: 51; RRF-only versus both component top-10 lists: 3.

## Paired cluster bootstrap

10,000 percentile replicates, seed 20260902, resampling the 12 frozen validation clusters. Deltas are paired reaction-micro Recall@10.

- bm25_minus_dense: 0.134159, 95% CI [0.071713, 0.306667]
- rrf_minus_bm25: 0.00516, 95% CI [-0.013158, 0.120482]
- rrf_minus_dense: 0.139319, 95% CI [0.102389, 0.371053]
- best_full_catalog_minus_phase2: 0.465428, 95% CI [0.197492, 0.724359] (best method: bm25_bge_m3_rrf)

Only 12 clusters are available, so intervals may be unstable. RRF versus BM25 includes zero and is not evidence of superiority.

## Runtime and recommendation

The cached BGE-M3 validation-query pass took 333.464 seconds on CPU-only PyTorch. The one-time catalog pass took approximately 56.7 minutes before the 50,430,080-byte embedding cache was atomically written; the downloaded model cache occupied 4.56 GB because Windows could not use Hugging Face symlinks.

Use a smaller dense encoder as the Phase 3B starting checkpoint on the RTX 3070. Keep BGE-M3 as the frozen off-the-shelf reference: its standalone Recall@10 trails BM25, and RRF's overall gain over BM25 is small and not statistically resolved, although the unseen-target gain is worth retaining as a comparison.
