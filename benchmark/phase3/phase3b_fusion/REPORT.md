# Phase 3B BM25 + trained bi-encoder fusion

The prespecified equal-weight RRF was built from frozen 100-deep BM25 and selected epoch-1 bi-encoder rankings, with `k=60`, one-indexed ranks, zero contribution for missing documents, and ascending KEGG-ID tie-breaking. Each output is a complete permutation of the frozen 12,312-reaction catalog. Rankings were hashed before validation truth was joined; no test row or label was read.

## Exact validation metrics

| Method | R@1 | R@3 | R@5 | R@10 | MRR@10 | R@10 model macro | R@10 cluster macro |
|---|---:|---:|---:|---:|---:|---:|---:|
| phase2_rule_based | 0.441692 | 0.459236 | 0.459236 | 0.459236 | 0.44926 | 0.396495 | 0.370843 |
| bm25 | 0.780186 | 0.884417 | 0.901961 | 0.919505 | 0.832932 | 0.722061 | 0.708131 |
| bge_m3_dense | 0.431373 | 0.626419 | 0.722394 | 0.785346 | 0.550519 | 0.530871 | 0.562368 |
| bm25_bge_m3_rrf | 0.736842 | 0.854489 | 0.897833 | 0.924665 | 0.802582 | 0.771005 | 0.778151 |
| trained_biencoder_epoch1 | 0.840041 | 0.897833 | 0.911249 | 0.921569 | 0.871584 | 0.740615 | 0.677518 |
| bm25_trained_epoch1_rrf | 0.876161 | 0.928793 | 0.937049 | 0.952528 | 0.902243 | 0.810343 | 0.803688 |

## BRITE/orthology-aware validation metrics

| Method | R@1 | R@3 | R@5 | R@10 | MRR@10 |
|---|---:|---:|---:|---:|---:|
| phase2_rule_based | 0.462332 | 0.464396 | 0.464396 | 0.464396 | 0.46302 |
| bm25 | 0.803922 | 0.904025 | 0.917441 | 0.931889 | 0.853961 |
| bge_m3_dense | 0.510836 | 0.684211 | 0.762642 | 0.814241 | 0.614642 |
| bm25_bge_m3_rrf | 0.780186 | 0.877193 | 0.913313 | 0.93808 | 0.833869 |
| trained_biencoder_epoch1 | 0.883385 | 0.914345 | 0.923633 | 0.932921 | 0.900451 |
| bm25_trained_epoch1_rrf | 0.907121 | 0.952528 | 0.95872 | 0.96904 | 0.928424 |

## Evidence-set strata

| Method | Seen R@10 (n=847) | Unseen R@10 (n=122) | True Phase 2 retrieval failures R@10 (n=524) | Phase 2 reranking failures R@10 (n=17) |
|---|---:|---:|---:|---:|
| phase2_rule_based | 0.460449 | 0.45082 | 0.0 | 1.0 |
| bm25 | 0.939787 | 0.778689 | 0.898855 | 0.470588 |
| bge_m3_dense | 0.821724 | 0.532787 | 0.734733 | 0.647059 |
| bm25_bge_m3_rrf | 0.939787 | 0.819672 | 0.90458 | 0.588235 |
| trained_biencoder_epoch1 | 0.961039 | 0.647541 | 0.875954 | 0.882353 |
| bm25_trained_epoch1_rrf | 0.968123 | 0.844262 | 0.94084 | 0.529412 |

## Component overlap and fusion transitions

At Recall@10: BM25 only 43; trained bi-encoder only 45; both 848; missed by both 33. Fusion recovered 34 beyond BM25 and 42 beyond the trained model, while losing 2 BM25 hits and 12 trained-model hits.

The 17-row Phase 2 reranking-failure subset is a real small-stratum tradeoff: fusion R@10 is 0.529412 versus 0.882353 for the trained model.

## Paired cluster bootstrap

10,000 percentile replicates use seed 20260902 and the 12 frozen validation clusters. Deltas are fusion minus reference reaction-micro recall.

- fusion_minus_bm25, recall_at_1: +0.095975, 95% CI [+0.013003, +0.243697]; includes zero: false.
- fusion_minus_bm25, recall_at_10: +0.033024, 95% CI [+0.004790, +0.190299]; includes zero: false.
- fusion_minus_trained_biencoder_epoch1, recall_at_1: +0.036120, 95% CI [-0.011299, +0.067278]; includes zero: true.
- fusion_minus_trained_biencoder_epoch1, recall_at_10: +0.030960, 95% CI [-0.011609, +0.245487]; includes zero: true.
- fusion_minus_bm25_bge_m3_rrf, recall_at_1: +0.139319, 95% CI [+0.100209, +0.225131]; includes zero: false.
- fusion_minus_bm25_bge_m3_rrf, recall_at_10: +0.027864, 95% CI [+0.014855, +0.077320]; includes zero: false.

Only 12 clusters are available, so percentile intervals may be unstable; no superiority claim is made where zero is included.

## Phase 3C recommendation

Fusion has the best validation evidence-set recall, including unseen targets, with no material (>0.01) Recall@1 harm; statistical uncertainty is retained.

Recommended Top-10 evidence retriever: `bm25_trained_epoch1_rrf`. Pareto frontier over Recall@1, Recall@10, and unseen-target Recall@10: bm25_trained_epoch1_rrf.

This milestone did not start Phase 3C, run held-out test evaluation, train another model, tune fusion, or call an API.
