# Phase 3B bi-encoder smoke milestone

This is a bounded engineering diagnostic, not a scientific performance result. No held-out test labels were read and no full training run was launched.

## Starting checkpoint

`BAAI/bge-small-en-v1.5` at immutable revision `5e62ea33e012fda8c02802b906664c915ebd1bb1` (MIT; BertModel; 12 layers; hidden size 384; 12 heads; approximately 33,360,000 parameters). This is a general English retrieval encoder, not a biology-specialized model.

## Leakage-safe data

3,466 usable frozen-training queries, 1,567 unique targets, 65 multi-positive queries, and 10,398 explicit negatives. 31 of the 3,497 assigned train rows are explicitly excluded because their only valid target has no document in the frozen catalog. All query/document text uses the frozen Phase 3 retrieval templates and passes the digit-bounded leakage scanner. EC/KO siblings are excluded as ambiguous negatives.

## Measured smoke results

Overfit loss: 1.341516 to 0.000000 (100.0% reduction).
Representative loss: 1.399312 to 0.610137; 168 examples in 4.23s (39.70/s); peak allocated VRAM 1.99 GiB.
Checkpoint and full optimizer/scaler/scheduler state were saved atomically; a separate resumed step is recorded in the smoke metrics.

## Validation smoke diagnostic

One frozen-ranking, validation-only diagnostic: R@1 0.4076, R@10 0.7616, unseen-target R@10 0.5328, and true-retrieval-failure R@10 0.6889. It was not used to select a checkpoint.

## Full-run recommendation (not executed)

Use batch 4, three explicit negatives, accumulation 2, maximum length 256, learning rate 2e-5, 10% warmup, and at most three epochs. Estimated training-only time is 1.5 minutes per epoch. Select on overall validation Recall@1 while retaining the other three prespecified metrics.

The local RTX 3070 is sufficient for this configuration; cloud rental is not currently justified.
