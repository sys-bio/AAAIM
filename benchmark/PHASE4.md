# Phase 4: bounded cached routing and uncertainty study

Protocol written 2026-11-05 before executing Phase 4 policy comparisons. Base revision: 53916fb54c36b834be0ec9b46a9571833a7d1354; branch benchmark/phase-4 in an isolated worktree. Phase 3C and the scoped single-reviewer audit remain complete and unchanged. No independent review is required for this milestone. No human review files or case judgments enter this implementation or public outputs.

## Questions and hypotheses

Can agreement between the frozen BM25 and selected trained bi-encoder, combined with the relative RRF Top-1 margin, safely bypass some grounded calls? Hypothesis: obvious retrieval decisions permit at least 20% fewer simulated calls while preserving overall exact accuracy, coverage, and answered accuracy. Can raw retrieval margin or LLM self-reported confidence rank correctness or reference-answer absence? These signals are not calibrated probabilities.

## Data and permitted computation

Use only the exact 163 validation reactions with complete frozen Phase 3C responses, evidence, requests, and post-freeze reference join. Full-catalog retrieval is already present for every reaction, including rule-based retrieval failures; do not duplicate it with a new recovery component. No fitting, training, new inference, provider calls, or external biological lookups. Training would be required for future calibration, but matched training confidence/output data are unavailable in these caches; do not fit a calibrator on this small reused pilot. Do not open any held-out reactions, labels, predictions, or diagnostics. Do not load the mixed-split assignment file merely to check separation: frozen validation sample IDs and manifests suffice.

This pilot and the prior human audit have informed development. The study is validation method selection with optimism, not an independent final evaluation. Do not use the 60 judgments as labels or optimize on audit cases. Do not import audit categories or reference membership into routing.

## Fixed policies and thresholds

Retrieve full catalog using frozen equal-weight BM25 + trained epoch-1 RRF, k=60, both components depth 100, Top-10 evidence. Relative margin = (score1-score2)/score1, with finite nonnegative descending scores required. Confidence gate requires both component ranks of fused Top-1 equal 1 AND relative margin >= t. Evaluate exactly t in {0.05, 0.10, 0.20}; no additional thresholds after seeing results.

Compare fusion Top-1 (always answer), frozen Phase 3C grounded (preserve original predictions/abstentions/compliance flags), simple threshold-only fusion (gate answers, otherwise abstains), and routed policy (gate takes fusion, otherwise reads frozen grounded output). Also describe a fixed post-LLM confidence >=0.80 abstention policy; it saves no provider calls. Do not interpret self-confidence as a calibrated probability. Do not fabricate absent route outputs: missing cache rows or alignment/digest mismatches abort replay. Do not change evidence, prompts, or outputs for routed calls. No controlled answer-absent experiment is needed; naturally absent reference answers already exist in evidence.

## Outcomes and uncertainty

Primary: exact reference match / all 163, with every abstention counted noncorrect and separately counted. Secondary: answered/n coverage, correct/answered selective accuracy, selective risk=1-selective accuracy, macro accuracy by cluster and model, per corrected retrieval stratum, and candidate-reference absence. Multi-label truth accepts any frozen reference ID and is never replaced with human judgments.

Mechanical abstention criteria: abstain with no reference ID in Top-10 = reference-absence-consistent; abstain when reference is present = reference-present abstention. These do not establish biological appropriateness or absence of defensible alternatives. Count answered reference-absent cases too. Correctness detection: AUROC/AP of confidence among native answered cases only (exclude abstentions from this detection target). Reference-answer absence detection: AUROC/AP of negative margin across all eligible cases; fixed low-confidence gate as a detector reports confusion counts. No calibrated-probability metrics or claims.

Risk-coverage curves remove answered outputs by score, admitting all exact-score ties as a group. Right-step area = sum((coverage_i-coverage_previous)*risk_i) from zero to native maximum coverage; report area and area/native_max_coverage. Never extrapolate abstentions into fabricated answers to reach full coverage. Retrieval-only uses relative margin; grounded-only uses its self-reported confidence; routed uses margin for bypasses and self-confidence otherwise. Mixed scales are a diagnostic limitation, not comparable probabilities.

Use paired cluster bootstrap of exact accuracy and simulated call rate with 2,000 replicates, seed 20261105, resampling the same 12 validation clusters for every method; report percentile 95% intervals. Reused validation and threshold selection mean intervals are descriptive, not multiplicity-adjusted independent confirmatory evidence. No significance fishing or per-case exceptions.

Resources: zero new calls, zero new training/inference; CPU-only cached calculation and plotting. Replay cost/tokens/serial provider latency sum existing per-call observations only on routes selecting grounded outputs; all calls are still historically paid. Retrieval/routing overhead is unmeasured, so there is no measured end-to-end latency or realized refund. Cached and fresh-call frequency are different; report both simulation and this run's actual zero calls.

## Retain/reject and stop rules

Retain a routed candidate only if correct count >= frozen grounded count, answered count >= grounded count, selective risk <= grounded risk, at least 20% simulated grounded-call reduction, and paired-bootstrap exact-accuracy delta lower bound >= -0.02. This -2 percentage-point tolerance is a development screening criterion, not demonstrated noninferiority. Among eligible candidates choose most correct, then fewest calls, then largest threshold. Threshold-only and post-LLM confidence policies are diagnostic controls, not automatically eligible for final selection. If no routed candidate qualifies, retain the stronger existing grounded baseline (or fusion if it has strictly more exact correct answers). Stop after these policies and intervals; no expanded search, training, API work, or all-969 expansion.

## Freeze and later test boundary

After selection record the exact rule, thresholds, retrieval/checkpoint/provider identities, prompt and schema, artifacts/source hashes and base revision plus uncommitted-source digest. A final operational freeze requires a working explicitly authorized held-out executor and approved hard budget; absent these, publish a draft freeze with concrete blockers, never a false completed freeze. A dry-run must use synthetic queries only and perform no sealed file access or provider imports. Execution must fail closed while the freeze is draft. Later test primary metric is exact micro accuracy; secondary metrics/paired bootstrap follow above, with BRITE/orthology secondary only. Freeze all choices before labels are joined, report all failures/abstentions, and prohibit test-driven changes or selective reruns. No test evaluation occurs in this task.
