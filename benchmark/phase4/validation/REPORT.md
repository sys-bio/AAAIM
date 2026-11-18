# Phase 4 cached validation routing results

Selected method: **grounded** by the criteria written in benchmark/PHASE4.md before execution.

All comparisons use the same 163 validation reactions and unchanged frozen references. No human review judgments were loaded. No training, new inference, provider calls, held-out data access, or external biological queries occurred.

| Method | Correct / 163 | Answered / 163 | Correct / answered | Simulated grounded calls | Replayed USD |
| --- | ---: | ---: | ---: | ---: | ---: |
| fusion | 93/163 | 163/163 | 93/163 | 0/163 | 0.000000 |
| grounded | 102/163 | 124/163 | 102/124 | 163/163 | 2.108151 |
| threshold_0.05 | 17/163 | 21/163 | 17/21 | 0/163 | 0.000000 |
| threshold_0.1 | 4/163 | 5/163 | 4/5 | 0/163 | 0.000000 |
| threshold_0.2 | 1/163 | 1/163 | 1/1 | 0/163 | 0.000000 |
| routed_0.05 | 102/163 | 124/163 | 102/124 | 142/163 | 1.826527 |
| routed_0.1 | 102/163 | 124/163 | 102/124 | 158/163 | 2.039711 |
| routed_0.2 | 102/163 | 124/163 | 102/124 | 162/163 | 2.093277 |
| post_llm_threshold_0.8 | 101/163 | 123/163 | 101/123 | 163/163 | 2.108151 |

## Selection and resource interpretation

No candidate may be promoted solely for saving calls. All five criteria in selection.json must pass. The three coarse gates were not expanded after results were observed. A negative development result retains the existing stronger method.
Frozen provider cost/usage/serial latency are summed only for replay routes choosing the grounded output. The present study's actual new calls and provider cost are zero. Historical calls were already paid; replay savings are not refunds. Both retrievers still run for all queries. Retrieval and routing overhead is unmeasured, so end-to-end speedup is not established.

## Uncertainty, abstention, and detection

metrics.json contains paired 2,000-replicate cluster-bootstrap intervals for accuracy, delta versus grounded and simulated call rate; 12 clusters, seed 20261105. Threshold selection and prior use of these validation cases make these intervals descriptive, with selection optimism and no multiplicity correction.
Reference-absence-consistent abstention means no accepted frozen reference ID appears in the Top-10 evidence. Reference-present abstention means one does. Neither category establishes biological appropriateness. The post-LLM threshold still consumes all 163 simulated calls.
Risk-coverage curves use native answered cases only, exact-score ties as groups, right-step partial area and area divided by maximum native coverage. Methods with different native coverage have different integration ranges. Grounded confidence and retrieval margins are uncalibrated. Routed mixed-score curves combine unlike scales and are diagnostic only. Correctness detection is evaluated among answered cases; absence detection uses negative retrieval margin and low gate confidence. Values and confusion counts are in metrics.json and answer_absence_detection.json. No calibrator was fit: compatible training confidence/output data are unavailable and the reused small pilot is unsuitable for both fitting and selecting one.

## Coverage and limitations

subgroups.csv reports denominators, exact accuracy, coverage, selective accuracy and call counts by corrected retrieval stratum, model cluster and model. The 163 pilot cases were stratified for development; unweighted micro accuracy describes this pilot, not population-wide accuracy. No weighting is invented.
This is a replay of cached outputs under unchanged requests and evidence, not a newly executed end-to-end system or fabricated counterfactual model answers. Two baseline abstentions carry evidence-compliance errors and are retained rather than repaired or excluded. Output compliance does not establish biological correctness. Missing or mismatched cache entries abort replay.
Full-catalog retrieval already handles rule-based empty/absent candidates; no redundant retrieval component was added. No controlled answer-absent data, new labels, calibration fitting, or audit-based policy optimization was performed. The single-reviewer audit remains complete; it is descriptive development evidence, not an independent evaluation. Thresholds/model choices require later sealed evaluation.

## Reproduction and freeze status

From the isolated Phase 4 worktree:

```powershell
python -m benchmark.scripts.phase4_routing --out benchmark/phase4/_validation_reproduced
python -m benchmark.scripts.phase4_freeze --prepare --study benchmark/phase4/_validation_reproduced --out benchmark/phase4/_freeze_reproduced
python -m benchmark.scripts.phase4_freeze --dry-run --out benchmark/phase4/_dry_run_reproduced.json
```

Choose new output paths on every replay; existing output directories are never overwritten. Runtime: existing NumPy/scikit-learn/Pydantic/benchmark dependencies plus Matplotlib for exported PNG/SVG plots. Source and input hashes, software versions, base revision and prior manifest results are in provenance.json. Checksum files pin the uncommitted Phase 4 source state; no commit was created.
The selected method is validation-selected. The evaluation package is explicitly a draft: no operational held-out executor, approved budget, or immutable hosted-provider identity has been finalized. The documented --execute command deliberately fails closed before reading data. See freeze/evaluation_freeze.json for exact blockers and proposed budget limits. Do not execute or unlock the test.
