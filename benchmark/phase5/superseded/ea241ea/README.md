# Phase 5 artifact entry point

Study: benchmark/PHASE5.md. Branch: benchmark/phase-5 in a separate worktree. Phase 4's uncommitted completed implementation was copied by verified artifact checksums; original checkouts and historical outputs were preserved. The original authoritative goals.md and this worktree's matching copy were updated as requested; a byte-for-byte starting goals snapshot is retained under ignored _provenance/.

The local retrieval experiment is complete. **Phase 5 is not fully complete:** its required degraded grounded-LLM comparison is blocked on new outputs and explicit run authorization. The selected grounded-fusion method stays unchanged; final evaluation freeze remains draft; held-out data stay sealed.

- validation/REPORT.md: results, uncertainty, limitations, realized doses and missing comparisons.
- validation/COMPLETION_CHECKLIST.md: completed, deferred and blocked work.
- validation/derived_inputs.jsonl and perturbation_manifest.jsonl: deterministic label-free derived queries and event-level provenance.
- validation/realized_corruptions.csv: actual bundle/link changes, unchanged/ineligible queries and masking.
- validation/frozen_rankings.jsonl and prediction_freeze.json: recomputed rankings and pre-reference freeze.
- validation/comparison_by_condition.csv and paired_family_summary.csv: aligned metrics and cluster/replicate intervals.
- validation/paired_method_robustness.csv: paired differences in Recall@10 changes across methods.
- validation/subgroups.csv: strata and model-cluster denominators/results.
- validation/identifier_removal.png/.svg and robustness_deltas.png/.svg: standalone plots.
- validation/historical_clean_comparison.json: all clean Top-1/10 lists match historical BM25/encoder and all fusion Top-10 lists match; one encoder Top-100 tail differs numerically.
- validation/llm_dry_run.json and llm_request_manifest.jsonl: exact next validation run scope, current price snapshot assumptions and missing output counts. Full prepared payloads are under ignored validation/_requests/.
- freeze/evaluation_freeze.json: updated draft final-evaluation package; prior Phase 4 package remains unchanged.
- reproduction_verification.json: all 3,097 derived queries and perturbation event records independently rebuilt identically.
- prior_verification.json and VERIFICATION.md: tests, seven read-only baseline verifiers and preservation checks.
- provenance.json and artifact_checksums.json: code/config/input/output identities and source revision plus uncommitted snapshot hashes.

Implementation: phase5_perturb.py, phase5_robustness.py, phase5_llm_plan.py and phase5_reporting.py under benchmark/scripts/. Synthetic tests are under tests/test_phase5_*.py. Reproduction commands and environment handoff are in validation/REPRODUCE.md. Existing normal and GPU benchmark environments and the existing Phase 4 plotting dependencies were used; no model/biological data were downloaded and nothing was trained.

Most stable does not mean best end-to-end method: trained retrieval's Recall@10 varies little under identifier removal with names retained. At nominal 75% removal, its paired change advantage over BM25 is 18.6 percentage points (descriptive 95% cluster/replicate interval 8.2 to 36.9 points). Its advantage in stability over fusion is 8.8 points with interval -1.8 to 29.4, so that comparison is uncertain. Do not change the end-to-end selection from retrieval-only evidence.

The proposed next comparison is exactly 163 validation requests, remove_0.5_s20261106, no cache reuses and no retries. Estimated Standard API cost is $2.60 using conservative input estimates plus historical mean output, or $6.33 allowing all output tokens; estimates are not guarantees. Proposed hard ceiling: $15. Current authorization: $0. Prices were verified using official OpenAI Docs on 2026-11-06; see the saved pricing JSON and https://developers.openai.com/api/docs/pricing. No paid execution command is enabled in this task.
