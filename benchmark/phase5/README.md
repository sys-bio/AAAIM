# Phase 5 artifact entry point

**Paid execution is paused; no calls are authorized.** The prior USD 40 cap proposal is withdrawn. The [preserved GPT-6.1 Terra runner proposal](validation/persistent_runner_gpt61_v2/EXECUTION_PROPOSAL.md) describes the unchanged 326-unit plan (317 distinct bodies), digest `2b80dd9e83108cdba2310121706af387bd770651a1f7648a87645a03ae0b0e8f`, and USD 17.994805 total request-bound reserve. User-reported total credit is approximately USD 20, not a confirmed balance or authorization. See [pause notice](validation/PAID_EXECUTION_PAUSED.md) before considering any future run. GPT-5.6 and intermediate GPT-6.1 v1 plans/results remain preserved and cannot be reused as these outputs.

Study protocol: `benchmark/PHASE5.md`. The Phase 4 validation study precedes this local robustness study. Historical model-specific plans and results are preserved; private reviewer exports and derived case-level judgments are excluded from this public artifact set.

The local retrieval experiment is complete and unchanged. **Phase 5 is not fully complete:** its required matched clean/degraded grounded-LLM comparison is paused by user decision and has no authorization. The validated architecture and historical baseline are retained; the proposed corrected v2 prompt requires matched validation before any final selection/freeze claim. Held-out data stay sealed.

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
- validation/llm_dry_run.json and llm_request_manifest.jsonl: active GPT-6.1 matched validation plan and pricing estimates. Its complete bodies are under ignored validation/matched_prompt_gpt61_v2/_requests/.
- validation/matched_prompt_gpt61_v2/CLEAN_CONTROL.md, clean_control_diagnostic.json and clean_control_differences.csv: actual 163-case clean-control reconstruction with every differing field, including component-rank/score drift. Complete historical body matches are zero; biological fields/order/settings are unchanged in the historical-construction diagnostic.
- validation/matched_prompt_gpt61_v2/provenance_sidecars.jsonl and matched_pair_differences.jsonl: accurate operational lineage outside model-visible text, and verification that matched arms share provider settings and differ only through perturbation-derived input fields.
- validation/matched_prompt_gpt61_v2/corrected_clean_control_diagnostic.json: all actual GPT-6.1 clean provider bodies compared field by field with historical requests; zero compatible bodies, provenance removed, numerical drift retained, and the model change explicitly recorded.
- validation/matched_prompt_v2/ and persistent_runner_v1/: preserved GPT-5.6 plans/results, superseded for the active model comparison; no responses are reused.
- superseded/ea241ea/: byte-preserved old plan/report/freeze/builder snapshots; the original request-only design is superseded, not silently rewritten as valid.
- freeze/evaluation_freeze.json: updated draft final-evaluation package; prior Phase 4 package remains unchanged.
- reproduction_verification.json: all 3,097 derived queries and perturbation event records independently rebuilt identically.
- review_fixes/VERIFICATION.md, prior_manifest_verification.json and artifact_checksums.json: review-fix verification, read-only checks and SHA-256 identities; private review paths and judgments are excluded.
- prior_verification.json, VERIFICATION.md, provenance.json and artifact_checksums.json at this directory's root are preserved historical ea241ea records. They describe the original code/reports and are not current checksums for the corrected mutable files.

Implementation: phase5_perturb.py, phase5_robustness.py, phase5_llm_plan.py and phase5_reporting.py under benchmark/scripts/. Synthetic tests are under tests/test_phase5_*.py. Reproduction commands and environment handoff are in validation/REPRODUCE.md. Existing normal and GPU benchmark environments and the existing Phase 4 plotting dependencies were used; no model/biological data were downloaded and nothing was trained.

Most stable does not mean best end-to-end method: trained retrieval's Recall@10 varies little under identifier removal with names retained. At nominal 75% removal, its paired change advantage over BM25 is 18.6 percentage points (descriptive 95% cluster/replicate interval 8.2 to 36.9 points). Its advantage in stability over fusion is 8.8 points with interval -1.8 to 29.4, so that comparison is uncertain. Do not change the end-to-end selection from retrieval-only evidence.

The matched comparison remains paused by user decision. It has 163 clean plus 163 remove_0.5_s20261106 requests with the same GPT-6.1 serialization/settings: 326 new calls, 317 distinct provider bodies, no historical compatible reuses and no retries. Its frozen complete-request reservation totals $17.994805, with a maximum unit reserve of $0.072820. The prior proposed $40 cap is withdrawn; no cap is authorized and no paid calls are authorized. The user reports approximately $20 total API credit; that estimate is not a confirmed balance and leaves limited allowance for excluded charges. See `validation/PAID_EXECUTION_PAUSED.md`. Any later execution requires fresh explicit authorization after rechecking pricing, balance, access and the exact plan digest. Usage-based reserve release is not invoice reconciliation; failed or uncertain attempts retain full reserve. Identical bodies do not guarantee unchanged hosted behavior.

Read the [Phase 3C support clarification](../PHASE3C_SUPPORT_ADDENDUM.md): 130 direct in-catalog answers without retrieval include 49 correct reference matches. Zero out-of-evidence grounded IDs measures compliance, not zero biological errors. The independently measured grounded 102/163 versus fusion 93/163 accuracy is unchanged. Historical reports, outputs and manifests remain intact.
