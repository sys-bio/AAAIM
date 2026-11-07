# GPT-6.1 Terra matched-run dry-run verification

**Status: prepared, dry-run verified, USD 0 authorized.** The plan contains 163 matched validation pairs, 326 one-attempt units and 317 distinct provider bodies. No GPT-5.6 response is reused. The complete-body model and experimental-unit cache namespace differs from the preserved GPT-5.6 plan.

The public model documentation supports `gpt-6.1-terra` on the Responses endpoint, `reasoning.effort: low`, and structured outputs. The exact request bodies use the same strict JSON schema, `max_output_tokens: 2048`, no tools, and matching non-input provider settings in both arms. Account/project access was not probed. SDK 1.78.1 and Pydantic 2.13.5 are recorded; reverify before any later live run.

## Dry-run result

`dry_run_verification.json` records zero provider constructions, zero paid attempts, zero authorized dollars, and zero held-out rows read. The separate all-units-at-maximum reservation is USD 4.640720 per dispatch, or USD 1,512.874720 for 326 attempts, while the proposed hard cap is USD 40.00. The cap may stop the run before completion. The estimated-input scenarios are USD 1.744317 at ordinary Standard input plus historical mean output; USD 2.121366 with all 30-minute cache writes at mean output; and USD 5.223486 with all 30-minute writes and all 2,048 output tokens. Estimates are not guarantees; exact-plan authorization remains USD 0.

## Reproduction

Run from the Phase 5 worktree, with the preserved source workspace as `SOURCE_ROOT`:

```powershell
python -m benchmark.scripts.phase5_llm_plan --dry-run --source-root SOURCE_ROOT --out benchmark/phase5/validation --plan-name NEW_GPT61_PLAN_NAME --model gpt-6.1-terra --no-publish-current
python -m benchmark.scripts.phase5_runner --package benchmark/phase5/validation/NEW_GPT61_RUNNER_NAME --build-from benchmark/phase5/validation/NEW_GPT61_PLAN_NAME
python -m benchmark.scripts.phase5_runner --package benchmark/phase5/validation/NEW_GPT61_RUNNER_NAME
```

Use new names; plan/package creation never overwrites existing artifacts. Live execution needs separate authorization matching the entire new plan digest. GPT-5.6 authorization and cached responses cannot be used. Hosted model identity or temporal drift may remain even with identical request bytes and returned aliases. No held-out access, paid call, inference or external biological query occurred.
