# GPT-6.1 matched-plan reproduction (no GPU or paid calls)

Run from the Phase 5 worktree. SOURCE_ROOT is the preserved original workspace containing the frozen local catalog; all retrieval rankings are reused unchanged. Choose new plan/package names.

```powershell
python -m benchmark.scripts.phase5_llm_plan --dry-run --source-root SOURCE_ROOT --out benchmark/phase5/validation --plan-name NEW_GPT61_PLAN_NAME --model gpt-6.1-terra --no-publish-current
python -m benchmark.scripts.phase5_runner --package benchmark/phase5/validation/NEW_GPT61_RUNNER_NAME --build-from benchmark/phase5/validation/NEW_GPT61_PLAN_NAME
python -m benchmark.scripts.phase5_runner --package benchmark/phase5/validation/NEW_GPT61_RUNNER_NAME
python -m benchmark.scripts.phase3c_support_clarification --source-root SOURCE_ROOT
python -m benchmark.scripts.phase5_reporting --source-root SOURCE_ROOT --out benchmark/phase5/validation --no-plots
```

The builder writes new plans only; --no-publish-current preserves current dry-run/manifest pointers during independent reproduction. It never overwrites a named existing plan. Provider bodies are complete SDK 1.78.1 JSON request bodies including the strict response schema, pinned to Pydantic 2.13.5. GPT-6.1 Terra is documented for Responses and structured outputs; account-level availability is not checked. No provider client or HTTP transport is created. GPT-5.6 plans/results are preserved and cannot be used as GPT-6.1 outputs. Reverify the serializer before any paid run. Historical retrieval reproduction commands and the original confounded plan are retained under superseded/. The reporting stage changes mutable reports/current draft only; historical retrieval tables/figures, Phase 3C artifacts and Phase 4 files remain unchanged. Paid/held-out execution stays disabled.
