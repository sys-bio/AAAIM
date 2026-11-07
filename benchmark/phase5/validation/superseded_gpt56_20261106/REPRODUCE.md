# Cached review-fix reproduction (no GPU or paid calls)

Run from the Phase 5 worktree. SOURCE_ROOT is the preserved original workspace containing the frozen local catalog; all retrieval rankings are reused unchanged. Choose a NEW plan name.

```powershell
python -m benchmark.scripts.phase5_llm_plan --dry-run --source-root SOURCE_ROOT --out benchmark/phase5/validation --plan-name _matched_prompt_v2_reproduced --no-publish-current
python -m benchmark.scripts.phase3c_support_clarification --source-root SOURCE_ROOT
python -m benchmark.scripts.phase5_reporting --source-root SOURCE_ROOT --out benchmark/phase5/validation --no-plots
```

The builder writes the named new plan; --no-publish-current preserves current dry-run/manifest pointers during independent reproduction. It never overwrites a named existing plan. Provider bodies are complete SDK 1.78.1 JSON request bodies including the strict response schema, pinned to Pydantic 2.13.5. No provider client or HTTP transport is created. The serializer must be kept pinned or reverified before a later paid run. Historical retrieval reproduction commands and the original confounded plan are retained under superseded/ea241ea/. The reporting stage changes mutable reports/current draft only; historical retrieval tables/figures, Phase 3C artifacts and Phase 4 files remain unchanged. Paid/held-out execution stays disabled.
