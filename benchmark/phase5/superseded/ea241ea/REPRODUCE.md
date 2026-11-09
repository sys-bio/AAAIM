# Local reproduction

Run from the isolated Phase 5 worktree, using a NEW output directory for preparation. SOURCE_ROOT is the preserved original workspace (no mixed-split loaders are called).

```powershell
python -m benchmark.scripts.phase5_robustness prepare --source-root SOURCE_ROOT --out benchmark/phase5/_reproduced
SOURCE_ROOT/benchmark/phase3/_phase3b_env/Scripts/python.exe -m benchmark.scripts.phase5_robustness dense --source-root SOURCE_ROOT --out benchmark/phase5/_reproduced
python -m benchmark.scripts.phase5_robustness analyze --source-root SOURCE_ROOT --out benchmark/phase5/_reproduced
python -m benchmark.scripts.phase5_llm_plan --dry-run --source-root SOURCE_ROOT --out benchmark/phase5/_reproduced
python -m benchmark.scripts.phase5_reporting --source-root SOURCE_ROOT --out benchmark/phase5/_reproduced --plot-deps PHASE4_WORKTREE/benchmark/phase4/_deps
```

SOURCE_ROOT and PHASE4_WORKTREE are recorded in the local handoff provenance. Plotting uses existing local Matplotlib dependencies; no downloads are needed. The existing GPU environment supplies pinned Torch/Transformers, while the normal benchmark environment supplies SBML/core matching dependencies. All trained weights/catalog/settings stay fixed. The reporting stage writes a new Phase 5 draft freeze and does not edit historical Phase 4 files. Paid execution is disabled.
