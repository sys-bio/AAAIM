# Canonical branches and archived integration work

Updated 2026-11-07. This is a local consolidation; no remote branch or PR is changed.

| Branch | Role |
|---|---|
| `benchmark/phase-1`, `benchmark/phase-2`, `benchmark/phase-3` | Preserve the completed historical research phases. |
| `benchmark/phase-3c` | Scoped Phase 3C validation/audit history plus the standalone evidence-support reporting clarification. |
| `benchmark/error-audit` | Original audit line and current cross-project roadmap/handoff documents; no private review exports added by cleanup. |
| `benchmark/phase-4` | Original bounded routing study, inheriting clarified Phase 3C history. |
| `benchmark/phase-5` | Original robustness study/results and preserved matched plans, the clarified earlier phases, pause/status documentation and portable runner test fix. Paid execution remains paused. |
| `feature/luna-model2507280001` | Current public-workflow fixes, selected RRF request preparation, output-budget/failure accounting, and Luna handoff. Renamed from `codex/luna-public-workflow`; requests remain unexecuted. |
| `main` | Original fork main; not updated or reset during this cleanup. |
| `backup/phase-1-before-rollback` | Pre-existing historical backup; retained. |

Use `goals.md` on Phase 5, the audit checkout or the Luna task branch for current project status. Earlier phase branches retain their stage-specific roadmap snapshots.

## Duplicate integration branches

The six local branches below are retired after their useful differences are integrated. Phase 4 implementation modules/tests matched both alternate versions after line-ending normalization. Phase 5 implementation modules/tests matched the integration copy; the main-based copy supplied a portable temporary-root test fix. The standalone Phase 3C clarification removes the old Phase 5 dependency. Current pause decisions are carried forward, while original experiment artifacts and model-specific request plans remain unchanged. Variant-specific provenance/serialization snapshots remain recoverable in the archived histories.

| Retired local branch | Original tip |
|---|---|
| `codex/phase3c-clarification` | `1fb44ed39e5de0b1d485449e5c11345d4e5e98d9` |
| `codex/phase3c-main` | `281684c39a67b45e6863bb8e37cca6e195b48a03` |
| `codex/phase4-integration` | `89cb4033183a5b5387b2e2b6b344550335466ad2` |
| `codex/phase4-main` | `a6d94c4befd868ff809f79db90d2fbcdf0f5b46c` |
| `codex/phase5-integration` | `ec8e4055cebf1f70a56e07945f1334a8db4e8237` |
| `codex/phase5-main` | `145f52d7305353d515d7877e36b42b6a1870a5b8` |

The corresponding worktrees are detached at their archived tips rather than deleted, preserving ignored exports, request payloads and local caches. The Luna physical worktree stays at `benchmark/integration_worktrees/luna_public`; renaming its branch does not change any frozen provider payload or cache key.

## Recovery

Every original local branch tip is retained at `refs/archive/20261107-branch-consolidation/<original-branch>`. A verified self-contained Git bundle and the full before/after inventory are local ignored files under `benchmark/integration_worktrees/_consolidation_20261107/`.

For example, restore an old integration branch with:

```powershell
git branch restore/phase5-integration refs/archive/20261107-branch-consolidation/codex/phase5-integration
```

No release tag or remote backup branch is created.

## Published upstream review branches

The existing `sys-bio/AAAIM` branches `codex/phase3c-support-clarification` (281684c), `codex/phase4-routing-validation` (a6d94c4), and `codex/phase5-paused-robustness` (145f52d) remain unchanged. Their names differ from the retired local working names. Local consolidation does not mean these upstream changes have been merged or their PRs approved; no remote deletion, push or PR mutation is included here.

## Boundaries

Phase 5 stays paused. Luna's preferred 541-unit GPT-6.1 Terra plan remains 2,048 maximum total output tokens with a proposed USD 16 cap, unapproved. No inference, new experiments, held-out access, original review-export changes, or biological judgments are part of this cleanup.
