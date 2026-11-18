# Phase 5 completion checklist

| Item | Status | Evidence / boundary |
| --- | --- | --- |
| Separate Phase 5 branch/worktree and verified Phase 4 handoff | Complete | original worktrees preserved; original goals snapshot retained in ignored _provenance; Phase 4 uncommitted implementation copied by checksum |
| Pre-experiment protocol and fixed grid | Complete | benchmark/PHASE5.md; study_config.json; 19 conditions, three fixed stochastic seeds, 163 validation reactions |
| Random 10/25/50/75% bundle removal | Complete locally | derived_inputs.jsonl; perturbation_manifest.jsonl; realized_corruptions.csv; names/stoichiometry retained |
| Reactant/product removal, reversibility and eligibility | Complete locally | same manifests; record SBML sides; retain unchanged cases in all denominators |
| ChEBI ancestor replacement | Complete locally | frozen local is_a graph, deterministic immediate parent, distance=1; direct KEGG retained and masking disclosed |
| Synthetic incorrect mappings | Complete locally | shared-parent sibling with disjoint original compound mapping; inability to replace counted; no biological plausibility claim beyond constraints |
| Rule-based retrieval comparison | Complete for explicit strict control | existing strict filter/scorer recomputed on direct mappings; does not certify historical iterative Phase 2 workflow |
| BM25, trained encoder and selected fusion | Complete locally | frozen_rankings.jsonl; fixed catalog/checkpoint, clean checks, no clean-ranking substitution |
| Paired metrics, subgroups and cluster/replicate uncertainty | Complete | comparison_by_condition.csv, paired_family_summary.csv, paired_method_robustness.csv, subgroups.csv, plots |
| Grounded selected-method robustness at 50% removal | Blocked, required | llm_dry_run.json and ignored _requests/primary_grounded_plan.jsonl: new responses require separate authorization/budget; retrieval-only is not full Phase 5 |
| Other grounded perturbation cells and direct LLM recovery | Deferred optional | no compatible changed-input outputs; not every method-condition combination is required |
| Iterative ontology relaxation / EM comparison | Deferred optional | existing expensive configuration not part of bounded strict-control study; no claim of execution |
| All-969 validation, adaptation or retraining | Deferred optional | not part of current fixed 163-query protocol |
| Human labels / audit reopening | Not required, excluded | completed scoped audit preserved; no private case judgments in Phase 5 artifacts |
| Selected method decision | Complete, unchanged | keep Phase 4 grounded-fusion selection; retrieval-only results do not select a new end-to-end method |
| Final method/evaluation freeze | Draft, pending | ../freeze/evaluation_freeze.json; required Phase 5 outputs and later operational test gates remain missing |
| Held-out evaluation | Sealed, not executed | no test reaction/label/prediction/diagnostic access; later separate authorization only |

Overall: retrieval robustness study complete; Phase 5 not fully complete. Do not close the final freeze until the required bounded grounded comparison and operational gates are resolved or an explicit later scope decision changes those requirements.
