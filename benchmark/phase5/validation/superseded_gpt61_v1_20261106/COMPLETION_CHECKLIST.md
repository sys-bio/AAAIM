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
| Real 163-case clean-control diagnostic | Complete | matched_prompt_v2/clean_control_diagnostic.json and field-level CSV; provenance changes in all cases plus explicit P3P0096 rank/score drift; full provider body checked |
| Corrected paired prompt/control design | Prepared, not executed | phase5-grounded-no-provenance-v2: 163 matched clean + 163 degraded requests, identical serialization/settings/schema, accurate provenance in sidecars; historical responses are not sole controls |
| Grounded selected-method robustness at 50% removal | Blocked, required | Active GPT-6.1 plan in llm_dry_run.json and matched_prompt_gpt61_v1/request_manifest.jsonl; 326 calls, USD 40 proposed cap, USD 0 authorized; no GPT-5.6 response reuse |
| Phase 3C support interpretation | Clarified | benchmark/PHASE3C_SUPPORT_ADDENDUM.md; 130 direct un-retrieved answers include 49 correct; zero outside-evidence IDs is compliance; 102 vs 93 accuracy preserved |
| Other grounded perturbation cells and direct LLM recovery | Deferred optional | no compatible changed-input outputs; not every method-condition combination is required |
| Iterative ontology relaxation / EM comparison | Deferred optional | existing expensive configuration not part of bounded strict-control study; no claim of execution |
| All-969 validation, adaptation or retraining | Deferred optional | not part of current fixed 163-query protocol |
| Human labels / audit reopening | Not required, excluded | completed scoped audit preserved; no private case judgments in Phase 5 artifacts |
| Selected method decision | Validated architecture retained; v2 prompt pending | keep Phase 4 architecture/historical baseline; changed prompt is not yet end-to-end validated and cannot be declared finally frozen |
| Final method/evaluation freeze | Draft, pending | ../freeze/evaluation_freeze.json; required Phase 5 outputs and later operational test gates remain missing |
| Held-out evaluation | Sealed, not executed | no test reaction/label/prediction/diagnostic access; later separate authorization only |

Overall: retrieval robustness study complete; Phase 5 not fully complete. Do not close the final freeze until the required bounded grounded comparison and operational gates are resolved or an explicit later scope decision changes those requirements.

Persistent runner and paired response-freeze/analysis entry point: implemented and synthetic-tested. Locked 326-unit gpt-6.1-terra proposal: persistent_runner_gpt61_v1/EXECUTION_PROPOSAL.md (plan digest c1c8cbec4df627983efc235d52c2a0e248689cf068ebc153327e0fdf7e3e95b0). Paid outputs and biological analysis remain missing; Phase 5 remains incomplete.
