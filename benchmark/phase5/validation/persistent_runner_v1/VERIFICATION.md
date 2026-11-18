# Persistent-runner verification and handoff

**128 tests passed, no skips**, including 32 runner/analysis tests. No live provider was constructed, no paid call occurred, and no live execution ledger exists. An HTTP MockTransport exercised the real SDK dispatch with a literal synthetic credential and compared the complete decoded provider JSON body; no request left the process. The actual production package was locked and independently reproduced without network access. See dry_run_verification.json and preservation_checks.json.

The tests cover:

- Complete frozen body dispatch; separate attempts for duplicate-body units; unit-specific resume; identical production-plan copies sharing one persistent ledger/OS lock.
- Exact recorded authorization, settings, model/endpoint, pricing and cap checks; USD 0 template rejected before dispatch; stale source/locked artifacts and incompatible caches rejected.
- Durable reservation before provider construction; interruption before/during dispatch; recovery after cache-write/ledger-update interruption; orphaned caches and inconsistent ledger charges failing closed.
- Failed/schema/compliance/missing-usage/model/tier outcomes, full failed-call reservations, no automatic repurchases, documented no-retry reconciliation, persistent attempt and dollar stops before dispatch.
- Cross-process lock exclusion; OS lock release; Windows long-path cache writes; maximum input/output/Fast/long-context cost arithmetic with reasoning counted once.
- Dry-run without provider/network/secret/held-out loading; frozen response alignment before lazy validation-reference loading; failures/abstentions retained in denominators; paired cluster bootstrap, undefined selective-accuracy draws and descriptive historical comparison.

Reproduce the suite from the Phase 5 worktree, choosing a fresh ignored test directory:

```powershell
python -m pytest tests/test_phase5_perturb.py tests/test_phase5_robustness.py tests/test_phase5_llm_plan.py tests/test_phase5_runner.py tests/test_phase5_reporting_review.py tests/test_phase3c_support_clarification.py tests/test_phase4_routing.py tests/test_phase4_freeze.py tests/test_phase3_error_analysis.py tests/test_phase3c_grounded.py::test_pydantic_schema_is_strict tests/test_phase3c_grounded.py::test_manifest_verification_is_read_only tests/test_phase3c_validation.py::test_retrieval_state_classification_accounting tests/test_phase3c_validation.py::test_paired_transition_accounting_is_exhaustive tests/test_phase3c_validation.py::test_cluster_bootstrap_is_deterministic tests/test_phase3c_validation.py::test_manifest_verification_is_read_only tests/test_phase3c_validation.py::test_decision_analysis_answers_all_prespecified_questions tests/test_phase3c_validation.py::test_committed_manifest_excludes_raw_scratch -q -p no:cacheprovider --basetemp benchmark/phase5/_pytest_runner_reproduced
python -m benchmark.scripts.phase5_runner --package benchmark/phase5/validation/persistent_runner_v1
python -c 'from benchmark.scripts.phase4_routing import verify_prior; r=verify_prior(); print(r); assert not any(r.values())'
git diff --check
```

All seven prior read-only verifiers pass. Twenty frozen retrieval artifacts retain their pre-review byte hashes; eleven original superseded snapshots still match reviewed Git blobs. All six source v2 input files match their locked hashes. Frozen Phase 3C code/reports/manifests, Phase 4 artifacts and the selected historical method have no tracked changes. Current modified reports/checklist/freeze and this proposal have new checksums in artifact_checksums.json; previous checksum packages remain historical and unchanged.

The Phase 5 worktree passes git diff --check. The original checkout was also checked and retains two **pre-existing user formatting flags** in goals.md: a Markdown hard-break at line 3 and a blank EOF line. This task preserved that file exactly (SHA-256 bd1b436b3404eddb16d86e45c0758191977b22d9b1ff265810361a2597ad7471), matching the previously pushed user update. These flags were not introduced by the runner work. New public text was checked separately because normal git diff omits untracked files. No unrelated files were changed.

SYNTHETIC_ANALYSIS.json is an executable-path demonstration on **two toy reactions**, four synthetic units, synthetic references and injected outputs. It contains no Phase 5 biological result and cannot be loaded as a live cohort. The synthetic cache/raw freeze remain ignored. The actual 163-pair cohort has no new outputs or analysis yet.

Authorization is USD 0; authorization.template.json remains unapproved. No approved production record exists. Proposed live/resume/response-freeze/analysis commands and the USD 40 ceiling are in EXECUTION_PROPOSAL.md. Raw bodies, responses, caches, authorization records and case-level analysis stay ignored. No held-out loader/executor, audit edits, retrieval/GPU rerun, paid inference, commit, push, PR, merge, tag or release occurred. Model availability is established only by public documentation, not by an account-specific inference probe. Phase 5 remains incomplete and the final freeze remains draft.
