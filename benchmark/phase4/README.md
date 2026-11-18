# Phase 4 artifact entry point

Read ../PHASE4.md for the protocol and ROADMAP_STATUS.md for the scoped completion statement. Work is on the isolated benchmark/phase-4 branch; the original benchmark/error-audit checkout remains unchanged. No commits, pushes, PRs, tags, releases or merges were performed.

- validation/REPORT.md: reproducible comparison, definitions, results and limitations.
- validation/comparison.csv: all 9 fixed policy settings on the same 163 cached validation cases.
- validation/subgroups.csv: corrected retrieval-stratum, cluster and model denominators/results.
- validation/metrics.json: risk-coverage curves/AURC, correctness detection and cluster-bootstrap intervals.
- validation/answer_absence_detection.json: mechanical reference-absence diagnostics.
- validation/selection.json: all prespecified retain/reject criteria and final selection.
- validation/validation_tradeoffs.svg and .png: standalone scientific figures.
- validation/provenance.json: exact source/input byte and portable hashes, code base revision and runtime versions; routing decisions frozen before the reference join.
- freeze/evaluation_freeze.json: explicitly draft operational evaluation contract and concrete blockers.
- freeze/selected_method.json, system_prompt.txt and response_schema.json: selected method, frozen baseline configuration, model/checkpoint/retrieval identity, and exact system instructions/schema.
- freeze/synthetic_dry_run.json: tested fake-provider exercise; zero held-out file access or network calls.
- freeze/checkpoint_availability.json: selected epoch-1 checkpoint verified by bytes in the original workspace, without loading the model or copying it into this worktree.
- VERIFICATION.md: tests, read-only manifest verification, reproduction checks and preservation.

Code: benchmark/scripts/phase4_routing.py and phase4_freeze.py. Tested additional dependencies: benchmark/requirements-phase4.txt. Plotting dependencies are local and ignored under _deps/. Scratch, caches, partial attempts and reproduction products beginning with _ are ignored. Public artifacts contain no personal review exports or audit-derived case judgments.

The --execute command is intentionally fail-closed. It must not be represented as an available held-out evaluator or as permission to run one.
