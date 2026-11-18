# Scoped human-audit analysis

The implemented Phase 3C grounded-fusion validation pilot and scoped single-reviewer audit are complete. The same reviewer assessed the frozen 60-case stratified sample in two stages: blinded source-label assessment in Pass 1, followed by system-informed assessment in Pass 2. These are not independent reviews. Independent second review and inter-reviewer adjudication are explicitly deferred for this scoped report, superseding the historical two-reviewer plan. No agreement statistic or independently adjudicated reference labels are claimed.

Findings are descriptive judgments from a development validation sample. Disclose uncertainty, possible anchoring after system evidence is revealed, and reviewer involvement in system development. Neither a flag nor an unresolved biological limitation automatically prevents scoped completion. Frozen reference labels and benchmark metrics remain unchanged; no corrected accuracy is calculated. The older broader same-set reranker proposal is not declared fulfilled by this milestone.

The reusable implementation is `benchmark/scripts/phase3_error_analysis.py`; focused tests use synthetic fixtures only. It validates schema/dataset/canonical digests, exact 60-case IDs, required fields, completion markers, the embedded original Pass 1, stable identifier joins, and prior read-only artifact manifests. Optional blanks and a completed unresolved judgment are handled according to the reviewer schema. Revision comparisons report every changed answer field without supplying biological judgments.

Example, with explicitly selected private local paths:

```powershell
python -m benchmark.scripts.phase3_error_analysis --pass1 benchmark/phase3/error_audit/review_work/inputs/pass1.json --pass2 benchmark/phase3/error_audit/review_work/inputs/pass2.json --scope-note benchmark/phase3/error_audit/review_work/scope.md --prior-analysis benchmark/phase3/error_audit/review_work/prior_analysis --out benchmark/phase3/error_audit/review_work/new_analysis
python -m pytest tests/test_phase3_error_analysis.py -q -p no:cacheprovider
```

Choose a new ignored output directory for each analysis. Personal exports, case-level judgments, joined tables, private reports and scope records remain under ignored `review_work/`; they are intentionally excluded from this publication. Existing historical frozen files and manifests are not rewritten. This note documents methodological scope without publishing the review answers.
