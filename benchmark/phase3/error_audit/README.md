# Phase 3 reaction-label error-audit inventories

This directory is an inventory-only milestone built deterministically from the frozen 163-reaction Phase 3C validation pilot. No held-out test data, new inference, API calls, live KEGG queries, or biological adjudication were used.

## Files

- `all_validation_outcomes.csv`: one provenance-rich row for every validation reaction.
- `phase3c_noncorrect_review.csv`: all 61 grounded outcomes that were not exactly correct; this is the main file to inspect next.
- `phase3c_incorrect_selections.csv`: the 22 non-abstained incorrect selections.
- `phase3c_abstentions.csv`: the 39 abstentions, kept separate because an abstention is not automatically an error.
- `phase3c_compliance_cases.csv`: the 2 mechanically detected evidence-compliance cases.
- `all_method_disagreements.csv`: 142 error-enriched cases where frozen methods or correctness notions disagree.
- `source_provenance.json`, `inventory_summary.json`, and `deterministic_rebuild.json`: frozen inputs, accounting, and byte-rebuild evidence.

These inventories are review queues, not estimates of label-error prevalence. System agreement or disagreement is not biological evidence that a label is right or wrong, and all human-review fields are intentionally blank.

Prompt 2 will select the formal 60-case audit and construct Pass 1/Pass 2 review packets under a separately authorized, prespecified sampling and adjudication protocol. That work has not started here.
