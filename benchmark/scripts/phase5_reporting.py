"""Render Phase 5 retrieval reports/plots and a new explicitly draft freeze."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

from benchmark.scripts.phase5_robustness import ROOT, load, write


def csv_rows(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def plot(out, plot_deps):
    if plot_deps:
        sys.path.append(str(plot_deps))
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "benchmark/phase5/_mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    table = csv_rows(out / "paired_family_summary.csv")
    clean = {r["method"]: r for r in csv_rows(out / "comparison_by_condition.csv") if r["condition"] == "clean"}
    methods = ["rule_strict", "bm25", "trained", "fusion"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), layout="constrained")
    for method in methods:
        for ax, metric in zip(axes, ["recall_at_1", "recall_at_10"]):
            values = [next(r for r in table if r["method"] == method and r["family"] == f"remove_{x:g}") for x in [.1, .25, .5, .75]]
            ax.plot([0, .1, .25, .5, .75], [float(clean[method][metric])] + [float(r[metric]) for r in values], marker="o", label=method)
            ax.set(xlabel="Nominal removed identifier-bundle fraction", ylabel=metric.replace("recall_at_", "Recall@"), ylim=(0, 1), title="163 validation reactions; names/equations retained")
            ax.grid(alpha=.25)
    axes[0].legend()
    fig.savefig(out / "identifier_removal.svg")
    fig.savefig(out / "identifier_removal.png", dpi=160)
    plt.close(fig)
    families = ["remove_0.1", "remove_0.25", "remove_0.5", "remove_0.75", "reactants_1", "products_1", "ancestor_1", "wrong_0.25"]
    fig, ax = plt.subplots(figsize=(12, 5), layout="constrained")
    for offset, method in enumerate(methods):
        values = [next(r for r in table if r["method"] == method and r["family"] == f) for f in families]
        y = [float(v["delta_recall_at_10"]) for v in values]
        low, high = [[float(v[f"recall_at_10_delta_ci_{bound}"]) for v in values] for bound in ("low", "high")]
        # Some discrete bootstrap intervals can lie just outside a Monte Carlo
        # point estimate: plot interval endpoints directly, never invent errors.
        x = [i+(offset-1.5)*.15 for i in range(len(families))]
        ax.vlines(x, low, high, alpha=.65, colors=f"C{offset}")
        ax.scatter(x, y, color=f"C{offset}", label=method, s=25)
    ax.set_xticks(range(len(families)), families, rotation=25, ha="right")
    ax.axhline(0, color="black", linewidth=.7)
    ax.set(ylabel="Paired change in Recall@10", title="95% paired cluster + perturbation-replicate bootstrap intervals")
    ax.grid(axis="y", alpha=.2)
    ax.legend()
    fig.savefig(out / "robustness_deltas.svg")
    fig.savefig(out / "robustness_deltas.png", dpi=160)
    plt.close(fig)


def report(out, source):
    family = csv_rows(out / "paired_family_summary.csv")
    clean = {r["method"]: r for r in csv_rows(out / "comparison_by_condition.csv") if r["condition"] == "clean"}
    realized = csv_rows(out / "realized_corruptions.csv")
    llm = load(out / "llm_dry_run.json")
    if llm.get("schema") != "phase5-matched-grounded-plan-v2":
        raise ValueError("superseded/confounded v1 plan cannot serve as current reporting input; prepare matched v2 arms")
    lines = ["# Phase 5 robustness results and experiment coverage", "",
             "Status: local retrieval study complete; Phase 5 overall remains blocked on the valid matched clean/degraded grounded-LLM comparison. The validated Phase 4/3C architecture and historical results are retained. The proposed comparison prompt is now phase5-grounded-no-provenance-v2 and requires new matched controls; it is not yet an end-to-end validated or final-frozen replacement. No retrieval, GPU inference, training or adaptation was rerun for the review fix.", "",
             "## Population and executed work", "",
             "Exactly 163 frozen validation reactions, 19 aligned query conditions (3,097 total): clean, four random removal fractions with three seeds each, reactant-side removal, product-side removal, one-hop ancestor replacement, and three sibling-corruption seeds. This is not all 969 validation reactions or a held-out evaluation. The 609 annotated participant bundles are query-level occurrences, not 609 unique model species; there are 1,306 distinct identifier links across these occurrences.",
             "All variants were actually rebuilt and reranked for strict rules, BM25, the trained encoder and the selected fusion. Only byte-identical query/mapping inputs were memoized; 1,573 unique query texts were encoded. One fixed catalog encoding was shared because the catalog did not change. No clean rankings were assigned to changed inputs.", "",
             "| Method | Clean R@1 | Clean R@10 | Clean empty rate |",
             "| --- | ---: | ---: | ---: |"]
    for method, row in clean.items():
        lines.append(f"| {method} | {float(row['recall_at_1']):.4f} | {float(row['recall_at_10']):.4f} | {float(row['empty']):.4f} |")
    lines += ["", "The strict-rule control uses existing core matching/scoring with direct mappings and no iterative ontology relaxation. It is not the historical ontology-relaxed Phase 2 configuration. Its clean baseline is explicitly recomputed here and all its deltas are paired against that baseline.",
              "BM25 reproduces all 163 historical Top-100 lists. The encoder reproduces all historical Top-1 and Top-10 lists; one Top-100 tail differs due to the disclosed numerical inference path. Fusion Top-10 historical agreement is recorded in historical_clean_comparison.json. Comparisons below use recomputed clean results throughout, so no historical population or precision mix is hidden.", "",
              "## Paired robustness findings", "",
              "| Method | Perturbation family | Replicates | R@10 | Change vs clean | 95% interval for change |",
              "| --- | --- | ---: | ---: | ---: | --- |"]
    for row in family:
        lines.append(f"| {row['method']} | {row['family']} | {row['perturbation_replicates']} | {float(row['recall_at_10']):.4f} | {float(row['delta_recall_at_10']):+.4f} | [{float(row['recall_at_10_delta_ci_low']):+.4f}, {float(row['recall_at_10_delta_ci_high']):+.4f}] |")
    lines += ["", "The trained encoder is descriptively most stable under identifier removal (R@10 near its clean 0.7423 across all four levels). BM25 loses substantial R@10 at high removal. Fusion starts from higher clean R@10 and degrades less than BM25 in point estimates, but its intervals are wide and often include zero. paired_method_robustness.csv provides paired differences in changes between methods; these distinguish stability from absolute retrieval performance. Do not infer universal superiority from one curve or overlapping intervals. These are reused-development-pilot results, not an independent final test.",
              "Removing constraints can improve strict-rule candidate recall by admitting more reactions; this is not proof that less biological evidence improves annotation. Candidate counts, emptiness, R@1/3/5/10/100 and MRR@10/100 must be read together. Full-catalog candidate recall is query-independent catalog coverage, not success at a useful rank.",
              "Names, equation text, stoichiometry and other participant evidence remain present. Neural stability here does not establish robustness to loss of all biological information. Ancestor replacement retains direct KEGG IDs and many changes are consequently masked. Sibling corruptions are synthetic graph-constrained perturbations, not independently verified biological misannotations.", "",
              "## Realized doses and unchanged cases", "",
              "| Condition | Changed bundles / eligible | Removed identifier links / original | Unchanged queries / 163 | Selected but unchanged bundles |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for row in realized:
        lines.append(f"| {row['condition']} | {row['changed_species']}/{row['eligible_species']} | {row['links_removed']}/{row['links_before']} | {row['unchanged_query']}/163 | {row['unchanged_selected_species']} |")
    lines += ["", "Random removal uses per-query half-up rounding, so nominal 10/25/50/75% corresponds to 8.05/27.59/56.65/78.33% of annotated bundles in this pilot. Each seed selects the same count but can remove different numbers of identifier links. One-sided removal has 8 reactant-side and 2 product-side unchanged/ineligible queries, retained in denominators. No shared-both-sides participant was selected in this pilot; the implementation/test handles that case. Ancestor replacement changes 324/609 bundles with 68 unchanged queries; 285 changed bundles retain direct KEGG evidence. Sibling replacement changes 78/84/82 bundles of 609 (12.81/13.79/13.46%), with 90/84/86 selected attempts unable to change; no outcome-driven refill is performed.",
              "The full event-level perturbation_manifest.jsonl records old/new IDs and one-hop distances or no-replacement outcomes. No species names, stoichiometry, sides or equations were altered. SBML and ontology/mapping source bytes remain unchanged.", "",
              "## Uncertainty and limitations", "",
              "Two thousand bootstrap draws, seed 20261106, pair clean and degraded query outcomes while resampling model clusters and stochastic perturbation replicate indices. The same 163 query denominator is used across conditions; three seeds are not 489 independent reactions. Deterministic side/ancestor conditions have only cluster variation. There are 12 clusters, three stochastic seeds and a development-stratified pilot; intervals are descriptive, not population-wide error bars or multiplicity-corrected confirmatory evidence. Individual-seed minima/maxima and stratum/cluster denominators are exported. Corruption generation never reads reference membership, audit flags or known correctness.", "",
              "## Missing end-to-end comparison and concrete resource decision", "",
              f"The corrected comparison has {llm['n_clean_requests']} clean and {llm['n_degraded_requests']} degraded requests ({llm['degraded_condition']}) for the same 163 validation pairs. The dry plan needs {llm['new_calls_needed']} unit-level calls and has {llm['compatible_historical_clean_cache_hits']} compatible historical clean reuses. There are {llm['unique_provider_bodies']} distinct provider bodies; existing per-experimental-unit cache policy is retained. No paid calls were made and no predictions fabricated. Historical clean responses are not the sole matched baseline for this changed prompt. Retrieval recall is not end-to-end correctness or abstention robustness.",
              "The real unchanged-input diagnostic reconstructed all 163 requests through the original ea241ea construction path. All 163 differ in visible provenance (4,890 leaf differences). P3P0096 also changes E07's trained rank 70 to 69 and fused score 0.020850202429 to 0.020909832721; no candidate order, biological field or provider setting changes. Numeric drift is retained, not silently reset. See matched_prompt_gpt61_v2/CLEAN_CONTROL.md, its complete field-difference JSON/CSV and paired request checks. The v2 serializer omits operational lineage from model-visible evidence and records accurate new lineage in sidecars. Both arms use the same selected model, complete provider settings and typed response schema; literal input strings and all actual provider fields determine cache identity. The GPT-5.6 plan and the intermediate GPT-6.1 v1 plan remain preserved under their superseded snapshots.",
              f"At the verified Standard short-context rate snapshot for the recorded model, the input-estimate/historical-mean-output scenario is ${llm['estimated_cost_usd_at_historical_mean_output']:.2f}; allowing all 2,048 output tokens gives ${llm['estimated_cost_usd_at_max_output_tokens']:.2f}. Estimates include both arms and the response schema and are not guarantees. A 32,768-input-unit pessimistic reservation is ${llm['conservative_32768_input_reservation_total_usd']:.2f}; proposed hard cap ${llm['proposed_hard_cap_usd']}, one attempt per unit-level request, zero currently authorized. No cache discount or regional modifier is assumed. [Official OpenAI API pricing](https://developers.openai.com/api/docs/pricing) was fetched on 2026-11-06; rates are recorded separately without changing old snapshots.",
              f"Expected value: measure paired grounded correctness, coverage, selective accuracy and reference-based abstention under identifier removal using a consistent prompt. Both clean and degraded responses must be collected in an interleaved matched cohort, recording returned model, SDK, endpoint and time and pausing for identity drift. Identical bodies and identical returned aliases cannot guarantee unchanged hosted-model behavior. The required decision is separate authorization for the {llm['new_calls_needed']}-unit validation plan and USD {llm['proposed_hard_cap_usd']} ceiling. Execution stays disabled pending a separate exact-plan authorization; no held-out executor is enabled.",
              "Iterative ontology relaxation, direct LLM perturbation runs, other grounded perturbation cells, all-969 expansion and retraining are deferred optional extensions. Required comparisons are not silently replaced by clean cached outputs.", "",
              "## Phase 3C support interpretation", "",
              "See the [Phase 3C support addendum](../../PHASE3C_SUPPORT_ADDENDUM.md). The historical 130 direct in-catalog answers without retrieved evidence include 49 correct reference matches. They are not 130 biological hallucinations. Zero out-of-evidence grounded IDs is evidence compliance, not zero errors; there are still 22 in-evidence reference-incorrect selections. The independent accuracy comparison, grounded 102/163 versus fusion 93/163, is preserved. A bootstrap of differently defined support indicators does not establish reduced biological hallucination. Frozen reports/outputs/manifests remain unchanged.", "",
              "## Completion and next step", "",
              "See COMPLETION_CHECKLIST.md and freeze/evaluation_freeze.json. Phase 5 retrieval work is complete, overall Phase 5 is not complete while the required degraded grounded comparison is blocked. Final method selection remains the Phase 4 grounded-fusion method; no adaptation is justified by this retrieval-only study. Resolve the bounded validation run before finalizing the method/evaluation freeze, then integrate/authorize a separate later one-time held-out evaluation. The held-out test stays sealed.", ""]
    runner_dir = "persistent_runner_gpt61_v2" if llm.get("model") == "gpt-6.1-terra" else "persistent_runner_v1"
    runner_path = ROOT / "benchmark/phase5/validation" / runner_dir / "plan.json"
    runner_plan = load(runner_path) if runner_path.exists() else None
    checklist = CHECKLIST
    if runner_plan:
        lines.insert(2, f"**Current execution proposal:** {runner_dir}/EXECUTION_PROPOSAL.md; model {llm['model']}, implemented runner, 326 unit attempts maximum, USD {runner_plan['dollar_cap_nusd']/1e9:.2f} proposed ceiling, USD 0 authorized. The earlier model-specific budget paragraphs are preserved planning history, not live authorization.")
        lines = [("Pre-runner planning snapshot, superseded for execution: " + line) if line.startswith("At the verified Standard") else line for line in lines]
        est = runner_plan["estimates"]
        lines += ["## Current persistent-runner proposal (supersedes the earlier GPT-5.6 execution plan)", "",
            f"The 326 exact matched requests use {llm['model']} in both clean and degraded arms and contain 317 distinct provider bodies. The locked package is {runner_dir}/; its plan, model-specific unit cache keys, fixed interleaved order, duplicate groups and source hashes are frozen separately. One attempt per (condition, sample_id); a duplicate body is a separate stochastic call, not another independent biological sample. GPT-5.6 responses are never substituted for GPT-6.1 outputs.",
            f"Plan digest: `{runner_plan['plan_digest']}`. The runner and response-freeze/paired-analysis entry point are implemented and tested locally with synthetic providers. Explicit recorded authorization is still required; current authorization is USD 0. See {runner_dir}/EXECUTION_PROPOSAL.md for live/resume commands and billing assumptions.",
            f"For GPT-6.1 Terra, public Standard pricing gives USD {est['standard_no_write_mean_output_usd']:.2f} at estimated input plus the historical mean output scenario; default 30-minute writes raise that scenario to USD {est['standard_all_30m_writes_mean_output_usd']:.2f}. Estimated input plus all 2,048 output tokens and default writes is USD {est['standard_all_30m_writes_max_output_usd']:.2f}. For dispatch reservations, each complete provider-body UTF-8 byte count plus a 4,096-token wrapper allowance bounds input; all input is priced at default 30-minute cache-write rates, and maximum output is 2,048 tokens including reasoning. The maximum request bound is {est['maximum_input_token_bound']:,} input tokens (largest complete body {est['maximum_complete_body_bytes']:,} bytes); all requests remain below the 272,000-token long-context threshold. Reservations include Fast's 2x multiplier. The largest per-unit reservation is USD {runner_plan['reservation_nusd']/1e9:.5f}; the sum of all 326 per-unit reservations, including the maximum output allowance on every unit, is USD {runner_plan['total_reservations_nusd']/1e9:.6f}. The proposed USD {runner_plan['dollar_cap_nusd']/1e9:.2f} cap covers this full reserved cohort with USD {runner_plan['dollar_cap_nusd']/1e9-runner_plan['total_reservations_nusd']/1e9:.2f} remaining under these locked assumptions. Only a successful response with expected identity/tier, schema/evidence compliance and in-bound provider-reported usage releases the unused portion to a write-priced usage ceiling; this is not invoice reconciliation. Failed, noncompliant or uncertain attempts retain their full per-unit reservation; if metered usage implies more, the larger amount is accounted and execution halts before another dispatch. No account-specific surcharges are included.",
            "All copies of the same production plan share an OS lock and durable ledger. Interrupted/uncertain attempts require documented no-retry reconciliation. The analysis first freezes all aligned responses, then opens only the frozen validation reference join. Failed and abstained units remain in denominators; cluster bootstrap pairs conditions on the same 163 reactions. The hosted model is documented publicly but account-specific availability remains unverified without inference. Held-out execution remains disabled.", ""]
        checklist = checklist.replace("proposed USD 30 cap, zero authorization", f"GPT-6.1 Terra; USD {runner_plan['dollar_cap_nusd']/1e9:.2f} proposed cap, zero authorization")
        checklist += f"\nPersistent runner and paired response-freeze/analysis entry point: implemented and synthetic-tested. Locked 326-unit {llm['model']} proposal: {runner_dir}/EXECUTION_PROPOSAL.md (plan digest {runner_plan['plan_digest']}). Paid outputs and biological analysis remain missing; Phase 5 remains incomplete.\n"
    (out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    (out / "COMPLETION_CHECKLIST.md").write_text(checklist, encoding="utf-8")
    freeze_dir = ROOT / "benchmark/phase5/freeze"
    freeze_dir.mkdir(exist_ok=True)
    prior = load(ROOT / "benchmark/phase4/freeze/evaluation_freeze.json")
    updated = {**prior, "schema": "phase5-evaluation-freeze-draft-v1", "status": "draft",
               "phase5_status": "retrieval study complete; valid matched v2 clean/degraded grounded comparison blocked",
               "method_selection": "validated architecture retained; v2 comparison prompt is proposed, not yet validated or finally frozen",
               "phase5_comparison_prompt_version": llm["prompt_version"],
               "review_fix_checksums": "benchmark/phase5/review_fixes/artifact_checksums.json",
               "historical_clean_responses_are_matched_controls": False,
               "phase5_clean_control_diagnostic": f"benchmark/phase5/validation/{llm['request_manifest'].split('/')[0]}/clean_control_diagnostic.json",
               "phase3c_support_clarification": "benchmark/PHASE3C_SUPPORT_ADDENDUM.md",
               "sequence": "Phase 5 development -> final method/evaluation freeze -> later separately authorized held-out evaluation",
               "phase5_protocol": "benchmark/PHASE5.md", "phase5_prediction_freeze": load(out / "prediction_freeze.json"),
               "phase5_grounded_run_plan": llm,
               "blockers": ["Both matched v2 clean and degraded validation outputs missing (326 case/arm requests); budget and live-run authorization not supplied.",
                            f"Record separate exact-plan authorization, execute both matched {llm['model']} arms, freeze responses and finish paired analysis under the proposed USD {llm['proposed_hard_cap_usd']}/326-attempt cap.",
                            "Later held-out executor/label-sealing integration and held-out budget remain unfinished; held-out execution disabled.",
                            "Record hosted model identity/SDK/endpoint/time within an interleaved cohort; unchanged request bodies or aliases do not rule out hidden provider/temporal drift."],
               "preserved_history": "Phase 4 draft retained unchanged under benchmark/phase4/freeze; new planning package only",
               "phase5_reference_policy": "unchanged benchmark labels, no human-audit corrected labels",
               "adaptation_or_retraining": False}
    if runner_plan:
        updated["phase5_persistent_execution_plan"] = {"path": f"benchmark/phase5/validation/{runner_dir}/plan.json",
            "plan_digest": runner_plan["plan_digest"], "attempt_cap": 326, "proposed_dollar_cap_usd": 40,
            "currently_authorized_usd": 0, "per_call_reservation_usd": runner_plan["reservation_nusd"]/1e9,
            "sum_worst_case_per_unit_reservations_usd": runner_plan["total_reservations_nusd"]/1e9,
            "state": "implemented, dry-run verified; no live responses; analysis only synthetic-tested"}
        updated["blockers"][1] = "Persistent runner implemented; record exact-plan authorization/billing assumptions, execute both matched arms, freeze responses and finish paired analysis. No authorization exists."
        updated["blockers"][0] = f"Both matched {llm['model']} arms missing (326 unit attempts maximum); current USD {llm['proposed_hard_cap_usd']} ceiling is a proposal, not authorization. GPT-5.6 plans/results remain historical and are not reused."
    write(freeze_dir / "evaluation_freeze.json", updated)
    write(freeze_dir / "selected_method.json", load(ROOT / "benchmark/phase4/freeze/selected_method.json"))
    (out / "REPRODUCE.md").write_text(
        "# GPT-6.1 matched-plan reproduction (no GPU or paid calls)\n\nRun from the Phase 5 worktree. SOURCE_ROOT is the preserved original workspace containing the frozen local catalog; all retrieval rankings are reused unchanged. Choose new plan/package names.\n\n```powershell\n"
        "python -m benchmark.scripts.phase5_llm_plan --dry-run --source-root SOURCE_ROOT --out benchmark/phase5/validation --plan-name NEW_GPT61_PLAN_NAME --model gpt-6.1-terra --no-publish-current\n"
        "python -m benchmark.scripts.phase5_runner --package benchmark/phase5/validation/NEW_GPT61_RUNNER_NAME --build-from benchmark/phase5/validation/NEW_GPT61_PLAN_NAME\n"
        "python -m benchmark.scripts.phase5_runner --package benchmark/phase5/validation/NEW_GPT61_RUNNER_NAME\n"
        "python -m benchmark.scripts.phase3c_support_clarification --source-root SOURCE_ROOT\n"
        "python -m benchmark.scripts.phase5_reporting --source-root SOURCE_ROOT --out benchmark/phase5/validation --no-plots\n"
        "```\n\nThe builder writes new plans only; --no-publish-current preserves current dry-run/manifest pointers during independent reproduction. It never overwrites a named existing plan. Provider bodies are complete SDK 1.78.1 JSON request bodies including the strict response schema, pinned to Pydantic 2.13.5. GPT-6.1 Terra is documented for Responses and structured outputs; account-level availability is not checked. No provider client or HTTP transport is created. GPT-5.6 plans/results are preserved and cannot be used as GPT-6.1 outputs. Reverify the serializer before any paid run. Historical retrieval reproduction commands and the original confounded plan are retained under superseded/. The reporting stage changes mutable reports/current draft only; historical retrieval tables/figures, Phase 3C artifacts and Phase 4 files remain unchanged. Paid/held-out execution stays disabled.\n", encoding="utf-8")


CHECKLIST = """# Phase 5 completion checklist

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
| Grounded selected-method robustness at 50% removal | Blocked, required | Active GPT-6.1 plan in llm_dry_run.json and matched_prompt_gpt61_v2/request_manifest.jsonl; 326 calls, $17.994805 total per-request reservation, USD 40 cap, USD 0 authorized; no GPT-5.6 response reuse |
| Phase 3C support interpretation | Clarified | benchmark/PHASE3C_SUPPORT_ADDENDUM.md; 130 direct un-retrieved answers include 49 correct; zero outside-evidence IDs is compliance; 102 vs 93 accuracy preserved |
| Other grounded perturbation cells and direct LLM recovery | Deferred optional | no compatible changed-input outputs; not every method-condition combination is required |
| Iterative ontology relaxation / EM comparison | Deferred optional | existing expensive configuration not part of bounded strict-control study; no claim of execution |
| All-969 validation, adaptation or retraining | Deferred optional | not part of current fixed 163-query protocol |
| Human labels / audit reopening | Not required, excluded | completed scoped audit preserved; no private case judgments in Phase 5 artifacts |
| Selected method decision | Validated architecture retained; v2 prompt pending | keep Phase 4 architecture/historical baseline; changed prompt is not yet end-to-end validated and cannot be declared finally frozen |
| Final method/evaluation freeze | Draft, pending | ../freeze/evaluation_freeze.json; required Phase 5 outputs and later operational test gates remain missing |
| Held-out evaluation | Sealed, not executed | no test reaction/label/prediction/diagnostic access; later separate authorization only |

Overall: retrieval robustness study complete; Phase 5 not fully complete. Do not close the final freeze until the required bounded grounded comparison and operational gates are resolved or an explicit later scope decision changes those requirements.
"""


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--plot-deps", type=Path)
    parser.add_argument("--no-plots", action="store_true", help="update mutable reports only; preserve historical retrieval figures")
    args = parser.parse_args(argv)
    if not args.no_plots:
        plot(args.out, args.plot_deps)
    report(args.out, args.source_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
