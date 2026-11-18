"""Reporting integration: no revival of the superseded control or support claim."""
import shutil

import pytest

from benchmark.scripts import phase5_reporting as reporting
from benchmark.scripts.phase5_robustness import ROOT, load, write


def cached_report_fixture(tmp_path, monkeypatch):
    out = tmp_path / "benchmark/phase5/validation"
    out.mkdir(parents=True)
    for name in ("paired_family_summary.csv", "comparison_by_condition.csv", "realized_corruptions.csv", "llm_dry_run.json", "prediction_freeze.json"):
        shutil.copyfile(ROOT / "benchmark/phase5/validation" / name, out / name)
    baseline = tmp_path / "benchmark/phase4/freeze"
    baseline.mkdir(parents=True)
    for name in ("evaluation_freeze.json", "selected_method.json"):
        shutil.copyfile(ROOT / "benchmark/phase4/freeze" / name, baseline / name)
    monkeypatch.setattr(reporting, "ROOT", tmp_path)
    return out


def test_current_report_and_freeze_require_both_new_control_arms(tmp_path, monkeypatch):
    out = cached_report_fixture(tmp_path, monkeypatch)
    reporting.report(out, ROOT)
    freeze = load(tmp_path / "benchmark/phase5/freeze/evaluation_freeze.json")
    assert freeze["status"] == "draft"
    assert freeze["historical_clean_responses_are_matched_controls"] is False
    assert freeze["phase5_comparison_prompt_version"] == "phase5-grounded-no-provenance-v2"
    assert freeze["phase5_grounded_run_plan"]["n_clean_requests"] == 163
    assert freeze["phase5_grounded_run_plan"]["n_degraded_requests"] == 163
    assert freeze["phase5_grounded_run_plan"]["attempt_cap"] == 326
    text = (out / "REPORT.md").read_text(encoding="utf-8")
    assert "not 130 biological hallucinations" in text
    assert "grounded 102/163 versus fusion 93/163" in text
    assert "does not establish reduced biological hallucination" in text


def test_superseded_one_arm_plan_is_not_silently_reported_as_current(tmp_path, monkeypatch):
    out = cached_report_fixture(tmp_path, monkeypatch)
    write(out / "llm_dry_run.json", {"schema": "phase5-grounded-validation-plan-v1"})
    with pytest.raises(ValueError, match="superseded/confounded"):
        reporting.report(out, ROOT)
    assert not (out / "REPORT.md").exists()


def test_persistent_runner_handoff_keeps_freeze_draft_and_budget_unapproved(tmp_path, monkeypatch):
    out = cached_report_fixture(tmp_path, monkeypatch)
    llm = load(out / "llm_dry_run.json")
    package_name = "persistent_runner_gpt61_v2" if llm["model"] == "gpt-6.1-terra" else "persistent_runner_v1"
    source_package = ROOT / "benchmark/phase5/validation" / package_name
    destination = tmp_path / "benchmark/phase5/validation" / package_name
    destination.mkdir()
    shutil.copyfile(source_package / "plan.json", destination / "plan.json")
    reporting.report(out, ROOT)
    freeze = load(tmp_path / "benchmark/phase5/freeze/evaluation_freeze.json")
    active = freeze["phase5_persistent_execution_plan"]
    assert freeze["status"] == "draft"
    assert active["attempt_cap"] == 326 and active["proposed_dollar_cap_usd"] == 40
    assert active["currently_authorized_usd"] == 0
    text = (out / "COMPLETION_CHECKLIST.md").read_text(encoding="utf-8")
    assert "implemented and synthetic-tested" in text and "Paid outputs and biological analysis remain missing" in text
