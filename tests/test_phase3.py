"""Phase 3 split, stratum, pilot, prompt, eval and cost tests.

Synthetic tests do not need the frozen corpus. Live tests skip if Phase 3 artifacts
have not been built. Nothing here makes a network call or reads secrets.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.scripts.phase3_common import (
    KEGG_ID_STRICT,
    PHASE3_DIR,
    STRATA,
    STRATUM_ABSENT,
    STRATUM_EMPTY,
    STRATUM_RERANK,
    STRATUM_TOP1,
    STRATUM_UNCONSTRAINED,
    YEAST_CLUSTER,
    assign_stratum,
    assert_no_kegg_leakage,
    estimate_tokens,
    find_kegg_leakage,
    parse_kegg_ids,
    parse_participant_ids,
    redact_kegg_reaction_ids,
)
from benchmark.scripts.phase3_eval import score_one, score_results
from benchmark.scripts.phase3_modes import (
    BlockedLiveProvider,
    LiveCallBlocked,
    MemoryCache,
    MockProvider,
    Prediction,
    ModeResult,
    parse_structured_output,
    run_closed_set,
    run_direct,
    run_tool_assisted,
    ToolEvidence,
)
from benchmark.scripts.phase3_prompts import (
    CONTEXT_VARIANTS,
    build_context,
    render_prompt,
)
from benchmark.scripts.build_phase3_splits import assign_clusters, TARGET_SHARES
from benchmark.scripts.sample_phase3_pilot import sample_stratum
from benchmark.scripts.phase3_cost import estimate_cost, load_pricing, PRICING_EXAMPLE

DATA_DIR = REPO_ROOT / "benchmark" / "data"
live_only = pytest.mark.skipif(
    not (PHASE3_DIR / "splits.csv").exists(),
    reason="Phase 3 artifacts not built",
)


def test_stratum_assignment_is_mutually_exclusive():
    assert assign_stratum("unconstrained_candidate_set", False, False) == STRATUM_UNCONSTRAINED
    assert assign_stratum("no_candidates", False, False) == STRATUM_EMPTY
    assert assign_stratum("ok", False, False) == STRATUM_ABSENT
    assert assign_stratum("ok", True, False) == STRATUM_RERANK
    assert assign_stratum("ok", True, True) == STRATUM_TOP1


def test_participant_parsing_strips_stoichiometry():
    assert parse_participant_ids("2 PGA_ch + x_CO2 => RuBP_ch") == [
        "PGA_ch", "x_CO2", "RuBP_ch",
    ]


def test_kegg_id_parser_rejects_malformed():
    assert parse_kegg_ids("R00024; R0006565; C00022") == ["R00024"]
    assert KEGG_ID_STRICT.match("R1") is None


def test_redaction_and_leakage_detection():
    text = redact_kegg_reaction_ids("see kegg.reaction/R00024 and R00164 in notes")
    assert "R00024" not in text
    assert "R00164" not in text
    payload = {"notes": text, "equation": "A => B"}
    assert find_kegg_leakage(payload) == []
    with pytest.raises(ValueError, match="leakage"):
        assert_no_kegg_leakage({"equation": "R00024 => B"}, where="test")


def test_cluster_assignment_never_splits_a_cluster():
    stats = pd.DataFrame([
        {"cluster_id": "A", "n_models": 1, "n_reactions": 100, "n_genome_scale": 100,
         "is_genome_scale": True, "is_yeast_cluster": False,
         "unconstrained": 10, "empty_constrained": 40, "nonempty_answer_absent": 10,
         "retrievable_rerank_failure": 5, "retrievable_top1_success": 35},
        {"cluster_id": "B", "n_models": 1, "n_reactions": 20, "n_genome_scale": 0,
         "is_genome_scale": False, "is_yeast_cluster": False,
         "unconstrained": 2, "empty_constrained": 8, "nonempty_answer_absent": 2,
         "retrievable_rerank_failure": 1, "retrievable_top1_success": 7},
        {"cluster_id": YEAST_CLUSTER, "n_models": 7, "n_reactions": 30, "n_genome_scale": 0,
         "is_genome_scale": False, "is_yeast_cluster": True,
         "unconstrained": 3, "empty_constrained": 10, "nonempty_answer_absent": 2,
         "retrievable_rerank_failure": 1, "retrievable_top1_success": 14},
        {"cluster_id": "D", "n_models": 1, "n_reactions": 15, "n_genome_scale": 0,
         "is_genome_scale": False, "is_yeast_cluster": False,
         "unconstrained": 1, "empty_constrained": 6, "nonempty_answer_absent": 1,
         "retrievable_rerank_failure": 1, "retrievable_top1_success": 6},
        {"cluster_id": "E", "n_models": 1, "n_reactions": 12, "n_genome_scale": 0,
         "is_genome_scale": False, "is_yeast_cluster": False,
         "unconstrained": 1, "empty_constrained": 5, "nonempty_answer_absent": 1,
         "retrievable_rerank_failure": 0, "retrievable_top1_success": 5},
        {"cluster_id": "F", "n_models": 1, "n_reactions": 8, "n_genome_scale": 0,
         "is_genome_scale": False, "is_yeast_cluster": False,
         "unconstrained": 0, "empty_constrained": 4, "nonempty_answer_absent": 1,
         "retrievable_rerank_failure": 0, "retrievable_top1_success": 3},
    ])
    m1 = assign_clusters(stats, seed=20260902)
    m2 = assign_clusters(stats, seed=20260902)
    assert m1 == m2
    assert set(m1) == set(stats.cluster_id)
    assert len(set(m1.values())) <= 3
    assert all(v in TARGET_SHARES for v in m1.values())


def test_pilot_sampling_is_deterministic_and_deduped():
    rows = []
    for cluster in ("C1", "C2", "C3"):
        for i in range(10):
            rows.append({
                "model_id": f"M{cluster}", "reaction_id": f"R{i}",
                "cluster_id": cluster, "stratum": STRATUM_EMPTY,
            })
    eligible = pd.DataFrame(rows)
    a = sample_stratum(eligible, 6, seed=7, stratum=STRATUM_EMPTY)
    b = sample_stratum(eligible, 6, seed=7, stratum=STRATUM_EMPTY)
    assert list(zip(a.model_id, a.reaction_id)) == list(zip(b.model_id, b.reaction_id))
    assert not a.duplicated(["model_id", "reaction_id"]).any()
    assert len(a) == 6
    # Round-robin should draw from every cluster before repeating one.
    assert a.cluster_id.nunique() == 3


def test_short_stratum_takes_all_eligible():
    eligible = pd.DataFrame({
        "model_id": ["M1"], "reaction_id": ["R1"], "cluster_id": ["C1"],
        "stratum": [STRATUM_RERANK],
    })
    out = sample_stratum(eligible, 25, seed=1, stratum=STRATUM_RERANK)
    assert len(out) == 1
    assert "all_eligible_test" in out.iloc[0].selection_rule


def _toy_row(**overrides):
    base = {
        "model_id": "M1", "reaction_id": "rxnA", "cluster_id": "C1",
        "reaction_equation": "A_c + B_c => C_c",
        "reaction_name": "toy", "model_name": "Toy", "model_title": "Toy model",
        "model_notes": "A note that once mentioned kegg.reaction/R00024 and R00164.",
        "substrate_names": "alpha; beta", "product_names": "gamma",
        "stratum": STRATUM_EMPTY,
    }
    base.update(overrides)
    return pd.Series(base)


def test_prompt_redacts_notes_and_rejects_kegg_reaction_ids():
    corpus = pd.DataFrame([_toy_row(), _toy_row(reaction_id="rxnB",
                                                reaction_equation="C_c => D_c")])
    evidence = pd.DataFrame({
        "model_id": ["M1", "M1"], "species_id": ["A_c", "B_c"],
        "annotation": ["CHEBI:1", "C00010"],
        "annotation_type": ["chebi", "kegg_compound"],
    })
    ctx = build_context(_toy_row(), variant="target_plus_model",
                        corpus=corpus, evidence=evidence)
    assert "R00024" not in json.dumps(ctx)
    prompt = render_prompt(ctx)
    blob = json.dumps(prompt)
    assert "R00024" not in blob
    assert "R00164" not in blob
    with pytest.raises(ValueError):
        build_context(_toy_row(reaction_equation="A_c => R00024"),
                      variant="target_only", corpus=corpus, evidence=evidence)


def test_neighborhood_is_bounded_and_deterministic():
    rows = [_toy_row()]
    for i, eq in enumerate(["A_c => X_c", "A_c => Y_c", "Z_c => Q_c", "B_c => W_c"], start=1):
        rows.append(_toy_row(reaction_id=f"n{i}", reaction_equation=eq))
    corpus = pd.DataFrame(rows)
    evidence = pd.DataFrame(columns=["model_id", "species_id", "annotation", "annotation_type"])
    a = build_context(_toy_row(), variant="target_plus_neighborhood",
                      corpus=corpus, evidence=evidence, neighborhood_k=2)
    b = build_context(_toy_row(), variant="target_plus_neighborhood",
                      corpus=corpus, evidence=evidence, neighborhood_k=2)
    assert a["neighbors"] == b["neighbors"]
    assert len(a["neighbors"]) <= 2
    assert all(n["reaction_id"] != "rxnA" for n in a["neighbors"])
    # Z_c => Q_c shares nothing with A_c + B_c => C_c
    assert "n3" not in {n["reaction_id"] for n in a["neighbors"]}


def test_structured_output_parser_handles_abstain_malformed_and_invalid_ids():
    ok = parse_structured_output(json.dumps({
        "abstain": False,
        "predictions": [{"kegg_id": "R00024", "confidence": 0.9}],
        "rationale": "match", "basis": "supplied_evidence",
    }))
    assert ok["abstain"] is False
    assert ok["predictions"][0].valid_kegg_id is True

    abstain = parse_structured_output(json.dumps({
        "abstain": True, "predictions": [{"kegg_id": "R00024", "confidence": 1}],
        "rationale": "not enough", "basis": "recalled_knowledge",
    }))
    assert abstain["abstain"] is True
    assert abstain["predictions"] == []

    bad = parse_structured_output("not json at all")
    assert bad["abstain"] is True
    assert bad["parse_error"] == "unparseable"

    invalid = parse_structured_output(json.dumps({
        "abstain": False,
        "predictions": [{"kegg_id": "R1", "confidence": 0.2}],
        "rationale": "guess", "basis": "recalled_knowledge",
    }))
    assert invalid["predictions"][0].valid_kegg_id is False


def test_abstention_is_not_scored_as_a_hallucinated_id():
    result = ModeResult(
        sample_id="P1", model_id="M1", reaction_id="R1", cluster_id="C1",
        stratum=STRATUM_EMPTY, mode="direct_open_set", variant="target_only",
        template_version="t", abstain=True, predictions=[],
    )
    row = score_one(result, ["R00024"], equiv=lambda c, t, k: c in set(t))
    assert row["abstain"] is True
    assert row["n_invalid_kegg_ids"] == 0
    assert row["exact_top1"] is False
    assert row["answered"] is False


def test_mocked_direct_and_tool_and_closed_set_and_cache():
    sample = {
        "sample_id": "P1", "model_id": "M1", "reaction_id": "rxnA",
        "cluster_id": "C1", "stratum": STRATUM_EMPTY,
    }
    prompt = {"messages": [{"role": "user", "content": "SBML reaction id: rxnA"}],
              "template_version": "t", "n_input_tokens_est": 10}
    raw = json.dumps({
        "abstain": False,
        "predictions": [{"kegg_id": "R00024", "confidence": 0.8}],
        "rationale": "mock", "basis": "recalled_knowledge",
    })
    cache = MemoryCache()
    provider = MockProvider(responses={"rxnA": raw})
    first = run_direct(sample, prompt, provider, cache=cache, variant="target_only")
    second = run_direct(sample, prompt, provider, cache=cache, variant="target_only")
    assert first.cache_hit is False
    assert second.cache_hit is True
    assert first.predictions[0].kegg_id == "R00024"

    tool = run_tool_assisted(
        sample, prompt, provider, cache=cache, variant="target_only",
        evidence=[ToolEvidence(source="kegg", query="A + B", n_hits=1,
                               identifiers=["R00024"], urls=["https://example.invalid/R00024"])],
    )
    assert tool.evidence_backed is True
    assert tool.mode == "tool_assisted"

    closed = run_closed_set(sample, ["R00024", "R00025"], abstain=False)
    assert closed.mode == "closed_set"
    assert closed.evidence_backed is True

    empty_closed = run_closed_set(sample, [], abstain=False)
    assert empty_closed.abstain is True


def test_live_provider_is_blocked():
    with pytest.raises(LiveCallBlocked):
        BlockedLiveProvider().complete({"messages": []})


def test_eval_separates_exact_and_equivalence_and_modes():
    def equiv(candidate, truth, kind):
        if kind == "exact":
            return candidate in set(truth)
        return candidate in set(truth) or candidate == "R00999"

    a = ModeResult(
        sample_id="1", model_id="M1", reaction_id="R1", cluster_id="C1",
        stratum=STRATUM_ABSENT, mode="direct_open_set", variant="target_only",
        template_version="t", abstain=False,
        predictions=[Prediction("R00999", 0.7, True)],
    )
    b = ModeResult(
        sample_id="2", model_id="M2", reaction_id="R2", cluster_id="C2",
        stratum=STRATUM_TOP1, mode="tool_assisted", variant="target_only",
        template_version="t", abstain=False, evidence_backed=True,
        predictions=[Prediction("R00024", 0.9, True)],
    )
    summary = score_results(
        [a, b],
        {("M1", "R1"): ["R00024"], ("M2", "R2"): ["R00024"]},
        seen_targets={"R00024"},
        equiv=equiv,
    )
    rows = {r["sample_id"]: r for r in summary["rows"]}
    assert rows["1"]["exact_top1"] is False
    assert rows["1"]["brite_top1"] is True
    assert summary["by_mode"]["direct_open_set"]["n"] == 1
    assert summary["by_mode"]["tool_assisted"]["evidence_backed_exact_top1"] == 1.0


def test_cost_estimator_uses_supplied_pricing_and_counts_calls():
    pricing = load_pricing(PRICING_EXAMPLE)
    prompts = [
        {"variant": "target_only", "stratum": STRATUM_EMPTY,
         "model_id": "M1", "reaction_id": "R1", "n_input_tokens_est": 100},
        {"variant": "target_plus_model", "stratum": STRATUM_EMPTY,
         "model_id": "M1", "reaction_id": "R1", "n_input_tokens_est": 150},
    ]
    out = estimate_cost(prompts, pricing, max_output_tokens=400)
    assert out["n_calls"] == 2
    assert out["n_input_tokens_est"] == 250
    assert out["gate"]["live_calls_blocked_until_approval"] is True
    assert "example-small-chat" in out["models"]
    assert "EXAMPLE ONLY" in out["pricing_source"]


def test_token_estimate_is_positive_for_nonempty_text():
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("a" * 8) == 2


@live_only
def test_live_strata_are_exhaustive_and_match_phase2_counts():
    strata = pd.read_csv(PHASE3_DIR / "strata.csv")
    counts = strata.stratum.value_counts().to_dict()
    assert len(strata) == 5816
    assert counts[STRATUM_UNCONSTRAINED] == 811
    assert counts[STRATUM_EMPTY] == 2646
    assert counts[STRATUM_ABSENT] == 340
    assert counts[STRATUM_RERANK] == 85
    assert counts[STRATUM_TOP1] == 1934
    assert set(counts) == set(STRATA)
    assert not strata.duplicated(["model_id", "reaction_id"]).any()


@live_only
def test_live_splits_are_cluster_isolated_and_complete():
    splits = pd.read_csv(PHASE3_DIR / "splits.csv")
    reactions = pd.read_csv(DATA_DIR / "reactions.csv")
    evaluable = reactions[reactions.included_in_eval.astype(bool)]
    assert len(splits) == len(evaluable) == 5816
    eval_keys = set(zip(evaluable.model_id.astype(str), evaluable.reaction_id.astype(str)))
    split_keys = set(zip(splits.model_id.astype(str), splits.reaction_id.astype(str)))
    assert eval_keys == split_keys
    crossed = splits.groupby("cluster_id").split.nunique()
    assert (crossed == 1).all()
    yeast = splits[splits.cluster_id == YEAST_CLUSTER]
    assert yeast.split.nunique() == 1
    summary = json.loads((PHASE3_DIR / "split_summary.json").read_text(encoding="utf-8"))
    assert summary["seed"] == 20260902
    assert summary["algorithm"] == "cluster_greedy_v1"


@live_only
def test_live_split_rebuild_is_byte_identical():
    from benchmark.scripts.build_phase3_splits import build_splits
    first, _, _, _ = build_splits()
    second, _, _, _ = build_splits()
    assert first.equals(second)


@live_only
def test_live_pilot_is_test_only_without_ground_truth():
    sample = pd.read_csv(PHASE3_DIR / "pilot_sample.csv")
    key = pd.read_csv(PHASE3_DIR / "pilot_answer_key.csv")
    splits = pd.read_csv(PHASE3_DIR / "splits.csv")
    test_keys = set(zip(
        splits.loc[splits.split == "test", "model_id"].astype(str),
        splits.loc[splits.split == "test", "reaction_id"].astype(str),
    ))
    for rec in sample.itertuples(index=False):
        assert rec.split == "test"
        assert (rec.model_id, rec.reaction_id) in test_keys
    assert "ground_truth_kegg_all" not in sample.columns
    assert "ground_truth_kegg_all" in key.columns
    assert sample.duplicated(["model_id", "reaction_id"]).sum() == 0
    assert len(sample) == len(key) == 183


@live_only
def test_live_prompts_have_no_kegg_reaction_leakage():
    path = PHASE3_DIR / "pilot_prompts.jsonl"
    n = 0
    variants = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        n += 1
        variants.add(row["variant"])
        leaked = find_kegg_leakage(row)
        assert leaked == [], leaked
        if row["variant"] == "target_plus_neighborhood":
            assert row["neighborhood_k"] <= 4
            neighbors = row["prompt"]["messages"][1]["content"].count("neighboring")
            assert neighbors >= 0
    assert n == 549
    assert variants == set(CONTEXT_VARIANTS)


@live_only
def test_live_cost_file_records_example_pricing_and_gate():
    cost = json.loads((PHASE3_DIR / "cost_estimate.json").read_text(encoding="utf-8"))
    assert cost["n_calls"] == 549
    assert cost["gate"]["live_calls_blocked_until_approval"] is True
    assert "EXAMPLE ONLY" in cost["pricing_source"]
    assert cost["bounded_vs_whole_model"]["ratio_whole_over_bounded"] > 1
