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
    ID_ABSENT,
    ID_IN_CATALOG,
    ID_MALFORMED,
    KEGG_ID_STRICT,
    KEGG_REACTION_LEGACY_WORD_BOUNDARY_RE,
    PHASE3_DIR,
    PILOT_SPLIT,
    PROMPT_TEMPLATE_VERSION,
    STRATA,
    STRATUM_ABSENT,
    STRATUM_EMPTY,
    STRATUM_RERANK,
    STRATUM_TOP1,
    STRATUM_UNCONSTRAINED,
    TOKENIZER_SCAFFOLD,
    YEAST_CLUSTER,
    assign_stratum,
    assert_no_kegg_leakage,
    classify_kegg_id,
    estimate_tokens,
    extract_kegg_reaction_ids,
    find_kegg_leakage,
    parse_kegg_ids,
    parse_participant_ids,
    redact_kegg_reaction_ids,
    require_live_tokenizer,
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
    SYSTEM_DIRECT,
    SYSTEM_TOOL_ASSISTED,
    audit_prompts_against_answer_key,
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


def test_embedded_kegg_ids_missed_by_word_boundaries_and_caught_after_fix():
    examples = {
        "R_R06861_C3_cytop": "R06861",
        "R00678_Tdo": "R00678",
        "prefixR00024_suffix": "R00024",
    }
    for text, kid in examples.items():
        assert KEGG_REACTION_LEGACY_WORD_BOUNDARY_RE.search(text) is None, text
        assert extract_kegg_reaction_ids(text) == [kid], text
        redacted = redact_kegg_reaction_ids(text)
        assert kid not in redacted, redacted
        assert extract_kegg_reaction_ids(redacted) == []
    # Six-or-more digits is not a KEGG reaction id.
    assert extract_kegg_reaction_ids("R000240") == []
    assert extract_kegg_reaction_ids("R00024") == ["R00024"]
    assert extract_kegg_reaction_ids("see kegg.reaction/R00024") == ["R00024"]


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
    assert "all_eligible_validation" in out.iloc[0].selection_rule


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
    names = pd.DataFrame({
        "model_id": ["M1", "M1", "M1"],
        "species_id": ["A_c", "B_c", "C_c"],
        "species_name": ["alpha", "beta", "gamma"],
    })
    ctx = build_context(_toy_row(), variant="target_plus_model",
                        corpus=corpus, evidence=evidence, species_names=names)
    assert "R00024" not in json.dumps(ctx)
    prompt = render_prompt(ctx)
    blob = json.dumps(prompt)
    assert "R00024" not in blob
    assert "R00164" not in blob
    leaked = build_context(_toy_row(reaction_equation="A_c => R00024"),
                           variant="target_only", corpus=corpus, evidence=evidence,
                           species_names=names)
    assert "R00024" not in json.dumps(leaked)
    assert "[REDACTED_KEGG_REACTION]" in leaked["equation"]
    prompt = render_prompt(ctx)
    assert "Use the supplied reaction context and your internal knowledge" in (
        prompt["messages"][0]["content"])
    assert "Use only the supplied reaction-local context" not in prompt["messages"][0]["content"]
    assert prompt["system_direct"] == SYSTEM_DIRECT
    assert prompt["system_tool_assisted"] == SYSTEM_TOOL_ASSISTED
    tool = render_prompt(ctx, mode="tool_assisted")
    assert tool["messages"][0]["content"] == SYSTEM_TOOL_ASSISTED
    assert "internal knowledge" not in tool["messages"][0]["content"]
    assert "recorded tool or search evidence" in tool["messages"][0]["content"]


def test_context_redacts_embedded_sbml_reaction_ids():
    row = _toy_row(reaction_id="R00678_Tdo", reaction_name="R_R06861_C3_cytop")
    corpus = pd.DataFrame([row])
    evidence = pd.DataFrame(columns=["model_id", "species_id", "annotation", "annotation_type"])
    ctx = build_context(row, variant="target_only", corpus=corpus, evidence=evidence,
                        species_names=pd.DataFrame(columns=["model_id", "species_id", "species_name"]))
    blob = json.dumps(ctx)
    assert "R00678" not in blob
    assert "R06861" not in blob
    assert "[REDACTED_KEGG_REACTION]" in ctx["reaction_id"]
    assert "[REDACTED_KEGG_REACTION]" in ctx["reaction_name"]


def test_participant_names_join_by_species_id_not_position():
    """Species on both sides, or names containing ';', must not be zipped positionally."""
    row = _toy_row(
        model_id="BIOMD0000000122",
        reaction_id="R1",
        reaction_equation="NFAT_Pi_Nuc + Act_C_Nuc <=> Act_C_Nuc + NFAT_Nuc",
        substrate_names="Phosphorylated NFAT in Nucleus; Active Calcineurin in Nucleus",
        product_names="Active Calcineurin in Nucleus; NFAT",
    )
    names = pd.DataFrame({
        "model_id": ["BIOMD0000000122"] * 4,
        "species_id": ["NFAT_Pi_Nuc", "Act_C_Nuc", "NFAT_Nuc", "weird"],
        "species_name": [
            "Phosphorylated NFAT in Nucleus",
            "Active Calcineurin in Nucleus",
            "NFAT",
            "foo; bar",
        ],
    })
    evidence = pd.DataFrame(columns=["model_id", "species_id", "annotation", "annotation_type"])
    ctx = build_context(
        row, variant="target_only", corpus=pd.DataFrame([row]),
        evidence=evidence, species_names=names,
    )
    by_id = {p["species_id"]: p["name"] for p in ctx["participants"]}
    assert by_id["NFAT_Nuc"] == "NFAT"
    assert by_id["Act_C_Nuc"] == "Active Calcineurin in Nucleus"
    assert "Calcineurin" not in by_id["NFAT_Nuc"]
    semicolon_row = _toy_row(
        reaction_equation="weird => C_c",
        model_id="BIOMD0000000122",
    )
    ctx2 = build_context(
        semicolon_row, variant="target_only",
        corpus=pd.DataFrame([semicolon_row]), evidence=evidence, species_names=names,
    )
    assert ctx2["participants"][0]["name"] == "foo; bar"


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
    catalog = {"R00024", "R00025"}
    ok = parse_structured_output(json.dumps({
        "abstain": False,
        "predictions": [{"kegg_id": "R00024", "confidence": 0.9}],
        "rationale": "match", "basis": "supplied_evidence",
    }), catalog=catalog)
    assert ok["abstain"] is False
    assert ok["predictions"][0].valid_kegg_id is True
    assert ok["predictions"][0].in_catalog is True
    assert ok["predictions"][0].id_class == ID_IN_CATALOG

    abstain = parse_structured_output(json.dumps({
        "abstain": True, "predictions": [{"kegg_id": "R00024", "confidence": 1}],
        "rationale": "not enough", "basis": "recalled_knowledge",
    }), catalog=catalog)
    assert abstain["abstain"] is True
    assert abstain["predictions"] == []

    bad = parse_structured_output("not json at all", catalog=catalog)
    assert bad["abstain"] is True
    assert bad["parse_error"] == "unparseable"

    invalid = parse_structured_output(json.dumps({
        "abstain": False,
        "predictions": [{"kegg_id": "R1", "confidence": 0.2}],
        "rationale": "guess", "basis": "recalled_knowledge",
    }), catalog=catalog)
    assert invalid["predictions"][0].valid_kegg_id is False
    assert invalid["predictions"][0].id_class == ID_MALFORMED
    assert invalid["predictions"][0].well_formed is False

    absent = parse_structured_output(json.dumps({
        "abstain": False,
        "predictions": [{"kegg_id": "R99999", "confidence": 0.5}],
        "rationale": "syntax only", "basis": "recalled_knowledge",
    }), catalog=catalog)
    assert absent["predictions"][0].well_formed is True
    assert absent["predictions"][0].in_catalog is False
    assert absent["predictions"][0].id_class == ID_ABSENT

    conf = parse_structured_output(json.dumps({
        "abstain": False,
        "predictions": [{"kegg_id": "R00024", "confidence": 1.5}],
        "rationale": "bad conf", "basis": "recalled_knowledge",
    }), catalog=catalog)
    assert "confidence_out_of_range" in conf["parse_error"]
    assert conf["predictions"][0].confidence is None

    dups = parse_structured_output(json.dumps({
        "abstain": False,
        "predictions": [
            {"kegg_id": "R00024", "confidence": 0.9},
            {"kegg_id": "R00024", "confidence": 0.1},
        ],
        "rationale": "dup", "basis": "recalled_knowledge",
    }), catalog=catalog)
    assert dups["parse_error"] == "duplicate_predicted_ids"
    assert len(dups["predictions"]) == 1

    empty = parse_structured_output(json.dumps({
        "abstain": False, "predictions": [],
        "rationale": "forgot ids", "basis": "recalled_knowledge",
    }), catalog=catalog)
    assert empty["abstain"] is False
    assert empty["predictions"] == []
    assert empty["parse_error"] == "abstain_false_without_predictions"


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
    assert tool.predictions[0].prediction_supported_by_evidence is True
    assert tool.predictions[0].supporting_evidence_ids == ["R00024"]

    closed = run_closed_set(sample, ["R00024", "R00025"], abstain=False)
    assert closed.mode == "closed_set"
    assert closed.evidence_backed is True

    empty_closed = run_closed_set(sample, [], abstain=False)
    assert empty_closed.abstain is True

    mismatch_raw = json.dumps({
        "abstain": False,
        "predictions": [{"kegg_id": "R99999", "confidence": 0.8}],
        "rationale": "guess", "basis": "supplied_evidence",
    })
    mismatch = run_tool_assisted(
        sample, prompt, MockProvider(responses={"rxnA": mismatch_raw}),
        variant="target_only",
        evidence=[ToolEvidence(source="kegg", query="A + B", n_hits=1,
                               identifiers=["R00024"])],
    )
    assert mismatch.evidence_backed is False
    assert mismatch.predictions[0].prediction_supported_by_evidence is False
    assert mismatch.predictions[0].supporting_evidence_ids == []
    row = score_one(mismatch, ["R00024"], equiv=lambda c, t, k: c in set(t))
    assert row["evidence_outcome"] == "incorrect_unsupported"


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
    b_pred = Prediction("R00024", 0.9, True)
    b_pred.prediction_supported_by_evidence = True
    b = ModeResult(
        sample_id="2", model_id="M2", reaction_id="R2", cluster_id="C2",
        stratum=STRATUM_TOP1, mode="tool_assisted", variant="target_only",
        template_version="t", abstain=False, evidence_backed=True,
        predictions=[b_pred],
    )
    summary = score_results(
        [a, b],
        {("M1", "R1"): ["R00024"], ("M2", "R2"): ["R00024"]},
        seen_targets={"R00024"},
        seen_definition="train",
        equiv=equiv,
    )
    rows = {r["sample_id"]: r for r in summary["rows"]}
    assert rows["1"]["exact_top1"] is False
    assert rows["1"]["brite_top1"] is True
    assert summary["by_mode"]["direct_open_set"]["n"] == 1
    assert summary["by_mode"]["tool_assisted"]["evidence_backed_exact_top1"] == 1.0
    assert summary["seen_target_definition"] == "train"
    assert summary["seen_fit_target"]["n"] == 2
    assert summary["unseen_fit_target"]["n"] == 0


def test_evidence_outcome_uses_top1_support_only():
    def pred(kid, supported):
        item = Prediction(kid, 0.8, True)
        item.prediction_supported_by_evidence = supported
        item.supporting_evidence_ids = [kid] if supported else []
        return item

    def result(preds):
        return ModeResult(
            sample_id="1", model_id="M1", reaction_id="rxn", cluster_id="C1",
            stratum=STRATUM_EMPTY, mode="tool_assisted", variant="target_only",
            template_version="t", abstain=False, predictions=preds,
            evidence=[ToolEvidence(source="kegg", query="q", identifiers=["R00024"])],
        )

    equiv = lambda c, t, k: c in set(t)
    correct_rank2_supported = result([pred("R00024", False), pred("R00025", True)])
    row = score_one(correct_rank2_supported, ["R00024"], equiv=equiv)
    assert row["exact_top1"] is True
    assert row["evidence_outcome"] == "correct_but_unsupported"
    assert row["top1_supported_by_evidence"] is False
    assert row["prediction_supported_by_evidence"] == [False, True]

    incorrect_rank2_supported = result([pred("R00025", False), pred("R00024", True)])
    row = score_one(incorrect_rank2_supported, ["R00024"], equiv=equiv)
    assert row["exact_top1"] is False
    assert row["evidence_outcome"] == "incorrect_unsupported"

    supported_top1 = result([pred("R00024", True), pred("R00025", False)])
    row = score_one(supported_top1, ["R00024"], equiv=equiv)
    assert row["evidence_outcome"] == "correct_and_evidence_supported"
    assert row["top1_supported_by_evidence"] is True

    wrong_supported_top1 = result([pred("R00025", True)])
    row = score_one(wrong_supported_top1, ["R00024"], equiv=equiv)
    assert row["evidence_outcome"] == "incorrect_despite_evidence"


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
    assert out["gate"]["tokenizer"]["method"] == TOKENIZER_SCAFFOLD
    assert out["gate"]["tokenizer"]["live_run_blocked_with_this_method"] is True
    assert "example-small-chat" in out["models"]
    assert "EXAMPLE ONLY" in out["pricing_source"]


def test_token_estimate_is_positive_for_nonempty_text():
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("a" * 8) == 2
    with pytest.raises(RuntimeError, match="tokenizer"):
        require_live_tokenizer(TOKENIZER_SCAFFOLD)


def test_classify_kegg_id_separates_syntax_from_catalog():
    catalog = {"R00024"}
    assert classify_kegg_id("R1", catalog) == ID_MALFORMED
    assert classify_kegg_id("R99999", catalog) == ID_ABSENT
    assert classify_kegg_id("R00024", catalog) == ID_IN_CATALOG


def test_gitignore_has_single_trailing_newline():
    raw = (REPO_ROOT / ".gitignore").read_bytes()
    assert raw.endswith(b"\n")
    assert not raw.endswith(b"\n\n")


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
def test_live_pilot_is_validation_only_without_ground_truth():
    sample = pd.read_csv(PHASE3_DIR / "pilot_sample.csv")
    key = pd.read_csv(PHASE3_DIR / "pilot_answer_key.csv")
    splits = pd.read_csv(PHASE3_DIR / "splits.csv")
    val_keys = set(zip(
        splits.loc[splits.split == "validation", "model_id"].astype(str),
        splits.loc[splits.split == "validation", "reaction_id"].astype(str),
    ))
    test_keys = set(zip(
        splits.loc[splits.split == "test", "model_id"].astype(str),
        splits.loc[splits.split == "test", "reaction_id"].astype(str),
    ))
    for rec in sample.itertuples(index=False):
        assert rec.split == PILOT_SPLIT == "validation"
        assert (rec.model_id, rec.reaction_id) in val_keys
        assert (rec.model_id, rec.reaction_id) not in test_keys
    assert "ground_truth_kegg_all" not in sample.columns
    assert "ground_truth_kegg_all" in key.columns
    assert sample.duplicated(["model_id", "reaction_id"]).sum() == 0
    assert len(sample) == len(key) == 163
    counts = sample.stratum.value_counts().to_dict()
    assert counts[STRATUM_UNCONSTRAINED] == 50
    assert counts[STRATUM_EMPTY] == 50
    assert counts[STRATUM_ABSENT] == 22
    assert counts[STRATUM_RERANK] == 16
    assert counts[STRATUM_TOP1] == 25
    summary = json.loads((PHASE3_DIR / "pilot_summary.json").read_text(encoding="utf-8"))
    assert summary["source_split"] == "validation"


@live_only
def test_live_prompts_have_no_kegg_reaction_leakage():
    path = PHASE3_DIR / "pilot_prompts.jsonl"
    n = 0
    variants = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        n += 1
        variants.add(row["variant"])
        leaked = find_kegg_leakage(row["prompt"])
        assert leaked == [], leaked
        assert row["template_version"] == PROMPT_TEMPLATE_VERSION == "phase3-open-set-v3"
        assert "Use only the supplied reaction-local context" not in json.dumps(row["prompt"])
        if row["variant"] == "target_plus_neighborhood":
            assert row["neighborhood_k"] <= 4
            neighbors = row["prompt"]["messages"][1]["content"].count("neighboring")
            assert neighbors >= 0
    assert n == 489
    assert variants == set(CONTEXT_VARIANTS)


@live_only
def test_live_prompts_answer_key_audit_and_rebuild_is_byte_identical():
    from benchmark.scripts.phase3_prompts import build_pilot_prompts
    path = PHASE3_DIR / "pilot_prompts.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    key = pd.read_csv(PHASE3_DIR / "pilot_answer_key.csv")
    sample = pd.read_csv(PHASE3_DIR / "pilot_sample.csv")
    assert set(zip(sample.model_id.astype(str), sample.reaction_id.astype(str))) == set(
        zip(key.model_id.astype(str), key.reaction_id.astype(str)))
    audit = audit_prompts_against_answer_key(rows, key)
    assert audit["n_prompts_checked"] == 489
    assert audit["n_samples_checked"] == 163
    assert audit["n_ground_truth_leaks"] == 0
    assert audit["n_any_kegg_id_leaks"] == 0
    first = build_pilot_prompts()
    second = build_pilot_prompts()
    blob1 = json.dumps(first, sort_keys=True, separators=(",", ":"))
    blob2 = json.dumps(second, sort_keys=True, separators=(",", ":"))
    assert blob1 == blob2


@live_only
def test_live_cost_file_records_example_pricing_and_gate():
    cost = json.loads((PHASE3_DIR / "cost_estimate.json").read_text(encoding="utf-8"))
    assert cost["n_calls"] == 489
    assert cost["n_reactions"] == 163
    assert cost["gate"]["live_calls_blocked_until_approval"] is True
    assert cost["gate"]["tokenizer"]["live_run_blocked_with_this_method"] is True
    assert "EXAMPLE ONLY" in cost["pricing_source"]
    assert cost["bounded_vs_whole_model"]["ratio_whole_over_bounded"] > 1


@live_only
def test_live_species_names_join_nfat_by_id():
    names = pd.read_csv(PHASE3_DIR / "species_names.csv")
    nfat = names[(names.model_id == "BIOMD0000000122") & (names.species_id == "NFAT_Nuc")]
    calc = names[(names.model_id == "BIOMD0000000122") & (names.species_id == "Act_C_Nuc")]
    assert len(nfat) == 1
    assert str(nfat.iloc[0].species_name) == "NFAT_nuc"
    assert "Calcineurin" in str(calc.iloc[0].species_name)
    assert "Calcineurin" not in str(nfat.iloc[0].species_name)
    assert not names.duplicated(["model_id", "species_id"]).any()


@live_only
def test_live_kegg_catalog_separates_syntax_from_existence():
    from benchmark.scripts.phase3_common import load_kegg_catalog_ids
    catalog = load_kegg_catalog_ids()
    assert "R00024" in catalog
    assert "R99999" not in catalog
    assert classify_kegg_id("R99999", catalog) == ID_ABSENT
    assert classify_kegg_id("R00024", catalog) == ID_IN_CATALOG
    payload = json.loads((PHASE3_DIR / "kegg_catalog_ids.json").read_text(encoding="utf-8"))
    assert payload["n"] == len(catalog) == len(payload["ids"])
