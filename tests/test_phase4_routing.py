"""Synthetic routing/accounting tests; no real review or test data are loaded."""
import copy
import math

import pytest

from benchmark.scripts import phase4_routing as routing


def evidence():
    return [{"kegg_id": "R99991", "fused_score": .03, "bm25_rank": 1, "trained_biencoder_rank": 1},
            {"kegg_id": "R99992", "fused_score": .024, "bm25_rank": 2, "trained_biencoder_rank": 2}]


def annotation():
    return {"abstain": False, "predicted_kegg_id": "R99992", "confidence": .8}


def test_routing_only_uses_prediction_time_signals():
    records = evidence()
    features = routing.extract_features(records)
    records[0].update(truth=["R99992"], audit_category="synthetic", correct=False)
    assert routing.extract_features(records) == features
    assert routing.predict(features, annotation(), "routed", .1)["prediction"] == "R99991"
    with pytest.raises(TypeError):
        routing.Features("R99991", .2, True, truth=["R99991"])


@pytest.mark.parametrize("scores", [(math.nan, .01), (.01, .02), (-1, -2), (math.inf, .1)])
def test_invalid_scores_fail_closed(scores):
    records = evidence()
    records[0]["fused_score"], records[1]["fused_score"] = scores
    with pytest.raises(ValueError):
        routing.extract_features(records)


def test_zero_score_and_disagreement_cannot_bypass():
    assert not routing.gate(routing.Features("R99991", 0.0, True), .05)
    assert not routing.gate(routing.Features("R99991", .9, False), .05)
    assert routing.gate(routing.Features("R99991", .1, True), .1)


def test_cached_route_missing_output_and_abstention():
    f = routing.Features("R99991", .01, False)
    with pytest.raises(ValueError, match="missing"):
        routing.predict(f, None, "routed", .1)
    assert routing.predict(f, None, "threshold", .1)["prediction"] is None
    value = annotation()
    value["abstain"] = True
    assert routing.predict(f, value, "routed", .1)["prediction"] is None
    assert routing.predict(f, value, "routed", .1)["llm_call"]
    assert routing.predict(f, annotation(), "post_llm_threshold", .9)["llm_call"]


def test_risk_curve_groups_ties_and_never_fabricates_abstained_outputs():
    curve = routing.risk_coverage([True, False, False], [True, True, False], [.8, .8, 1])
    assert len(curve["curve"]) == 1
    assert curve["native_max_coverage"] == pytest.approx(2/3)
    assert curve["aurc_partial"] == pytest.approx(1/3)
    assert curve["aurc_normalized"] == .5
    assert routing.risk_coverage([False], [False], [1])["aurc_normalized"] is None


def test_confidence_ranking_is_not_probability_calibration():
    first = routing.detection([True, False, True], [.8, .1, .6])
    transformed = routing.detection([True, False, True], [8, 1, 6])
    assert first == transformed  # no reliability/probability claim inferred
    assert routing.detection([True, True], [.1, .2])["auroc"] is None


def synthetic_scored():
    base = {"sample_id": "SYN-1", "prediction": "R99991", "correct": True,
            "reference_present": True, "score": .8, "llm_call": True,
            "cluster_id": "SYN-C1", "model_id": "SYN-M1", "cost_usd": .01,
            "input_tokens": 100, "cached_input_tokens": 0, "output_tokens": 10, "latency_ms": 100}
    abstain = {**base, "sample_id": "SYN-2", "prediction": None, "correct": False, "reference_present": False}
    return [base, abstain]


def test_abstention_denominator_and_mechanical_appropriateness():
    m = routing.metrics(synthetic_scored())
    assert (m["n"], m["correct"], m["answered"], m["abstained"]) == (2, 1, 1, 1)
    assert m["accuracy"] == m["coverage"] == .5
    assert m["selective_accuracy"] == 1
    assert m["reference_absence_consistent_abstention"] == 1
    assert m["reference_present_abstention"] == 0
    assert m["replayed_cost_usd"] == .02


def test_cluster_bootstrap_keeps_pairs_and_reproduces_seed():
    data = synthetic_scored()
    conf = {"bootstrap_seed": 1, "bootstrap_replicates": 20}
    result = routing.paired_bootstrap(data, data, conf)
    assert result["accuracy_delta_vs_grounded_ci95"] == [0, 0]
    assert result == routing.paired_bootstrap(data, data, conf)
    with pytest.raises(ValueError):
        routing.paired_bootstrap(data, data[::-1], conf)


def test_selection_rejects_accuracy_loss_even_when_calls_saved():
    baseline = {"correct": 10, "answered": 12, "selective_risk": 2/12}
    candidate = {"correct": 9, "answered": 12, "selective_risk": 3/12,
                 "simulated_call_rate": .5, "simulated_llm_calls": 8,
                 "bootstrap": {"accuracy_delta_vs_grounded_ci95": [-.01, .02]}}
    result = routing.select_method({"grounded": baseline, "fusion": {"correct": 8}, "routed_0.1": candidate},
                                   {"min_call_reduction": .2, "min_delta_lower_bound": -.02})
    assert result["selected_method"] == "grounded"
    candidate.update(correct=10, selective_risk=2/12)
    assert routing.select_method({"grounded": baseline, "fusion": {"correct": 8}, "routed_0.1": candidate},
                                 {"min_call_reduction": .2, "min_delta_lower_bound": -.02})["selected_method"] == "routed_0.1"


def test_duplicate_cache_ids_rejected():
    with pytest.raises(ValueError):
        routing.keyed([{"sample_id": "SYN"}, {"sample_id": "SYN"}])
