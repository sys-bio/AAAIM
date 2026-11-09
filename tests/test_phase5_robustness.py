"""Synthetic-only metric, paired-uncertainty and affected-stage tests."""
import pytest

from benchmark.scripts import phase5_robustness as study
from benchmark.scripts.phase5_perturb import digest


def test_multilabel_and_empty_candidate_accounting():
    value = study.retrieval_metrics(["R99991", "R99992"], ["R99992", "R99993"], {"R99991", "R99992"}, False)
    assert value["recall_at_1"] == 0 and value["recall_at_3"] == 1
    assert value["mrr_at_10"] == .5 and value["recall_any_candidate"] == 1
    empty = study.retrieval_metrics([], ["R99992"], set(), False)
    assert empty["empty"] == 1 and empty["recall_at_10"] == 0
    assert empty["reference_absent_top10"] == 1


def test_full_catalog_presence_not_confused_with_top10_recall():
    value = study.retrieval_metrics(["R99991"], ["R99992"], {"R99991", "R99992"}, True)
    assert value["recall_any_candidate"] == 1 and value["recall_at_10"] == 0
    assert value["candidate_count"] == 2


def test_downstream_rejects_clean_rankings_for_changed_inputs():
    clean = {"query": "Synthetic clean query", "query_digest": digest("Synthetic clean query")}
    study.validate_rank_input_alignment({"query_digest": clean["query_digest"]}, clean)
    changed = {"query": "Synthetic altered query", "query_digest": digest("Synthetic altered query")}
    with pytest.raises(ValueError, match="not recomputed"):
        study.validate_rank_input_alignment({"query_digest": clean["query_digest"]}, changed)


def test_cluster_replicate_interval_deterministic_and_paired():
    clean = [1, 0, 1, 0]
    degraded = [[1, 0, 1, 0], [1, 0, 1, 0]]
    args = clean, degraded, ["SYN-C1", "SYN-C1", "SYN-C2", "SYN-C2"], 123, 100
    assert study.paired_interval(*args) == [0, 0]
    assert study.paired_interval(*args) == study.paired_interval(*args)


def test_condition_grid_is_bounded_and_complete():
    values = study.conditions({"removal_fractions": [.1, .25, .5, .75], "seeds": [1, 2, 3]})
    assert len(values) == len({v[0] for v in values}) == 19
    assert 19*163 == 3097


def test_difference_in_changes_bootstrap_keeps_methods_paired():
    # A and B are equally correct clean; only B loses all hits under corruption.
    assert study.paired_interval([0, 0], [[1, 1], [1, 1]], ["SYN-C1", "SYN-C2"], 1, 50) == [1, 1]


def test_strict_adapter_recomputes_changed_mappings_and_only_reuses_identical_inputs(monkeypatch):
    from types import SimpleNamespace
    from core.reaction import matching
    calls = []
    def normalize(equations, ids, frame, spectators):
        compounds = frame.KEGG_ID.tolist()
        calls.append(compounds)
        return compounds
    def recommend(compounds, **kwargs):
        return [SimpleNamespace(metadata={"filtered_species_count": 1}, candidates=["R99991" if "C00001" in compounds else "R99992"], match_score=[1])]
    monkeypatch.setattr(matching, "map_reactions_to_kegg", normalize)
    adapter = object.__new__(study.StrictRules)
    adapter.mapping = {"CHEBI:1": ["C00001"], "CHEBI:2": ["C00002"]}
    adapter.db = SimpleNamespace(_get_kegg_recommendations_rulebased=recommend)
    adapter.cofactors, adapter.cache = set(), {}
    record = {"participants": [{"species_id": "s1", "chebi": ["CHEBI:1"], "kegg": []}], "reactants": {"s1": 1}, "products": {}}
    assert adapter.rank(record) == (["R99991"], False)
    record["participants"][0]["chebi"] = ["CHEBI:2"]
    assert adapter.rank(record) == (["R99992"], False)
    assert adapter.rank(record) == (["R99992"], True)
    assert calls == [["C00001"], ["C00002"]]
