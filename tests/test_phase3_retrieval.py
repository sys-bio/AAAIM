import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from benchmark.scripts import phase3_retrieval as pr


def test_validation_population_is_committed_and_label_free():
    pop=pr.load_query_population()
    expected=json.loads(pr.SPLIT_SUMMARY.read_text())["splits"]["validation"]["n_reactions"]
    assert len(pop)==expected==969
    assert set(pop.split)=={"validation"}
    assert not any(c.startswith("ground_truth") for c in pop)


def test_test_split_rejected():
    with pytest.raises(ValueError,match="validation"):
        pr.load_query_population("test")
    with pytest.raises(ValueError,match="test/non-validation"):
        pr.reject_test_rows(pd.DataFrame({"split":["test"]}))


def test_answer_key_isolation():
    with pytest.raises(ValueError,match="answer key"):
        pr.query_text({"reaction_equation":"A => B","ground_truth_kegg_all":"R00001"})


@pytest.mark.parametrize("leak",["R00678_Tdo","R_R06861_C3_cytop","prefixR00024_suffix"])
def test_embedded_kegg_ids_are_redacted(leak):
    text=pr.query_text({"reaction_equation":f"{leak} => B","substrate_names":"A","product_names":"B","query_text":"A -> B"})
    assert leak not in text and not pr.KEGG_REACTION_RE.search(text)


def test_query_and_document_deterministic_and_document_key_hidden():
    row={"reaction_equation":"2 ATP-x + H+ => ADP","substrate_names":"ATP-x; H+","product_names":"ADP","query_text":"ATP-x -> ADP"}
    assert pr.query_text(row)==pr.query_text(dict(row))
    doc=pr.document_text("R00001",{"NAME":"name","DEFINITION":"R00001 reaction","EQUATION":"C1 => C2"})
    assert "R00001" not in doc and doc==pr.document_text("R99999",{"NAME":"name","DEFINITION":"R00001 reaction","EQUATION":"C1 => C2"})


def test_tokenizer_preserves_biochemical_tokens():
    assert pr.tokenise("ATP-dependent H+ 2-phospho-D-glycerate") == ["atp-dependent","h+","2-phospho-d-glycerate"]


def test_bm25_fixture_ranking_and_tie_break():
    docs=[{"kegg_id":"R00002","text":"water kinase"},{"kegg_id":"R00001","text":"water kinase"},{"kegg_id":"R00003","text":"glucose oxidase"}]
    assert pr.BM25(docs).rank("water kinase",3)==["R00001","R00002","R00003"]


def test_dense_ranking_mocked_encoder_and_ties():
    q=np.array([[1.,0.]])
    d=np.array([[1.,0.],[0.,1.],[1.,0.]])
    assert pr.dense_rank(q,d,["R00002","R00003","R00001"],3)==[["R00001","R00002","R00003"]]


def test_rrf_one_indexed_missing_and_tie_break():
    # Two symmetric scores tie; identifier ascending resolves it.
    assert pr.rrf([["R00002"],["R00001"]],k=60,topn=2)==["R00001","R00002"]
    assert pr.rrf([["R00001"]],k=0,topn=1)==["R00001"]


def test_rank_validation_unique_and_minimum_depth():
    row={"model_id":"m","reaction_id":"r","method":"bm25","ranked_ids":[f"R{i:05d}" for i in range(10)]}
    pr._validate_ranked([row],1)
    row["ranked_ids"][-1]=row["ranked_ids"][0]
    with pytest.raises(ValueError,match="duplicate ranked"):
        pr._validate_ranked([row],1)


def test_metric_denominator_and_consecutive_implicit_ranks():
    truth=pd.DataFrame([{"model_id":"m","reaction_id":"a","cluster_id":"c","stratum":"unconstrained","seen_in_train":False,"truth":["R00002"]},{"model_id":"m","reaction_id":"b","cluster_id":"c","stratum":"empty_constrained","seen_in_train":True,"truth":["R00009"]}])
    rows=[{"model_id":"m","reaction_id":"a","method":"x","ranked_ids":["R00001","R00002"]}]
    score=pr.score_rankings(rows,truth)
    assert len(score)==2 and score.recall_at_3_exact.mean()==.5
    assert score.loc[0,"first_hit_rank_exact"]==2 and bool(score.loc[1,"missing_output"])


def test_true_retrieval_failure_definition_unchanged():
    from benchmark.scripts.phase3_common import TRUE_RETRIEVAL_FAILURE_STRATA
    assert set(TRUE_RETRIEVAL_FAILURE_STRATA)=={"unconstrained","empty_constrained","nonempty_answer_absent"}
    assert "retrievable_rerank_failure" not in TRUE_RETRIEVAL_FAILURE_STRATA


def test_multilabel_phase2_stratum_correction_is_derived_only():
    truth=pr._truth_and_metadata()
    changed=truth[truth.frozen_stratum.ne(truth.stratum)]
    assert list(changed.reaction_id)==["reaction_1","reaction_3"]
    assert set(changed.stratum)=={"retrievable_top1_success","retrievable_rerank_failure"}
    assert truth.stratum.isin({"unconstrained","empty_constrained","nonempty_answer_absent"}).sum()==524


def test_cache_key_invalidation():
    base={"catalog":"a","template":"v1","model":"m","revision":"r","pooling":"cls"}
    keys=set()
    for field in base:
        changed=dict(base); changed[field]+="x"; keys.add(pr.cache_key(changed))
    assert len(keys)==len(base) and pr.cache_key(base) not in keys


def test_manifest_read_only_and_posix(monkeypatch):
    tmp_path=pr.PHASE3_DIR/"_test_manifest"
    tmp_path.mkdir(parents=True,exist_ok=True)
    artifact=tmp_path/"x.json"; artifact.write_text("{}\n")
    monkeypatch.setattr(pr,"REPO_ROOT",tmp_path)
    from benchmark.scripts.phase3_common import write_artifact_manifest
    write_artifact_manifest(tmp_path,[artifact],root=tmp_path)
    before=(tmp_path/"artifact_manifest.json").read_bytes()
    assert pr.verify_manifest(tmp_path)==[]
    assert before==(tmp_path/"artifact_manifest.json").read_bytes()
    payload=json.loads(before); assert len({x["path"] for x in payload["files"]})==payload["n_files"] and all("\\" not in x["path"] for x in payload["files"])
    (tmp_path/"artifact_manifest.json").unlink(); artifact.unlink(); tmp_path.rmdir()
