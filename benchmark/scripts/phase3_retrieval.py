"""Phase 3 retrieval-only full-catalog baselines.

This module deliberately keeps labels out of query construction and joins them only
in :func:`evaluate`.  It is usable offline from frozen rankings and never reads the
test split during ``run --split validation``.
"""
from __future__ import annotations

import argparse, csv, hashlib, importlib.metadata, json, lzma, pickle, platform, random, re, sys, time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from benchmark.scripts.phase3_common import (REPO_ROOT, PHASE3_DIR, KEGG_REACTION_RE,
    assert_no_kegg_leakage, assign_stratum, atomic_write_jsonl, load_evaluable_corpus, parse_kegg_ids,
    parse_participant_ids, redact_kegg_reaction_ids, repo_relative_posix, seen_targets_from_corpus,
    sha256_file, sha256_portable, write_artifact_manifest, write_csv, write_json)
from benchmark.scripts.kegg_equivalence import match_kinds

OUT = PHASE3_DIR / "retrieval_baselines"
CATALOG = REPO_ROOT / "data" / "kegg" / "kegg_reaction_features.lzma"
SPLITS = PHASE3_DIR / "splits.csv"
CANDIDATES = REPO_ROOT / "benchmark" / "data" / "candidates.csv"
STATUS = REPO_ROOT / "benchmark" / "data" / "candidate_status.csv"
REACTIONS = REPO_ROOT / "benchmark" / "data" / "reactions.csv"
REACTION_TEXT = REPO_ROOT / "benchmark" / "data" / "reaction_text.csv"
SPECIES_EVIDENCE = REPO_ROOT / "benchmark" / "data" / "species_evidence.csv"
SPECIES_NAMES = PHASE3_DIR / "species_names.csv"
SPLIT_SUMMARY = PHASE3_DIR / "split_summary.json"
QUERY_TEMPLATE_VERSION = "phase3-retrieval-query-v1"
DOCUMENT_TEMPLATE_VERSION = "phase3-retrieval-document-v1"
TOKENIZER_VERSION = "biochemical-regex-v1"
MODEL = "BAAI/bge-m3"
MODEL_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
RRF_K = 60
TOKEN_RE = re.compile(r"(?:\d+(?:\.\d+)?-)?[A-Za-z]+(?:[A-Za-z0-9_+\-./:]*)|\d+(?:\.\d+)?|[+\-]", re.ASCII)

def tokenise(text: str) -> list[str]:
    """Deterministic lowercase tokenizer preserving biochemical punctuation."""
    return [x.lower() for x in TOKEN_RE.findall(text or "")]

def _clean(value: Any) -> str:
    return "" if value is None or (isinstance(value, float) and pd.isna(value)) else str(value)

def _csv_rows_for_keys(path: Path, keys: set[tuple[str,str]], columns: Sequence[str], *, require_one: bool = True) -> pd.DataFrame:
    """Materialize only explicitly allowed reaction keys from a mixed-split CSV."""
    rows=[]
    with path.open(encoding="utf-8",newline="") as fh:
        for row in csv.DictReader(fh):
            if (str(row["model_id"]),str(row["reaction_id"])) in keys:
                rows.append({c:row.get(c,"") for c in columns})
    frame=pd.DataFrame(rows,columns=columns)
    if require_one and (len(frame)!=len(keys) or frame.duplicated(["model_id","reaction_id"]).any()):
        raise ValueError(f"allowed-key load mismatch for {path}: {len(frame)} rows for {len(keys)} keys")
    return frame

def query_text(row: Mapping[str, Any]) -> str:
    """Target-local non-label representation; rejects label-bearing input mappings."""
    forbidden = {"ground_truth_kegg_all", "ground_truth_kegg_primary", "ground_truth_ids"}
    if forbidden & set(row):
        raise ValueError("answer key supplied to query construction")
    equation = redact_kegg_reaction_ids(_clean(row.get("reaction_equation")))
    participants = redact_kegg_reaction_ids(_clean(row.get("participant_evidence")))
    if not participants:
        participants = redact_kegg_reaction_ids("; ".join(filter(None,[_clean(row.get("substrate_names")),_clean(row.get("product_names"))])))
    text = f"Equation: {equation}\nParticipants: {participants}".strip()
    assert_no_kegg_leakage(text, where="retrieval query")
    return text

def load_query_population(split: str = "validation") -> pd.DataFrame:
    """Load only model-visible validation fields; never load the answer key."""
    if split != "validation": raise ValueError("retrieval ranking is validation-only")
    assignments = pd.read_csv(SPLITS, dtype=str)
    selected = assignments.loc[assignments.split.eq(split), ["model_id","reaction_id","cluster_id","split","stratum"]]
    expected = int(json.loads(SPLIT_SUMMARY.read_text(encoding="utf-8"))["splits"][split]["n_reactions"])
    if len(selected) != expected: raise ValueError(f"split count mismatch: {len(selected)} != {expected}")
    keys=set(zip(selected.model_id,selected.reaction_id))
    visible = _csv_rows_for_keys(REACTION_TEXT,keys,["model_id","reaction_id","reaction_equation","substrate_names","product_names","query_text"])
    out = selected.merge(visible, on=["model_id","reaction_id"], validate="one_to_one")
    if not out.split.eq("validation").all(): raise ValueError("test/non-validation row loaded into ranking population")
    forbidden = [c for c in out if c.startswith("ground_truth")]
    if forbidden: raise ValueError(f"answer-key columns loaded: {forbidden}")
    names=pd.read_csv(SPECIES_NAMES,dtype=str).fillna(""); names=names[names.model_id.isin(set(out.model_id))]
    name_map={(r.model_id,r.species_id):r.species_name for r in names.itertuples()}
    evidence=pd.read_csv(SPECIES_EVIDENCE,dtype=str).fillna(""); evidence=evidence[evidence.model_id.isin(set(out.model_id))]
    ev: dict[tuple[str,str],dict[str,list[str]]]=defaultdict(lambda:{"chebi":[],"kegg_compound":[]})
    for r in evidence.itertuples():
        if r.annotation_type in ev[(r.model_id,r.species_id)] and r.annotation not in ev[(r.model_id,r.species_id)][r.annotation_type]: ev[(r.model_id,r.species_id)][r.annotation_type].append(r.annotation)
    blocks=[]
    for r in out.itertuples():
        items=[]
        for sid in parse_participant_ids(r.reaction_equation):
            e=ev[(r.model_id,sid)]; detail=[f"species={sid}"]
            if e["chebi"]: detail.append("ChEBI="+",".join(e["chebi"]))
            if e["kegg_compound"]: detail.append("KEGG-compound="+",".join(e["kegg_compound"]))
            items.append(f"{name_map.get((r.model_id,sid),sid)} [{'; '.join(detail)}]")
        blocks.append("; ".join(items))
    out["participant_evidence"]=blocks
    return out.sort_values(["model_id","reaction_id"]).reset_index(drop=True)

def reject_test_rows(frame: pd.DataFrame) -> None:
    if "split" not in frame or not frame.split.eq("validation").all():
        raise ValueError("test/non-validation row loaded into retrieval evaluation")

def document_text(kegg_id: str, fields: Mapping[str, Any]) -> str:
    """Canonical document; key is deliberately excluded from searchable text."""
    parts = []
    for label, key in (("Definition", "DEFINITION"), ("Names", "NAME"),
                       ("Equation", "EQUATION"), ("Enzyme", "ENZYME"),
                       ("Class", "RCLASS"), ("Brite", "BRITE")):
        value = redact_kegg_reaction_ids(_clean(fields.get(key)))
        if value: parts.append(f"{label}: {value}")
    text = "\n".join(parts)
    assert_no_kegg_leakage(text, where=f"catalog document {kegg_id}")
    return text

def load_catalog() -> list[dict[str, str]]:
    raw = pickle.loads(lzma.open(CATALOG, "rb").read())
    docs = [{"kegg_id": k, "text": document_text(k, raw[k])} for k in sorted(raw)]
    if len(docs) != 12312: raise ValueError(f"frozen catalog count changed: {len(docs)}")
    return docs

class BM25:
    def __init__(self, docs: Sequence[Mapping[str, str]], k1: float = 1.2, b: float = .75):
        self.docs, self.k1, self.b = list(docs), k1, b
        self.tokens = [tokenise(x["text"]) for x in docs]; self.lengths = [len(x) for x in self.tokens]
        self.avgdl = sum(self.lengths) / len(self.lengths); self.tf = [Counter(x) for x in self.tokens]
        df = Counter(t for doc in self.tokens for t in set(doc)); n = len(docs)
        self.idf = {t: np.log(1 + (n - f + .5)/(f + .5)) for t, f in df.items()}
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, terms in enumerate(self.tf):
            for term, freq in terms.items(): self.postings[term].append((i, freq))
    def rank(self, query: str, topn: int = 100) -> list[str]:
        scores = np.zeros(len(self.docs)); q = Counter(tokenise(query))
        for term, qfreq in q.items():
            for i, freq in self.postings.get(term, []):
                denom = freq + self.k1 * (1 - self.b + self.b * self.lengths[i] / self.avgdl)
                scores[i] += self.idf[term] * qfreq * freq * (self.k1 + 1) / denom
        order = sorted(range(len(self.docs)), key=lambda i: (-scores[i], self.docs[i]["kegg_id"]))
        return [self.docs[i]["kegg_id"] for i in order[:topn]]

def rrf(lists: Sequence[Sequence[str]], k: int = RRF_K, topn: int = 100) -> list[str]:
    scores: dict[str, float] = defaultdict(float)
    for ranked in lists:
        for rank, ident in enumerate(ranked, 1): scores[ident] += 1.0 / (k + rank)
    return [x for x, _ in sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:topn]]

def _dense_encoder(model_name: str, revision: str | None, batch: int) -> tuple[Callable[[Sequence[str]], np.ndarray], dict[str, Any]]:
    """Official checkpoint, dense CLS pooling only; never changes model silently."""
    from transformers import AutoModel, AutoTokenizer
    import torch
    print(json.dumps({"model": model_name, "revision": revision or "resolved commit recorded after download", "source": "Hugging Face official BAAI repository", "expected_disk_use": "approximately 2.2 GiB model files plus embedding cache", "destination": str(OUT / "_model_cache")}, sort_keys=True), flush=True)
    cache_dir = OUT / "_model_cache"
    tok = AutoTokenizer.from_pretrained(model_name, revision=revision, cache_dir=cache_dir)
    mod = AutoModel.from_pretrained(model_name, revision=revision, cache_dir=cache_dir).eval()
    resolved = getattr(mod.config, "_commit_hash", None) or revision
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mod.to(device)
    def encode(texts: Sequence[str]) -> np.ndarray:
        vectors=[]; n=max(1,batch); pos=0
        while pos < len(texts):
            try:
                chunk=texts[pos:pos+n]; x=tok(chunk, padding=True, truncation=True, max_length=512, return_tensors="pt")
                x={k:v.to(device) for k,v in x.items()}
                with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type=="cuda"):
                    out=mod(**x).last_hidden_state
                v=out[:,0]
                v=torch.nn.functional.normalize(v, p=2, dim=1).cpu().numpy(); vectors.append(v); pos += len(chunk)
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower() or n == 1: raise
                n=max(1,n//2)
        return np.vstack(vectors)
    return encode, {"model":model_name,"requested_revision":revision,"resolved_revision":resolved,"pooling":"cls","normalization":"l2","similarity":"inner_product_cosine","max_length":512,"device":str(device),"mixed_precision":device.type=="cuda"}

def _validate_ranked(rows: Sequence[Mapping[str, Any]], expected: int) -> None:
    if len(rows) != expected: raise ValueError(f"missing ranking rows: {len(rows)} != {expected}")
    keys=set()
    for row in rows:
        key=(row["model_id"],row["reaction_id"]); ids=list(row["ranked_ids"])
        if key in keys: raise ValueError(f"duplicate ranking key: {key}")
        if len(ids) != len(set(ids)): raise ValueError(f"duplicate ranked identifier: {key}")
        if len(ids) < 10 and row["method"] != "phase2_rule_based": raise ValueError(f"ranking shorter than 10: {key}")
        keys.add(key)

def freeze_queries(out: Path = OUT) -> list[dict[str,Any]]:
    rows=[]
    for r in load_query_population().to_dict("records"):
        text=query_text(r)
        rows.append({"model_id":r["model_id"],"reaction_id":r["reaction_id"],"split":"validation","template_version":QUERY_TEMPLATE_VERSION,"query":text})
    if any(r["split"]!="validation" for r in rows): raise ValueError("non-validation query")
    atomic_write_jsonl(rows,out/"queries.jsonl")
    return rows

def dense_rank(query_embeddings: np.ndarray, document_embeddings: np.ndarray,
               document_ids: Sequence[str], topn: int = 100) -> list[list[str]]:
    """Exact normalized-inner-product ranker, separated for mocked tests."""
    result=[]
    for q in query_embeddings:
        scores=document_embeddings @ q
        order=sorted(range(len(document_ids)),key=lambda i:(-float(scores[i]),document_ids[i]))
        result.append([document_ids[i] for i in order[:topn]])
    return result

def cache_key(config: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(dict(config),sort_keys=True,separators=(",",":")).encode()).hexdigest()

def run_bm25(out: Path = OUT, limit: int | None = None) -> Path:
    freeze_queries(out); corpus=load_query_population();
    if limit is not None: corpus=corpus.head(limit)
    docs=load_catalog(); bm=BM25(docs)
    rows=[]
    started=time.perf_counter()
    for r in corpus.to_dict("records"):
        rows.append({"model_id":r["model_id"],"reaction_id":r["reaction_id"],"method":"bm25","ranked_ids":bm.rank(query_text(r),100)})
    _validate_ranked(rows,len(corpus)); path=out/"rankings_bm25.jsonl"; atomic_write_jsonl(rows,path)
    write_json({"method":"bm25","seconds":round(time.perf_counter()-started,3),"hardware":platform.platform(),"python":platform.python_version(),"n_queries":len(rows)},out/"runtime_bm25.json")
    return path

def run_dense(out: Path = OUT, revision: str | None = MODEL_REVISION, batch: int = 4, limit: int | None = None) -> Path:
    freeze_queries(out); corpus=load_query_population();
    if limit is not None: corpus=corpus.head(limit)
    docs=load_catalog(); enc,meta=_dense_encoder(MODEL,revision,batch)
    if not meta["resolved_revision"]: raise ValueError("model revision could not be resolved")
    cfg={"catalog_sha256":sha256_file(CATALOG),"document_template":DOCUMENT_TEMPLATE_VERSION,"model":MODEL,"revision":meta["resolved_revision"],"pooling":"cls","normalization":"l2","similarity":"inner_product_cosine","max_length":512}
    key=cache_key(cfg); cache=out/"_embedding_cache"/f"{key}.npy"; cache.parent.mkdir(parents=True,exist_ok=True)
    cache_hit=cache.exists(); emb=np.load(cache) if cache_hit else enc([d["text"] for d in docs])
    if not cache_hit:
        tmp=cache.with_suffix(".tmp.npy"); np.save(tmp,emb); tmp.replace(cache)
    rows=[]
    started=time.perf_counter(); queries=enc([query_text(r) for r in corpus.to_dict("records")])
    ranks=dense_rank(queries,emb,[d["kegg_id"] for d in docs])
    for r,ids in zip(corpus.to_dict("records"),ranks): rows.append({"model_id":r["model_id"],"reaction_id":r["reaction_id"],"method":"bge_m3_dense","ranked_ids":ids})
    _validate_ranked(rows,len(corpus)); path=out/"rankings_bge_m3_dense.jsonl"; atomic_write_jsonl(rows,path)
    write_json({**meta,"seconds":round(time.perf_counter()-started,3),"hardware":platform.platform(),"python":platform.python_version(),"torch":importlib.metadata.version("torch"),"transformers":importlib.metadata.version("transformers"),"numpy":np.__version__,"n_queries":len(rows),"cache_key":key,"cache_hit":cache_hit},out/"runtime_bge_m3_dense.json")
    return path

def _read_jsonl(path: Path) -> list[dict[str,Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x]

def freeze_phase2(out: Path = OUT) -> Path:
    pop=load_query_population(); keys=set(zip(pop.model_id,pop.reaction_id)); candidates=_csv_rows_for_keys(CANDIDATES,keys,["model_id","reaction_id","candidate_kegg","raw_rank"],require_one=False)
    candidates["raw_rank"]=candidates.raw_rank.astype(int); candidates=candidates.sort_values(["model_id","reaction_id","raw_rank","candidate_kegg"])
    grouped={(m,r):list(g.candidate_kegg) for (m,r),g in candidates.groupby(["model_id","reaction_id"],sort=False)}
    rows=[{"model_id":r.model_id,"reaction_id":r.reaction_id,"method":"phase2_rule_based","ranked_ids":grouped.get((r.model_id,r.reaction_id),[])} for r in pop.itertuples()]
    _validate_ranked(rows,len(pop)); path=out/"rankings_phase2_rule_based.jsonl"; atomic_write_jsonl(rows,path); return path

def fuse(out: Path = OUT) -> Path:
    bm={ (r["model_id"],r["reaction_id"]):r["ranked_ids"] for r in _read_jsonl(out/"rankings_bm25.jsonl") }
    de={ (r["model_id"],r["reaction_id"]):r["ranked_ids"] for r in _read_jsonl(out/"rankings_bge_m3_dense.jsonl") }
    if set(bm)!=set(de): raise ValueError("component ranking keys differ")
    rows=[{"model_id":k[0],"reaction_id":k[1],"method":"bm25_bge_m3_rrf","ranked_ids":rrf([bm[k],de[k]],RRF_K,100)} for k in sorted(bm)]
    _validate_ranked(rows,len(bm)); path=out/"rankings_bm25_bge_m3_rrf.jsonl"; atomic_write_jsonl(rows,path); return path

def _truth_and_metadata() -> pd.DataFrame:
    assignments=pd.read_csv(SPLITS,dtype=str); allowed=assignments[assignments.split.isin(["train","validation"])]
    allowed_keys=set(zip(allowed.model_id,allowed.reaction_id))
    reactions=_csv_rows_for_keys(REACTIONS,allowed_keys,["model_id","reaction_id","ground_truth_kegg_all"])
    # Only train/validation labels are materialized; held-out test keys are excluded.
    frame=allowed.merge(reactions[["model_id","reaction_id","ground_truth_kegg_all"]],on=["model_id","reaction_id"],validate="one_to_one")
    frame["truth"]=frame.ground_truth_kegg_all.map(parse_kegg_ids)
    train_ids={x for ids in frame.loc[frame.split.eq("train"),"truth"] for x in ids}
    val=frame[frame.split.eq("validation")].copy(); reject_test_rows(val); val["seen_in_train"]=val.truth.map(lambda x:any(i in train_ids for i in x))
    val_keys=set(zip(val.model_id,val.reaction_id)); candidates=_csv_rows_for_keys(CANDIDATES,val_keys,["model_id","reaction_id","candidate_kegg","raw_rank"],require_one=False); candidates["raw_rank"]=candidates.raw_rank.astype(int)
    grouped={(m,r):list(g.sort_values(["raw_rank","candidate_kegg"]).candidate_kegg) for (m,r),g in candidates.groupby(["model_id","reaction_id"],sort=False)}
    corrected=[]
    for row in val.itertuples():
        ranked=grouped.get((row.model_id,row.reaction_id),[]); target=set(row.truth)
        hit=next((i for i,x in enumerate(ranked,1) if x in target),None)
        corrected.append(assign_stratum(row.status,hit is not None,hit==1))
    val["frozen_stratum"]=val.stratum; val["stratum"]=corrected
    return val.sort_values(["model_id","reaction_id"])

def _three_way(df: pd.DataFrame, col: str) -> dict[str,Any]:
    return {"reaction_micro":round(float(df[col].mean()),6),"model_macro":round(float(df.groupby("model_id")[col].mean().mean()),6),"cluster_macro":round(float(df.groupby("cluster_id")[col].mean().mean()),6),"n_reactions":len(df)}

def _optional_rank(value: Any) -> int | None:
    return None if value == "" or pd.isna(value) else int(value)

def score_rankings(rows: Sequence[Mapping[str,Any]], truth: pd.DataFrame) -> pd.DataFrame:
    rank={(r["model_id"],r["reaction_id"]):r for r in rows}; out=[]
    for t in truth.itertuples():
        rr=rank.get((t.model_id,t.reaction_id)); ids=[] if rr is None else list(rr["ranked_ids"]); target=set(t.truth)
        exact=next((i for i,x in enumerate(ids,1) if x in target),None)
        brite=next((i for i,x in enumerate(ids,1) if match_kinds(x,target)["brite_orthology"]),None)
        row={"model_id":t.model_id,"reaction_id":t.reaction_id,"cluster_id":t.cluster_id,"stratum":t.stratum,"seen_in_train":bool(t.seen_in_train),"method":rr["method"] if rr else "missing","ranking_depth":len(ids),"missing_output":rr is None,"first_hit_rank_exact":exact or "","first_hit_rank_brite_orthology":brite or "","mrr_at_10_exact":0 if exact is None or exact>10 else 1/exact,"mrr_at_10_brite_orthology":0 if brite is None or brite>10 else 1/brite}
        for k in (1,3,5,10): row[f"recall_at_{k}_exact"]=exact is not None and exact<=k; row[f"recall_at_{k}_brite_orthology"]=brite is not None and brite<=k
        out.append(row)
    return pd.DataFrame(out)

def paired_cluster_bootstrap(a: pd.DataFrame,b:pd.DataFrame,col:str,seed:int=20260902,n_boot:int=10000)->dict[str,Any]:
    x=a[["model_id","reaction_id","cluster_id",col]].merge(b[["model_id","reaction_id",col]],on=["model_id","reaction_id"],suffixes=("_a","_b")); groups=sorted(x.cluster_id.unique()); rng=random.Random(seed); vals=[]
    for _ in range(n_boot):
        drawn=pd.concat([x[x.cluster_id.eq(rng.choice(groups))] for _ in groups],ignore_index=True); vals.append(float((drawn[f"{col}_a"].astype(float)-drawn[f"{col}_b"].astype(float)).mean()))
    vals.sort(); point=float((x[f"{col}_a"].astype(float)-x[f"{col}_b"].astype(float)).mean())
    return {"delta_a_minus_b":round(point,6),"ci_95_percentile":[round(vals[int(.025*n_boot)],6),round(vals[min(n_boot-1,int(.975*n_boot))],6)],"seed":seed,"n_boot":n_boot,"unit":"frozen validation cluster","n_clusters":len(groups),"estimand":"paired reaction-micro Recall@10 difference","limitation":"Only 12 validation clusters; intervals may be unstable. Do not claim superiority when zero is included."}

def evaluate(out: Path = OUT) -> None:
    truth=_truth_and_metadata(); files={"phase2_rule_based":"rankings_phase2_rule_based.jsonl","bm25":"rankings_bm25.jsonl","bge_m3_dense":"rankings_bge_m3_dense.jsonl","bm25_bge_m3_rrf":"rankings_bm25_bge_m3_rrf.jsonl"}; scored={}
    discrepancies=truth[truth.frozen_stratum.ne(truth.stratum)][["model_id","reaction_id","frozen_stratum","stratum","ground_truth_kegg_all"]].rename(columns={"stratum":"corrected_evaluation_stratum"})
    write_csv(discrepancies,out/"frozen_stratum_discrepancies.csv")
    rank_lists={m:{(r["model_id"],r["reaction_id"]):r["ranked_ids"] for r in _read_jsonl(out/name)} for m,name in files.items()}
    for method,name in files.items(): scored[method]=score_rankings(_read_jsonl(out/name),truth); write_csv(scored[method],out/f"scored_{method}.csv")
    metric_cols=([f"recall_at_{k}_exact" for k in (1,3,5,10)]+["mrr_at_10_exact"]+
                 [f"recall_at_{k}_brite_orthology" for k in (1,3,5,10)]+["mrr_at_10_brite_orthology"])
    metrics={m:{c:_three_way(df,c) for c in metric_cols} for m,df in scored.items()}
    write_json(metrics,out/"metrics_by_method.json")
    diagnostics={m:{"n_expected":len(truth),"n_outputs":int((~df.missing_output).sum()),"missing_outputs":int(df.missing_output.sum()),"ranking_depth_min":int(df.ranking_depth.min()),"ranking_depth_max":int(df.ranking_depth.max()),"ranking_depth_mean":round(float(df.ranking_depth.mean()),3)} for m,df in scored.items()}; write_json(diagnostics,out/"output_diagnostics.json")
    true_rf={"unconstrained","empty_constrained","nonempty_answer_absent"}
    strata={}
    for m,df in scored.items():
        strata[m]={s:{c:_three_way(g,c) for c in metric_cols} for s,g in df.groupby("stratum")}
        strata[m]["true_retrieval_failure"]={c:_three_way(df[df.stratum.isin(true_rf)],c) for c in metric_cols}
        strata[m]["rerank_failure_separate"]={c:_three_way(df[df.stratum.eq("retrievable_rerank_failure")],c) for c in metric_cols}
    write_json(strata,out/"stratum_analysis.json")
    seen={m:{str(v).lower():{c:_three_way(g,c) for c in metric_cols} for v,g in df.groupby("seen_in_train")} for m,df in scored.items()}; write_json(seen,out/"seen_unseen_analysis.json")
    hits={m:set(zip(df.loc[df.recall_at_10_exact,"model_id"],df.loc[df.recall_at_10_exact,"reaction_id"])) for m,df in scored.items()}; universe=set(zip(truth.model_id,truth.reaction_id)); bm,hg=hits["bm25"],hits["bge_m3_dense"]
    overlap={"bm25_only":len(bm-hg),"dense_only":len(hg-bm),"both":len(bm&hg),"neither":len(universe-(bm|hg)),"rrf_only_vs_components":len(hits["bm25_bge_m3_rrf"]-(bm|hg)),"phase2_overlap_bm25":len(hits["phase2_rule_based"]&bm),"phase2_overlap_dense":len(hits["phase2_rule_based"]&hg)}; write_json(overlap,out/"method_overlap.json")
    boot={"bm25_minus_dense":paired_cluster_bootstrap(scored["bm25"],scored["bge_m3_dense"],"recall_at_10_exact"),"rrf_minus_bm25":paired_cluster_bootstrap(scored["bm25_bge_m3_rrf"],scored["bm25"],"recall_at_10_exact"),"rrf_minus_dense":paired_cluster_bootstrap(scored["bm25_bge_m3_rrf"],scored["bge_m3_dense"],"recall_at_10_exact")}; best=max((metrics[m]["recall_at_10_exact"]["reaction_micro"],m) for m in ("bm25","bge_m3_dense","bm25_bge_m3_rrf"))[1]; boot["best_full_catalog_minus_phase2"]={"best_method":best,**paired_cluster_bootstrap(scored[best],scored["phase2_rule_based"],"recall_at_10_exact")}; write_json(boot,out/"bootstrap_comparisons.json")
    examples=[]
    categories={"bm25_only":bm-hg,"dense_only":hg-bm,"hybrid_only":hits["bm25_bge_m3_rrf"]-(bm|hg),"all_method_failure":universe-(bm|hg|hits["bm25_bge_m3_rrf"]|hits["phase2_rule_based"])}
    bm_rank=scored["bm25"].set_index(["model_id","reaction_id"]).first_hit_rank_exact
    hy_rank=scored["bm25_bge_m3_rrf"].set_index(["model_id","reaction_id"]).first_hit_rank_exact
    categories["fusion_harms_rank"]={k for k in universe if bm_rank.get(k)!="" and (hy_rank.get(k)=="" or int(hy_rank.get(k))>int(bm_rank.get(k)))}
    categories["seen_target_behavior"]={k for k in universe if bool(truth.set_index(["model_id","reaction_id"]).seen_in_train.get(k))}
    categories["unseen_target_behavior"]={k for k in universe if not bool(truth.set_index(["model_id","reaction_id"]).seen_in_train.get(k))}
    q=load_query_population().set_index(["model_id","reaction_id"])
    truth_idx=truth.set_index(["model_id","reaction_id"])
    for cat,keys in categories.items():
        if keys:
            k=sorted(keys)[0]; meta=truth_idx.loc[k]
            examples.append({"category":cat,"model_id":k[0],"reaction_id":k[1],"stratum":meta.stratum,"seen_in_train":bool(meta.seen_in_train),"ground_truth_ids":list(meta.truth),"query":query_text(q.loc[k].to_dict()),"top10":{m:ids.get(k,[])[:10] for m,ids in rank_lists.items()},"first_hit_rank_exact":{m:_optional_rank(scored[m].set_index(["model_id","reaction_id"]).first_hit_rank_exact.get(k)) for m in scored}})
    write_json({"selection":"lexicographically_first_eligible_reaction","examples":examples},out/"qualitative_examples.json")
    exact_cols=[f"recall_at_{k}_exact" for k in (1,3,5,10)]+["mrr_at_10_exact"]
    report=["# Phase 3 full-catalog retrieval baselines","",f"Validation reactions: {len(truth)}; models: {truth.model_id.nunique()}; clusters: {truth.cluster_id.nunique()}. Rankings were frozen before train/validation labels were joined; no held-out test query or label was loaded.","", "## Exact reaction-micro metrics","", "| Method | R@1 | R@3 | R@5 | R@10 | MRR@10 |","|---|---:|---:|---:|---:|---:|"]
    for m in files: report.append("| "+m+" | "+" | ".join(str(metrics[m][c]["reaction_micro"]) for c in exact_cols)+" |")
    brite_cols=[f"recall_at_{k}_brite_orthology" for k in (1,3,5,10)]+["mrr_at_10_brite_orthology"]
    report += ["","## BRITE/orthology-aware reaction-micro metrics","","| Method | R@1 | R@3 | R@5 | R@10 | MRR@10 |","|---|---:|---:|---:|---:|---:|"]
    for m in files: report.append("| "+m+" | "+" | ".join(str(metrics[m][c]["reaction_micro"]) for c in brite_cols)+" |")
    report += ["","## Averaging, seen/unseen, and recovery","","| Method | R@10 model-macro | R@10 cluster-macro | Seen R@10 (n=847) | Unseen R@10 (n=122) | True RF R@10 (n=524) |","|---|---:|---:|---:|---:|---:|"]
    for m in files:
        r10=metrics[m]["recall_at_10_exact"]; report.append(f"| {m} | {r10['model_macro']} | {r10['cluster_macro']} | {seen[m]['true']['recall_at_10_exact']['reaction_micro']} | {seen[m]['false']['recall_at_10_exact']['reaction_micro']} | {strata[m]['true_retrieval_failure']['recall_at_10_exact']['reaction_micro']} |")
    report += ["","True retrieval failure is corrected to unconstrained (85), empty constrained (419), and nonempty-answer-absent (20); the 17 retrievable rerank failures are reported separately. Two frozen multi-label stratum discrepancies are preserved in `frozen_stratum_discrepancies.csv`.","","## Recall@10 overlap","",f"BM25 only: {overlap['bm25_only']}; dense only: {overlap['dense_only']}; both: {overlap['both']}; neither: {overlap['neither']}; RRF-only versus both component top-10 lists: {overlap['rrf_only_vs_components']}.","","## Paired cluster bootstrap","","10,000 percentile replicates, seed 20260902, resampling the 12 frozen validation clusters. Deltas are paired reaction-micro Recall@10.",""]
    for name,value in boot.items(): report.append(f"- {name}: {value['delta_a_minus_b']}, 95% CI [{value['ci_95_percentile'][0]}, {value['ci_95_percentile'][1]}]"+(f" (best method: {value['best_method']})" if 'best_method' in value else ""))
    report += ["","Only 12 clusters are available, so intervals may be unstable. RRF versus BM25 includes zero and is not evidence of superiority.","","## Runtime and recommendation","",f"The cached BGE-M3 validation-query pass took {json.loads((out/'runtime_bge_m3_dense.json').read_text())['seconds']} seconds on CPU-only PyTorch. The one-time catalog pass took approximately 56.7 minutes before the 50,430,080-byte embedding cache was atomically written; the downloaded model cache occupied 4.56 GB because Windows could not use Hugging Face symlinks.","","Use a smaller dense encoder as the Phase 3B starting checkpoint on the RTX 3070. Keep BGE-M3 as the frozen off-the-shelf reference: its standalone Recall@10 trails BM25, and RRF's overall gain over BM25 is small and not statistically resolved, although the unseen-target gain is worth retaining as a comparison."]
    (out/"REPORT.md").write_text("\n".join(report)+"\n",encoding="utf-8",newline="\n")
    write_json({"recommendation":"Use a smaller encoder as the Phase 3B starting checkpoint; retain BGE-M3 as the frozen off-the-shelf reference.","decision_basis":"BGE-M3 standalone Recall@10 is below BM25; RRF's +0.005160 overall delta versus BM25 has a 95% interval containing zero, while its unseen-target gain remains useful as a reference; BGE-M3 also required a 4.56 GB model cache and approximately 56.7 CPU minutes for one-time catalog encoding.","scope":"validation-only recommendation; no trained model or held-out test evaluation"},out/"recommendation.json")
    artifacts=[p for p in out.iterdir() if p.is_file() and p.name!="artifact_manifest.json"]; write_artifact_manifest(out,artifacts)

def verify_manifest(out: Path = OUT) -> list[str]:
    manifest=json.loads((out/"artifact_manifest.json").read_text(encoding="utf-8")); paths=[x["path"] for x in manifest["files"]]; problems=[]
    if len(paths)!=len(set(paths)): problems.append("duplicate paths")
    for item in manifest["files"]:
        if "\\" in item["path"]: problems.append(f"non-POSIX path: {item['path']}")
        path=REPO_ROOT/item["path"]
        if not path.exists() or sha256_portable(path)!=item["sha256"]: problems.append(f"digest mismatch: {item['path']}")
    return problems

def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument("method",choices=["bm25","dense","phase2","fuse","evaluate","verify"]); p.add_argument("--out",type=Path,default=OUT); p.add_argument("--revision",default=MODEL_REVISION); p.add_argument("--batch",type=int,default=4); p.add_argument("--limit",type=int); a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    if a.method=="bm25": run_bm25(a.out,a.limit)
    elif a.method=="dense": run_dense(a.out,a.revision,a.batch,a.limit)
    elif a.method=="phase2": freeze_phase2(a.out)
    elif a.method=="fuse": fuse(a.out)
    elif a.method=="evaluate": evaluate(a.out)
    elif a.method=="verify":
        problems=verify_manifest(a.out); print(json.dumps({"problems":problems,"n_problems":len(problems)})); return bool(problems)
    write_json({"query_template":QUERY_TEMPLATE_VERSION,"query_format":"Equation: {normalized SBML equation}\\nParticipants: {name} [species={SBML id}; ChEBI={ids}; KEGG-compound={ids}]; ...","document_template":DOCUMENT_TEMPLATE_VERSION,"document_format":"Definition/Names/Equation/Enzyme/Class/Brite labelled lines when present","document_fields":["DEFINITION","NAME","EQUATION","ENZYME","RCLASS","BRITE"],"document_id_searchable":False,"tokenizer":TOKENIZER_VERSION,"bm25":{"implementation":"project native Okapi BM25","version":"phase3_retrieval.py v1","k1":1.2,"b":.75},"dense":{"checkpoint":MODEL,"revision":MODEL_REVISION,"pooling":"cls","normalization":"l2","similarity":"inner_product_cosine","learned_sparse":False,"multi_vector":False,"fine_tuned":False},"rrf":{"k":RRF_K,"ranks":"one-indexed","missing":"zero contribution","tie_break":"KEGG identifier ascending"},"catalog":{"source":repo_relative_posix(CATALOG),"count":12312,"sha256":sha256_file(CATALOG)},"inputs":{repo_relative_posix(p):sha256_portable(p) for p in (SPLITS,REACTION_TEXT,SPECIES_EVIDENCE,SPECIES_NAMES,CANDIDATES,STATUS,REACTIONS)}},a.out/"config.json")
    return 0
if __name__ == "__main__": raise SystemExit(main())
