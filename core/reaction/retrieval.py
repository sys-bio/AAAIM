"""Selected epoch-1 full-catalog retrieval, independent of the benchmark harness."""
from __future__ import annotations

import hashlib
import json
import lzma
import pickle
import re
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

CATALOG_SHA256 = "00acb4de1bbfb1ddae0298a09d218f93eb22241844179439ff34a2912fba046c"
ARCHIVE_SHA256 = "3412a3fa546347d8209ab62ca7ef55fb490e148b5f90c25cf33616328a0d0f53"
MANIFEST_SHA256 = "cad31482d31112d9ad403c00cb832209f9bf5d2556675dfd1bc6007b11d904b5"
CONFIG = {"method": "bm25-trained-epoch1-rrf-v1", "component_depth": 100,
          "rrf_k": 60, "top_k": 10, "bm25_k1": 1.2, "bm25_b": .75,
          "query_template": "phase3-retrieval-query-v1", "document_template": "phase3-retrieval-document-v1",
          "tokenizer": "biochemical-regex-v1", "max_length": 256, "pooling": "cls",
          "normalization": "l2", "tie_break": "KEGG identifier ascending"}
TOKEN_RE = re.compile(r"(?:\d+(?:\.\d+)?-)?[A-Za-z]+(?:[A-Za-z0-9_+\-./:]*)|\d+(?:\.\d+)?|[+\-]", re.ASCII)


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def redact(value):
    value = "" if value is None else str(value)
    value = re.sub(r"kegg\.reaction[/:]R\d{5}(?!\d)", "[REDACTED_KEGG_REACTION]", value, flags=re.IGNORECASE)
    return re.sub(r"(?<!\d)(R\d{5})(?!\d)", "[REDACTED_KEGG_REACTION]", value)


def query_text(row):
    if {"ground_truth_ids", "ground_truth_kegg_all", "ground_truth_kegg_primary"} & set(row):
        raise ValueError("answer key supplied to query construction")
    participants = row.get("participant_evidence") or "; ".join(filter(None, [row.get("substrate_names"), row.get("product_names")]))
    return f"Equation: {redact(row.get('reaction_equation'))}\nParticipants: {redact(participants)}".strip()


def document_text(kegg_id, fields):
    parts = []
    for label, key in (("Definition", "DEFINITION"), ("Names", "NAME"), ("Equation", "EQUATION"),
                       ("Enzyme", "ENZYME"), ("Class", "RCLASS"), ("Brite", "BRITE")):
        value = redact(fields.get(key))
        if value:
            parts.append(f"{label}: {value}")
    return "\n".join(parts)


def tokenise(value):
    return [x.lower() for x in TOKEN_RE.findall(value or "")]


class BM25:
    def __init__(self, docs, k1=1.2, b=.75):
        if not docs:
            raise ValueError("empty catalog")
        self.docs, self.k1, self.b = list(docs), k1, b
        self.tokens = [tokenise(x["text"]) for x in docs]
        self.lengths = [len(x) for x in self.tokens]
        self.avgdl = sum(self.lengths) / len(self.lengths) or 1
        df = Counter(t for doc in self.tokens for t in set(doc))
        self.idf = {t: np.log(1 + (len(docs) - f + .5) / (f + .5)) for t, f in df.items()}
        self.postings = defaultdict(list)
        for i, terms in enumerate(map(Counter, self.tokens)):
            for term, freq in terms.items():
                self.postings[term].append((i, freq))

    def rank(self, query, topn=100):
        scores = np.zeros(len(self.docs))
        for term, qfreq in Counter(tokenise(query)).items():
            for i, freq in self.postings.get(term, []):
                denom = freq + self.k1 * (1 - self.b + self.b * self.lengths[i] / self.avgdl)
                scores[i] += self.idf[term] * qfreq * freq * (self.k1 + 1) / denom
        order = sorted(range(len(self.docs)), key=lambda i: (-scores[i], self.docs[i]["kegg_id"]))
        return [self.docs[i]["kegg_id"] for i in order[:topn]]


def fusion(components, catalog, k=60):
    if k < 0 or len(catalog) != len(set(catalog)):
        raise ValueError("invalid fusion configuration")
    scores = defaultdict(float)
    for ids in components:
        if len(ids) != len(set(ids)) or set(ids) - set(catalog):
            raise ValueError("invalid component ranking")
        for rank, identifier in enumerate(ids, 1):
            scores[identifier] += 1 / (k + rank)
    return sorted(catalog, key=lambda identifier: (-scores[identifier], identifier))


def evidence_for(catalog, bm25_ids, dense_ids):
    fused = fusion([bm25_ids, dense_ids], sorted(catalog))[:10]
    bm = {kid: i for i, kid in enumerate(bm25_ids, 1)}
    dense = {kid: i for i, kid in enumerate(dense_ids, 1)}
    records = []
    for rank, kid in enumerate(fused, 1):
        fields = catalog[kid]
        records.append({"evidence_id": f"E{rank:02d}", "kegg_id": kid, "fused_rank": rank,
            "fused_score": round((1 / (60 + bm[kid]) if kid in bm else 0) +
                                 (1 / (60 + dense[kid]) if kid in dense else 0), 12),
            "bm25_rank": bm.get(kid), "trained_biencoder_rank": dense.get(kid),
            **{name: str(fields.get(key) or "") for name, key in
               (("name", "NAME"), ("definition", "DEFINITION"), ("equation", "EQUATION"),
                ("enzyme", "ENZYME"), ("rclass", "RCLASS"), ("brite", "BRITE"))}})
    return records


def verify_model_dir(directory):
    directory = Path(directory)
    manifest = directory / "archive_manifest.json"
    if not manifest.is_file() or sha256(manifest) != MANIFEST_SHA256:
        raise ValueError("Missing or unrecognized selected encoder manifest; install the verified inference archive")
    entries = json.loads(manifest.read_text())["files"]
    for entry in entries:
        path = directory / entry["path"]
        if Path(entry["path"]).name != entry["path"] or path.is_symlink() or not path.is_file() or sha256(path) != entry["sha256"]:
            raise ValueError("Encoder resource digest mismatch: " + entry["path"])
    # Extra model/tokenizer configuration could change from_pretrained behavior.
    if {p.name for p in directory.iterdir()} != {x["path"] for x in entries} | {"archive_manifest.json"}:
        raise ValueError("Unexpected files in selected encoder directory")


def install_resources(archive, destination):
    """Explicit local installation only. Never downloads or overwrites resources."""
    if sha256(archive) != ARCHIVE_SHA256:
        raise ValueError("Selected inference archive SHA-256 mismatch")
    destination = Path(destination)
    with zipfile.ZipFile(archive) as source:
        if any(Path(name).name != name or name in {"", ".", ".."} for name in source.namelist()):
            raise ValueError("Unsafe inference archive path")
        destination.mkdir(parents=True, exist_ok=False)
        for name in source.namelist():
            (destination / name).write_bytes(source.read(name))
    verify_model_dir(destination)


@dataclass(frozen=True)
class RetrievalResources:
    catalog_path: str
    encoder_dir: str
    device: str = "cpu"
    batch_size: int = 64


@lru_cache(maxsize=2)
def load_encoder(directory, device="cpu"):
    """Shared, local-only encoder load, also usable for the small CPU fixture."""
    from transformers import AutoModel, AutoTokenizer
    verify_model_dir(directory)
    tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
    model = AutoModel.from_pretrained(directory, local_files_only=True).eval().to(device)
    return model, tokenizer


class ReactionRetriever:
    """One verified catalog/model instance; catalog embeddings built once per instance."""
    def __init__(self, resources):
        import torch
        if resources.device not in {"cpu", "cuda"}:
            raise ValueError("device must be cpu or cuda")
        if resources.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if resources.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable; choose cpu explicitly")
        if sha256(resources.catalog_path) != CATALOG_SHA256:
            raise ValueError("Frozen KEGG catalog SHA-256 mismatch")
        verify_model_dir(resources.encoder_dir)
        with lzma.open(resources.catalog_path, "rb") as handle:
            self.catalog = pickle.load(handle)  # trusted bytes verified above before unpickling
        if len(self.catalog) != 12312 or any(not re.fullmatch(r"R\d{5}", kid) for kid in self.catalog):
            raise ValueError("Frozen catalog membership mismatch")
        self.ids = sorted(self.catalog)
        docs = [{"kegg_id": kid, "text": document_text(kid, self.catalog[kid])} for kid in self.ids]
        self.bm25 = BM25(docs)
        self.resources = resources
        self.model, self.tokenizer = load_encoder(resources.encoder_dir, resources.device)
        self.embeddings = self.encode([doc["text"] for doc in docs])
        self.provenance = {**CONFIG, "catalog_sha256": CATALOG_SHA256,
            "encoder_manifest_sha256": MANIFEST_SHA256, "archive_sha256": ARCHIVE_SHA256,
            "device": resources.device, "dtype": "float16 autocast" if resources.device == "cuda" else "float32"}

    def encode(self, texts):
        import torch
        vectors = []
        for pos in range(0, len(texts), self.resources.batch_size):
            tokens = self.tokenizer(texts[pos:pos + self.resources.batch_size], padding=True,
                                    truncation=True, max_length=256, return_tensors="pt")
            tokens = {key: value.to(self.resources.device) for key, value in tokens.items()}
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=self.resources.device == "cuda"):
                vector = self.model(**tokens).last_hidden_state[:, 0]
                vector = torch.nn.functional.normalize(vector, p=2, dim=1)
            vectors.append(vector.float().cpu().numpy())
        return np.vstack(vectors)

    def retrieve(self, query):
        scores = self.embeddings @ self.encode([query])[0]
        order = sorted(range(len(self.ids)), key=lambda i: (-float(scores[i]), self.ids[i]))[:100]
        return evidence_for(self.catalog, self.bm25.rank(query, 100), [self.ids[i] for i in order])


@lru_cache(maxsize=2)
def load_retriever(resources):
    return ReactionRetriever(resources)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Install a local, verified AAAIM selected encoder archive; no downloads")
    parser.add_argument("archive")
    parser.add_argument("destination")
    args = parser.parse_args()
    install_resources(args.archive, args.destination)
