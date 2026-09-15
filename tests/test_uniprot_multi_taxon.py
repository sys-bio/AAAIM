"""Regression checks for organism-specific UniProt label loading."""

import lzma
import pickle

from core import database_search


def _write_labels(path, labels):
    path.parent.mkdir(parents=True, exist_ok=True)
    with lzma.open(path, "wb") as handle:
        pickle.dump(labels, handle)


def test_uniprot_label_loader_merges_taxon_list(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    _write_labels(data_dir / "uniprot" / "uniprot2label_tax9606.lzma", {"P1": "HUMAN"})
    _write_labels(data_dir / "uniprot" / "uniprot2label_tax10090.lzma", {"P2": "MOUSE"})

    monkeypatch.setattr(database_search, "get_data_dir", lambda: data_dir)
    monkeypatch.setattr(database_search.load_uniprot_label_dict, "_cache", {}, raising=False)

    assert database_search.load_uniprot_label_dict(["9606", "10090"]) == {
        "P1": "HUMAN",
        "P2": "MOUSE",
    }


def test_uniprot_label_cache_is_taxon_specific(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    _write_labels(data_dir / "uniprot" / "uniprot2label_tax9606.lzma", {"P1": "HUMAN"})
    _write_labels(data_dir / "uniprot" / "uniprot2label_tax10090.lzma", {"P2": "MOUSE"})

    monkeypatch.setattr(database_search, "get_data_dir", lambda: data_dir)
    monkeypatch.setattr(database_search.load_uniprot_label_dict, "_cache", {}, raising=False)

    assert database_search.load_uniprot_label_dict("9606") == {"P1": "HUMAN"}
    assert database_search.load_uniprot_label_dict("10090") == {"P2": "MOUSE"}
