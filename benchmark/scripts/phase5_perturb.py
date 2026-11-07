"""Label-free, deterministic query-local identifier perturbations.

No reference reactions, audit answers, retrieval outcomes or provider are used.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections import defaultdict


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def render(record):
    parts = []
    for p in record["participants"]:
        details = ["species=" + p["species_id"]]
        if p["chebi"]:
            details.append("ChEBI=" + ",".join(p["chebi"]))
        if p["kegg"]:
            details.append("KEGG-compound=" + ",".join(p["kegg"]))
        parts.append(p["name"] + " [" + "; ".join(details) + "]")
    text = "Equation: " + record["equation"] + "\nParticipants: " + "; ".join(parts)
    if re.search(r"(?<![A-Za-z0-9])R\d{5}(?!\d)", text, flags=re.I):
        raise ValueError("reaction-ID leakage in query")
    return text


def parse_query(text):
    equation, body = text.split("\nParticipants: ", 1)
    if not equation.startswith("Equation: "):
        raise ValueError("unsupported query template")
    participants = []
    for match in re.finditer(r"(?:^|; )(.*?) \[species=([^;\]]+)((?:; [^\]]*)?)\]", body):
        name, sid, attrs = match.groups()
        detail = dict(v.split("=", 1) for v in attrs.split("; ") if v)
        if set(detail) - {"ChEBI", "KEGG-compound"}:
            raise ValueError("unknown participant field")
        chebi = detail.get("ChEBI", "").split(",") if detail.get("ChEBI") else []
        kegg = detail.get("KEGG-compound", "").split(",") if detail.get("KEGG-compound") else []
        if any(not re.fullmatch(r"CHEBI:\d+", c) for c in chebi) or any(not re.fullmatch(r"C\d{5}", c) for c in kegg):
            raise ValueError("invalid species annotation identifier")
        participants.append({"name": name, "species_id": sid, "chebi": chebi, "kegg": kegg})
    record = {"equation": equation[len("Equation: "):], "participants": participants}
    if render(record) != text or len({p["species_id"] for p in participants}) != len(participants):
        raise ValueError("query reconstruction is not lossless/unique")
    return record


class Ontology:
    def __init__(self, parents, mapping):
        self.parents = {c: sorted(set(v), key=lambda x: int(x.split(":")[1])) for c, v in parents.items()}
        self.mapping = {c: sorted(set([v] if isinstance(v, str) else v)) for c, v in mapping.items()}
        self.children = defaultdict(set)
        for child, values in self.parents.items():
            for parent in values:
                self.children[parent].add(child)

    def ancestor(self, term):
        return self.parents.get(term, [None])[0] if self.parents.get(term) else None

    def sibling(self, participant, key):
        original_ids = set(participant["chebi"])
        original_compounds = set(participant["kegg"])
        for term in original_ids:
            original_compounds.update(self.mapping.get(term, []))
        # Without a known original compound mapping we cannot establish disjointness.
        if not original_compounds:
            return None
        siblings = set()
        for term in original_ids:
            for parent in self.parents.get(term, []):
                siblings.update(self.children[parent])
        choices = [c for c in siblings-original_ids if self.mapping.get(c) and not original_compounds.intersection(self.mapping[c])]
        return min(choices, key=lambda c: digest([key, c])) if choices else None


def annotation_links(record):
    return {(p["species_id"], ns, value) for p in record["participants"] for ns in ("chebi", "kegg") for value in p[ns]}


def perturb(record, kind, level, seed, ontology):
    forbidden = {"ground_truth_ids", "truth", "audit_category", "correct", "label", "stratum"}
    if forbidden.intersection(record):
        raise ValueError("label/evaluation metadata supplied to perturbation generator")
    result = copy.deepcopy(record)
    eligible = [p["species_id"] for p in record["participants"] if p["chebi"] or p["kegg"]]
    key = [seed, record["model_id"], record["reaction_id"]]
    ordered = sorted(eligible, key=lambda sid: digest([key, sid]))
    if kind in {"remove", "wrong"}:
        if not 0 <= level <= 1:
            raise ValueError("invalid corruption fraction")
        selected = set(ordered[:math.floor(level*len(ordered)+.5)])
    elif kind in {"reactants", "products"}:
        selected = set(record[kind]).intersection(eligible)
    elif kind == "ancestor":
        selected = set(eligible)
    elif kind == "clean":
        selected = set()
    else:
        raise ValueError("unknown perturbation")
    events, changed = [], []
    for p in result["participants"]:
        sid = p["species_id"]
        if sid not in selected:
            continue
        before = {ns: list(p[ns]) for ns in ("chebi", "kegg")}
        distances = []
        if kind in {"remove", "reactants", "products"}:
            p["chebi"], p["kegg"] = [], []
        elif kind == "ancestor":
            terms = []
            for term in p["chebi"]:
                parent = ontology.ancestor(term)
                terms.append(parent or term)
                distances.append({"original": term, "replacement": parent or term, "distance": 1 if parent else 0})
            p["chebi"] = list(dict.fromkeys(terms))
        elif kind == "wrong":
            sibling = ontology.sibling(p, [key, sid])
            if sibling:
                p["chebi"] = [sibling]
                p["kegg"] = list(ontology.mapping[sibling]) if before["kegg"] else []
        after = {ns: list(p[ns]) for ns in ("chebi", "kegg")}
        if before != after:
            changed.append(sid)
        events.append({"species_id": sid, "before": before, "after": after, "changed": before != after, "ancestor_distances": distances})
    # Everything except identifiers must be exactly preserved.
    for original, modified in zip(record["participants"], result["participants"]):
        if {k: v for k, v in original.items() if k not in {"chebi", "kegg"}} != {k: v for k, v in modified.items() if k not in {"chebi", "kegg"}}:
            raise ValueError("non-annotation participant evidence changed")
    old_links, new_links = annotation_links(record), annotation_links(result)
    accounting = {"eligible_species": len(eligible), "selected_species": len(selected), "changed_species": len(changed),
                  "unchanged_selected_species": len(selected)-len(changed), "links_before": len(old_links),
                  "links_after": len(new_links), "links_removed": len(old_links-new_links), "links_added": len(new_links-old_links),
                  "unchanged_query": render(record) == render(result),
                  "zero_eligible": not eligible, "zero_eligible_selected_side": kind in {"reactants", "products"} and not selected,
                  "selected_both_sides_species": len(selected.intersection(record["reactants"]).intersection(record["products"])),
                  "ancestor_changes_with_direct_kegg_retained": sum(bool(e["before"]["kegg"]) and e["changed"] for e in events) if kind == "ancestor" else 0,
                  "events": events}
    render(result)
    return result, accounting


def conditions(config):
    values = [("clean", "clean", 0, 0)]
    values += [(f"remove_{fraction:g}_s{seed}", "remove", fraction, seed) for fraction in config["removal_fractions"] for seed in config["seeds"]]
    values += [("reactants", "reactants", 1, 0), ("products", "products", 1, 0), ("ancestor", "ancestor", 1, 0)]
    values += [(f"wrong_0.25_s{seed}", "wrong", .25, seed) for seed in config["seeds"]]
    return values
