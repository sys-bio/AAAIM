"""Phase 3 experiment modes, result schema, mocks, and response cache.

No network. Live providers raise until the user explicitly approves a budgeted run.
Tests and scaffolding use ``MockProvider`` and ``MemoryCache``.

Modes
-----
``direct_open_set``
    No candidate list, no tools. Measures parametric identification.
``tool_assisted``
    May query an explicitly configured resource; evidence is recorded. Unsupported
    guesses are labelled separately from evidence-backed answers.
``closed_set``
    Reorders the frozen Phase 2 candidate set. Not open-set recovery.
``learned_retriever``
    Query against the full KEGG catalog. Training is out of scope here; the schema
    is ready for cached retrieval results.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import AbstractSet, Any, Callable, Dict, Iterable, List, Optional, Protocol, Sequence

from benchmark.scripts.phase3_common import (
    ID_ABSENT,
    ID_IN_CATALOG,
    ID_MALFORMED,
    KEGG_ID_STRICT,
    PHASE3_DIR,
    PROMPT_TEMPLATE_VERSION,
    TOKENIZER_SCAFFOLD,
    classify_kegg_id,
    estimate_tokens,
    load_kegg_catalog_ids,
)

logger = logging.getLogger("phase3_modes")

MODES = ("direct_open_set", "tool_assisted", "closed_set", "learned_retriever")
BASIS_VALUES = ("recalled_knowledge", "supplied_evidence", "mixed")
CACHE_DIR = PHASE3_DIR / "_response_cache"

_LIVE_CALL_ERROR = (
    "Refusing to make a live API call. Freeze the sample, pass leakage tests, print "
    "the cost estimate, and obtain explicit approval for provider, model, sample size, "
    "and budget before enabling live calls."
)


class LiveCallBlocked(RuntimeError):
    """Raised by any provider that would otherwise hit the network."""


@dataclass
class Prediction:
    kegg_id: str
    confidence: Optional[float] = None
    valid_kegg_id: bool = False  # in the frozen KEGG catalog, not merely well-formed
    well_formed: bool = False
    in_catalog: bool = False
    id_class: str = ID_MALFORMED
    duplicate: bool = False
    prediction_supported_by_evidence: bool = False
    supporting_evidence_ids: List[str] = field(default_factory=list)


def make_prediction(
    kegg_id: str,
    confidence: Optional[float],
    catalog: AbstractSet[str],
    *,
    duplicate: bool = False,
) -> Prediction:
    id_class = classify_kegg_id(kegg_id, catalog)
    well_formed = id_class != ID_MALFORMED
    in_catalog = id_class == ID_IN_CATALOG
    return Prediction(
        kegg_id=kegg_id,
        confidence=confidence,
        valid_kegg_id=in_catalog,
        well_formed=well_formed,
        in_catalog=in_catalog,
        id_class=id_class,
        duplicate=duplicate,
    )


def evidence_identifier_list(evidence: Sequence["ToolEvidence"]) -> List[str]:
    ids: List[str] = []
    for ev in evidence:
        ids.extend(str(x) for x in (ev.identifiers or []) if x)
    return ids


def link_predictions_to_evidence(
    predictions: Sequence[Prediction],
    evidence_ids: Sequence[str],
) -> None:
    """Mark each prediction if its identifier appears in recorded evidence ids."""
    id_set = set(evidence_ids)
    for pred in predictions:
        supporting = [i for i in evidence_ids if i == pred.kegg_id]
        pred.supporting_evidence_ids = supporting
        pred.prediction_supported_by_evidence = pred.kegg_id in id_set


def apply_tool_evidence(result: "ModeResult", evidence: Sequence["ToolEvidence"]) -> "ModeResult":
    result.evidence = list(evidence)
    link_predictions_to_evidence(result.predictions, evidence_identifier_list(result.evidence))
    result.evidence_backed = (
        (not result.abstain)
        and any(p.prediction_supported_by_evidence for p in result.predictions)
    )
    if result.evidence_backed:
        result.basis = "supplied_evidence"
    return result


@dataclass
class ToolEvidence:
    source: str
    query: str
    n_hits: int = 0
    identifiers: List[str] = field(default_factory=list)
    urls: List[str] = field(default_factory=list)
    snippet: str = ""


@dataclass
class ModeResult:
    sample_id: str
    model_id: str
    reaction_id: str
    cluster_id: str
    stratum: str
    mode: str
    variant: str
    template_version: str
    abstain: bool
    predictions: List[Prediction]
    rationale: str = ""
    basis: str = "recalled_knowledge"
    evidence_backed: bool = False
    evidence: List[ToolEvidence] = field(default_factory=list)
    raw_text: str = ""
    parse_error: Optional[str] = None
    n_input_tokens: int = 0
    n_output_tokens: int = 0
    latency_ms: Optional[float] = None
    provider: str = "mock"
    model_name: str = "mock"
    cache_hit: bool = False
    cached: bool = False

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        return payload


def cache_key(prompt: Dict[str, Any], *, mode: str, provider: str, model_name: str) -> str:
    blob = json.dumps(
        {"mode": mode, "provider": provider, "model": model_name, "prompt": prompt},
        sort_keys=True, separators=(",", ":"),
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class ResponseCache:
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        raise NotImplementedError

    def put(self, key: str, value: Dict[str, Any]) -> None:
        raise NotImplementedError


class MemoryCache(ResponseCache):
    def __init__(self) -> None:
        self._store: Dict[str, Dict[str, Any]] = {}

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        return self._store.get(key)

    def put(self, key: str, value: Dict[str, Any]) -> None:
        self._store[key] = value


class FileCache(ResponseCache):
    def __init__(self, root: Path = CACHE_DIR) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.root / f"{key}.json"

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        path = self._path(key)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def put(self, key: str, value: Dict[str, Any]) -> None:
        path = self._path(key)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp, path)


def parse_structured_output(
    raw: str,
    *,
    catalog: Optional[AbstractSet[str]] = None,
) -> Dict[str, Any]:
    """Parse a model response into abstain/predictions/rationale/basis.

    Malformed JSON, missing fields, confidence outside [0, 1], duplicate ids,
    and identifiers absent from the frozen KEGG catalog are recorded rather than
    coerced into a fake answer. An explicit abstention is never turned into an id.
    ``abstain=false`` with no predictions is a compliance error, not a silent abstention.
    """
    catalog_ids: AbstractSet[str] = (
        catalog if catalog is not None else load_kegg_catalog_ids()
    )
    empty = {
        "abstain": True, "predictions": [], "rationale": "",
        "basis": "recalled_knowledge", "parse_error": None,
    }
    text = (raw or "").strip()
    if not text:
        empty["parse_error"] = "empty_response"
        return empty
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            empty["rationale"] = text[:500]
            empty["parse_error"] = "unparseable"
            return empty
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            empty["rationale"] = text[:500]
            empty["parse_error"] = "unparseable"
            return empty

    if not isinstance(payload, dict):
        empty["parse_error"] = "not_an_object"
        return empty

    errors: List[str] = []
    abstain = bool(payload.get("abstain", False))
    preds_in = payload.get("predictions") or []
    preds: List[Prediction] = []
    if not abstain:
        if not isinstance(preds_in, list):
            preds_in = []
            errors.append("predictions_not_a_list")
        seen_ids: set = set()
        for item in preds_in:
            if len(preds) >= 3:
                break
            if isinstance(item, str):
                kid = item.strip()
                conf = None
            elif isinstance(item, dict):
                kid = str(item.get("kegg_id") or item.get("id") or "").strip()
                conf = item.get("confidence")
                try:
                    conf = float(conf) if conf is not None else None
                except (TypeError, ValueError):
                    conf = None
                    errors.append("confidence_unparseable")
            else:
                continue
            if not kid:
                errors.append("empty_kegg_id")
                continue
            if conf is not None and not (0.0 <= conf <= 1.0):
                errors.append("confidence_out_of_range")
                conf = None
            duplicate = kid in seen_ids
            if duplicate:
                errors.append("duplicate_predicted_ids")
                continue
            seen_ids.add(kid)
            preds.append(make_prediction(kid, conf, catalog_ids, duplicate=False))
        if not preds:
            errors.append("abstain_false_without_predictions")
    basis = payload.get("basis") or "recalled_knowledge"
    if basis not in BASIS_VALUES:
        basis = "recalled_knowledge"
    parse_error = ";".join(dict.fromkeys(errors)) if errors else None
    return {
        "abstain": abstain,
        "predictions": [] if abstain else preds,
        "rationale": str(payload.get("rationale") or ""),
        "basis": basis,
        "parse_error": parse_error,
    }


class Provider(Protocol):
    name: str
    model_name: str

    def complete(self, prompt: Dict[str, Any]) -> str: ...


class BlockedLiveProvider:
    """Stand-in that documents the live-call gate. Tests assert it never succeeds."""

    name = "blocked"
    model_name = "unapproved"

    def complete(self, prompt: Dict[str, Any]) -> str:
        raise LiveCallBlocked(_LIVE_CALL_ERROR)


class MockProvider:
    """Deterministic offline provider for tests and scaffolding."""

    name = "mock"

    def __init__(self, responses: Optional[Dict[str, str]] = None, default: str = "") -> None:
        self.model_name = "mock-phase3"
        self.responses = responses or {}
        self.default = default or json.dumps({
            "abstain": True, "predictions": [],
            "rationale": "mock default abstention", "basis": "recalled_knowledge",
        })

    def complete(self, prompt: Dict[str, Any]) -> str:
        key = json.dumps(prompt, sort_keys=True)
        if key in self.responses:
            return self.responses[key]
        # Allow lookup by reaction id embedded in the user message.
        user = ""
        for msg in prompt.get("messages") or []:
            if msg.get("role") == "user":
                user = msg.get("content") or ""
        for needle, text in self.responses.items():
            if needle and needle in user:
                return text
        return self.default


def prompt_for_mode(prompt: Dict[str, Any], mode: str) -> Dict[str, Any]:
    """Copy a stored prompt and swap in the mode-specific system instruction."""
    out = json.loads(json.dumps(prompt))
    if mode == "tool_assisted":
        system = out.get("system_tool_assisted")
    else:
        system = out.get("system_direct")
    if system:
        for msg in out.get("messages") or []:
            if msg.get("role") == "system":
                msg["content"] = system
                break
        out["mode"] = mode
    return out


def run_direct(
    sample: Dict[str, Any],
    prompt: Dict[str, Any],
    provider: Provider,
    *,
    cache: Optional[ResponseCache] = None,
    variant: str,
) -> ModeResult:
    return _run(
        "direct_open_set", sample, prompt_for_mode(prompt, "direct_open_set"),
        provider, cache=cache, variant=variant,
    )


def run_tool_assisted(
    sample: Dict[str, Any],
    prompt: Dict[str, Any],
    provider: Provider,
    *,
    evidence: Optional[Sequence[ToolEvidence]] = None,
    cache: Optional[ResponseCache] = None,
    variant: str,
) -> ModeResult:
    result = _run(
        "tool_assisted", sample, prompt_for_mode(prompt, "tool_assisted"),
        provider, cache=cache, variant=variant,
    )
    return apply_tool_evidence(result, evidence or [])


def run_closed_set(
    sample: Dict[str, Any],
    ranked_kegg_ids: Sequence[str],
    *,
    variant: str = "phase2_candidates",
    abstain: bool = False,
) -> ModeResult:
    catalog = load_kegg_catalog_ids()
    preds = [
        make_prediction(k, None, catalog)
        for k in list(ranked_kegg_ids)[:3]
    ]
    link_predictions_to_evidence(preds, list(ranked_kegg_ids))
    abstain_flag = abstain or not preds
    return ModeResult(
        sample_id=sample["sample_id"],
        model_id=sample["model_id"],
        reaction_id=sample["reaction_id"],
        cluster_id=sample.get("cluster_id", ""),
        stratum=sample.get("stratum", ""),
        mode="closed_set",
        variant=variant,
        template_version="phase2-frozen-candidates",
        abstain=abstain_flag,
        predictions=[] if abstain else preds,
        rationale="Frozen Phase 2 candidate order (heuristic).",
        basis="supplied_evidence",
        evidence_backed=(not abstain_flag) and any(
            p.prediction_supported_by_evidence for p in preds),
        provider="phase2",
        model_name="heuristic",
    )


def run_learned_retriever(
    sample: Dict[str, Any],
    ranked_kegg_ids: Sequence[str],
    *,
    variant: str = "biencoder_full_kegg",
) -> ModeResult:
    catalog = load_kegg_catalog_ids()
    preds = [
        make_prediction(k, None, catalog)
        for k in list(ranked_kegg_ids)[:10]
    ]
    link_predictions_to_evidence(preds, list(ranked_kegg_ids))
    return ModeResult(
        sample_id=sample["sample_id"],
        model_id=sample["model_id"],
        reaction_id=sample["reaction_id"],
        cluster_id=sample.get("cluster_id", ""),
        stratum=sample.get("stratum", ""),
        mode="learned_retriever",
        variant=variant,
        template_version="learned-retriever-design",
        abstain=not preds,
        predictions=preds,
        rationale="Full-catalog retrieval (not yet trained; schema only).",
        basis="supplied_evidence",
        evidence_backed=any(p.prediction_supported_by_evidence for p in preds),
        provider="untrained",
        model_name="pending",
    )


def _run(
    mode: str,
    sample: Dict[str, Any],
    prompt: Dict[str, Any],
    provider: Provider,
    *,
    cache: Optional[ResponseCache],
    variant: str,
) -> ModeResult:
    key = cache_key(prompt, mode=mode, provider=provider.name, model_name=provider.model_name)
    hit = cache.get(key) if cache is not None else None
    if hit is not None:
        parsed = parse_structured_output(hit["raw_text"])
        return ModeResult(
            sample_id=sample["sample_id"],
            model_id=sample["model_id"],
            reaction_id=sample["reaction_id"],
            cluster_id=sample.get("cluster_id", ""),
            stratum=sample.get("stratum", ""),
            mode=mode,
            variant=variant,
            template_version=prompt.get("template_version", PROMPT_TEMPLATE_VERSION),
            abstain=parsed["abstain"],
            predictions=parsed["predictions"],
            rationale=parsed["rationale"],
            basis=parsed["basis"],
            raw_text=hit["raw_text"],
            parse_error=parsed["parse_error"],
            n_input_tokens=int(hit.get("n_input_tokens") or 0),
            n_output_tokens=int(hit.get("n_output_tokens") or 0),
            provider=provider.name,
            model_name=provider.model_name,
            cache_hit=True,
            cached=True,
        )

    raw = provider.complete(prompt)
    parsed = parse_structured_output(raw)
    n_in = int(prompt.get("n_input_tokens_est") or estimate_tokens(
        " ".join(m.get("content", "") for m in prompt.get("messages") or [])))
    n_out = estimate_tokens(raw)
    if prompt.get("token_estimate_method") not in (None, TOKENIZER_SCAFFOLD):
        n_in = int(prompt.get("n_input_tokens_est") or n_in)
    if cache is not None:
        cache.put(key, {
            "raw_text": raw,
            "n_input_tokens": n_in,
            "n_output_tokens": n_out,
            "stored_at": datetime.now(timezone.utc).isoformat(),
        })
    return ModeResult(
        sample_id=sample["sample_id"],
        model_id=sample["model_id"],
        reaction_id=sample["reaction_id"],
        cluster_id=sample.get("cluster_id", ""),
        stratum=sample.get("stratum", ""),
        mode=mode,
        variant=variant,
        template_version=prompt.get("template_version", PROMPT_TEMPLATE_VERSION),
        abstain=parsed["abstain"],
        predictions=parsed["predictions"],
        rationale=parsed["rationale"],
        basis=parsed["basis"],
        raw_text=raw,
        parse_error=parsed["parse_error"],
        n_input_tokens=n_in,
        n_output_tokens=n_out,
        provider=provider.name,
        model_name=provider.model_name,
        cache_hit=False,
        cached=cache is not None,
    )
