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
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Protocol, Sequence

from benchmark.scripts.phase3_common import (
    KEGG_ID_STRICT,
    PHASE3_DIR,
    PROMPT_TEMPLATE_VERSION,
    estimate_tokens,
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
    valid_kegg_id: bool = False


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
        path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_structured_output(raw: str) -> Dict[str, Any]:
    """Parse a model response into abstain/predictions/rationale/basis.

    Malformed JSON, missing fields, and non-R##### identifiers are recorded rather
    than coerced into a fake answer. An explicit abstention is never turned into an id.
    """
    text = (raw or "").strip()
    if not text:
        return {"abstain": True, "predictions": [], "rationale": "", "basis": "recalled_knowledge",
                "parse_error": "empty_response"}
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return {"abstain": True, "predictions": [], "rationale": text[:500],
                    "basis": "recalled_knowledge", "parse_error": "unparseable"}
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {"abstain": True, "predictions": [], "rationale": text[:500],
                    "basis": "recalled_knowledge", "parse_error": "unparseable"}

    if not isinstance(payload, dict):
        return {"abstain": True, "predictions": [], "rationale": "",
                "basis": "recalled_knowledge", "parse_error": "not_an_object"}

    abstain = bool(payload.get("abstain", False))
    preds_in = payload.get("predictions") or []
    preds: List[Prediction] = []
    if not abstain:
        if not isinstance(preds_in, list):
            preds_in = []
        for item in preds_in[:3]:
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
            else:
                continue
            preds.append(Prediction(
                kegg_id=kid,
                confidence=conf,
                valid_kegg_id=bool(KEGG_ID_STRICT.match(kid)),
            ))
    basis = payload.get("basis") or "recalled_knowledge"
    if basis not in BASIS_VALUES:
        basis = "recalled_knowledge"
    return {
        "abstain": abstain or not preds,
        "predictions": [] if abstain else preds,
        "rationale": str(payload.get("rationale") or ""),
        "basis": basis,
        "parse_error": None,
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


def run_direct(
    sample: Dict[str, Any],
    prompt: Dict[str, Any],
    provider: Provider,
    *,
    cache: Optional[ResponseCache] = None,
    variant: str,
) -> ModeResult:
    return _run("direct_open_set", sample, prompt, provider, cache=cache, variant=variant)


def run_tool_assisted(
    sample: Dict[str, Any],
    prompt: Dict[str, Any],
    provider: Provider,
    *,
    evidence: Optional[Sequence[ToolEvidence]] = None,
    cache: Optional[ResponseCache] = None,
    variant: str,
) -> ModeResult:
    result = _run("tool_assisted", sample, prompt, provider, cache=cache, variant=variant)
    result.evidence = list(evidence or [])
    result.evidence_backed = bool(result.evidence) and not result.abstain
    if result.evidence_backed:
        result.basis = "supplied_evidence"
    return result


def run_closed_set(
    sample: Dict[str, Any],
    ranked_kegg_ids: Sequence[str],
    *,
    variant: str = "phase2_candidates",
    abstain: bool = False,
) -> ModeResult:
    preds = [
        Prediction(kegg_id=k, confidence=None, valid_kegg_id=bool(KEGG_ID_STRICT.match(k)))
        for k in list(ranked_kegg_ids)[:3]
    ]
    return ModeResult(
        sample_id=sample["sample_id"],
        model_id=sample["model_id"],
        reaction_id=sample["reaction_id"],
        cluster_id=sample.get("cluster_id", ""),
        stratum=sample.get("stratum", ""),
        mode="closed_set",
        variant=variant,
        template_version="phase2-frozen-candidates",
        abstain=abstain or not preds,
        predictions=[] if abstain else preds,
        rationale="Frozen Phase 2 candidate order (heuristic).",
        basis="supplied_evidence",
        evidence_backed=not abstain and bool(preds),
        provider="phase2",
        model_name="heuristic",
    )


def run_learned_retriever(
    sample: Dict[str, Any],
    ranked_kegg_ids: Sequence[str],
    *,
    variant: str = "biencoder_full_kegg",
) -> ModeResult:
    preds = [
        Prediction(kegg_id=k, valid_kegg_id=bool(KEGG_ID_STRICT.match(k)))
        for k in list(ranked_kegg_ids)[:10]
    ]
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
        evidence_backed=bool(preds),
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
