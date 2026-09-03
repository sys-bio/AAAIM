"""Phase 3A OpenAI Responses-API runner for the nine-call operational smoke test.

Reuses the existing Phase 3 prompt builders, structured-output schema, leakage
scanner, result parser, cache, and offline evaluator. Does not invent a parallel
format.

Required OpenAI Python SDK: openai==1.78.1 (Responses.parse + Pydantic
text_format). See requirements.txt.

Live spending is off unless ``--execute`` is passed. This module's live ceiling
is nine API calls and the supplied ``--max-cost-usd`` cap (smoke default $1.00).
The 489-call validation pilot is not authorized here.

Usage::

    python benchmark/scripts/phase3_openai_run.py
    python benchmark/scripts/phase3_openai_run.py --execute --max-cost-usd 1.00
    python benchmark/scripts/phase3_openai_run.py --cache-only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Mapping, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
from pydantic import BaseModel, Field, ValidationError

from benchmark.scripts.phase3_common import (
    DEFAULT_MODEL,
    OUT_PILOT,
    OUT_PILOT_KEY,
    OUT_PROMPTS,
    OUT_SMOKE_DIR,
    OUTPUT_SCHEMA_VERSION,
    PILOT_SEED,
    PILOT_SPLIT,
    PRICING_OPENAI_TERRA,
    PROMPT_TEMPLATE_VERSION,
    SMOKE_N_REACTIONS,
    SMOKE_N_REQUESTS,
    SMOKE_SELECTION_RULE,
    STRATA,
    TOKENIZER_CONSERVATIVE,
    TOKENIZER_SCAFFOLD,
    assert_no_kegg_leakage,
    atomic_write_json,
    atomic_write_jsonl,
    estimate_tokens_conservative,
    find_kegg_leakage,
    load_kegg_catalog_ids,
    parse_kegg_ids,
    redact_kegg_in_obj,
    redact_kegg_reaction_ids,
    require_live_tokenizer,
)
from benchmark.scripts.phase3_cost import load_pricing
from benchmark.scripts.phase3_eval import score_results
from benchmark.scripts.phase3_modes import (
    CACHE_DIR,
    FileCache,
    ModeResult,
    Prediction,
    parse_structured_output,
    prompt_for_mode,
)
from benchmark.scripts.phase3_prompts import CONTEXT_VARIANTS, STRUCTURED_OUTPUT_SCHEMA

logger = logging.getLogger("phase3_openai_run")

REQUIRED_OPENAI_SDK = "1.78.1"
PROVIDER = "openai"
API_NAME = "responses"
MODE = "direct_open_set"
DEFAULT_MAX_OUTPUT_TOKENS = 1024
DEFAULT_MAX_RETRIES = 2
DEFAULT_MAX_COST_USD = 1.00
DEFAULT_REASONING_EFFORT = "low"
LIVE_REQUEST_CEILING = SMOKE_N_REQUESTS
SECRET_ENV = "OPENAI_API_KEY"
ENV_FILENAME = ".env"

FORBIDDEN_PAYLOAD_KEYS = frozenset({
    "ground_truth_kegg_all",
    "ground_truth_kegg_primary",
    "ground_truth_ids",
    "num_ground_truth_ids",
    "answer_key",
    "candidates",
    "candidate_ids",
    "candidate_set",
    "ranked_kegg_ids",
    "pilot_answer_key",
})

TRANSIENT_STATUS_CODES = frozenset({408, 409, 429, 500, 502, 503, 504})


class Phase3PredictionItem(BaseModel):
    """Pydantic mirror of the existing Phase 3 prediction object."""

    kegg_id: str = Field(pattern=r"^R[0-9]{5}$")
    confidence: float = Field(ge=0, le=1)


class Phase3StructuredOutput(BaseModel):
    """Pydantic mirror of STRUCTURED_OUTPUT_SCHEMA for Responses.parse."""

    abstain: bool
    predictions: List[Phase3PredictionItem] = Field(default_factory=list, max_length=3)
    rationale: str
    basis: Literal["recalled_knowledge", "supplied_evidence", "mixed"]


class EnvProtectionError(RuntimeError):
    """Raised when .env is tracked or not gitignored before a live call."""


class MissingAPIKeyError(RuntimeError):
    """Raised only for live execution when OPENAI_API_KEY is absent."""


class BudgetExceeded(RuntimeError):
    """Raised when a run-level cost cap would be exceeded."""


class LeakageBlocked(RuntimeError):
    """Raised when a model-visible field still contains a KEGG reaction id."""


class ClosedListBlocked(RuntimeError):
    """Raised when a request would include a candidate list or answer-key field."""


@dataclass
class InferenceSettings:
    model: str = DEFAULT_MODEL
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS
    reasoning_effort: str = DEFAULT_REASONING_EFFORT
    output_schema_version: str = OUTPUT_SCHEMA_VERSION
    template_version: str = PROMPT_TEMPLATE_VERSION
    tokenizer: str = TOKENIZER_CONSERVATIVE
    max_retries: int = DEFAULT_MAX_RETRIES
    provider: str = PROVIDER
    api: str = API_NAME
    mode: str = MODE


@dataclass
class PlannedRequest:
    sample_id: str
    model_id: str
    reaction_id: str
    cluster_id: str
    stratum: str
    variant: str
    template_version: str
    system: str
    user: str
    prompt_hash: str
    cache_key: str
    n_input_tokens_est: int
    token_estimate_method: str
    settings: InferenceSettings

    def cache_identity(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "model_id": self.model_id,
            "reaction_id": self.reaction_id,
            "variant": self.variant,
            "template_version": self.template_version,
            "prompt_hash": self.prompt_hash,
            "model": self.settings.model,
            "max_output_tokens": self.settings.max_output_tokens,
            "reasoning_effort": self.settings.reasoning_effort,
            "output_schema_version": self.settings.output_schema_version,
            "mode": self.settings.mode,
            "provider": self.settings.provider,
            "api": self.settings.api,
        }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_dotenv_if_present(repo_root: Path = REPO_ROOT) -> None:
    """Load OPENAI_API_KEY from .env without logging file contents."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    path = repo_root / ENV_FILENAME
    if path.is_file():
        load_dotenv(path)


def _git(repo_root: Path, args: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo_root), *args],
        capture_output=True,
        text=True,
        check=False,
    )


def env_file_is_tracked(repo_root: Path = REPO_ROOT) -> bool:
    proc = _git(repo_root, ["ls-files", "--error-unmatch", ENV_FILENAME])
    return proc.returncode == 0


def env_file_is_ignored(repo_root: Path = REPO_ROOT) -> bool:
    proc = _git(repo_root, ["check-ignore", "-q", ENV_FILENAME])
    return proc.returncode == 0


def assert_env_file_protected(repo_root: Path = REPO_ROOT) -> None:
    """Refuse live calls if .env is tracked or exists without being gitignored."""
    if env_file_is_tracked(repo_root):
        raise EnvProtectionError(
            ".env is tracked by Git. Refusing live API calls until it is untracked."
        )
    env_path = repo_root / ENV_FILENAME
    if env_path.exists() and not env_file_is_ignored(repo_root):
        raise EnvProtectionError(
            ".env exists but is not gitignored. Refusing live API calls."
        )


def require_api_key_for_live() -> None:
    if not os.environ.get(SECRET_ENV):
        raise MissingAPIKeyError(
            "OPENAI_API_KEY is not set. Live mode requires it; dry-run does not."
        )


def _usd(n_tokens: int, per_million: float) -> float:
    return (n_tokens / 1_000_000.0) * per_million


def model_rates(pricing: Mapping[str, Any], model: str) -> Dict[str, float]:
    models = pricing.get("models") or {}
    if model not in models:
        raise ValueError(
            f"pricing file has no rates for {model}; refusing to guess. "
            f"Record a snapshot for this model before a live run."
        )
    rates = models[model]
    for key in ("input_per_million", "output_per_million"):
        if key not in rates:
            raise ValueError(f"pricing for {model} is missing {key}")
    return {
        "input_per_million": float(rates["input_per_million"]),
        "output_per_million": float(rates["output_per_million"]),
        "cached_input_per_million": float(rates.get("cached_input_per_million") or 0.0),
    }


def estimate_call_cost(
    *,
    n_input: int,
    n_output: int,
    n_cached_input: int = 0,
    rates: Mapping[str, float],
) -> float:
    billable_input = max(0, n_input - n_cached_input)
    return (
        _usd(billable_input, rates["input_per_million"])
        + _usd(n_cached_input, rates["cached_input_per_million"])
        + _usd(n_output, rates["output_per_million"])
    )


def usage_from_response(response: Any) -> Dict[str, Optional[int]]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {
            "input_tokens": None,
            "cached_input_tokens": None,
            "output_tokens": None,
            "reasoning_tokens": None,
            "usage_missing": True,
        }
    details_in = getattr(usage, "input_tokens_details", None)
    details_out = getattr(usage, "output_tokens_details", None)
    cached = getattr(details_in, "cached_tokens", None) if details_in is not None else None
    reasoning = getattr(details_out, "reasoning_tokens", None) if details_out is not None else None
    return {
        "input_tokens": getattr(usage, "input_tokens", None),
        "cached_input_tokens": cached,
        "output_tokens": getattr(usage, "output_tokens", None),
        "reasoning_tokens": reasoning,
        "usage_missing": False,
    }


def cost_from_usage(
    usage: Mapping[str, Any],
    *,
    rates: Mapping[str, float],
    fallback_usd: float,
) -> Dict[str, Any]:
    if usage.get("usage_missing"):
        return {
            "usd": fallback_usd,
            "used_fallback": True,
            "reason": "missing_usage_charged_at_preflight_maximum",
        }
    n_in = int(usage.get("input_tokens") or 0)
    n_cached = int(usage.get("cached_input_tokens") or 0)
    n_out = int(usage.get("output_tokens") or 0)
    return {
        "usd": estimate_call_cost(
            n_input=n_in, n_output=n_out, n_cached_input=n_cached, rates=rates,
        ),
        "used_fallback": False,
        "reason": "api_usage",
    }


def is_retryable(exc: BaseException) -> bool:
    """Retry only demonstrably transient failures."""
    try:
        import openai
    except ImportError:
        openai = None  # type: ignore

    if openai is not None:
        if isinstance(exc, (
            openai.AuthenticationError,
            openai.PermissionDeniedError,
            openai.BadRequestError,
            openai.UnprocessableEntityError,
            openai.NotFoundError,
            openai.LengthFinishReasonError,
            openai.ContentFilterFinishReasonError,
        )):
            return False
        if isinstance(exc, (
            openai.RateLimitError,
            openai.APIConnectionError,
            openai.APITimeoutError,
            openai.InternalServerError,
        )):
            return True
    status = getattr(exc, "status_code", None)
    if status in TRANSIENT_STATUS_CODES:
        return True
    return False


def backoff_seconds(attempt: int, *, initial: float = 1.0, cap: float = 8.0) -> float:
    return min(cap, initial * (2 ** attempt))


def load_prompt_rows(path: Path = OUT_PROMPTS) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def select_smoke_reactions(
    sample: pd.DataFrame,
    *,
    seed: int = PILOT_SEED,
    n: int = SMOKE_N_REACTIONS,
) -> pd.DataFrame:
    """Pick ``n`` validation reactions, one per stratum in STRATA order.

    Rule ``seeded_round_robin_one_per_stratum_v1``: walk Phase 3 strata in the
    frozen order. Within each stratum, sort by (cluster_id, model_id,
    reaction_id, sample_id), shuffle with a single RNG seeded at ``seed``, and
    take the first row. Test-split rows are refused.
    """
    if sample.empty:
        raise ValueError("pilot sample is empty")
    if "split" in sample.columns:
        splits = set(sample["split"].astype(str))
        if splits != {PILOT_SPLIT}:
            raise ValueError(
                f"smoke-test sample must be validation-only; found splits {sorted(splits)}"
            )
    rng = random.Random(seed)
    picked: List[pd.Series] = []
    for stratum in STRATA:
        sub = sample[sample["stratum"].astype(str) == stratum].copy()
        if sub.empty:
            continue
        sub = sub.sort_values(
            ["cluster_id", "model_id", "reaction_id", "sample_id"]
        ).reset_index(drop=True)
        order = list(range(len(sub)))
        rng.shuffle(order)
        picked.append(sub.iloc[order[0]])
        if len(picked) >= n:
            break
    if len(picked) != n:
        raise ValueError(f"could not select {n} smoke-test reactions; got {len(picked)}")
    return pd.DataFrame(picked).reset_index(drop=True)


def attach_variants(
    selected: pd.DataFrame,
    prompt_rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    wanted = {
        (str(rec.sample_id), str(rec.model_id), str(rec.reaction_id))
        for rec in selected.itertuples(index=False)
    }
    by_key: Dict[tuple, Dict[str, Dict[str, Any]]] = {}
    for row in prompt_rows:
        key = (str(row["sample_id"]), str(row["model_id"]), str(row["reaction_id"]))
        if key not in wanted:
            continue
        by_key.setdefault(key, {})[str(row["variant"])] = dict(row)
    planned: List[Dict[str, Any]] = []
    for rec in selected.itertuples(index=False):
        key = (str(rec.sample_id), str(rec.model_id), str(rec.reaction_id))
        variants = by_key.get(key, {})
        missing = [v for v in CONTEXT_VARIANTS if v not in variants]
        if missing:
            raise ValueError(f"missing variants for {key}: {missing}")
        for variant in CONTEXT_VARIANTS:
            planned.append(variants[variant])
    expected = len(selected) * len(CONTEXT_VARIANTS)
    if len(planned) != expected:
        raise ValueError(f"expected {expected} planned prompts, got {len(planned)}")
    return planned


def _walk_forbidden_keys(obj: Any, *, where: str) -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in FORBIDDEN_PAYLOAD_KEYS:
                raise ClosedListBlocked(
                    f"forbidden field {key!r} in model-visible request ({where})"
                )
            _walk_forbidden_keys(value, where=where)
    elif isinstance(obj, list):
        for item in obj:
            _walk_forbidden_keys(item, where=where)


def assert_no_closed_list_or_answer_key(payload: Any, *, where: str) -> None:
    _walk_forbidden_keys(payload, where=where)
    blob = json.dumps(payload, default=str).lower()
    for needle in ("ground_truth_kegg", "pilot_answer_key", '"candidates":'):
        if needle in blob:
            raise ClosedListBlocked(
                f"answer-key or candidate-list content in model-visible request ({where})"
            )


def prompt_hash(system: str, user: str) -> str:
    blob = json.dumps({"system": system, "user": user}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def live_cache_key(identity: Mapping[str, Any]) -> str:
    blob = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def finalize_request(
    prompt_row: Mapping[str, Any],
    settings: InferenceSettings,
) -> PlannedRequest:
    """Redact, scan, and hash a stored prompt into an API-ready request.

    Does not read the answer key. Join keys on the jsonl row stay off the wire.
    """
    stored = prompt_row.get("prompt") or prompt_row
    prepared = prompt_for_mode(dict(stored), MODE)
    messages = list(prepared.get("messages") or [])
    system = ""
    user = ""
    for msg in messages:
        if msg.get("role") == "system":
            system = str(msg.get("content") or "")
        elif msg.get("role") == "user":
            user = str(msg.get("content") or "")
    system = redact_kegg_reaction_ids(system)
    user = redact_kegg_reaction_ids(user)
    visible = redact_kegg_in_obj({"system": system, "user": user})
    where = f"{prompt_row.get('sample_id')}/{prompt_row.get('variant')}"
    leaked = find_kegg_leakage(visible)
    if leaked:
        raise LeakageBlocked(f"KEGG reaction-id leakage in {where}: {leaked[:10]}")
    assert_no_kegg_leakage(visible, where=where)
    assert_no_closed_list_or_answer_key(visible, where=where)
    digest = prompt_hash(visible["system"], visible["user"])
    n_in = estimate_tokens_conservative(visible["system"]) + estimate_tokens_conservative(
        visible["user"]
    )
    planned = PlannedRequest(
        sample_id=str(prompt_row["sample_id"]),
        model_id=str(prompt_row["model_id"]),
        reaction_id=str(prompt_row["reaction_id"]),
        cluster_id=str(prompt_row.get("cluster_id") or ""),
        stratum=str(prompt_row.get("stratum") or ""),
        variant=str(prompt_row["variant"]),
        template_version=str(
            prompt_row.get("template_version") or stored.get("template_version")
            or settings.template_version
        ),
        system=visible["system"],
        user=visible["user"],
        prompt_hash=digest,
        cache_key="",
        n_input_tokens_est=n_in,
        token_estimate_method=TOKENIZER_CONSERVATIVE,
        settings=settings,
    )
    if planned.template_version != settings.template_version:
        raise ValueError(
            f"prompt template {planned.template_version} != runner {settings.template_version}"
        )
    planned.cache_key = live_cache_key(planned.cache_identity())
    return planned


def preflight_cost(
    requests: Sequence[PlannedRequest],
    pricing: Mapping[str, Any],
    *,
    max_retries: int,
) -> Dict[str, Any]:
    if not requests:
        raise ValueError("no requests to estimate")
    model = requests[0].settings.model
    rates = model_rates(pricing, model)
    n_input = sum(r.n_input_tokens_est for r in requests)
    max_out = requests[0].settings.max_output_tokens
    n_calls = len(requests)
    attempts = n_calls * (1 + max_retries)
    n_output_max = attempts * max_out
    expected_no_retry = estimate_call_cost(
        n_input=n_input, n_output=n_calls * max_out, rates=rates,
    )
    worst = estimate_call_cost(
        n_input=n_input * (1 + max_retries), n_output=n_output_max, rates=rates,
    )
    per_call_max = estimate_call_cost(
        n_input=max(r.n_input_tokens_est for r in requests),
        n_output=max_out,
        rates=rates,
    )
    return {
        "model": model,
        "pricing_date": pricing.get("pricing_date"),
        "pricing_source": pricing.get("source"),
        "rates": rates,
        "n_calls": n_calls,
        "max_attempts_including_retries": attempts,
        "n_input_tokens_est": n_input,
        "planned_max_output_tokens_per_call": max_out,
        "planned_max_output_tokens_total_no_retry": n_calls * max_out,
        "planned_max_output_tokens_total_with_retries": n_output_max,
        "token_estimate_method": TOKENIZER_CONSERVATIVE,
        "expected_usd_no_retry_at_max_output": round(expected_no_retry, 6),
        "worst_case_usd": round(worst, 6),
        "per_call_max_usd": round(per_call_max, 6),
    }


def extract_refusal(response: Any) -> Optional[str]:
    output = getattr(response, "output", None) or []
    for item in output:
        for part in getattr(item, "content", None) or []:
            if getattr(part, "type", None) == "refusal":
                return str(getattr(part, "refusal", "") or "refused")
    return None


def serialize_prediction(pred: Prediction) -> Dict[str, Any]:
    return asdict(pred)


def interpret_parsed_payload(
    payload: Mapping[str, Any],
    *,
    catalog,
    raw_text: str,
) -> Dict[str, Any]:
    parsed = parse_structured_output(json.dumps(payload), catalog=catalog)
    status = "succeeded"
    if parsed.get("parse_error"):
        status = "compliance_error"
    return {
        "terminal_status": status,
        "abstain": parsed["abstain"],
        "predictions": parsed["predictions"],
        "rationale": parsed["rationale"],
        "basis": parsed["basis"],
        "parse_error": parsed["parse_error"],
        "raw_text": raw_text,
        "refusal": None,
    }


def interpret_response(response: Any, *, catalog) -> Dict[str, Any]:
    refusal = extract_refusal(response)
    if refusal:
        return {
            "terminal_status": "refused",
            "abstain": False,
            "predictions": [],
            "rationale": "",
            "basis": "recalled_knowledge",
            "parse_error": None,
            "raw_text": "",
            "refusal": refusal,
        }
    parsed_obj = getattr(response, "output_parsed", None)
    if parsed_obj is not None:
        if hasattr(parsed_obj, "model_dump"):
            payload = parsed_obj.model_dump()
            raw = parsed_obj.model_dump_json()
        elif isinstance(parsed_obj, dict):
            payload = parsed_obj
            raw = json.dumps(payload)
        else:
            payload = dict(parsed_obj)
            raw = json.dumps(payload)
        return interpret_parsed_payload(payload, catalog=catalog, raw_text=raw)
    return {
        "terminal_status": "schema_invalid",
        "abstain": False,
        "predictions": [],
        "rationale": "",
        "basis": "recalled_knowledge",
        "parse_error": "unparseable",
        "raw_text": "",
        "refusal": None,
    }


def result_row(
    planned: PlannedRequest,
    *,
    interpreted: Mapping[str, Any],
    usage: Mapping[str, Any],
    cost: Mapping[str, Any],
    attempt_count: int,
    cache_hit: bool,
    model_requested: str,
    model_returned: Optional[str],
    response_id: Optional[str],
    api_error: Optional[str],
    pricing: Mapping[str, Any],
) -> Dict[str, Any]:
    preds = interpreted.get("predictions") or []
    pred_dicts = [
        serialize_prediction(p) if isinstance(p, Prediction) else p for p in preds
    ]
    return {
        "cache_id": planned.cache_key,
        "sample_id": planned.sample_id,
        "model_id": planned.model_id,
        "reaction_id": planned.reaction_id,
        "cluster_id": planned.cluster_id,
        "stratum": planned.stratum,
        "variant": planned.variant,
        "mode": MODE,
        "template_version": planned.template_version,
        "prompt_hash": planned.prompt_hash,
        "output_schema_version": planned.settings.output_schema_version,
        "model_requested": model_requested,
        "model_returned": model_returned,
        "timestamp": utc_now(),
        "attempt_count": attempt_count,
        "terminal_status": interpreted.get("terminal_status") or "api_error",
        "cache_hit": cache_hit,
        "n_input_tokens": usage.get("input_tokens"),
        "n_cached_input_tokens": usage.get("cached_input_tokens"),
        "n_output_tokens": usage.get("output_tokens"),
        "n_reasoning_tokens": usage.get("reasoning_tokens"),
        "usage_missing": bool(usage.get("usage_missing")),
        "cost_usd": round(float(cost.get("usd") or 0.0), 8),
        "cost_used_fallback": bool(cost.get("used_fallback")),
        "cost_reason": cost.get("reason"),
        "pricing_source": pricing.get("source"),
        "pricing_date": pricing.get("pricing_date"),
        "abstain": interpreted.get("abstain"),
        "predictions": pred_dicts,
        "rationale": interpreted.get("rationale") or "",
        "basis": interpreted.get("basis") or "recalled_knowledge",
        "parse_error": interpreted.get("parse_error"),
        "refusal": interpreted.get("refusal"),
        "api_error": api_error,
        "response_id": response_id,
        "raw_text": interpreted.get("raw_text") or "",
        "max_output_tokens": planned.settings.max_output_tokens,
        "reasoning_effort": planned.settings.reasoning_effort,
        "tokenizer": planned.token_estimate_method,
        "n_input_tokens_est": planned.n_input_tokens_est,
    }


def row_to_mode_result(row: Mapping[str, Any]) -> ModeResult:
    preds = []
    for item in row.get("predictions") or []:
        if isinstance(item, Prediction):
            preds.append(item)
        else:
            preds.append(Prediction(
                kegg_id=str(item.get("kegg_id") or ""),
                confidence=item.get("confidence"),
                valid_kegg_id=bool(item.get("valid_kegg_id")),
                well_formed=bool(item.get("well_formed")),
                in_catalog=bool(item.get("in_catalog")),
                id_class=str(item.get("id_class") or ""),
                duplicate=bool(item.get("duplicate")),
            ))
    return ModeResult(
        sample_id=str(row["sample_id"]),
        model_id=str(row["model_id"]),
        reaction_id=str(row["reaction_id"]),
        cluster_id=str(row.get("cluster_id") or ""),
        stratum=str(row.get("stratum") or ""),
        mode=str(row.get("mode") or MODE),
        variant=str(row["variant"]),
        template_version=str(row.get("template_version") or PROMPT_TEMPLATE_VERSION),
        abstain=bool(row.get("abstain")),
        predictions=preds,
        rationale=str(row.get("rationale") or ""),
        basis=str(row.get("basis") or "recalled_knowledge"),
        raw_text=str(row.get("raw_text") or ""),
        parse_error=row.get("parse_error"),
        n_input_tokens=int(row.get("n_input_tokens") or 0),
        n_output_tokens=int(row.get("n_output_tokens") or 0),
        provider=PROVIDER,
        model_name=str(row.get("model_returned") or row.get("model_requested") or ""),
        cache_hit=bool(row.get("cache_hit")),
        cached=True,
    )


def build_api_kwargs(planned: PlannedRequest) -> Dict[str, Any]:
    return {
        "model": planned.settings.model,
        "instructions": planned.system,
        "input": planned.user,
        "text_format": Phase3StructuredOutput,
        "max_output_tokens": planned.settings.max_output_tokens,
        "reasoning": {"effort": planned.settings.reasoning_effort},
        "store": False,
        "tools": [],
    }


def call_with_retries(
    parse_fn: Callable[[Dict[str, Any]], Any],
    kwargs: Dict[str, Any],
    *,
    max_retries: int,
    sleep: Callable[[float], None],
) -> tuple[Any, int]:
    attempt = 0
    last_exc: Optional[BaseException] = None
    while attempt <= max_retries:
        try:
            return parse_fn(kwargs), attempt + 1
        except Exception as exc:  # noqa: BLE001 — classified immediately
            last_exc = exc
            if not is_retryable(exc) or attempt >= max_retries:
                raise
            sleep(backoff_seconds(attempt))
            attempt += 1
    raise last_exc or RuntimeError("retry loop exited without a response")


def make_openai_parse_fn() -> Callable[[Dict[str, Any]], Any]:
    try:
        import openai
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            f"openai SDK is required (pinned {REQUIRED_OPENAI_SDK})"
        ) from exc
    if getattr(openai, "__version__", "") != REQUIRED_OPENAI_SDK:
        logger.warning(
            "openai SDK version is %s; this runner was pinned at %s",
            openai.__version__, REQUIRED_OPENAI_SDK,
        )
    require_api_key_for_live()
    client = OpenAI(api_key=os.environ.get(SECRET_ENV))

    def _parse(kwargs: Dict[str, Any]) -> Any:
        return client.responses.parse(**kwargs)

    return _parse


def evaluate_persisted_rows(
    rows: Sequence[Mapping[str, Any]],
    answer_key_path: Path,
) -> Dict[str, Any]:
    """Join ground truth only after responses have been persisted."""
    key = pd.read_csv(answer_key_path)
    mapping = {
        (str(rec.model_id), str(rec.reaction_id)): parse_kegg_ids(rec.ground_truth_kegg_all)
        for rec in key.itertuples(index=False)
    }
    results = [row_to_mode_result(row) for row in rows]
    return score_results(results, mapping, seen_definition="not_applicable_smoke")


def _status_counts(rows: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for row in rows:
        status = str(row.get("terminal_status") or "unknown")
        out[status] = out.get(status, 0) + 1
    return out


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    attempted = [
        r for r in rows
        if not r.get("cache_hit") and r.get("terminal_status") not in {"dry_run", "budget_stopped"}
    ]
    cached = [r for r in rows if r.get("cache_hit")]
    succeeded = [r for r in rows if r.get("terminal_status") == "succeeded"]
    failed = [r for r in rows if r.get("terminal_status") in {
        "schema_invalid", "api_error", "leakage_blocked", "budget_stopped",
    }]
    refused = [r for r in rows if r.get("terminal_status") == "refused"]
    compliance = [r for r in rows if r.get("terminal_status") == "compliance_error"]
    cost = sum(float(r.get("cost_usd") or 0.0) for r in rows if not r.get("cache_hit"))
    return {
        "n_planned": len(rows),
        "n_attempted": len(attempted),
        "n_succeeded": len(succeeded),
        "n_failed": len(failed),
        "n_refused": len(refused),
        "n_compliance_error": len(compliance),
        "n_cached": len(cached),
        "status_counts": _status_counts(rows),
        "actual_input_tokens": sum(int(r["n_input_tokens"]) for r in rows if r.get("n_input_tokens") is not None),
        "actual_cached_input_tokens": sum(
            int(r["n_cached_input_tokens"]) for r in rows if r.get("n_cached_input_tokens") is not None
        ),
        "actual_output_tokens": sum(int(r["n_output_tokens"]) for r in rows if r.get("n_output_tokens") is not None),
        "actual_reasoning_tokens": sum(
            int(r["n_reasoning_tokens"]) for r in rows if r.get("n_reasoning_tokens") is not None
        ),
        "n_missing_usage": sum(1 for r in rows if r.get("usage_missing")),
        "actual_cost_usd": round(cost, 8),
    }


@dataclass
class RunConfig:
    sample_path: Path = OUT_PILOT
    prompts_path: Path = OUT_PROMPTS
    answer_key_path: Path = OUT_PILOT_KEY
    pricing_path: Path = PRICING_OPENAI_TERRA
    out_dir: Path = OUT_SMOKE_DIR
    cache_dir: Path = CACHE_DIR
    model: str = DEFAULT_MODEL
    execute: bool = False
    cache_only: bool = False
    max_cost_usd: float = DEFAULT_MAX_COST_USD
    max_requests: int = SMOKE_N_REQUESTS
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS
    max_retries: int = DEFAULT_MAX_RETRIES
    reasoning_effort: str = DEFAULT_REASONING_EFFORT
    evaluate: bool = True
    repo_root: Path = REPO_ROOT
    parse_fn: Optional[Callable[[Dict[str, Any]], Any]] = None
    sleep: Callable[[float], None] = time.sleep
    load_env: bool = True
    n_reactions: int = SMOKE_N_REACTIONS
    seed: int = PILOT_SEED


def plan_smoke_run(config: RunConfig) -> Dict[str, Any]:
    """Select, validate, and estimate without reading the answer key or test split."""
    require_live_tokenizer(TOKENIZER_CONSERVATIVE)
    sample = pd.read_csv(config.sample_path)
    prompts = load_prompt_rows(config.prompts_path)
    selected = select_smoke_reactions(sample, seed=config.seed, n=config.n_reactions)
    prompt_subset = attach_variants(selected, prompts)
    settings = InferenceSettings(
        model=config.model,
        max_output_tokens=config.max_output_tokens,
        reasoning_effort=config.reasoning_effort,
        max_retries=config.max_retries,
    )
    planned = [finalize_request(row, settings) for row in prompt_subset]
    if len(planned) != SMOKE_N_REQUESTS:
        raise ValueError(f"smoke test must plan exactly {SMOKE_N_REQUESTS} calls, got {len(planned)}")
    if config.max_requests < len(planned):
        raise ValueError(
            f"--max-requests {config.max_requests} is below the planned {len(planned)} smoke calls"
        )
    if config.execute and config.max_requests > LIVE_REQUEST_CEILING:
        raise ValueError(
            f"this runner's live ceiling is {LIVE_REQUEST_CEILING} calls; "
            "the 489-call validation pilot is not authorized here"
        )
    pricing = load_pricing(config.pricing_path)
    estimate = preflight_cost(planned, pricing, max_retries=config.max_retries)
    if estimate["worst_case_usd"] > config.max_cost_usd + 1e-12:
        raise BudgetExceeded(
            f"preflight worst-case ${estimate['worst_case_usd']:.6f} exceeds "
            f"cap ${config.max_cost_usd:.2f}"
        )
    return {
        "profile": "smoke",
        "selection_rule": SMOKE_SELECTION_RULE,
        "seed": config.seed,
        "split": PILOT_SPLIT,
        "n_reactions": len(selected),
        "n_requests": len(planned),
        "variants": list(CONTEXT_VARIANTS),
        "selected": selected[["sample_id", "model_id", "reaction_id", "cluster_id", "stratum"]].to_dict(
            orient="records"
        ),
        "planned": planned,
        "pricing": pricing,
        "estimate": estimate,
        "max_cost_usd": config.max_cost_usd,
        "model": config.model,
        "template_version": PROMPT_TEMPLATE_VERSION,
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "tokenizer": TOKENIZER_CONSERVATIVE,
        "live_request_ceiling": LIVE_REQUEST_CEILING,
        "answer_key_read": False,
        "test_split_read": False,
    }


def _cache_payload(row: Mapping[str, Any]) -> Dict[str, Any]:
    return dict(row)


def run_planned(
    plan: Mapping[str, Any],
    config: RunConfig,
) -> Dict[str, Any]:
    planned: Sequence[PlannedRequest] = plan["planned"]
    pricing = plan["pricing"]
    estimate = plan["estimate"]
    rates = estimate["rates"]
    per_call_max = float(estimate["per_call_max_usd"])
    cache = FileCache(config.cache_dir)
    config.out_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    spent = 0.0
    api_calls = 0
    parse_fn = config.parse_fn
    interrupted = False

    def persist() -> None:
        atomic_write_jsonl(rows, config.out_dir / "results.jsonl")
        if not config.cache_only:
            atomic_write_json(summarize_rows(rows), config.out_dir / "summary.json")

    try:
        for item in planned:
            hit = cache.get(item.cache_key)
            if hit is not None:
                row = dict(hit)
                row["cache_replayed_at"] = utc_now()
                row["cache_hit"] = True
                if hit.get("timestamp"):
                    row["timestamp"] = hit["timestamp"]
                rows.append(row)
                persist()
                continue
            if config.cache_only:
                raise RuntimeError(
                    f"cache-only mode would require an API call for {item.cache_key}"
                )
            if not config.execute:
                rows.append({
                    "cache_id": item.cache_key,
                    "sample_id": item.sample_id,
                    "model_id": item.model_id,
                    "reaction_id": item.reaction_id,
                    "cluster_id": item.cluster_id,
                    "stratum": item.stratum,
                    "variant": item.variant,
                    "template_version": item.template_version,
                    "prompt_hash": item.prompt_hash,
                    "terminal_status": "dry_run",
                    "cache_hit": False,
                    "attempt_count": 0,
                    "model_requested": item.settings.model,
                    "model_returned": None,
                    "n_input_tokens_est": item.n_input_tokens_est,
                    "max_output_tokens": item.settings.max_output_tokens,
                    "cost_usd": 0.0,
                })
                persist()
                continue
            remaining = config.max_cost_usd - spent
            if per_call_max > remaining + 1e-12:
                row = result_row(
                    item,
                    interpreted={
                        "terminal_status": "budget_stopped",
                        "abstain": False,
                        "predictions": [],
                        "rationale": "",
                        "basis": "recalled_knowledge",
                        "parse_error": None,
                        "raw_text": "",
                        "refusal": None,
                    },
                    usage={"usage_missing": True},
                    cost={"usd": 0.0, "used_fallback": False, "reason": "not_attempted"},
                    attempt_count=0,
                    cache_hit=False,
                    model_requested=item.settings.model,
                    model_returned=None,
                    response_id=None,
                    api_error=f"runtime budget stop: remaining ${remaining:.6f} < per-call max ${per_call_max:.6f}",
                    pricing=pricing,
                )
                rows.append(row)
                persist()
                break
            if parse_fn is None:
                parse_fn = make_openai_parse_fn()
            kwargs = build_api_kwargs(item)
            try:
                response, attempts = call_with_retries(
                    parse_fn, kwargs,
                    max_retries=config.max_retries, sleep=config.sleep,
                )
            except Exception as exc:  # noqa: BLE001 — persist, then classify
                api_calls += 1
                name = exc.__class__.__name__
                retryable = is_retryable(exc)
                if name in {"LengthFinishReasonError", "ValidationError"}:
                    status = "schema_invalid"
                    fallback = per_call_max
                    spent += fallback
                elif name == "ContentFilterFinishReasonError":
                    status = "refused"
                    fallback = per_call_max
                    spent += fallback
                elif name in {"AuthenticationError", "PermissionDeniedError"}:
                    status = "api_error"
                    fallback = 0.0
                else:
                    status = "api_error"
                    fallback = per_call_max if retryable else 0.0
                    spent += fallback
                row = result_row(
                    item,
                    interpreted={
                        "terminal_status": status,
                        "abstain": False, "predictions": [], "rationale": "",
                        "basis": "recalled_knowledge",
                        "parse_error": "unparseable" if status == "schema_invalid" else None,
                        "raw_text": "",
                        "refusal": "content_filter" if status == "refused" else None,
                    },
                    usage={"usage_missing": True},
                    cost={
                        "usd": fallback,
                        "used_fallback": bool(fallback),
                        "reason": "api_error_fallback" if status == "api_error" else status,
                    },
                    attempt_count=1 + config.max_retries if retryable else 1,
                    cache_hit=False,
                    model_requested=item.settings.model, model_returned=None,
                    response_id=None, api_error=name, pricing=pricing,
                )
                if name not in {"AuthenticationError", "PermissionDeniedError"}:
                    cache.put(item.cache_key, _cache_payload(row))
                rows.append(row)
                persist()
                if name in {
                    "AuthenticationError", "PermissionDeniedError",
                    "BadRequestError", "UnprocessableEntityError",
                }:
                    raise
                continue
            interpreted = interpret_response(response, catalog=load_kegg_catalog_ids())
            usage = usage_from_response(response)
            cost = cost_from_usage(usage, rates=rates, fallback_usd=per_call_max)
            spent += float(cost["usd"])
            api_calls += 1
            row = result_row(
                item,
                interpreted=interpreted,
                usage=usage,
                cost=cost,
                attempt_count=attempts,
                cache_hit=False,
                model_requested=item.settings.model,
                model_returned=getattr(response, "model", None),
                response_id=getattr(response, "id", None),
                api_error=None,
                pricing=pricing,
            )
            cache.put(item.cache_key, _cache_payload(row))
            rows.append(row)
            persist()
    except KeyboardInterrupt:
        interrupted = True
        persist()
        raise
    summary = summarize_rows(rows)
    summary.update({
        "interrupted": interrupted,
        "api_calls": api_calls,
        "spent_usd": round(spent, 8),
        "max_cost_usd": config.max_cost_usd,
        "cache_only": config.cache_only,
        "execute": config.execute,
        "model_requested": config.model,
        "out_dir": str(config.out_dir),
        "cache_dir": str(config.cache_dir),
    })
    if config.cache_only:
        atomic_write_json(summary, config.out_dir / "cache_verify.json")
    else:
        atomic_write_json(summary, config.out_dir / "summary.json")
    eval_payload = None
    if config.evaluate and config.execute and any(
        r.get("terminal_status") in {"succeeded", "compliance_error", "refused", "schema_invalid"}
        for r in rows
    ):
        eval_payload = evaluate_persisted_rows(rows, config.answer_key_path)
        atomic_write_json(eval_payload, config.out_dir / "eval.json")
    return {"rows": rows, "summary": summary, "eval": eval_payload}


def write_plan_artifacts(plan: Mapping[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    planned: Sequence[PlannedRequest] = plan["planned"]
    public_plan = {
        "profile": plan["profile"],
        "selection_rule": plan["selection_rule"],
        "seed": plan["seed"],
        "split": plan["split"],
        "n_reactions": plan["n_reactions"],
        "n_requests": plan["n_requests"],
        "variants": plan["variants"],
        "selected": plan["selected"],
        "model": plan["model"],
        "template_version": plan["template_version"],
        "output_schema_version": plan["output_schema_version"],
        "tokenizer": plan["tokenizer"],
        "max_cost_usd": plan["max_cost_usd"],
        "estimate": plan["estimate"],
        "pricing_date": plan["pricing"].get("pricing_date"),
        "pricing_source": plan["pricing"].get("source"),
        "live_request_ceiling": plan["live_request_ceiling"],
        "answer_key_read": False,
        "test_split_read": False,
        "required_openai_sdk": REQUIRED_OPENAI_SDK,
        "out_dir": str(out_dir),
        "requests": [
            {
                "cache_id": item.cache_key,
                "sample_id": item.sample_id,
                "model_id": item.model_id,
                "reaction_id": item.reaction_id,
                "cluster_id": item.cluster_id,
                "stratum": item.stratum,
                "variant": item.variant,
                "template_version": item.template_version,
                "prompt_hash": item.prompt_hash,
                "n_input_tokens_est": item.n_input_tokens_est,
                "token_estimate_method": item.token_estimate_method,
                "max_output_tokens": item.settings.max_output_tokens,
                "model": item.settings.model,
                "reasoning_effort": item.settings.reasoning_effort,
                "output_schema_version": item.settings.output_schema_version,
            }
            for item in planned
        ],
    }
    atomic_write_json(public_plan, out_dir / "plan.json")
    atomic_write_jsonl(public_plan["requests"], out_dir / "requests.jsonl")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--execute", action="store_true",
                        help="Contact OpenAI. Default is dry-run with zero API calls.")
    parser.add_argument("--cache-only", action="store_true",
                        help="Replay from cache; fail if any request would need the API.")
    parser.add_argument("--max-cost-usd", type=float, default=DEFAULT_MAX_COST_USD)
    parser.add_argument("--max-requests", type=int, default=SMOKE_N_REQUESTS)
    parser.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument("--reasoning-effort", default=DEFAULT_REASONING_EFFORT)
    parser.add_argument("--out-dir", type=Path, default=OUT_SMOKE_DIR)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--pricing", type=Path, default=PRICING_OPENAI_TERRA)
    parser.add_argument("--sample", type=Path, default=OUT_PILOT)
    parser.add_argument("--prompts", type=Path, default=OUT_PROMPTS)
    parser.add_argument("--answer-key", type=Path, default=OUT_PILOT_KEY)
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--no-dotenv", action="store_true")
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> RunConfig:
    return RunConfig(
        sample_path=args.sample,
        prompts_path=args.prompts,
        answer_key_path=args.answer_key,
        pricing_path=args.pricing,
        out_dir=args.out_dir,
        cache_dir=args.cache_dir,
        model=args.model,
        execute=bool(args.execute),
        cache_only=bool(args.cache_only),
        max_cost_usd=float(args.max_cost_usd),
        max_requests=int(args.max_requests),
        max_output_tokens=int(args.max_output_tokens),
        max_retries=int(args.max_retries),
        reasoning_effort=str(args.reasoning_effort),
        evaluate=not bool(args.skip_eval),
        load_env=not bool(args.no_dotenv),
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = config_from_args(args)
    if config.load_env:
        load_dotenv_if_present(config.repo_root)
    live = config.execute and not config.cache_only
    if live:
        assert_env_file_protected(config.repo_root)
        require_api_key_for_live()
    plan = plan_smoke_run(config)
    write_plan_artifacts(plan, config.out_dir)
    estimate = plan["estimate"]
    logger.info(
        "plan model=%s requests=%d input_tokens_est=%d max_output_total=%d "
        "worst_case_usd=%.6f cap=%.2f out=%s",
        plan["model"], plan["n_requests"], estimate["n_input_tokens_est"],
        estimate["planned_max_output_tokens_total_with_retries"],
        estimate["worst_case_usd"], config.max_cost_usd, config.out_dir,
    )
    outcome = run_planned(plan, config)
    summary = outcome["summary"]
    logger.info(
        "done attempted=%s cached=%s succeeded=%s failed=%s refused=%s cost_usd=%s api_calls=%s",
        summary.get("n_attempted"), summary.get("n_cached"),
        summary.get("n_succeeded"), summary.get("n_failed"),
        summary.get("n_refused"), summary.get("actual_cost_usd"),
        summary.get("api_calls"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
