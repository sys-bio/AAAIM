"""Conservative per-request billing bounds for frozen Phase 5 provider bodies."""
from __future__ import annotations

import json
from decimal import Decimal, ROUND_CEILING


# Provider-side Responses framing is not part of the locally serialized body.
# Reserve an additional 4,096 input tokens per unit, then validate returned
# usage against that bound. This is an explicit conservative allowance, not a
# claim that the provider bills exactly this many input tokens.
PROVIDER_WRAPPER_ALLOWANCE_TOKENS = 4096


def canonical_body_bytes(body: dict) -> int:
    """UTF-8 bytes in the entire deterministic JSON body, including schema."""
    return len(json.dumps(body, sort_keys=True, ensure_ascii=False,
                          separators=(",", ":"), allow_nan=False).encode("utf-8"))


def request_input_token_bound(body: dict, pricing: dict) -> tuple[int, int]:
    """Return (body bytes, conservative token bound) for this complete request."""
    body_bytes = canonical_body_bytes(body)
    bound = body_bytes + PROVIDER_WRAPPER_ALLOWANCE_TOKENS
    if bound > pricing["max_input_tokens"]:
        raise ValueError("complete request and framing allowance exceed documented model input maximum")
    return body_bytes, bound


def request_reservation_nusd(body: dict, pricing: dict, input_token_bound: int) -> int:
    """Reserve write-priced complete input plus max output at applicable maxima.

    All input is charged at the default 30-minute cache-write rate (no cache
    read discount). The request specifies default service, but reserve includes
    Fast's 2x multiplier as additional protection. Long-context multipliers
    apply only when this request's calculated bound exceeds the published
    threshold. Output cap already includes reasoning tokens.
    """
    long_context = input_token_bound > pricing["long_context_threshold"]
    write_rate = pricing.get("implicit_30m_write_per_million", pricing.get("implicit_30m_cache_write_per_million"))
    output_standard_rate = pricing.get("standard_output_per_million", pricing.get("output_per_million"))
    if write_rate is None or output_standard_rate is None:
        raise ValueError("billing rates missing for request reservation")
    input_rate = (Decimal(str(write_rate))
                  * (Decimal(str(pricing["long_context_input_multiplier"])) if long_context else 1))
    output_rate = (Decimal(str(output_standard_rate))
                   * (Decimal(str(pricing["long_context_output_multiplier"])) if long_context else 1))
    fast = Decimal(str(pricing["fast_multiplier"]))
    maximum_output = int(body["max_output_tokens"])
    usd = (Decimal(input_token_bound) * input_rate + Decimal(maximum_output) * output_rate) * fast / Decimal(1_000_000)
    return int((usd * Decimal(1_000_000_000)).to_integral_value(rounding=ROUND_CEILING))
