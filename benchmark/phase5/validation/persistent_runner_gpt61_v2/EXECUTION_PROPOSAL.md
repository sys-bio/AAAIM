# GPT-6.1 Terra matched-run execution proposal

**PAUSED: dry-run only. USD 0 authorized; no paid execution has occurred.** The user withdrew the previous USD 40 cap proposal and has not authorized paid calls. The user reports approximately USD 20 total API credit, which is not a confirmed available balance. This package is preserved for later review. The prior GPT-5.6 authorization and both earlier model-specific plans remain historical and do not authorize this package.

Plan digest: `2b80dd9e83108cdba2310121706af387bd770651a1f7648a87645a03ae0b0e8f`.

The plan covers 163 validation reactions with one clean and one `remove_0.5_s20261106` unit each: 326 attempts maximum, 317 complete provider bodies. Both arms use GPT-6.1 Terra and the same instructions, strict response schema, reasoning effort, maximum output, no-tools setting and serializer. The complete matched requests are in `matched_prompt_gpt61_v2/`; operational provenance stays in sidecars. No GPT-5.6 result is reused.

## Per-request reservation calculation

The prior USD 4.640720 reservation applied the model's maximum input capacity, long-context rates and Fast pricing to every dispatch. That is not an appropriate bound for these frozen short requests.

For each unit, the runner calculates:

1. Serialize the **entire frozen provider JSON body** canonically, including instructions, evidence, settings and JSON schema; count its UTF-8 bytes.
2. Add a 4,096-token allowance for provider-side framing. This is a deliberately conservative input-token bound, not a claim about exact tokenization. The locked runner checks provider-reported input usage against it.
3. Price that bound at GPT-6.1 Terra's default 30-minute cache-write rate of $1.25 per million input tokens, with no cache-read discount, and add the request's full 2,048-token output cap at $5 per million. The output cap includes reasoning tokens.
4. Multiply both input and output charges by 2 to reserve for Fast processing. Apply long-context multipliers only if that unit's bound exceeds 272,000 tokens.

For these exact requests, the largest canonical body is 16,840 bytes, so the largest input bound is 20,936 tokens. The sum of the 326 request bounds is 4,527,330 tokens. Every bound is below the 272,000-token long-context threshold, so no long-context multiplier applies. The largest per-unit reservation is **$0.072820**. Summing each unit's reservation, including 2,048 output tokens for every attempt, gives **$17.994805**.

The earlier **$40.00 cap proposal is withdrawn** and is not authorized. Under this frozen request-bound arithmetic, the maximum reservation for the complete 326-unit plan is **$17.994805**, which is below the approximately $20 total credit reported by the user. That does not confirm the available balance or cover excluded account charges, taxes, pricing changes or modifiers; it is not a current cap or authorization. No paid calls may proceed under this proposal. A later request must recheck credit/pricing/access and present a new cap and authorization.

## When unused reservation is released

After dispatch, a reservation is reduced only when the response succeeds and passes the expected model/tier, schema, evidence-compliance and usage-bound checks. The runner then replaces the reservation with a conservative usage-based ceiling derived from provider-reported input/output tokens: all input is still priced at the 30-minute cache-write rate, and output includes reasoning. The difference is released for later dispatches.

This is **not invoice reconciliation**: the runner does not query an account invoice. It trusts validated usage fields in the completed provider response and computes an upper-cost estimate under the locked public rates. Missing usage, provider failure, identity/tier drift, schema failure, evidence-compliance failure or any uncertain attempt retains the full per-unit reservation. For a recognized requested model/service tier, if valid metered usage implies a charge greater than the local request-bound reserve, the runner records the larger amount and stops before another dispatch. A bound breach is an exceptional post-dispatch exposure and could exceed the cap; it is not treated as a reason to retry or continue. An unrecognized returned model/tier cannot be safely priced by this ledger; it pauses with the full reserve retained and requires external billing reconciliation before resuming.

The API endpoint is fixed to the global `https://api.openai.com/v1`; regional processing, account-specific modifiers, taxes and other surcharges are excluded and must be confirmed before authorization. The expected service tier is `default`; Fast is reserved as extra cost protection, but a returned non-default tier pauses the run. Reconfirm published pricing and project-level model access before authorizing.

## Estimate scenarios

| Scenario across 326 units | Estimated USD |
| --- | ---: |
| Estimated input plus historical mean output, ordinary Standard rates | 1.74 |
| Same output scenario with all input at default 30-minute cache-write rates | 2.12 |
| All estimated input at cache-write rates plus all 2,048 output tokens | 5.22 |
| Sum of per-request reservations: full body-byte bound + 4,096 framing allowance, max output, cache writes and Fast | 17.994805 |
| **Historical proposed cap; withdrawn** | **40.00** |
| **Current paid authorization** | **0.00; execution paused** |

The historical output mean is a scenario, not a forecast for GPT-6.1. These estimates exclude account-specific charges and do not verify an invoice.

## Persistent execution policy

The plan fixes the interleaved order and unit-specific keys from condition, sample, model and complete provider-body digest. Duplicate bodies remain separate stochastic calls. One attempt per unit; no retries or cross-unit response reuse. The persistent ledger reserves the unit amount before provider construction, and successful completed response caches are written atomically. Failed/uncertain units retain their reservation and require documented no-retry reconciliation before continuing. The 326-call reservation total fits the cap without depending on successful-call releases.

The package uses SDK 1.78.1 and Pydantic 2.13.5 and posts the locked body without regeneration. Hosted model behavior can still drift over time even with identical body bytes and returned aliases. Account-level availability has not been tested.

## Dry-run and later commands

Dry-run only; this creates no provider client, reads no API secret, makes no network call and reads no held-out data:

```powershell
python -m benchmark.scripts.phase5_runner --package benchmark/phase5/validation/persistent_runner_gpt61_v2
```

Paid execution remains disabled until the user separately authorizes the exact digest and billing assumptions. Later commands are listed here for reproducibility only; they are not authorization:

```powershell
python -m benchmark.scripts.phase5_runner --package benchmark/phase5/validation/persistent_runner_gpt61_v2 --execute --authorization benchmark/phase5/validation/persistent_runner_gpt61_v2/_authorization/approved.json --confirm-plan 2b80dd9e83108cdba2310121706af387bd770651a1f7648a87645a03ae0b0e8f
python -m benchmark.scripts.phase5_runner --package benchmark/phase5/validation/persistent_runner_gpt61_v2 --execute --resume --authorization benchmark/phase5/validation/persistent_runner_gpt61_v2/_authorization/approved.json --confirm-plan 2b80dd9e83108cdba2310121706af387bd770651a1f7648a87645a03ae0b0e8f
```

No held-out execution is enabled. Phase 5 remains incomplete until this exact matched run is separately authorized, executed and analyzed.
