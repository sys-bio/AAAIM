# Phase 5 persistent matched-run execution proposal

**Dry-run only; USD 0 authorized. Phase 5 remains incomplete.** No live authorization record has been created. The earlier USD 30 planning ceiling is not approved and is superseded by this proposal.

Plan digest: `bd9a4bae63905a8ceff854d99937f637f7734877444aa6d20092315f8d7513a5`.

Scope: 163 biological validation reactions, each with one clean and one remove_0.5_s20261106 unit; **326 attempts maximum**, 317 distinct complete provider bodies. No new sample, perturbation, biological judgment, retrieval experiment, human review or held-out operation is included. The v2 request JSON bodies are copied exactly from matched_prompt_v2; source and locked file hashes are in plan.json. Model-visible serialization, instructions, strict output schema and settings are identical across conditions. Accurate operational provenance remains in sidecars. Historical clean responses cannot supply this changed-prompt baseline.

## Duplicate-body and order policy, fixed before execution

One attempt per `(condition, sample_id)`, with cache identity also including the complete canonical provider-body digest. The 8 duplicate-body groups contain 9 excess units over the 317 distinct bodies; see duplicate_bodies.json for every member. Separate units still receive separate calls. These are repeated stochastic observations, not additional independent biological samples. Resume reuses only the same unit's completed result. Historical and cross-unit caches are never consulted.

order.json fixes all 326 units. Matched units are adjacent; pair order and within-pair arm order use the preserved SHA-256 seed rule (20261106). No runtime shuffling or prompt changes occur. The full exact bodies remain local under ignored _requests/. Unit digests/order and sidecar hashes are public.

## Cost and availability recheck

Official OpenAI documentation was fetched on 2026-11-06 using OpenAI Docs, without provider inference. The [model page](https://developers.openai.com/api/docs/models/gpt-5.6-terra) documents gpt-5.6-terra, Responses, structured outputs and low reasoning. Account/project availability has not been verified; no model-list client was constructed. There is no separately documented immutable dated snapshot here. The runner initially accepts only returned model `gpt-5.6-terra` and service tier `default`; another identity/tier pauses for documented reconciliation. An unchanged returned alias cannot establish unchanged backend weights.

[Pricing](https://developers.openai.com/api/docs/pricing): Standard short-context ordinary input USD 2/M, cached reads USD 0.20/M, default 30-minute cache writes USD 2.50/M, output USD 12/M. [Prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching) is implicit by default; write pricing replaces ordinary input pricing for those tokens. No write is added on top of another input charge. The frozen body does not request explicit breakpoints, extended lifetime or a service tier. The runner never adds those fields. Implicit caching therefore needs a budget allowance even though it was not explicitly requested.

| Scenario, all 326 units | USD |
| --- | ---: |
| Estimated input plus historical mean output; ordinary input rate | 3.58 |
| Same estimate, treating all input as default 30-minute cache writes | 4.34 |
| Estimated input, all cache writes, maximum 2,048 output tokens | 11.78 |
| Every attempt at documented model input maximum, Fast and long-context rates | 3,029.76 |
| **Proposed enforced run ceiling** | **40.00, unapproved** |

The first three are planning estimates, not enforced bounds. Input estimate includes the strict schema. The historical output mean is a scenario only, not a prediction for the new matched cohort. Output limits include hidden reasoning; [reasoning documentation](https://developers.openai.com/api/docs/guides/reasoning) describes this and the possibility of a charged incomplete response without visible output. No tools or extra tool charges are allowed.

The enforced **per-dispatch reservation is USD 9.293728**: documented maximum accepted input 922,000 tokens at USD 10/M (30-minute writes, Fast 2x, long-context input 2x), plus 2,048 output tokens at USD 36/M (Fast 2x, long-context output 1.5x). This deliberately uses the documented input maximum rather than an unproved local tokenizer/wrapper estimate. All actual complete bodies are much smaller, but their byte sizes are not used as a hard billable-token bound. Reasoning is already included in output tokens and is never added twice.

Before dispatch, persistent accounted charges plus this reservation must fit USD 40. A successful response with complete in-bound usage releases excess reservation to a usage-based upper cost (all input at cache-write rate, no read discount). Failed, interrupted, missing-usage, parse/compliance or identity-mismatched responses retain their full reservation. Thus USD 40 may stop the run before 326 units; it does not authorize the theoretical USD 3,029.76 extreme or promise completion. No silent cap increase, sample reduction or repeat is allowed.

Billing assumptions requiring confirmation before approval: direct global `https://api.openai.com/v1`, no account-specific surcharges/taxes in the model-call budget, default 30-minute implicit writes, Standard or at most Fast exposure, published pricing still applicable. Regional/FedRAMP endpoints and custom base URLs are excluded. Unexpected tier/model/usage pauses execution; unsupported pricing or configuration requires a revised locked proposal. These are operational guards under documented rates, not a guarantee about an unverified account's invoice.

## Persistent state and recovery

All copies of this plan use the same ignored directory, `benchmark/phase5/_runner_state/<plan_digest>/run/`, and the same OS execution lock. A copied package cannot create another purchase cohort. SDK 1.78.1 and Pydantic 2.13.5 are pinned; SDK retries are zero; live HTTP timeout is 120 seconds. The SDK low-level POST sends the frozen body without regeneration. A mock HTTP transport verifies complete JSON equality.

The runner fsyncs a reservation ledger before provider construction/dispatch, writes a complete unit cache atomically, then completes the ledger. A crash between cache and ledger completion recovers without repurchase. An uncertain attempt without a completed cache fails closed. Orphaned caches, missing state, stale plans, changed settings, incompatible caches and ledger cost mismatches stop execution. OS locking is released on process death; do not delete the lock pathname. Windows extended paths keep hashed cache files usable in this nested worktree.

Each result records annotation/schema/evidence status, usage, requested/returned identity, service tier, endpoint, SDK, UTC times, latency, response ID and cost ceiling. Raw provider outputs stay ignored. Evidence validation uses the exact unit's supplied candidates. Failed and abstained units remain in the denominator. Missing usage never becomes zero cost.

An uncertain attempt can be explicitly reconciled without retry using a JSON record with plan_digest, unit_key, operator, reason, evidence_reference and `decision: retain_reservation_no_retry`. Run `python -m benchmark.scripts.phase5_runner --package benchmark/phase5/validation/persistent_runner_v1 --reconcile-record PATH_TO_RECORD.json`. This preserves the full charge reservation, records the missing/failing unit, and permits resume of later units. V1 provides no invoice-based refund/release or second-attempt mechanism. A model/tier pause needs substantive review before reconciliation; it is not permission to silently mix cohorts.

## Commands

Run from the Phase 5 worktree. Dry-run constructs no provider client, reads no secrets, makes no network/provider call, and creates no live ledger:

```powershell
python -m benchmark.scripts.phase5_runner --package benchmark/phase5/validation/persistent_runner_v1
```

**Only after explicit user approval:** create the ignored `_authorization/approved.json` from authorization.template.json, recording actual approval, operator, timestamp and billing confirmation. Do not mark the template approved now. The authorization must match the entire plan digest, settings digest, model, endpoint, expected identity, pricing digest, 326-attempt cap and USD 40 cap. `OPENAI_API_KEY` is read only by the later live adapter; no dotenv file is loaded automatically.

Later live command, currently unapproved:

```powershell
python -m benchmark.scripts.phase5_runner --package benchmark/phase5/validation/persistent_runner_v1 --execute --authorization benchmark/phase5/validation/persistent_runner_v1/_authorization/approved.json --confirm-plan bd9a4bae63905a8ceff854d99937f637f7734877444aa6d20092315f8d7513a5
```

Later resume command, currently unapproved; uses the same authorization and state:

```powershell
python -m benchmark.scripts.phase5_runner --package benchmark/phase5/validation/persistent_runner_v1 --execute --resume --authorization benchmark/phase5/validation/persistent_runner_v1/_authorization/approved.json --confirm-plan bd9a4bae63905a8ceff854d99937f637f7734877444aa6d20092315f8d7513a5
```

After all units are terminal and uncertainties reconciled, freeze responses **before** opening references:

```powershell
python -m benchmark.scripts.phase5_matched_analysis --package benchmark/phase5/validation/persistent_runner_v1 --freeze-only --response-freeze benchmark/phase5/validation/persistent_runner_v1/_analysis/response_freeze.json
python -m benchmark.scripts.phase5_matched_analysis --package benchmark/phase5/validation/persistent_runner_v1 --response-freeze benchmark/phase5/validation/persistent_runner_v1/_analysis/response_freeze.json --source-root C:/Users/janis/Documents/software_projects/janisshin/AAAIM --out benchmark/phase5/validation/persistent_runner_v1/_analysis/matched_analysis.json
```

Analysis uses only the frozen 163-row validation reference join, with exact sample/model/reaction alignment. It reports correctness, coverage, selective accuracy, abstentions, failures, reference-based abstentions, strata and paired changes; the 2,000-draw bootstrap resamples biological model clusters with both conditions together. Undefined zero-coverage selective-accuracy draws are reported. Historical clean accuracy/coverage are descriptive comparisons only. Paired timing, provider identity, duplicate policy and temporal/stochastic limitations accompany results. Interruptions and resumes can separate matched calls in time; report actual pair gaps. The reused stratified pilot does not establish population-wide accuracy. No audit corrections enter labels.

Synthetic analysis and fault-injection tests exercise this path without generating biological findings. Phase 5 and the evaluation freeze remain incomplete/draft until an authorized matched run and analysis finish. No held-out executor, commit, push, PR, merge, tag or release is authorized by this proposal.
