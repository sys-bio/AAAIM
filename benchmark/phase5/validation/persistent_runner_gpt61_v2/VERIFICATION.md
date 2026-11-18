# GPT-6.1 Terra request-bound budget verification

**Dry-run only. USD 0 authorized.** No paid calls or held-out rows were accessed. The plan locks 163 pairs / 326 attempts, 317 distinct provider bodies, and digest `2b80dd9e83108cdba2310121706af387bd770651a1f7648a87645a03ae0b0e8f`.

## Reservation bound

For every frozen unit, `units.json` records canonical full-provider-body UTF-8 bytes, an input bound equal to those bytes plus 4,096 provider-framing tokens, and the resulting reservation. This bound counts instructions, all evidence and the strict output schema through the full JSON body. It is intentionally much more conservative than the token estimate used only for planning. Returned provider usage is checked against each unit's bound.

The maximum body is 16,840 bytes and the maximum input bound is 20,936 tokens; the sum of all 326 bounds is 4,527,330 tokens. Every per-request bound is below the 272,000-token long-context pricing threshold. The reservation prices all input at the default 30-minute cache-write rate, assumes no cache-read discount, includes each request's maximum 2,048 output tokens (reasoning included), and includes Fast's 2x multiplier. Long-context input/output uplifts are applied only if that request's own bound crosses the threshold.

The largest single reservation is USD 0.072820. The sum of all 326 pessimistic reservations is **USD 17.994805**, leaving **USD 22.005195** below the proposed USD 40 cap. Thus the cap covers all attempts even if every attempt keeps its entire reservation, under the frozen request bound and published pricing assumptions.

Only a successful response that passes expected model/tier, schema, evidence compliance and usage-bound checks releases unused reserve. Settlement uses validated provider-reported usage at a conservative write-priced ceiling; it does not query or reconcile the account invoice. Failed, missing-usage, noncompliant or uncertain attempts retain the full per-unit reserve. If returned metered usage indicates a higher cost than the local bound, the runner books the greater cost and halts before another dispatch. Account-specific charges and unverified pricing changes are outside this cap and must be checked before authorization.

## Verification

The persistent runner's dry-run reported 326 units, 317 bodies, zero provider constructions, zero paid attempts, USD 0 authorized, and zero held-out rows read. Synthetic runner tests exercise all-326 cap coverage, release of unused reservation after validated successful usage, and retention of the full reservation after an uncertain dispatch. No production response cache or authorization record exists.

Before any live execution, recheck pricing, project-level model access and billing assumptions, then obtain separate authorization for this exact plan digest. Earlier GPT-5.6 and GPT-6.1 v1 authorization does not transfer. GPT-5.6 responses are not reused. Identical request bodies and aliases do not rule out hosted-model temporal drift.
