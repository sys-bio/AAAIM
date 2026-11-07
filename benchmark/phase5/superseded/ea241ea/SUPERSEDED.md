# Superseded request plan and mutable-report snapshot

These byte-for-byte copies preserve the original plan, manifests, reports, README, draft freeze and builder from reviewed commit ea241ea163e3fb68f56ebd3fe801fa6f1fd66827. The original retrieval experiment is still valid; no retrieval/GPU run is repeated or replaced.

The one-arm plan was confounded because its changed model-visible provenance/serialization was not shared by the historical clean controls. Running the actual original builder on clean inputs changes all 163 historical prompts; one case also has a component-rank/score difference. `legacy_clean_request_spec_digests.json` pins independently captured original-builder results on all 163 clean cases and supports regression tests for the diagnostic.

The active replacement is the matched no-visible-provenance v2 plan under `validation/matched_prompt_v2/`, with 163 fresh clean plus 163 degraded case/arm requests. Historical clean responses are not the sole matched baseline. Full old payloads remain locally preserved under ignored `_requests/superseded_ea241ea/`; personal audit exports are neither copied nor used.

The copied artifact_checksums.json/provenance.json describe the historical ea241ea state, not the mutable current reports and revised source. The original root manifests are also preserved as historical records. Current review-fix hashes are recorded separately; do not silently regenerate historical manifests to conceal the distinction.
