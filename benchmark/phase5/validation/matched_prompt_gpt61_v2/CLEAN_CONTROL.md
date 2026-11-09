# Phase 5 clean-control diagnostic and corrected paired design

The actual original ea241ea construction was run on all 163 unperturbed inputs; all reconstructed request-spec digests match independently captured original-builder digests in superseded/ea241ea/legacy_clean_request_spec_digests.json.
Complete historical provider-body matches: 0/163. All body fields, literal input strings, candidate fields/order, component ranks and scores are checked. SDK 1.78.1 is intercepted before transport; its structured-output JSON schema is included. Only the two internal version tags excluded by the historical runner itself are omitted from provider bodies.

## Differing fields

```json
{
  "schema": "phase5-real-clean-control-v1",
  "n": 163,
  "complete_equal_count": 0,
  "differing_cases": 163,
  "leaf_difference_counts": {
    "provenance": 4890,
    "numerical": 2,
    "biological_evidence": 0,
    "provider_settings": 0,
    "serialization": 0
  },
  "cases_with_equal_candidate_fields": {
    "candidate_ids_and_order": 163,
    "evidence_id": 163,
    "definition": 163,
    "bm25_rank": 163,
    "trained_biencoder_rank": 162,
    "fused_rank": 163,
    "name": 163,
    "enzyme": 163,
    "kegg_id": 163,
    "equation": 163,
    "fused_score": 162,
    "brite": 163,
    "rclass": 163
  },
  "cases_with_non_provenance_differences": [
    "P3P0096"
  ]
}
```

P3P0096

```json
[
  {
    "path": "input.EVIDENCE[6].fused_score",
    "before_present": true,
    "after_present": true,
    "before": 0.020850202429,
    "after": 0.020909832721,
    "category": "numerical"
  },
  {
    "path": "input.EVIDENCE[6].trained_biencoder_rank",
    "before_present": true,
    "after_present": true,
    "before": 70,
    "after": 69,
    "category": "numerical"
  }
]
```

The 4,890 provenance leaf changes are new fusion path, fusion artifact digest and query digest in each of 1,630 records. No biological field, candidate ID/order or provider setting changed in the legacy clean diagnostic. P3P0096 retains its recomputed rank 69 and score 0.020909832721 at E07; these are not silently replaced with historical values.
Full leaf differences, including missing-versus-null flags, are in clean_control_differences.csv and clean_control_diagnostic.json. Literal input-string digest differences are also recorded, so semantic or tolerance comparisons never establish false complete-body equality.

## Corrected design

Prompt phase5-grounded-no-provenance-v2 includes only scientific evidence fields. Accurate operational lineage is in provenance_sidecars.jsonl and never copied from historical clean rankings into changed evidence. Both clean and degraded arms are rebuilt from the same frozen Phase 5 retrieval artifact with the same serializer, model (gpt-6.1-terra), instructions, schema and settings. Only actual perturbation-derived target/evidence fields differ within this model's paired arms. No retrieval experiments were rerun.
Historical v1 clean responses are not a matched baseline for v2. Both arms need new outputs. matched_pair_differences.jsonl records all actual paired input changes. Complete provider JSON bodies, including exact input strings and strict schema, determine payload hashes; experimental-unit identity also remains part of the cache policy. Identical operational sidecar changes do not invalidate an unchanged body; actual body changes always do.
corrected_clean_control_diagnostic.json separately compares all 163 actual v2 clean-arm bodies with historical requests and expands every changed field. This is distinct from the legacy diagnostic: v2 removes historical visible provenance instead of substituting new metadata. No complete historical bodies match.
The 326 case/arm calls contain 317 distinct provider bodies. We retain the existing experimental-unit cache policy rather than silently changing cross-reaction reuse; duplicate prompts do not establish independent biological observations. Identical requests still do not guarantee identical hosted-model behavior: run matched pairs interleaved, record returned model/SDK/endpoint/time and stop on declared identity drift. Even an unchanged returned alias cannot rule out hidden backend changes. Final inference remains development evidence on a reused pilot.
