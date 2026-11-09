# Public grounded API integration: verification record

This local implementation starts from freshly fetched upstream main
`866be205d45eb7c3f7eb1ab3be05e97af1dfb311` on
`feature/rrf-grounded-public-api`, in the isolated
`benchmark/integration_worktrees/rrf_grounded_api` worktree. Existing phase
branches, the Luna feature branch, PRs and the original checkout are preserved.
No applicable AGENTS.md was found in the workspace/ancestor instruction search.

## Species contract and implementation boundary

The [contract](reaction_species_contract.md) was written before implementation
after reading upstream identity-support, qualifier, tied-accession, public
workflow, result and writer code. It retains unresolved/legacy provenance,
deduplicates ties, prevents `hasPart` from asserting whole-species identity,
permits direct KEGG compounds, and keeps names when IDs are unusable.

Runtime files under `core/reaction/` import no benchmark modules. The selected
query/document templates, regex tokenizer, BM25, complete-catalog RRF and
structured grounded schema/prompt were adapted without changing benchmark
scripts or frozen artifacts. Existing research scripts remain reproducible.
Retrieval never reads labels, splits, judgments, sample IDs or old rankings.

## Luna compatibility

The supplied preserved SBML and recorded legacy species CSV were reconstructed
through the **new** query builder. Every one of 541 query strings matches the
prepared run. Using the verified full catalog, the new BM25 implementation
reproduces all 541 archived Top-100 lists. Once query identity was established,
the permitted frozen dense component lists were used to reconstruct evidence.
There was no dense reranking/inference of Luna's reactions.

All 5,410 evidence records match, including candidate IDs/order, evidence IDs,
both component ranks, fused ranks/scores and all six readable catalog fields.
All 541 **complete provider payloads** match the preferred 2,048-token plan,
including input text, instructions, schema, model, reasoning effort, output
limit, tools, store and service tier. No provider-received field is ignored.
The local SDK's actual HTTP serialization was also checked with MockTransport,
not just the payload helper.

Her preferred plan remains
`ce8ea533e1b435627eb2302add49c74a9e16ce33d5c8b3e53e84bf48049e7d4f`,
GPT-6.1 Terra, low reasoning, max output 2,048, default service tier, no tools,
store false. It can be used unchanged **by its existing guarded runner after
separate authorization**. The preserved alternative 512-token plan remains
`c5773ca60ea0b17cb916adae3060c06f82623899bb1e81b5cbbfdf3f3915d59b`.
The prior read-only plan verifier confirms both versions, disjoint cache keys,
and retained-reservation accounting. No new execution plan is needed because
no semantics changed. The public API uses a separate complete-request hash
namespace and does not migrate caches or purchase responses automatically.

The detailed before/after field report, checksums and diagnostic reconstructed
requests are private/ignored under
`_rrf_grounded_work/luna_compatibility_v2_final/` (the earlier v1 check is also
preserved). The reconstruction envelope has its
own diagnostic digest; it does not replace the preserved execution-plan digest.
No original plan, request, manifest, export or historical result was overwritten.

Reproduce with explicit local inputs, from this worktree:

```powershell
python -m scripts.verify_rrf_grounded `
  --model ../luna_public/tests/useCase/MODEL2507280001.xml `
  --species ../luna_public/tests/useCase/results/MODEL2507280001/selected_rrf_analysis/reused_source_artifacts/species_input.csv `
  --source ../luna_public/tests/useCase/results/MODEL2507280001/selected_rrf_analysis `
  --plan ../luna_public/tests/useCase/results/MODEL2507280001/selected_rrf_output_budget_20261107/max_output_2048 `
  --catalog data/kegg/kegg_reaction_features.lzma `
  --out _rrf_grounded_work/luna_compatibility_NEW
```

The verifier refuses changed queries instead of reusing rankings for a changed
input. Future changed semantics require fresh retrieval and a separately
versioned candidate plan; changed scores/settings/schema/text require new
complete-request cache identity. Re-encoding on another device/runtime may
change numerical near ties, and equal request payloads cannot rule out hosted
model/provider-version or temporal drift.

## Tests and read-only checks

The new tests block network/provider access and use generated SBML, synthetic
evidence, mocked responses and permitted validation artifacts. They cover all
reaction coverage, stoichiometry, identifier/qualifier/status policies,
alternatives/conflicts/CSV handoff, reaction-only/combined/dry-run paths,
missing-resource preflight, settings/usage, output files and explicit SBML
round-trip writing. Technical failures remain distinct from abstentions.
Actual SDK MockTransport serialization checks the entire wire body and
`max_retries=0`. Complete-request hashes change with model/settings, candidate
order/scores and biological evidence, but not operational sidecar provenance.

New tests also compare complete RRF rankings for all **969 frozen validation
queries**, without loading answer keys or held-out reactions. The selected
archive is installed/verified locally and the small real CPU encoder fixture
reproduces its expected rankings with normalized float32 embeddings.

The latest targeted run passes **46 tests**, with the full-catalog CPU case
deselected. The separate full-catalog public CPU dry-run also passes: actual
verified resources, all 12,312 documents, three synthetic previously unseen
reactions, Top-10 evidence for each, and zero provider attempts. The first
in-memory index build took about eight minutes on this machine; later calls
reuse that retriever. This is a functional/coverage check, not biological
evaluation. The full-catalog run's suite had 44 tests at launch; subsequent
handoff/serialization tests account for the larger latest targeted count.

```powershell
$env:AAAIM_TEST_ENCODER_DIR=(Resolve-Path _rrf_grounded_work/encoder).Path
python -m pytest tests/test_public_rrf_grounded.py -q -k 'not full_catalog_cpu'
# Explicit, slower full-catalog CPU smoke check on synthetic unseen SBML:
$env:AAAIM_TEST_FULL_CATALOG='1'
python -m pytest tests/test_public_rrf_grounded.py -q -k full_catalog_cpu
```

The targeted existing suite covers identity support, qualifiers, annotation
writing, usage, Phase 3B release/fusion, and Phase 3C grounded/validation:
**64 passed** with the preserved local checkpoint and five existing smoke
response fixtures. A fresh worktree initially lacks ignored resources: one
checkpoint test fails and eleven validation tests cannot initialize without
those five fixtures. No paid responses were generated to resolve this. Only
the five digest-named existing response fixtures were copied to this worktree's
ignored cache; original responses and attempt ledgers were untouched. The
checkpoint verifier was pointed at the preserved original `best.pt` in memory.
Use a short temporary path on Windows to avoid its filename-length limit.

Three existing `test_auto_complex_components.py` failures reproduce outside
this integration in code identical to current upstream: their monkeypatched
NCBI label loaders omit the `tax_id` keyword now supplied by upstream. The
other nine tests in that file pass. These failures were not changed here.

Read-only retrieval, fusion, grounded smoke and grounded validation manifest
verifiers all pass. The prior Phase 3 scanner checks eight manifests / 151
listed files with zero problems. The selected archive verifier also passes.
The canonical Phase 4 prior verifier passes its inventory, sampling, reviewer,
Phase 3B and Phase 3C manifest checks without reopening any personal review.
Luna's existing output-budget artifact
verifier passes for both options with zero calls. `git diff --check` passes.
Ignored resources and case reports were checked with `git check-ignore`.

## Limits

No real GPT-6.1 grounded call, new biological judgment, unseen-model accuracy
claim, held-out evaluation or GPU inference occurred. Existing selected-retrieval
results and the completed scoped human audit remain valid and unchanged.
Phase 5 remains paused/incomplete. This integration does not supply independent
review, finish the broader roadmap, or authorize Luna's pending 541-call run.
No commits, pushes, merges, tags, releases, PR changes or external messages were
made. See [usage and remaining work](rrf_grounded_api.md).
