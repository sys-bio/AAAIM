# Grounded full-catalog reaction recommendations

`annotate_model(..., reaction_method="rrf_grounded")` integrates the selected
research pipeline into the public API. The default reaction method remains
`rulebased`. `method="direct"`/`"rag"` continues to control species retrieval.
An omitted `llm_model` still means `gpt-4o-mini` for existing paths; it means
`gpt-6.1-terra` for the new grounded path. Pass the model explicitly in saved
examples and execution plans.

## Local resources

The selected encoder ZIP is available on the existing
[MODEL2507280001 reaction review release](https://github.com/sys-bio/AAAIM/releases/tag/model2507280001-reaction-review).
Download it explicitly (no API inference is involved):

```bash
gh release download model2507280001-reaction-review --repo sys-bio/AAAIM \
  --pattern aaaim-phase3b-selected-epoch1-inference.zip --dir _rrf_grounded_work
```

The frozen catalog is tracked at `data/kegg/kegg_reaction_features.lzma` in the
repository. The [restore instructions](../benchmark/dist/RESTORE_phase3b-selected-epoch1.md)
also record archive verification details. No new release/tag is required.
The runtime never downloads models or reads benchmark populations, labels,
judgments or frozen rankings. It needs only a local catalog and extracted
inference resources. From the repository root:

```python
from core.reaction.retrieval import install_resources

install_resources("_rrf_grounded_work/aaaim-phase3b-selected-epoch1-inference.zip",
                  "_rrf_grounded_work/encoder")
```

Installation refuses an existing destination. It checks the archive's SHA-256,
extracts only flat file names, and verifies the pinned manifest and every
payload hash. Model loading verifies the manifest and all eleven files again,
rejecting extra files. Catalog bytes are verified before trusted deserialization.
Catalog SHA-256 is
`00acb4de1bbfb1ddae0298a09d218f93eb22241844179439ff34a2912fba046c`;
archive SHA-256 is
`3412a3fa546347d8209ab62ca7ef55fb490e148b5f90c25cf33616328a0d0f53`.
The manifest pins model weights, architecture, tokenizer and tokenizer settings;
it identifies the selected epoch-1 weights, not the original pretrained model.

The installed environment must provide the repository dependencies, including
PyTorch and Transformers. The small CPU verification passed with PyTorch
`2.13.0+cpu`, Transformers `5.16.1`, and NumPy `2.2.2`. The archive contains
`requirements-inference.txt` and the original environment record. No dependency
installation occurs during annotation. Model loading uses `local_files_only=True`.

CPU is the default. CUDA is explicit and fails clearly when unavailable. The
full 12,312-document embedding matrix is computed locally once per cached
retriever instance; constructing it can be slow on CPU. Models/tokenizers are
cached separately, with bounded in-process caches. There is no implicit disk
embedding cache or cross-process cache. CPU float32 and CUDA float16 autocast
follow the selected encoder implementation; numerical near ties can differ
across devices, libraries and batch sizes. Actual ranks/settings are provenance.

## Public usage

```python
from core import annotate_model, RetrievalResources

resources = RetrievalResources(
    catalog_path="data/kegg/kegg_reaction_features.lzma",
    encoder_dir="_rrf_grounded_work/encoder",
    device="cpu",
)
result = annotate_model(
    "unseen_model.xml",
    annotate="reactions",
    reaction_method="rrf_grounded",
    reaction_resources=resources,
    llm_model="gpt-6.1-terra",
    reaction_reasoning_effort="low",
    max_completion_tokens=2048,
    dry_run=True,                       # retrieval and request preparation only
    save_to="_rrf_grounded_work/unseen",
)
print(result.reaction_recommendations_df[["id", "status", "annotation"]])
print(result.metrics["coverage"])
```

The [command-line example](../examples/grounded_reactions.py) always performs
a dry-run and accepts any local SBML plus an optional species CSV.

Pass `species_recommendations_df=frame` or a CSV path for reaction-only handoff.
With no table, source SBML CVTerms are used. `annotate="both"` runs the normal
species workflow first and hands the **complete** table to reaction retrieval,
including direct KEGG compounds and uncertainty fields. `predict_qualifiers=True`
enables upstream's separate identity-support/qualifier stages. Its `effort`
remains independent of `reaction_reasoning_effort`.

Combined `dry_run=True` deliberately skips species inference: it uses the
supplied table or SBML and records `species_stage.status="skipped_dry_run"`.
It previews existing evidence, not hypothetical future species predictions.
Dry-run guarantees no provider calls, including the species stage. Removing
`dry_run` is a live API operation; the present integration task did not execute
or authorize it. The ordinary public API is not Luna's persistent spending
runner: it has no cross-invocation purchase ledger, budget cap or automatic
response reuse. Use her separately frozen, guarded runner for her prepared run.

The new path covers every SBML Reaction, including boundary/exchange reactions,
reactions without compound IDs and reactions omitted by older extraction. It
does not use `max_entities` to trim reaction coverage (`max_entities` still
limits the species stage in combined mode). `top_k`, `n_return`, `message` and
EM/ontology options retain their species/baseline meanings; grounded reactions
always use Top-10 evidence and one selection/abstention, with the frozen prompt.
SBML Level 2/3 reaction elements,
including FBC models, are supported; qualitative transitions are not reaction
elements. Invalid/unparseable SBML raises an input error because reaction
coverage cannot be enumerated. A bad species table yields `input_error` records
for every enumerable reaction, without retrieving or calling a provider.

## Evidence and response contract

The [species evidence contract](reaction_species_contract.md) defines supported,
unresolved, missing, legacy, tied, conflicting and component evidence. Names
survive unusable IDs. Identifier presence does not establish correctness.
Unresolved candidates and complex components remain in the sidecar audit but
do not become whole-species compound mappings.
An eligible whole-species `is` mapping survives the same URI under `hasPart`,
another non-identity qualifier, or a component row. Only conflicting
whole-species identity assertions can veto it. Row provenance distinguishes
pre-conflict `eligible` from final `included`/`usable` and records exclusion
reasons; different qualifiers alone are not identity conflicts.

Queries use the selected target-local equation/participant template, including
stoichiometry and names, with KEGG reaction accessions redacted. Reaction names
are display-only. Catalog documents use the selected six-field template and
biochemical regex tokenizer. BM25 uses `k1=1.2`, `b=.75`; the selected encoder
uses CLS pooling, L2 normalization, max length 256 and normalized inner product.
Both components search the full catalog and contribute their Top-100 equally
to RRF with `k=60`. Ties resolve by ascending KEGG ID; omitted IDs have zero
contribution and form the ascending zero-score tail. The model sees Top-10
records with candidate IDs, readable fields, ranks and scores. Operational
hashes, paths, species status records and resource provenance stay in sidecars.

Responses use strict `GroundedAnnotation` JSON schema, followed by semantic
checking of identifiers, ranks, citations, selection/abstention consistency,
and a nonempty summary. All technical failures have `biological_abstain=None`.

| Status | Meaning |
| --- | --- |
| `planned` | Dry-run request prepared; no biological answer |
| `selected` | Valid selection from supplied evidence; correctness not established |
| `abstained` | Valid biological/evidence abstention |
| `input_error`, `resource_error`, `input_resource_error` | Input, resource or retrieval construction failure |
| `provider_error` | Provider/transport failure or unexpected response model/status |
| `incomplete` | Truncated/incomplete provider response, possibly with no visible answer |
| `empty_response`, `refusal`, `schema_invalid`, `evidence_noncompliant` | Distinct unsuccessful answer outcomes |

The default 2,048 output tokens cover both reasoning and visible structured
output. Incompleteness is checked before JSON parsing. There are no automatic
retries, repairs or fallback methods in grounded reaction inference. Combined
species inference retains upstream's normal retry policy. GPT-6.1 Terra support is based on current
[official model documentation](https://developers.openai.com/api/docs/models/gpt-6.1-terra),
[Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs)
and [reasoning/output-limit semantics](https://developers.openai.com/api/docs/guides/reasoning),
plus local SDK serialization and mocked transport checks. No actual GPT-6.1
response was generated or empirically validated here. Identical requests alone
do not guarantee identical hosted behavior across time or provider versions.

The result is the usual `AnnotationResult`, supporting tuple unpacking.
It adds `reaction_evidence` (queries, evidence, complete requests, hashes,
response/status and provenance). CSV rows contain coverage/status and selections.
`<prefix>_reaction_evidence.jsonl` and `<prefix>_reaction_metrics.json` contain the
full audit and token usage, including reasoning-token details when reported.
Combined species usage is retained separately under `metrics['species_stage']`.
Generated reports and `_rrf_grounded_work/` resources are ignored by Git.
Generic species `.revise()` is blocked for these results because it would bypass
the grounded schema and evidence checks.

## Explicit SBML writing

```python
from core import write_reaction_annotations

reviewed = result.reaction_recommendations_df.copy()
# After reviewing actual successful selections, approve the intended rows:
reviewed.loc[reviewed.id.isin(["my_reaction"]) & reviewed.status.eq("selected"),
             "update_annotation"] = "add"
write_reaction_annotations("unseen_model.xml", reviewed, "reviewed_model.xml")
```

Annotation never writes SBML automatically. This additive writer requires
`selected` status plus explicit `add`, validates reaction IDs/KEGG IDs, targets
`model.getReaction`, and uses `bqbiol:is`. It preserves existing CVTerms and
meta IDs, creates a unique meta ID if needed, rejects species/component
qualifiers and source-path overwrite, and refuses an existing output file.
If the same reaction URI already occurs under another qualifier, the explicit
writer adds the reviewed `is` relationship while retaining the other CVTerms,
including model qualifiers. Repeated writes do not duplicate the identity.
Upstream's species writer and identity/qualifier workflow remain unchanged.

## Local evidence and remaining work

See [verification and compatibility](rrf_grounded_verification.md). This is
public API integration of completed research components, not a new accuracy
experiment. Real inference on unseen models and biological review remain
pending. The completed human audit is not reopened. Phase 5 remains paused;
the held-out test remains sealed. The feature branch is proposed for review;
PR creation does not mean the functionality is merged or Luna's request is complete.
