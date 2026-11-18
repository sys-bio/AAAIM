# AAAIM — Auto-Annotator via AI for Modeling

AAAIM is an LLM-powered tool that annotates biosimulation models (SBML) with standardized ontology terms from ChEBI, NCBI Gene, UniProt, and KEGG. You can annotate **species**, **reactions**, or **both**.

Reaction annotation now defaults to `reaction_method="rrf_grounded"`: the
selected full-catalog BM25 + trained bi-encoder RRF retrieval feeding
evidence-constrained reaction selection/abstention. It requires explicit local
resources (see [Reaction annotation](#reaction-annotation)). The rule-based
reaction method is deprecated but still available with
`reaction_method="rulebased"`. See [the public grounded API, local resources, dry-run example and migration notes](docs/rrf_grounded_api.md)
and [local verification/compatibility evidence](docs/rrf_grounded_verification.md).

![AAAIM Workflow](docs/AAAIM%20workflow.png)

---

## Installation

**Requirements**: Python 3.12

```bash
pip install -r requirements.txt
```

Set at least one LLM API key (in your shell or a `.env` file):

```bash
OPENAI_API_KEY=<your-openai-key>          # gpt-4o-mini, gpt-5-mini
OPENROUTER_API_KEY=<your-openrouter-key>  # llama-3.3-70b (free tier available)
```

To start with a free OpenRouter model:

```python
result = annotate_model(
    model_file="path/to/model.xml",
    llm_model="openrouter/free",
)
```

For other models like OpenAI models, just pass the model name, e.g. "gpt-5-mini".

---

## Quick Start

```python
from core import annotate_model

result = annotate_model(
    model_file="path/to/model.xml",
    annotate="species",               # "species" (default) | "reactions" | "both"
    entity_type="auto",               # detects chemical / gene / protein / complex
    database=["chebi", "uniprot"],    # databases to search for species
    message="This is a model for human metabolism", # optional user message to the LLM
    top_k=3,                         # identity retrieval pool per species/component
    n_return=1,                      # final identities kept per species/component
    validation=True,                 # optional structured output and candidate checks
    source_fidelity=True,            # optional exact SBML chemical-name priority
    verbose=False,                    # True for a short progress summary
)
# result.species_recommendations_df    → species
# result.reaction_recommendations_df   → KEGG reactions (when annotate is "reactions" or "both")
```

Requesting reactions (`"reactions"` or `"both"`) uses the grounded reaction
method by default and needs `reaction_resources`; see
[Reaction annotation](#reaction-annotation).


Run the bundled example (uses a test SBML model):

```bash
python examples/simple_example.py
```

For models with existing annotations (curation/validation workflow):

```python
from core import curate_model

result = curate_model(
    model_file="path/to/model.xml",
    entity_type="chemical",
    database="chebi"
)
print(f"Accuracy: {result.metrics['accuracy']:.1%}")
```

---

## Reaction annotation

`annotate="reactions"` or `"both"` maps SBML reactions to KEGG reaction IDs.
The default `reaction_method="rrf_grounded"` needs two explicit local
resources: the frozen KEGG catalog tracked in this repository and the verified
selected encoder. Download the encoder archive yourself, then install it once
(the API never downloads anything):

```bash
gh release download model2507280001-reaction-review --repo sys-bio/AAAIM \
  --pattern aaaim-phase3b-selected-epoch1-inference.zip --dir _rrf_grounded_work
python -m core.reaction.retrieval _rrf_grounded_work/aaaim-phase3b-selected-epoch1-inference.zip _rrf_grounded_work/encoder
```

Preview with a zero-provider-call dry run:

```python
from core import annotate_model, RetrievalResources

resources = RetrievalResources(
    catalog_path="data/kegg/kegg_reaction_features.lzma",
    encoder_dir="_rrf_grounded_work/encoder",
)
result = annotate_model(
    "path/to/model.xml",
    annotate="reactions",          # reaction_method defaults to "rrf_grounded"
    reaction_resources=resources,
    llm_model="gpt-6.1-terra",
    max_completion_tokens=2048,
    dry_run=True,                  # no provider calls; remove only for an approved live run
    save_to="_rrf_grounded_work/model",
)
print(result.reaction_recommendations_df[["id", "status"]])
```

Without valid `reaction_resources`, each reaction is reported as
`resource_error` with setup instructions; AAAIM never falls back to rule-based
matching. A live run makes up to one provider request per reaction and the
public API has no spending cap, purchase ledger or response cache.

**Migrating:** calls that requested reactions without `reaction_method`
previously used rule-based matching and now use the grounded method (and, when
`llm_model` is omitted, `gpt-6.1-terra` instead of `gpt-4o-mini`). To keep the
old behavior pass `reaction_method="rulebased"`; it still works but is
deprecated and emits a `DeprecationWarning`. No removal date is set. See the
[migration notes](docs/rrf_grounded_api.md#migration-from-the-rule-based-default).

---

## User Feedback

After reviewing the initial recommendations, you can refine them with
natural-language feedback. The LLM receives your feedback together with
the full conversation history and produces revised synonyms, which are
then re-matched against the database.

```python
result = annotate_model("model.xml", entity_type="chemical", database="chebi")

# Single revision
result = result.revise("Species X should be glucose-6-phosphate, not glucose")

# Interactive loop (console prompt; press Enter to accept)
result.feedback_loop()
```

Each revision saves a versioned CSV (`*_recommendations_v1.csv`,
`*_recommendations_v2.csv`, …) so you can always revert to an earlier
version.


---

## Applying Annotation Recommendations

After reviewing the output CSV, edit the `update_annotation` column for each row:

| Value                 | Effect                         |
| --------------------- | ------------------------------ |
| `add`               | Add the recommended annotation |
| `delete`            | Remove the existing annotation |
| `ignore` / `keep` | Leave unchanged                |

Then write the updated model. `add_all=True` adds every annotation in the table and ignores the `update_annotation` column.

```python
from core.update_model import update_annotation

update_annotation(
    original_model_path="model.xml",
    recommendation_table="recommendations.csv",
    new_model_path="model_updated.xml",
    add_all=False,
)
```

---

## Optional: RAG-based Search

By default, AAAIM uses direct dictionary matching (`method="direct"`). For semantic (embedding-based) search, use `method="rag"` — but you must build the vector index first.

**One-time setup** (builds embeddings for all databases, this may take a while depending on the size of your database):

```bash
python setup_rag.py                        # all databases, human (tax_id=9606)
python setup_rag.py --databases chebi      # ChEBI only
python setup_rag.py --tax_id 10090         # mouse
```

Then pass `method="rag"` to `annotate_model()` or `curate_model()`.

---
## Databases Supported

| Database     | Annotates                         | `entity_type` / `annotate`      |
| ------------ | --------------------------------- | ------------------------------- |
| **ChEBI**    | Chemicals, metabolites            | `chemical`                      |
| **NCBI Gene**| Genes                             | `gene`                          |
| **UniProt**  | Proteins                          | `protein`                       |
| **KEGG**     | Metabolic reactions               | `annotate="reactions"` or `"both"` |

Use `entity_type="auto"` with a `database` list (for example `["chebi", "uniprot"]`) to assign each species to the matching database. Reaction annotation uses species evidence from a prior species run, a CSV, or annotations already in the model; the default grounded method also uses participant names when identifiers are missing, while the deprecated rule-based method requires species ChEBI terms.

---

## Full Documentation

See [docs/README.md](docs/README.md) for:

- All parameters for `annotate_model` / `curate_model`
- Species, reaction, and combined annotation workflows
- Per-database annotation examples
- Feedback API reference
- Evaluation utilities
- Data file descriptions
- Supported embedding models for RAG
