# AAAIM — Auto-Annotator via AI for Modeling

AAAIM is an LLM-powered tool that annotates biosimulation models (SBML) with standardized ontology terms from ChEBI, NCBI Gene, UniProt, and KEGG. You can annotate **species**, **reactions**, or **both**.

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

For other models like OpenAI models, just pass the model name, e.g. "gpt-4o-mini".

---

## Quick Start

```python
from core import annotate_model

result = annotate_model(
    model_file="path/to/model.xml",
    annotate="both",                  # "species" (default) | "reactions" | "both"
    entity_type="auto",               # detects chemical / gene / protein / complex
    database=["chebi", "uniprot"],    # databases to search for species
    message="This is a model for human metabolism", # optional user message to the LLM
    top_k=3,                         # identity retrieval pool per species/component
    n_return=1,                      # final identities kept per species/component
    validation=True,                 # optional structured output and candidate checks
    verbose=False,                    # True for a short progress summary
)
# result.species_recommendations_df    → species
# result.reaction_recommendations_df   → KEGG reactions (when annotate is "reactions" or "both")
```

For parsed complexes, the species table preserves `component_id`,
`component_name`, and `component_type`. `top_k` and `n_return` apply to each
component independently. For multi-organism gene/protein searches, accessions
are grouped by biological identity before ranking and emitted with a `tax_id`;
several accession rows can therefore share `identity_rank=1`.
Component IDs are species-scoped: species `X` uses `X_1`, `X_2`, and so on.
With `validation=True`, species normalization and ranking use structured LLM
responses, and applicable entity-type and chemical-property checks run before
the final top-one selection. The default is `False` for reproducibility with
earlier AAAIM runs.

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

Then write the updated model:

```python
from core.update_model import update_annotation

update_annotation(
    original_model_path="model.xml",
    recommendation_table="recommendations.csv",
    new_model_path="model_updated.xml"
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

Use `entity_type="auto"` with a `database` list (for example `["chebi", "uniprot"]`) to assign each species to the matching database. Reaction annotation needs species ChEBI terms — from a prior species run, a CSV, or annotations already in the model.

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
