# AAAIM (Auto-Annotator via AI for Modeling)

AAAIM is a LLM-powered system for annotating biosimulation models with standardized ontology terms. It supports chemical, gene, and protein annotation for species, and KEGG annotation for reactions. You can annotate species only, reactions only, or both in one call.

## Installation

```bash
# python = 3.12

# Install dependencies
pip install -r requirements.txt
```

Set up your LLM provider API keys:

```bash
# For OpenAI models (gpt-4o-mini, gpt-4.1-nano)
export OPENAI_API_KEY="your-openai-key"

# For OpenRouter models (meta-llama/llama-3.3-70b-instruct:free)
export OPENROUTER_API_KEY="your-openrouter-key"
```

Alternatively, you can setup an `.env` file that looks like the following:

```bash
OPENAI_API_KEY=<your-openai-api-key-here>
OPENROUTER_API_KEY=<your-openrouter-api-key-here>
```

OpenRouter models are also supported. Reasoning is enabled automatically for
OpenRouter requests when the provider supports it:

```python
result = annotate_model(
    model_file="path/to/model.xml",
    llm_model="openrouter/free",
)
```

## Usage

AAAIM provides species annotation, reaction annotation, a combined species+reaction workflow, a curation workflow, and an interactive feedback mechanism.

Both `annotate_model` and `curate_model` return an `AnnotationResult` object.
It holds `recommendations_df` (primary table), `species_recommendations_df`,
`reaction_recommendations_df`, and `metrics`, and also supports tuple unpacking:

```python
# Both styles work:
result = annotate_model("model.xml", ...)
recommendations_df, metrics = annotate_model("model.xml", ...)
```

A recommendations CSV is saved automatically after every run and after every
feedback revision. Pass ``save_to`` to set the file prefix (default: the model
filename). Species go to ``<save_to>_species.csv`` and reactions to
``<save_to>_reactions.csv``.

### 1. Annotation Workflow (for new models)

Set `annotate` to choose what to annotate:

| `annotate`     | What runs                                      | Result attributes                                              |
| -------------- | ---------------------------------------------- | -------------------------------------------------------------- |
| `"species"`    | Species only (default)                         | `species_recommendations_df`                                   |
| `"reactions"`  | KEGG reactions only                            | `reaction_recommendations_df` (+ input `species_recommendations_df`) |
| `"both"`       | Species first, then reactions from those terms | `species_recommendations_df` and `reaction_recommendations_df` |

- **Purpose**: Annotate models with no or limited existing annotations
- **Metrics**: Accuracy is NA when no existing annotations available
- **Large Models**: Automatically splits models with >50 species into chunks to avoid LLM context limits

Species annotation uses two size knobs. `top_k` is the number of biological identities retained by direct/RAG retrieval per ordinary species or per parsed complex component (default 3). `n_return` is the number of identities retained by the final LLM ranking (default 3). Taxon-specific accessions belonging to one identity do not consume additional identity slots. Synonym generation is fixed at 3 and is not controlled by either parameter. The LLM ranking step runs only when a unit has more identities than `n_return`, and all such units are ranked in one LLM call. For example, `top_k=3, n_return=1` retrieves three identities and reports the top identity; a three-component complex reports one identity for each component. Reactions retain their existing candidate-based interpretation of these parameters.

Optional `validation=True` changes the species workflow; it is off by default.
It requests structured JSON for name normalization and candidate ranking,
checks that every requested species/component is represented, and disallows
ranked identifiers outside the retrieved pool. A malformed response gets one
repair request; ranking then falls back to retrieval order if still invalid.
For automatic entity typing, a perfect direct match in an allowed alternative
database can override the LLM's route when it scores strictly higher than the
selected route. Forced entity types are not changed. For ChEBI candidates,
SBML FBC formula and charge are compared with local ChEBI data. Mismatched
candidates are removed only if the same retrieval pool contains a fully
specified heavy-atom-formula-and-charge match; otherwise the conflict is inconclusive and
the pool remains available for review. These checks do not establish biological
correctness and are recorded under `result.metrics["validation"]`.

Optional `source_fidelity=True` requires `validation=True` and applies to
ordinary ChEBI chemical species only. The literal SBML display name is kept
ahead of LLM-generated synonyms; a trailing alternate-name parenthetical may
also be omitted and standalone `ACP` expanded to `[acyl-carrier protein]` for
exact lookup. Leading `(R)`/`(S)` stereochemistry is never discarded. Exact
SBML-name hits reserve retrieval slots before generated aliases. After the
formula/charge guard, a unique exact source-name candidate is selected
deterministically; explicit stereo/ACP names exclude generic alternatives if
an exact source-name candidate remains. Other ambiguous pools still use the
structured ranking LLM, which is instructed to prefer the SBML name. This is
source-specific lexical evidence, not an independent ontology validation.
Complex components, gene/protein searches, and reaction ranking are unchanged.

For a three-run variability check of the BioModels use cases, the bundled
`run_model250115_calibration.py` and `run_biomodel_usecase.py` scripts accept
`--source-fidelity --repeat-id 2` or `3`. Each repeat uses a separate output
prefix and saves a `*_trace.json` file with normalization, candidates before
ranking, and final rows. The original source-fidelity result is repeat 1.
`tests/useCase/evaluate_variability.py` writes a combined variability JSON;
the per-model publication-backed evaluators are reused unchanged. An empty
ranking selection can currently omit a species row, so the combined result
records those omissions separately as abstentions.

#### Chemical Annotation (ChEBI)

```python
from core import annotate_model

result = annotate_model(
    model_file="path/to/model.xml",
    entity_type="chemical",
    database="chebi",
    message="user message to the LLM here (optional)",
)
```

#### Gene Annotation (NCBI Gene)

```python
from core import annotate_model

result = annotate_model(
    model_file="path/to/model.xml",
    entity_type="gene",
    database="ncbigene",
    tax_id=["9606", "10090", "10116"]  # human, mouse, and rat
)
```

For multi-taxon searches, candidate identities are ranked before their
taxon-specific accessions are expanded. The output records `identity`,
`identity_rank`, `candidate_rank`, and `tax_id`, so a top-one identity may have
one accession row per organism.

#### Protein Annotation (UniProt)

```python
from core import annotate_model

result = annotate_model(
    model_file="path/to/model.xml",
    entity_type="protein",
    database="uniprot",
    tax_id="9606"  # for human
)
```

#### Automatic Entity Type Detection

AAAIM supports automatic detection of entity types (chemical, gene, protein, complex, or unknown) for models with mixed entity types:

```python
from core import annotate_model

result = annotate_model(
    model_file="path/to/model.xml",
    entity_type="auto",
    database=["chebi", "uniprot"]  # choose from these databases
)

# The results will include a 'type' column indicating the detected entity type
print(result.recommendations_df[['species_id', 'type', 'synonyms_LLM', 'predictions_names']])
```

**How it works:**

- The LLM analyzes each species in context (display names, reactions, model notes, and optional user message) to determine its type
- Detected types: `chemical`, `gene`, `protein`, `complex`, or `unknown`
- Database matching is performed using the appropriate database for each detected type
- The `database` parameter accepts a list to specify which databases to use:
  - Chemicals → ChEBI
  - Genes → NCBI Gene
  - Proteins → UniProt
  - Complexes → each component is searched in the database for that component's type (chemical → ChEBI, protein → UniProt, gene → NCBI Gene). The LLM lists all components, with a synonym group and type per component. Component boundaries are retained in the output and ranking is performed per component. Untyped (legacy) complex replies still search every allowed database as one unresolved unit.
- Species with `unknown` type are included in results with their LLM-suggested synonyms but no database matches

#### Reaction Annotation (KEGG)

Reaction annotation maps model reactions to KEGG reaction IDs. It does not start from reaction names. Species-level ChEBI (or KEGG compound) annotations are projected into KEGG compound space, candidate KEGG reactions are generated by matching reactants and products, and an LLM re-ranks those candidates.

Species annotations can come from:

1. A previous AAAIM species run (`annotate="both"`, or pass `species_recommendations_df`)
2. A species recommendation CSV
3. ChEBI terms already present in the SBML file

```python
from core import annotate_model

# Use ChEBI annotations already in the model
result = annotate_model(
    model_file="path/to/model.xml",
    annotate="reactions",
    n_return=3,
)

# Or pass a species recommendation table / CSV
result = annotate_model(
    model_file="path/to/model.xml",
    annotate="reactions",
    species_recommendations_df="model.xml_species.csv",
    n_return=3,
)
```

`entity_type="reaction"` and `method="rulebased"` still select this path for backward compatibility.

**How it works:**

1. Map species ChEBI IDs to KEGG compound IDs (ontology walk when there is no direct map)
2. Match each model reaction to KEGG reactions by participant sets, with staged ChEBI relaxation and optional cofactor filtering
3. Collapse near-redundant candidates and score them
4. Re-rank remaining candidates with an LLM using the model reaction equation and KEGG DEFINITION text, keeping `n_return` IDs

The output CSV lists candidate KEGG reaction IDs per model reaction, ordered by rank. Use the same `update_annotation` column (`add` / `delete` / `ignore`) as for species, then `update_annotation()` to write them into the SBML.

#### Combined species and reaction annotation

```python
result = annotate_model(
    model_file="path/to/model.xml",
    annotate="both",
    entity_type="chemical",   # or "auto"
    database="chebi",
    top_k=10,                 # retrieval pool per species
    n_return=3,               # final LLM-ranked IDs per entity
    save_to="modelA_gpt5",    # writes modelA_gpt5_species.csv and modelA_gpt5_reactions.csv
)

print(result.species_recommendations_df)    # species
print(result.reaction_recommendations_df)   # KEGG reactions
print(result.metrics["reaction"])           # reaction-only metrics
```

Species recommendations are saved to `<save_to>_species.csv`. Reaction recommendations are saved to `<save_to>_reactions.csv`. If `save_to` is omitted, the model filename is used as the prefix.

If the species step produces no ChEBI rows, reaction annotation is skipped and `reaction_recommendations_df` is empty.

### 2. Curation Workflow (for models with existing annotations)

- **Purpose**: Evaluate and improve existing annotations
- **Input**: Only species that already have annotations
- **Output**: Validation and improvement recommendations
- **Metrics**: Accuracy calculated against existing annotations
- **Large Models**: Automatically splits models with >50 entities into chunks to avoid LLM context limits

#### Chemical Curation

```python
from core import curate_model

result = curate_model(
    model_file="path/to/model.xml",
    entity_type="chemical",
    database="chebi"
)

print(f"Chemical entities with existing annotations: {result.metrics['total_entities']}")
print(f"Accuracy: {result.metrics['accuracy']:.1%}")
```

#### Gene Curation

```python
from core import curate_model

result = curate_model(
    model_file="path/to/model.xml",
    entity_type="gene",
    database="ncbigene"
)

print(f"Gene entities with existing annotations: {result.metrics['total_entities']}")
print(f"Accuracy: {result.metrics['accuracy']:.1%}")
```

#### Protein Curation

```python
from core import curate_model

result = curate_model(
    model_file="path/to/model.xml",
    entity_type="protein",
    database="uniprot",
    tax_id=9606  # for human
)

print(f"Protein entities with existing annotations: {result.metrics['total_entities']}")
print(f"Accuracy: {result.metrics['accuracy']:.1%}")
```

### 3. User Feedback

After reviewing the initial recommendations you can iteratively refine them.
The previous prompt, LLM response, and your feedback are sent back to the
LLM for a revised set of synonyms, which then go through the retrieval
pipeline again.

#### Single revision

```python
result = annotate_model("model.xml", entity_type="chemical", database="chebi")

# Provide feedback
result = result.revise("Species X should be glucose-6-phosphate, not glucose")
```

#### Interactive loop

```python
result.feedback_loop()          # console prompt; press Enter to accept
```

A custom callback can replace the console prompt (useful for GUIs or
notebooks):

```python
def my_feedback(df, iteration):
    # display df in your UI, collect text
    return user_text   # return "" or None to stop

result.feedback_loop(get_feedback_fn=my_feedback)
```

#### Versioned CSV output

Every revision writes a new file so that earlier recommendations are
never lost:

```
model.xml_species.csv       ← initial run
model.xml_species_v1.csv    ← after 1st revision
model.xml_species_v2.csv    ← after 2nd revision
```

#### Advanced control

`result.revise()` and `result.feedback_loop()` use the same database,
method, LLM model, and other parameters from the initial run by default.
For advanced use you can also call the lower-level helpers directly; see
`core/feedback.py` for the full API.

### 4. Updating Model Annotations After Review

After running `annotate_model` or `curate_model`, you can review the resulting CSV file and edit the `update_annotation` column for each entity:

- `add`: Add the recommended annotation to the model for that entity.
- `delete`: Remove the annotation for that entity.
- `ignore` or `keep`: Leave the annotation unchanged. Whether keep the existing one, or ignore the new suggestion.

To apply your changes and save a new SBML model:

```python
from core.update_model import update_annotation

update_annotation(
    original_model_path="path/to/original_model.xml",
    recommendation_table="recommendations.csv",  # or a pandas DataFrame
    new_model_path="path/to/updated_model.xml",
    qualifier="is"  # (optional) bqbiol qualifier, default is 'is'
)
```

A summary of added/removed annotations will be printed after update.

### 5. Advanced Usage

```python
# More control over parameters
result = annotate_model(
    model_file = "path/to/model.xml",
    annotate = "species",				 # "species", "reactions", or "both"
    llm_model = "gpt-4o-mini",       # the LLM model used to predict annotations
    # llm_model = "openrouter/free",			 # OpenRouter free model; reasoning is automatic when supported
    max_entities = 100,					 # maximum number of entities to annotate (None for all)
    entity_type = "gene",				 # type of entities to annotate ("chemical", "gene", "protein", "auto", "reaction")
    database = "ncbigene",				 # database to use ("chebi", "ncbigene", "uniprot", "kegg") or list for auto mode
    method = "direct",					 # species search: "direct" or "rag"; reactions always use rule-based matching + LLM ranking
    top_k = 3,						 # identity pool per species or complex component
    n_return = 1,					 # final identities kept per species/component
    validation = False,             # optional species validation guards
    chunk_size = 50,					 # split large models into chunks of 50 entities (None for no chunking)
    species_recommendations_df = None,			 # species table or CSV; used when annotate="reactions"
    save_to = None,					 # output prefix; writes <save_to>_species.csv / <save_to>_reactions.csv
    verbose = False,					 # True for a short progress summary
    em_max_iterations = 5,				 # reaction EM rematch rounds; 0 skips EM
    message = "This is a model for human metabolism", # optional user message to the LLM
)

# Direct access to qualifier tracking functions
from core.model_info import find_species_with_annotations_and_qualifiers

# Get annotations and qualifiers for any supported database
annotations, qualifiers = find_species_with_annotations_and_qualifiers(
    model_file="path/to/model.xml",
    database="chebi",  # or "ncbigene", "uniprot"
    bqbiol_qualifiers=['is', 'isVersionOf']  # optional: filter by specific qualifiers
)

print(f"Found {len(annotations)} species with annotations")
for species_id, annotation_ids in annotations.items():
    if species_id in qualifiers:
        print(f"{species_id}: {annotation_ids}")
        for ann_id, qualifier in qualifiers[species_id].items():
            print(f"  {ann_id} -> {qualifier}")
    else:
        print(f"{species_id}: {annotation_ids} (no qualifier info)")
```

### Example

```python
# Using "tests/test_models/BIOMD0000000190.xml"
python examples/simple_example.py
```

### 6. Evaluation and Results Analysis

AAAIM provides tools for evaluating annotation quality and analyzing results:

```python
from utils.evaluation import evaluate_single_model, print_evaluation_results

# Evaluate a single model and get detailed results with a 'type' column
result_df = evaluate_single_model(
    model_file="path/to/model.xml",
    llm_model="gpt-4o-mini",
    method="direct",
    top_k=3,
    entity_type="auto",  # or "chemical", "gene", "protein"
    database=["chebi", "uniprot"]  # for auto mode, or single database for specific type
)

# The result DataFrame includes a 'type' column showing detected entity types
print(result_df[['species_id', 'type', 'synonyms_LLM', 'predictions_names', 'accuracy']])

# Print summary statistics from a results CSV file
print_evaluation_results(
    results_csv="results.csv",
    ref_results_csv="reference_results.csv",  # optional: filter to only species in reference
    bqbiol_qualifiers=['is', 'isVersionOf'],  # optional: filter by annotation qualifiers
    entity_types=['chemical', 'protein']  # optional: filter by detected entity types
)
```

**Output columns:**

- `detected_entity_type`: Detected entity type (chemical, gene, protein, complex, or unknown)
- `synonyms_LLM`: LLM-suggested synonyms for the species
- `predictions`: Top-k database IDs matched for this species
- `predictions_names`: Corresponding names for the predicted IDs
- `exist_annotation_id`: Existing annotation IDs from the model (if any)
- `exist_annotation_name`: Names of existing annotations
- `accuracy`: Match accuracy between predictions and existing annotations

## Methods

### Direct matching

After LLM name normalization, direct matching expands composite gene/protein fields into lookup-safe forms. For example, `MAP2K1 (MEK1), MAP2K2 (MEK2)` contributes the original text plus `MAP2K1`, `MEK1`, `MAP2K2`, and `MEK2`; transcript descriptors are removed from additional lookup variants. Chemical punctuation is preserved.

Retrieval ranks `top_k` biological identities per ordinary species or parsed complex component. NCBI Gene and UniProt candidates are grouped across requested taxa before accessions are expanded. The LLM then retains up to `n_return` identities per unit. Output columns preserve `component_id`, `component_name`, `component_type`, `identity`, `identity_rank`, `retrieval_identity_rank`, `candidate_rank`, and `tax_id`. An unmatched parsed component receives its own empty row so complex completeness can be evaluated.

When `validation=True`, the ranking LLM is also a constrained selection check:
it must choose from retrieved IDs. This is not an independent validation of
entity type or molecular identity; cross-database matching and FBC chemistry
checks supply separate evidence before ranking. The validation mode applies
to species annotation, not the reaction ranking method.

Cross-taxon identity grouping uses exact normalized canonical labels, not fuzzy
symbol similarity. This keeps ortholog labels together while preventing distinct
near-symbol genes such as `PDPK1`/`PDK1` or `PXN`/`PXDN` from being merged.

### Rule-based reaction matching + LLM ranking

Used when `annotate="reactions"` or `annotate="both"`. After species ChEBI IDs are mapped to KEGG compounds, candidate KEGG reactions are retrieved by aligning substrate and product sets (the model participants must be a subset of the KEGG reaction). ChEBI ontology relaxation reconciles differences in chemical granularity (for example glucose vs hexose). Common cofactors can be excluded from matching. The LLM then ranks the remaining KEGG DEFINITION strings against the model reaction equation, using `n_return` and the model notes in the prompt, and keeps `n_return` IDs. A reaction with `n_return` or fewer candidates skips this call. The LLM does not invent IDs that were not generated as candidates.

### Retrival augmented generation (RAG)

After LLM name normalization, RAG finds similar ontology terms by cosine similarity. Multi-taxon NCBI Gene and UniProt hits use the same identity grouping and component-aware ranking as direct matching; `top_k` and `n_return` therefore limit identities rather than raw taxon accessions.

To use RAG, create embeddings of the ontology first:

```bash
cd data
# for ChEBI:
python load_data.py --database chebi --model default
# for NCBI gene, specify the taxnomy id:
python load_data.py --database ncbigene --model default --tax_id 9606
# for uniprot, specify the taxnomy id:
python load_data.py --database uniprot --model default --tax_id 9606
# for KEGG:
python load_data.py --database kegg --model default
```

## Databases

### Currently Supported

- **ChEBI**: Chemical Entities of Biological Interest
  - **Entity Type**: `chemical`
  - All terms in ChEBI are included.
  - Used for: small molecules, metabolites, compounds
- **NCBI Gene**: Gene annotation
  - **Entity Type**: `gene`
  - Only genes for common species are supported (those included in bigg models).
  - Used for: genes, DNA sequences, gene symbols
- **UniProt**: Protein annotation
  - **Entity Type**: `protein`
  - Only proteins for human (9606) and mouse (10090) are supported for now.
  - Used for: proteins, enzymes
- **KEGG**: Reaction annotation
  - **Entity Type**: `reaction` (via `annotate="reactions"` or `annotate="both"`)
  - Candidate KEGG reactions are generated from species ChEBI/KEGG compound annotations, then ranked by an LLM.
  - Used for: metabolic reactions in SBML models

### Entity Type to Database Mapping

When using `entity_type="auto"`, AAAIM automatically selects the appropriate database(s) based on the detected entity type:


| Detected Type | Default Databases         | Usage                                          |
| ------------- | ------------------------- | ---------------------------------------------- |
| `chemical`    | ChEBI                     | Small molecules, metabolites, compounds        |
| `gene`        | NCBI Gene                 | Genes, DNA sequences, gene symbols             |
| `protein`     | UniProt                   | Proteins, enzymes                              |
| `complex`     | Per component (ChEBI / UniProt / NCBI Gene) | Each component is routed to its type's database |
| `unknown`     | None                      | LLM synonyms included but no database matching |


You can restrict which databases are used by providing a `database` list parameter. For example, `database=["chebi", "uniprot"]` will only use ChEBI for chemicals and UniProt for proteins, but will not search NCBI Gene even if genes are detected.

### Future Support

- **Rhea**, **MetaCyc**, **BiGG**: additional reaction databases
- **GO**: Gene Ontology terms

## Data Files

### ChEBI Data

- **Location**: `data/chebi/`
- **Files**:
  - `cleannames2chebi.lzma`: Mapping from clean names to ChEBI IDs
  - `chebi2label.lzma`: Mapping from ChEBI IDs to labels
  - `chebi2names.lzma`: ChEBI synonyms used for RAG approach
- **Source**: ChEBI ontology downloaded from [https://ftp.ebi.ac.uk/pub/databases/chebi/ontology/chebi.owl.gz](https://ftp.ebi.ac.uk/pub/databases/chebi/ontology/chebi.owl.gz).

### NCBI gene Data

- **Location**: `data/ncbigene/`
- **Files**:
  - `names2ncbigene_bigg_organisms_protein-coding.lzma`: Mapping from names to NCBI gene IDs, only include protein-coding genes from 18 species covered in Bigg models for file size considerations
  - `ncbigene2label_bigg_organisms_protein-coding.lzma`: Mapping from NCBI gene IDs to labels (primary name)
  - `names2ncbigene_tax{tax_id}_protein-coding.lzma`: organism-specific names used for direct matching
  - `ncbigene2names_tax{tax_id}_protein-coding.lzma`: organism-specific names used for RAG indexing
  - Included manuscript organisms: human (`9606`), mouse (`10090`), and rat (`10116`)
- **Source**: Data are obtained from the NCBI gene FTP site: [https://ftp.ncbi.nih.gov/gene/DATA/GENE_INFO/](https://ftp.ncbi.nih.gov/gene/DATA/GENE_INFO/).

### UniProt Data

- **Location**: `data/uniprot/`
- **Files**:
  - `names2uniprot_human+mouse.lzma`: Mapping from synonyms to UniProt IDs, only include human and mouse proteins for now
  - `uniprot2label_human+mouse.lzma`: Mapping from UniProt IDs to labels (primary name)
  - `uniprot2names_tax{tax_id}.lzma`: Uniprot synonyms for tax_id used for RAG approach
- **Source**: Data are obtained from the UniProt site: [https://www.uniprot.org/help/downloads](https://www.uniprot.org/help/downloads) (Reviewed (Swiss-Prot) xml).

### KEGG Data

- **Location**: `data/kegg/`
- **Files**:
  - `chebi_to_kegg_map.lzma`: Mapping from ChEBI IDs to KEGG compound IDs.
  - `parsed_kegg_reactions.lzma`: Dict of KEGG reactions and their attributes
- **Source**: Data are obtained from the KEGG site: [https://rest.kegg.jp](https://rest.kegg.jp).

## File Structure

```
aaaim/
├── core/
│   ├── __init__.py              # Main interface exports
│   ├── annotation_workflow.py   # Annotation workflow (species, reactions, or both)
│   ├── curation_workflow.py     # Curation workflow (models with annotations)
│   ├── feedback.py              # Feedback module (AnnotationResult, revise, feedback_loop)
│   ├── model_info.py            # Model parsing and context
│   ├── llm_interface.py         # LLM interaction
│   ├── database_search.py       # Database search functions
│   ├── update_model.py          # put annotations into model
│   └── reaction/                # KEGG reaction matching, relaxation, scoring, LLM ranking
├── utils/
│   ├── constants.py
│   ├── evaluation.py            # functions for evaluation
├── examples/
│   ├── simple_example.py        # Species annotation / curation demo
│   └── kegg_rulebased_annotation_example.py  # Reaction / combined demo
├── data/
│   ├── chebi/                   # ChEBI compressed dictionaries
│   ├── ncbigene/                # NCBIgene compressed dictionaries
│   ├── uniprot/                 # UniProt compressed dictionaries
│   ├── kegg/                    # KEGG reaction features and ChEBI→KEGG map
│   ├── chroma_storage/          # Database embeddings for RAG
└── tests/
    ├── test_models              # Test models
    └── aaaim_evaluation.ipynb   # evaluation notebook
```

## Future Development

### Planned Features

- **Multi-Database Support**: GO, Rhea, mapping between ontologies
- **Improve RAG for NCBI Gene**: Test on other embedding models for genes
- **Web Interface**: User-friendly annotation tool
