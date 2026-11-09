# AAAIM BioModels use-case fixes

This document records the prediction, validation, evaluation, and reproducibility changes made while testing four BioModels use cases. It describes the initial problem, the implemented approach, and the observed outcome. The final use-case configuration was a pinned GPT-5 mini snapshot, direct retrieval with `top_k=3`, and `n_return=1` biological identity per ordinary species or parsed complex component. Validation and chemical source fidelity were enabled for the final runs.

The changes below concern annotation prediction. The separate SBML qualifier question (`is` versus `hasPart`) was intentionally not changed for this evaluation.

## Summary table

| Category | Issue | Change | Meaning | Final outcome / finding |
|---|---|---|---|---|
| Output and ranking | **Complex top-k semantics** | Component-aware retrieval and ranking | `top_k` builds a candidate-identity pool for each component, and `n_return` limits the final identities for each component. A three-component complex with `n_return=1` therefore returns one identity for each component, not one identity for the whole complex. | Top one can now be used consistently for ordinary species and complexes. With `top_k=3`, a three-component complex can contain up to nine candidate identities before ranking. |
| Output and ranking | **Component traceability and missing components** | Species-scoped component IDs and explicit empty component rows | Components are labeled `<species_id>_1`, `<species_id>_2`, and so on. A parsed component with no match remains visible as an empty row instead of silently disappearing. | Complex rows can be traced back to their parent species, and component completeness can be measured. |
| Output and ranking | **Synonym wording and composite names** | Synonym normalization, de-duplication, and lookup-safe expansion | Case, punctuation, and trivial wording variants are consolidated. Composite gene/protein text such as `MAP2K1 (MEK1), MAP2K2 (MEK2)` also contributes the individual symbols and aliases needed for lookup. | Reduces duplicate identities and missed direct matches caused only by wording. It does not remove LLM variability in complex decomposition. |
| Output and ranking | **Taxon duplication** | Identity/taxon separation | Candidate accessions are first grouped by biological identity; identity ranking happens before the selected identity is expanded to the requested taxa. | One top-ranked gene can be emitted consistently as human, mouse, and rat Entrez rows. These three rows are one identity, not three top-k choices. |
| Entity routing | **Mixed complexes searched as one unresolved object** | Per-component database routing | Each typed component is searched in its matching database: chemical to ChEBI, protein to UniProt, and gene to NCBI Gene. Legacy untyped complex output still searches the allowed databases. | Mixed complexes no longer force every component through the same database. |
| Model handling | **SBML-qual species were not handled like ordinary SBML species** | Qualitative-species and transition extraction for automatic annotation | The model reader now uses qualitative species and transitions when the SBML `qual` package is present. | MODEL2506050001 is processed as 165 qualitative species and 165 transitions rather than as an empty or incomplete core-SBML model. |
| Validation | **Free-form LLM output could be incomplete, malformed, or contain IDs outside the candidate pool** | Structured LLM output with application validation | Normalization and ranking use JSON schemas, require every requested unit, and restrict ranking to retrieved IDs. A failed response gets one repair attempt; ranking then falls back to retrieval order if necessary. | Parsing is more reliable and the ranker cannot invent an identifier. An empty valid selection is still an abstention, not evidence that the candidates are biologically wrong. |
| Validation | **EGF routed as a chemical** | Cross-database entity-type validation | In automatic mode, the normalized name is checked in the allowed databases. A strictly stronger exact match in another database can change the route before candidate ranking; forced entity types are not changed. | With Table S3 context, five EGF components were rerouted from ChEBI to human UniProt P01133, giving 113/113 expected components and 46/46 complete species. SBML-only context still does not identify EGF reliably. |
| Validation | **Acetate ester selected instead of acetate** | Formula/charge validation | SBML FBC formula and charge are compared with local ChEBI properties. A definite mismatch is removed only when the same pool contains a fully specified heavy-atom-formula-and-charge match; missing or generic chemistry remains inconclusive. | The four acetate species now return acetate, ChEBI 30089, instead of acetate ester. The guard is conservative and is not a universal identity validator. |
| Source fidelity | **LLM wording lost ACP conjugation or stereochemistry** | Source-fidelity retrieval and priority | The literal SBML display name is retained ahead of generated synonyms; standalone `ACP` is expanded for exact lookup; leading `(R)`/`(S)` is preserved; exact source-name ChEBI hits receive retrieval priority. A unique exact hit surviving chemistry checks is selected deterministically. | All six targeted species across five error classes were fixed, and MODEL2507280001 increased from 511/539 (94.8%) to **532/539 (98.7%)** supported top-one identities. |
| Context | **Important names were present only in a publication or supplement** | Optional context input and separate calibration conditions | Publication text can be supplied through the annotation message without modifying the SBML. SBML-only and context-augmented runs are saved and evaluated separately. | MODEL2501150001 recovered 113/113 Table S3 components in all three augmented runs; SBML-only recovery varied from 82 to 105 of 113. This shows that missing context, not only routing logic, causes failures. |
| Evaluation | **One generic accuracy definition was misleading across the four models** | Publication-backed, model-specific evaluators | Complexes are scored by component identity, gene predictions by exact Entrez identifiers plus verified requested-taxon orthologs, and metabolites by supported molecular identity rather than only a direct ChEBI-to-KEGG cross-reference. Process/state nodes are excluded from molecular-ID accuracy. | The reported measures match each publication's available reference and clearly separate coverage, precision, component completeness, and exact-set recovery. |
| Reproducibility | **Single runs could hide LLM variability or overwrite earlier outputs** | Pinned configuration, condition-specific output prefixes, repeat IDs, and stage traces | Each condition saves predictions, metrics, evaluation, and—on repeats—normalization, pre-ranking candidates, and final rows. Repeats refuse to overwrite existing results. | Three-run comparison localized variation to normalization/decomposition, retrieval, or ranking. It did not support choosing the best run or treating majority vote as the original top-one method. |
| Reliability | **Failed or excessively long LLM responses could trigger uncontrolled work or obscure an incomplete run** | Bounded repair/retry behavior, output controls, and completeness checks | The runner limits recovery attempts, records validation events, checks the expected number of species, and distinguishes ranking abstention from earlier pipeline failure. | Runs terminate predictably and incomplete output is visible. A current limitation remains: a ranking abstention can omit the final species row, although it is retained in the audit trace. |

## 1. Output unit and ranking semantics

The main ranking change was to define the unit of prediction before applying `top_k` or `n_return`:

1. An ordinary species is one ranking unit.
2. A parsed complex is split into typed component units.
3. Direct or RAG retrieval keeps up to `top_k` **biological identities** for each unit.
4. The ranking step keeps up to `n_return` identities for each unit.
5. Only after identity selection are taxon-specific accessions expanded.

This order resolves two earlier ambiguities. First, top one no longer means one result for an entire complex. Second, three ortholog accessions do not consume three identity positions. For example, a three-component complex with `top_k=3` and `n_return=1` may have nine identities in its internal candidate pools but reports one identity for each component. A selected gene identity may then produce three Entrez rows when human, mouse, and rat are requested.

The output retains `component_id`, `component_name`, `component_type`, `identity`, `identity_rank`, `retrieval_identity_rank`, `candidate_rank`, and `tax_id`. Species-scoped component IDs make rows unambiguous across the model. Empty component rows preserve failed retrievals for review instead of making a complex appear complete.

## 2. Name normalization, component typing, and database routing

Name normalization now serves retrieval without changing the biological unit being ranked. Trivial duplicate variants are removed, while useful pieces of composite fields are added as lookup variants. Explicit gene symbols are kept distinct: exact normalized canonical labels are used for identity grouping, so near-symbol genes such as `PDPK1` and `PDK1` are not merged.

For an automatically typed complex, the LLM returns a list of components, a synonym group for each component, and a component type. AAAIM then routes each component to its matching database. This is different from treating a complex as one string and searching every database indiscriminately. When the LLM supplies only the older untyped complex form, the broad multi-database search is retained as a compatibility fallback.

The SBML reader was also corrected for qualitative models. When the `qual` package is present, qualitative species and transitions supply the model context used for annotation. This is required for MODEL2506050001.

## 3. Optional validation

`validation=True` adds three checks to species annotation. It is optional and remains off by default for compatibility with earlier runs.

### Structured LLM output

Normalization must return every requested species exactly once. A complex must contain typed components rather than an unstructured sentence. Ranking must return only identifiers offered for that species or component. AAAIM validates these conditions in code, requests at most one structured repair, and uses deterministic retrieval order if the ranking response is still unusable.

The ranking LLM is therefore a **constrained selection check**, not an independent biological validator. It can reject or reorder retrieved candidates, but it cannot correct a candidate that was sent to the wrong database or recover an identity absent from the pool. Cross-database routing and formula/charge checks act before ranking and supply separate evidence.

### Cross-database entity-type validation

This check applies only to `entity_type="auto"`. AAAIM searches the selected route and allowed alternative databases using the normalized label. It changes the route only when an alternative has a strictly stronger exact match. This conservative rule avoids changing forced configurations such as the all-gene NCBI Gene run for MODEL2506050001.

The EGF test demonstrates both the benefit and the limit. With the Table S3 description “epidermal growth factor,” UniProt provides strong exact protein evidence and the route changes from ChEBI to UniProt P01133. Without Table S3, the SBML labels and interaction context do not consistently expose the EGF identity, so validation cannot manufacture the missing information.

### Formula/charge validation

For ChEBI candidates, AAAIM reads formula and charge from the SBML FBC species extension and compares them with local ChEBI data. Hydrogen differences are handled through a heavy-atom formula comparison because protonation may vary. A mismatching candidate is removed only when the same candidate pool contains a fully specified formula-and-charge match. If the SBML or ChEBI chemistry is absent, contains a generic group such as `R`, or has no exact alternative, the result is marked inconclusive and retained for review.

This rule directly fixes acetate versus acetate ester: the ester is chemically incompatible, while acetate ChEBI 30089 matches the supplied properties. Other improvements observed in the validation rerun cannot all be assigned to this guard because LLM normalization and ranking also varied between runs.

## 4. Chemical source fidelity

Formula and charge cannot distinguish every chemically important detail. In MODEL2507280001, generated synonyms sometimes removed an ACP conjugate or an explicit stereochemical prefix before retrieval. `source_fidelity=True` was added for ordinary ChEBI chemical species and requires `validation=True`.

The approach is deliberately narrow:

1. Keep the literal SBML display name before all LLM-generated aliases.
2. Preserve leading `(R)` or `(S)` stereochemistry.
3. Expand standalone `ACP` to `[acyl-carrier protein]` as an additional exact-lookup form.
4. Reserve retrieval positions for exact source-name ChEBI matches.
5. Apply formula/charge validation.
6. Select a unique surviving exact source-name match deterministically; otherwise use constrained LLM ranking with an instruction to prefer the source label.

This is lexical evidence from the model source, not independent ontology validation. It does not apply to complex decomposition, genes, proteins, or reaction ranking. It fixed the six targeted MODEL2507280001 species: two `(S)-3-methyl-2-oxopentanoate` occurrences and four ACP-bound compounds. The only remaining known wrong top-one identity in that run was the broad NAD(P)H term for `M_nadph_c`.

## 5. Evaluation and interpretation

The evaluators were made specific to the reference available for each model:

- **MODEL2501150001:** Table S3 defines the expected unique components for 46 aligned SBML species. “Complete species” means that every expected component was recovered; extra components are counted separately and do not make an otherwise complete species incomplete.
- **MODEL2503190002:** the predicted identity set must exactly match the publication-backed set. Three process/state species are expected to remain without molecular identifiers.
- **MODEL2506050001:** predictions are forced to NCBI Gene for human, mouse, and rat and checked against exact Entrez identifiers. Verified requested-taxon orthologs missing from the deposited `NodeGenes` list are counted as supported but reported separately. The 245-ID recall denominator measures recovery of long family/complex lists and is not a species-accuracy percentage.
- **MODEL2507280001:** the primary score is supported molecular identity. Direct ChEBI-to-KEGG mapping, short ontology relations, names/synonyms, formula evidence, and reviewed identity equivalence are used because a direct cross-reference alone undercounts charged, tautomeric, and alternate ontology forms.

### Final observed performance

| Model / condition | Final observation |
|---|---|
| MODEL2501150001, SBML + Table S3 | 113/113 expected components and 46/46 complete species in all three runs. Extra components still varied across runs. |
| MODEL2501150001, SBML only | 82–105/113 expected components across three runs; EGF was missing from the same five complexes in all three runs. |
| MODEL2503190002 | 27/28 exact molecular identity sets in the first run and 28/28 in repeats 2–3; only CK2 component expansion varied. |
| MODEL2506050001 | Supported predictions on mapped nodes were 171/183 (93.4%), 147/160 (91.9%), and 156/168 (92.9%) across three runs. CCNA2 versus CCNA1 and CDH1 versus FZR1 remained the main identity errors. |
| MODEL2507280001 | 532/539 (98.7%) supported top-one identities, with one known wrong identity and six scorable references without a candidate. |

These numbers should not be collapsed into one universal accuracy score because the reference unit differs by model: complex components, exact identity sets, taxon-specific Entrez rows, or metabolite identities.

## 6. Reproducibility and remaining limitations

The use-case runners pin the LLM snapshot and save condition-specific files so baseline, validation, source-fidelity, and repeat outputs cannot be confused. The MODEL2501150001 calibration keeps the SBML-only and Table S3 conditions separate. Repeat traces record:

- normalized names, types, and complex components;
- the retrieved candidate pool before ranking; and
- final selected rows.

The three-run comparison showed that name wording can change without changing the selected identity, but complex decomposition and gene ranking can materially change results. Repetition is useful for estimating this variability and locating its stage; it does not justify selecting the most favorable run or adding majority voting without defining a new ensemble method.

Remaining limitations are:

- EGF is still not recoverable reliably from MODEL2501150001 SBML-only context.
- Cross-database validation cannot help a forced single-entity-type run.
- Formula/charge validation is inconclusive when source or ChEBI chemistry is missing or generic.
- Source fidelity currently applies only to ordinary chemical species.
- A valid ranking abstention can omit the final species row; traces and variability evaluation record these omissions, but an explicit empty output row would be clearer.
- MODEL2506050001 retains context-sensitive homonym/family errors, especially Cyclin A and Cdh1.
- MODEL2507280001 retains the overly broad NAD(P)H prediction and six scorable no-candidate species.

## Implementation and evidence files

- Core workflow: [annotation_workflow.py](../../core/annotation_workflow.py)
- Method documentation: [docs/README.md](../../docs/README.md)
- Standard use-case runner: [run_biomodel_usecase.py](run_biomodel_usecase.py)
- MODEL2501150001 calibration runner: [run_model250115_calibration.py](run_model250115_calibration.py)
- Publication-backed evaluator: [evaluate_biomodel_usecases.py](evaluate_biomodel_usecases.py)
- Variability evaluator: [evaluate_variability.py](evaluate_variability.py)
- Final comparison: [source_fidelity_comparison.md](reports/source_fidelity_comparison.md)
- Three-run analysis: [variability_source_fidelity.md](reports/variability_source_fidelity.md)

