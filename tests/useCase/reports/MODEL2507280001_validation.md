# MODEL2507280001 — validation run

This *Buchnera aphidicola* metabolic model has **593 species and 541 reactions**. The run forces chemical identity and ChEBI retrieval, uses GPT-5 mini with direct `top_k=3`, and returns one identity per species. The publication supplement's metabolite names, formulas, charges, and KEGG IDs were withheld during prediction and used for review. Of 556 KEGG-referenced model species, 539 have a locally scorable reference.

| Measure | Baseline | Validation |
|---|---:|---:|
| Model species with a candidate | 552/593 | 540/593 |
| Direct ChEBI→KEGG agreement | 439/539 | 316/539 |
| Supported top-1 identities after ontology, name/synonym, and manual review | 508/539 (94.2%) | **511/539 (94.8%)** |
| Wrong or incomplete top-1 identities | 18 | **6** |
| Scorable references without a candidate | 13 | 22 |

The validation run predicts **acetate** (ChEBI 30089) rather than acetate ester in all four compartments. It also predicts formate, cobalt(II), NADPH, 2-isopropylmaleate, and the previously missing ACP/folate conjugates in their corresponding tested species. Formula/charge filtering directly excludes the acetate ester; some other improvements also reflect different LLM normalization/ranking in this single rerun, so they cannot all be attributed to that filter. The structured ranking response was valid after bounded repairs.

Direct KEGG cross-reference agreement falls sharply because many selected ChEBI terms represent charged forms without the same local KEGG mapping. The supported-identity result additionally checks ChEBI ontology, exact charge-stripped names, ChEBI synonyms, and eight reviewed species–accession pairs. It assesses molecular identity, not exact protonation state. For example, [ChEBI identifies ferroheme b(2−) as the conjugate base of ferroheme b](https://www.ebi.ac.uk/chebi/CHEBI:60344), which is associated with protoheme.

## Wrong or incomplete top-1 predictions

| Species ID(s) | Validation prediction | Supplement reference/correct identity | Assessment |
|---|---|---|---|
| `M_3mop_c`, `M_3mop_s` | 3-methyl-2-oxovalerate, ChEBI 28654 | **(S)-3-Methyl-2-oxopentanoate**, KEGG C00671 | Missing required stereochemistry. [ChEBI distinguishes the (S) child from this generic term](https://www.ebi.ac.uk/chebi/CHEBI:28654). |
| `M_3hhexACP_c` | 3-hydroxyhexanoic acid, ChEBI 37035 | **(R)-3-Hydroxyhexanoyl-[acyl-carrier protein]**, KEGG C05747 | Free acid instead of ACP-bound, and no (R) specificity. |
| `M_3ooctACP_c` | 3-oxooctanoic acid, ChEBI 44680 | **3-Oxooctanoyl-[acyl-carrier protein]**, KEGG C05750 | Free acid instead of ACP-bound. |
| `M_3hddecACP_c` | 3-hydroxylaurate, ChEBI 76616 | **(R)-3-Hydroxydodecanoyl-[acyl-carrier protein]**, KEGG C05757 | Free acid instead of ACP-bound, and no (R) specificity. |
| `M_myrsACP_c` | tetradecanoic acid, ChEBI 28875 | **Myristoyl-ACP**, KEGG C05761 | Free acid instead of ACP-bound. |

Twenty-two scorable references have no candidate, including common NAD, 2-phosphoglycerate, and PAP entries whose structured normalized names did not retain an exact lookup form, plus several ACP-bound and phospholipid species. Preserving the SBML display name as an additional retrieval synonym is a plausible next experiment. Missing ChEBI formula/structure data also limits what chemistry validation can exclude; it is not a universal correctness check.

Artifacts: [predictions](../results/MODEL2507280001/full_validation_species.csv), [evaluation](../results/MODEL2507280001/evaluation_validation.json), [held-out supplement](../context/publications/MODEL2507280001/supplement.xlsx).
