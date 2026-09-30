# Source-fidelity rerun — four BioModels cases

All runs used GPT-5 mini, direct retrieval of three identities per species/component, and top-one output. The new `source_fidelity=True` option runs with the existing validation guards: for **ordinary chemical species**, it keeps the literal SBML name alongside LLM synonyms, expands standalone ACP for lookup, gives exact source-name ChEBI hits retrieval priority, and selects a unique exact hit after chemistry checks. Gene/protein identities and complex-component decomposition do not use this new rule. This is one new run per condition, not a variance-controlled ablation.

| Model (size) | Publication-backed measure | Prior validation | Source fidelity |
|---|---|---:|---:|
| MODEL2501150001 (47 species, 51 reactions), SBML only | Expected Table S3 components recovered; species with all expected components | 85/113; 21/46 | 105/113; 40/46 |
| MODEL2501150001, Table S3 supplied | Same measures | 113/113; 46/46 | 113/113; 46/46 |
| MODEL2503190002 (31 species, 40 reactions) | Molecular species with exactly the expected identities | 28/28 | 27/28 |
| MODEL2506050001 (165 qualitative species, 165 transitions) | Exact deposited Entrez IDs recovered; supported precision on mapped nodes | 151/245; 159/171 (93.0%) | 163/245; 171/183 (93.4%) |
| MODEL2507280001 (593 species, 541 reactions) | Supported top-one identities against 539 scorable supplement references; species with candidates | 511/539 (94.8%); 540/593 | **532/539 (98.7%); 563/593** |

For MODEL2507280001, all six species in the five previously identified stereo/ACP error classes now have the specific ChEBI identity: `(S)-3-methyl-2-oxopentanoate` **35146** (`M_3mop_c/s`), `(R)-3-hydroxyhexanoyl-ACP` **326**, `3-oxooctanoyl-ACP` **1646**, `(R)-3-hydroxydodecanoyl-ACP` **325**, and myristoyl-ACP **50651**. Each has a recorded unique-source-match selection event. Among the 539 scorable species, 23 previously unsupported entries became supported and two regressed; known wrong/incomplete top-one predictions fell from six to one, while no-candidate references fell from 22 to six. The charged/tautomer predictions for meso-diaminopimelate and dephospho-CoA were manually counted as the same molecular identities based on [ChEBI's tautomer relation](https://www.ebi.ac.uk/chebi/searchId.do?chebiId=CHEBI%3A16488) and [conjugate-base relation](https://www.ebi.ac.uk/chebi/searchId.do?chebiId=CHEBI%3A57328). Accuracy is identity-level, not exact protonation-state agreement.

## Remaining wrong or over-expanded predictions

| Model; species/node | New prediction | Publication/supplement reference and issue |
|---|---|---|
| 115; `EG2SOS` (both conditions) | Extra GRB2, UniProt **P62993** | Table S3 specifies EGF–EGFR–SOS, without GRB2. |
| 115; `BRAF_dimer`, `iBRAF_dimer`, `BRAF_iBRAF_dimer` (Table S3 condition) | Extra RAS, UniProt **P01116**, and GTP, ChEBI **15996** | These are not Table S3 components of the BRAF-containing dimers. |
| 319; `CK2` | Additional CSNK2A2 and CSNK2B components | Operational reference is CSNK2A1 alone; this may be biologically plausible family/complex expansion but fails the defined exact-identity test. |
| 605; `CyclinA`, `CyclinA_mRNA` | CCNA2, human/mouse/rat Entrez **890/12428/114494** | Deposited CCNA1, **8900/12427/295052**. |
| 605; `pAPC` | Additional CDC20, **991/107995/64515** | Deposited APC/C components ANAPC1, **64682/17222/311412**, and ANAPC2, **29882/99152/296558**; CDC20 is not among them. |
| 605; `Cdh1` | E-cadherin CDH1, **999/12550/83502** | APC/C co-activator FZR1, **51343/56371/314642**. |
| 728; `M_nadph_c` | NAD(P)H, ChEBI **13392** | Reduced NADP/NADPH, KEGG **C00005**; [ChEBI defines NAD(P)H as including either NADH or NADPH](https://www.ebi.ac.uk/chebi/CHEBI%3A13392), so this is too broad. |

Further gaps: SBML-only 115 still misses eight expected component occurrences (EGF/EGFR in `mE`, `mEL`, `mELmEL`, `E`, `EG2`, and `EG2SOS`); Table S3 recovers all expected occurrences but has the extra components above. In 728, six scorable references still lack a candidate: `M_sl2a6o_c`, `M_3hoctaACP_c`, `M_4r5au_c`, `M_apoACP_c`, `M_ugmd_c`, and `M_uaagmda_c`. `M_3hoctaACP_c` and `M_nadph_c` are the two regressions versus the prior validation run. The 115/319/605 changes cannot be attributed to source fidelity because their discrepant units are proteins, genes, or complexes; they illustrate single-run LLM variability. A paired repeat with fixed normalization output would better isolate the new chemical retrieval/ranking rules.

Artifacts: evaluations for [115](../results/MODEL2501150001/evaluation_source_fidelity.json), [319](../results/MODEL2503190002/evaluation_source_fidelity.json), [605](../results/MODEL2506050001/evaluation_source_fidelity.json), and [728](../results/MODEL2507280001/evaluation_source_fidelity.json). Each model folder also contains the corresponding `*_source_fidelity_species.csv` and `*_source_fidelity_metrics.json` files; prior validation files are unchanged.
Review sources: [115 Table S3](../context/MODEL2501150001_TableS3.txt), [319 supplement](../context/publications/MODEL2503190002/supplement.pdf), [605 deposited node mapping](../context/publications/MODEL2506050001/Hypoxia_EMT_Model.dmms), and [728 metabolite supplement](../context/publications/MODEL2507280001/supplement.xlsx).
