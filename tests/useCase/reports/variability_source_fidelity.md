# AAAIM top-one variability — three runs per condition

We repeated the current `validation=True, source_fidelity=True` workflow with the pinned `gpt-5-mini-2025-08-07` snapshot, direct retrieval (`top_k=3`), and one identity per species or complex component. Run 1 is the previously saved source-fidelity result; runs 2–3 are new. The SBML, Table S3 text, taxon settings, prompts, and evaluation rules were held fixed. The four conditions are MODEL2501150001 with and without Table S3, MODEL2503190002, and MODEL2506050001. These are descriptive results from three runs, not confidence intervals.

| Condition (model size) | Run 1 | Run 2 | Run 3 |
|---|---:|---:|---:|
| 115 SBML only (47 species, 51 reactions): expected components recovered | 105/113 | 83/113 | 82/113 |
| 115 SBML only: species with **all expected** components | 40/46 | 20/46 | 20/46 |
| 115 + Table S3: expected components recovered; extra predictions | 113/113; 7 | 113/113; 1 | 113/113; 1 |
| 319 (31 species, 40 reactions): species with **exactly** the expected identity set | 27/28 | 28/28 | 28/28 |
| 605 (165 qualitative species, 165 transitions): deposited Entrez IDs recovered | 163/245 | 140/245 | 150/245 |
| 605: supported predictions on mapped nodes | 171/183 (93.4%) | 147/160 (91.9%) | 156/168 (92.9%) |

“All expected” in 115 allows extra components, which are counted separately; “exactly” in 319 does not. The 605 denominator is predicted human/mouse/rat Entrez rows on 51 nodes with deposited gene mappings, not all 165 nodes. Verified orthologs missing from the deposited mapping count as supported, not wrong.

## What changed, and what did not

- **Publication context is decisive for 115.** With Table S3, all 113 expected components were recovered in every run. The extra GRB2 in `EG2SOS` persisted in **3/3** runs; extra RAS and GTP in three BRAF-containing dimers appeared only in run 1. Without Table S3, EGF was missing from `E`, `EG2`, `EG2SOS`, `mEL`, and `mELmEL` in **3/3** runs. The large recall swing came mainly from GTP being omitted in numerous Ras-containing complexes in runs 2–3, not from a change to chemical source-name validation.
- **319 is mostly stable.** Only `CK2` varied at the scored identity level: run 1 added CSNK2A2 and CSNK2B beyond the operational CSNK2A1 reference; runs 2–3 returned the reference identity alone. The three process/state species remained unannotated.
- **605 has meaningful gene-identity variance.** `CyclinA` and `CyclinA_mRNA` returned CCNA2 rather than deposited CCNA1 in **3/3** runs. `Cdh1` returned E-cadherin CDH1 rather than FZR1 in **2/3**; the correct FZR1 was in the retrieved pool even when ranking chose CDH1. `pAPC` added CDC20 in one run, CDC27 in another, and neither in the third. A nonreference PODXL2 prediction for `GF_High` occurred once. These are assessed against the deposited Entrez mapping, which is incomplete for other model nodes.

Runs 2–3 include a stage trace. Between those two runs, normalized names/types/components changed for **36/47** species in 115 SBML-only, **28/47** in 115 + Table S3, **15/31** in 319, and **138/165** in 605. The retrieved candidate pool changed for **12, 0, 2, and 46** species, respectively. Across all three runs, the final *biological identity set* varied for **21, 3, 1, and 43** species, respectively. Thus synonym wording often varies without changing candidate identities, but complex decomposition and gene ranking can change the final result. Run 1 predates trace capture, so stage-change counts compare runs 2–3 only.

The second 605 run also exposes an output issue: the ranking LLM abstained on `Ca2p` and `DAG`, which had retrieved gene candidates but are non-gene chemical inputs. The current top-one table drops those rows instead of emitting explicit empty rows. They are retained as ranking abstentions in the [machine-readable variability result](../results/variability_source_fidelity.json), not treated as failed normalization or excluded from this analysis.

The strongest next targets are therefore stable context errors (EGF without Table S3 and Cyclin A/CCNA1) and ambiguous ranking when the correct gene is already retrieved (`Cdh1`/FZR1). Repeating the same workflow did not justify choosing a “best” run or majority-voting predictions as though that were the original top-one method.

Artifacts: [combined results](../results/variability_source_fidelity.json); per-model `*_source_fidelity_repeat02/03_species.csv`, `*_metrics.json`, `*_trace.json`, and `evaluation_source_fidelity_repeat02/03.json` in the [115](../results/MODEL2501150001/), [319](../results/MODEL2503190002/), and [605](../results/MODEL2506050001/) result folders. The original source-fidelity files remain run 1.
