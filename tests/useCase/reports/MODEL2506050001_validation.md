# MODEL2506050001 — validation run

This hypoxia/EMT qualitative model has **165 qualitative species and 165 transitions**. It forces gene identity and NCBI Gene retrieval for human, mouse, and rat, using GPT-5 mini, direct retrieval (`top_k=3`), and one final biological identity per species/component; that identity may yield three Entrez rows. The publication's `NodeGenes` lists are incomplete across taxa, so verified orthologs omitted from those lists are counted separately.

| Measure | Baseline | Validation |
|---|---:|---:|
| Species with candidates | 160/165 | 154/165 |
| Exact deposited Entrez IDs recovered | 145/245 | **151/245** |
| Supported Entrez predictions on 51 mapped nodes | 153/168 (91.1%) | **159/171 (93.0%)** |

Eight validation predictions are valid requested-taxon orthologs absent from `NodeGenes`. Structured output improved supported precision in this single run but reduced overall coverage; it did not fix the following identity mismatches.

| Model node(s) | Wrong top-1 prediction, human / mouse / rat Entrez IDs | Publication reference/correct identity, human / mouse / rat Entrez IDs | Assessment |
|---|---|---|---|
| `CyclinA`, `CyclinA_mRNA` | CCNA2: 890 / 12428 / 114494 | CCNA1: 8900 / 12427 / 295052 | “Cyclin A” is ambiguous, but the deposited mapping specifies A1. |
| `pAPC` | CDC27: 996 / 217232 / 360643 | ANAPC1: 64682 / 17222 / 311412; ANAPC2: 29882 / 99152 / 296558 | CDC27 is an APC/C subunit, but not one of the deposited components. |
| `Cdh1` | CDH1: 999 / 12550 / 83502 | FZR1: 51343 / 56371 / 314642 | Homonym: the model means APC/C co-activator Cdh1/FZR1, not E-cadherin. |

This is a top-one identity test, not full recall of every gene in a family/complex. Eleven species receive no candidate, and most model nodes lack a deposited gene reference; predictions for those nodes are not included in identifier-level precision. The ranking LLM can choose among retrieved genes but does not independently verify their function or resolve an ambiguous model name without contextual evidence.

Artifacts: [predictions](../results/MODEL2506050001/full_validation_species.csv), [evaluation](../results/MODEL2506050001/evaluation_validation.json), [source mapping](../context/publications/MODEL2506050001/Hypoxia_EMT_Model.dmms).
