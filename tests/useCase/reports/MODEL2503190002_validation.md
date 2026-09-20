# MODEL2503190002 — validation run

This human DNA-damage-response model has **31 species and 40 reactions**. It uses GPT-5 mini, direct ChEBI/UniProt retrieval (`top_k=3`), and one final identity per species or complex component. Predictions were reviewed against the publication and supplement.

| Measure | Baseline | Validation |
|---|---:|---:|
| Species with candidates | 28/31 | 28/31 |
| Molecular species with complete supported top-1 identity | 28/28 | **28/28** |
| Process/state species without a molecular ID | 3/3 | 3/3 |

**Wrong predictions:** none among the 28 molecular species. `DNA_damage`, `Autophagy_active`, and `Autophagy_inactive` remain appropriately unannotated as molecular entities. The PP2A–I2PP2A complex retains separately ranked components. PP2A and CK2 still resolve to representative catalytic subunits rather than complete families; that interpretation should be stated explicitly if used in the manuscript. Validation did not change measured accuracy here.

Artifacts: [predictions](../results/MODEL2503190002/full_validation_species.csv), [evaluation](../results/MODEL2503190002/evaluation_validation.json), [saved supplement](../context/publications/MODEL2503190002/supplement.pdf).
