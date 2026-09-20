# MODEL2501150001 — validation run

This human EGFR–ERK signaling model has **47 species and 51 reactions**. Both runs use GPT-5 mini, direct retrieval (`top_k=3`), and one final identity per species or complex component. The augmented condition supplies publication Table S3 names; the SBML-only condition does not. Accuracy below compares predicted components with the 46 SBML species aligned to Table S3 (113 expected components).

| Condition | Baseline complete species | Validation complete species | Baseline / validation components recovered |
|---|---:|---:|---:|
| SBML only | 23/46 | 21/46 | 90/113 → 85/113 |
| SBML + Table S3 | 41/46 | **46/46** | 108/113 → **113/113** |

Validation rerouted five EGF components from ChEBI to human UniProt **P01133** in the Table S3 run. This fixes the systematic EGF identity error there, but the SBML-only run still lacks enough context to identify five EGF components and performs slightly worse overall. The extra GRB2 prediction below is not counted among the 113 expected components.

| Condition / species | Wrong top-1 prediction | Publication Table S3 identity | Issue |
|---|---|---|---|
| Both: `EG2SOS` | GRB2, UniProt P62993 | EGF–EGFR–SOS (no GRB2) | Extra complex component |
| SBML only: `BRAF_dimer`, `iBRAF_dimer`, `BRAF_iBRAF_dimer` | RAS, UniProt P01116 | BRAF-containing dimers | Extra/misparsed component |

The Table S3 result supports the routing fix; one run per condition does not isolate its effect from LLM variability. The SBML-only decomposition and inhibitor-bound complex interpretation merit review. The second ranking LLM call cannot rescue an EGF candidate sent only to ChEBI; the cross-database check is the relevant guard.

Artifacts: [SBML-only predictions](../results/MODEL2501150001/sbml_only_validation_species.csv), [Table S3 predictions](../results/MODEL2501150001/table_s3_validation_species.csv), [evaluation](../results/MODEL2501150001/evaluation_validation.json), [saved Table S3](../context/MODEL2501150001_TableS3.txt).
