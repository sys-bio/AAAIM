# MODEL2507280001 reaction review (Phase 3C branch)

Branch: `test/MODEL2507280001-reaction-review`
Base: `debc52978603532fe8cc25a1c9e1e4b3070df836` (`codex/benchmark-phase-3c`)
SBML model: BioModels `MODEL2507280001`; internal model id `iLG335`

## Scope and outcome

The SBML parses cleanly with 593 species and 541 reactions. No reaction has an
existing KEGG reaction identifier; 265 reactions have an EC annotation. The Phase 3
BM25 full-catalog retriever generated a Top-10 list for all 541 reactions in 9.2 s.

The model contains 261 reactions that should not be treated as ordinary KEGG chemical
transformations: 257 compartment transports, one biomass objective, and three amino-
acid supply pseudo-reactions. BM25 nevertheless returns identifiers for them, but the
identifiers are not valid annotations of the modeled process.

Of 280 remaining internal transformations, 16 lack enough held-out metabolite mapping
for a confident transformation-level review. Among the 264 assessable reactions:

| Result | Count | Rate |
|---|---:|---:|
| Strong transformation match at BM25 Top-1 | 147 | 55.7% |
| Additional biologically plausible Top-1 (partial representation plus matching EC) | 20 | 7.6% |
| Supported/plausible Top-1 total | 167 | 63.3% |
| Strong transformation match somewhere in Top-10 | 194 | 73.5% |
| Strong candidate present but misranked | 47 | 17.8% |
| Only weak/no matching transformation in Top-10 | 50 | 18.9% |

The independent chemistry review ignores only water and proton. It does not forgive
ATP/ADP, NAD(P)(H), phosphate, oxygen, or other material cofactors. The pipeline's
broader default cofactor-ignored score is retained in the detailed CSV for comparison.

## Stage review

| Stage | Result | Assessment |
|---|---:|---|
| SBML parsing/extraction | 541/541 | Succeeded; no parser errors or duplicate reaction ids. |
| Existing reaction annotations | 0 KEGG; 265 EC | EC values were review evidence, not query input. |
| AAAIM species mapping input | 563/593 ChEBI predictions | Existing source-fidelity result; 30 species lack ChEBI output. |
| Held-out metabolite review map | 556/593 KEGG references | Supplement used only after prediction. |
| Database scope | 261/541 out of ordinary KEGG-reaction scope | 257 transports, one biomass objective, three model pseudo-reactions. |
| Metabolite mapping | 16/280 internal reactions insufficient | Primarily Fe-S, murein, lipid/cardiolipin, and model pool species. |
| Full-catalog retrieval | Top-10 returned for 541/541 | Mechanically complete, but 50/264 assessable internal reactions lack a strong Top-10 transformation match. |
| Candidate ranking | 47/264 clear misranks | A strong candidate is present below rank 1. |
| Final prediction | BM25 component only | Phase 3B trained RRF and Phase 3C grounded selection could not run from this checkout. |

## Representative chemistry checks

| Model reaction | BM25 Top-1 | Review |
|---|---|---|
| `R_UAAGDS` | `R02788` | Correct: ATP + UDP-MurNAc-L-Ala-D-Glu + meso-DAP produces ADP, phosphate, and UDP-MurNAc-tripeptide; proton bookkeeping is immaterial. |
| `R_PRAMPC` | `R04640` | Wrong Top-1. `R04037` at rank 2 matches phosphoribosyl-AMP hydrolysis. Ranking failure. |
| `R_EAR160x` | `R00119` | Wrong Top-1 (NADP/nicotinate chemistry). The matching C16 enoyl-ACP reduction `R04969` is rank 9. Ranking failure. |
| `R_NADK` | `R00119` | Wrong transformation; no strong match in Top-10. Retrieval failure. |
| `R_ADK2` | `R11319` | Same EC family is not sufficient: `R11319` is thiamin diphosphate/ADP chemistry, not AMP + inorganic triphosphate. Wrong transformation. |
| `R_COBALT2tps` | `R00165` | Modeled process is compartment transport; `R00165` is protein/alanyl-tRNA chemistry. Database-scope/no-call case. |

## Limitations of this run

- The selected Phase 3B checkpoint and 111 MB inference archive are absent locally;
  the repository registry records that the archive was never uploaded. Therefore the
  trained-neural + BM25 RRF stage cannot infer on this new model.
- `AAAIM_PHASE3C_OPENAI_API_KEY` is unset, so the evidence-grounded Phase 3C selector
  cannot run. Reported identifiers are BM25 component predictions, not complete
  fusion/grounded final annotations.
- The installed parsed KEGG cache drops coefficients. Transformation review therefore
  compares participant presence/direction while retaining raw model and KEGG equations
  in the detailed table for multiplicity review.

## Artifacts and reproduction

- `reaction_review.csv`: one row for every modeled reaction, including equation,
  annotations, Top-1/Top-10, scores, KEGG definition/equation, stage classification,
  and warnings.
- `phase3_bm25_top10_candidates.csv`: all 5,410 candidate rows with BM25 score,
  transformation scores, EC evidence, definition, and equation.
- `summary.json`: machine-readable aggregate.
- `run_phase3_review.py`: reproducible driver; it does not modify model or pipeline.

Run:

```powershell
git switch test/MODEL2507280001-reaction-review
python tests/results/MODEL2507280001_reaction_review/run_phase3_review.py
python -m pytest tests/test_phase2_candidates.py tests/test_phase3_retrieval.py tests/test_phase3c_grounded.py -q --basetemp tests/results/MODEL2507280001_reaction_review/_pytest_core -p no:cacheprovider
```

Focused runnable tests: 77 passed. The broader Phase 3 test command produced 47
passes, one failure, and 11 setup errors because the ignored Phase 3B checkpoint and
Phase 3C smoke response cache are not present; these are the same missing runtime
assets that prevent the full fusion/grounded inference run.
