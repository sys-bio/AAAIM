# MODEL2507280001 selected Phase 3 review

## Outcome

The exact validation-selected Phase 3B epoch-1 inference artifact was found in a prior
local checkout and verified byte-for-byte against the committed registry.  It was run
on all 541 model reactions with the selected 100-deep BM25 + trained-bi-encoder RRF.
Phase 3C was prepared as a zero-call preflight for the 264 adequately mapped internal
reactions, but was not called because neither `OPENAI_API_KEY` nor `.env` is present.

Branch and code under evaluation:

- branch: `test/MODEL2507280001-reaction-review`
- commit: `debc52978603532fe8cc25a1c9e1e4b3070df836`
- commit subject: `benchmark: freeze phase 3c validation pilot`
- this commit descends from Phase 3B release commit `ca9a0f69ee23063eff2788485ddf19680f5d4395`

No model, core prediction code, or benchmark methodology was modified.  All new files
are ignored evaluation artifacts in this review directory.

## Access from a fresh clone

The review files are committed on branch `test/MODEL2507280001-reaction-review`.
The portable selected Phase 3B model is published separately as the GitHub Release
asset `aaaim-phase3b-selected-epoch1-inference.zip` under release tag
`model2507280001-reaction-review`; the 399 MB optimizer/resume checkpoint is not
required for inference and is intentionally omitted.

```powershell
git clone https://github.com/sys-bio/AAAIM.git
Set-Location AAAIM
git switch test/MODEL2507280001-reaction-review
gh release download model2507280001-reaction-review --repo sys-bio/AAAIM --pattern aaaim-phase3b-selected-epoch1-inference.zip --dir benchmark/dist
python -m benchmark.scripts.phase3b_release verify-archive --archive benchmark/dist/aaaim-phase3b-selected-epoch1-inference.zip
```

The expected archive SHA-256 is
`3412a3fa546347d8209ab62ca7ef55fb490e148b5f90c25cf33616328a0d0f53`.
After installing the pinned Phase 3B environment, rerun this model with:

```powershell
benchmark\phase3\_phase3b_env\Scripts\python.exe tests\results\MODEL2507280001_reaction_review\run_phase3b_full_review.py --archive benchmark\dist\aaaim-phase3b-selected-epoch1-inference.zip --batch 64
```

## Selected workflow

| Stage | Selected implementation |
|---|---|
| Phase 3A | Direct `gpt-5.6-terra` open-set inference was an earlier development comparator. Its `target_only` variant was retained only for method development, not as the selected final core, and no new Phase 3A call is part of the selected workflow. |
| Phase 3 retrieval | Project-native Okapi BM25 (`k1=1.2`, `b=0.75`) over the frozen 12,312-reaction KEGG catalog, using `phase3-retrieval-query-v1` and `phase3-retrieval-document-v1`; depth 100. |
| Phase 3B | Shared BERT bi-encoder initialized from `BAAI/bge-small-en-v1.5@5e62ea33e012fda8c02802b906664c915ebd1bb1`, fine-tuned with multi-positive InfoNCE, CLS pooling, L2 normalization, and inner-product similarity. Validation selected epoch 1. |
| Fusion | Equal-weight reciprocal-rank fusion of BM25 depth 100 and trained bi-encoder depth 100; `k=60`, one-indexed ranks, zero contribution when absent, ascending KEGG ID tie-break. Selected method: `bm25_trained_epoch1_rrf`. |
| Phase 3C | `grounded_llm_on_every_reaction`: one stateless Responses API request over the fusion Top-10; `gpt-5.6-terra`, reasoning `low`, 2,048 max output tokens, `tools=[]`, `store=false`, strict `GroundedAnnotation`, prompt `phase3c-grounded-prompt-v1`. The model may select only supplied evidence or abstain. |

Phase 3C requires `OPENAI_API_KEY`.  The project optionally loads it from a protected,
gitignored `.env`; neither the process variable nor `.env` was available in this run.

## Checkpoint and artifact audit

The current checkout and all Git object names/history contain neither `best.pt` nor the
inference ZIP.  Both are intentionally ignored.  They do exist at the repository's
documented paths in the prior local checkout:

| Artifact | Local source | Bytes | SHA-256 | Result |
|---|---|---:|---|---|
| Selected full checkpoint | `C:\Users\janis\Documents\software_projects\janisshin\AAAIM\benchmark\phase3\phase3b_full\_checkpoints\best.pt` | 399,385,639 | `8773b04f09916889b74c956e708044fe2c653fa764fe73c422ecc376ecae81c1` | Exact committed selected-checkpoint hash |
| Portable selected inference archive | `C:\Users\janis\Documents\software_projects\janisshin\AAAIM\benchmark\dist\aaaim-phase3b-selected-epoch1-inference.zip` | 111,163,286 | `3412a3fa546347d8209ab62ca7ef55fb490e148b5f90c25cf33616328a0d0f53` | Exact registry hash; archive manifest valid |

A second byte-identical ZIP was also found in the temporary Phase 3 integration assets.
The archive was preferred for inference because it is the intended portable selected
artifact and contains no optimizer, scheduler, scaler, training cursor, or cache.
Restoration reproduced the committed 969-query validation ranking exactly on CUDA,
including ranking digest `660c7ff55d787928050ba963cf4236788de21f661a523a5ebc1c47abbfc2f304`.

Because the exact artifact exists, retraining was neither necessary nor performed.
If it were absent, the current repository can reproduce it without a methodology
change, provided the pinned initializer cache and CUDA environment are available.
The full run is deterministic and takes an empty output directory:

```powershell
python -m venv benchmark\phase3\_phase3b_env
benchmark\phase3\_phase3b_env\Scripts\python.exe -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
benchmark\phase3\_phase3b_env\Scripts\python.exe -m pip install -r benchmark\requirements-phase3b.txt
benchmark\phase3\_phase3b_env\Scripts\python.exe -m benchmark.scripts.phase3_biencoder full --out tests\results\MODEL2507280001_reaction_review\phase3b_reproduction
```

The full trainer deliberately uses `local_files_only=True` at
`benchmark/phase3/phase3b_smoke/_model_cache`; that cache must contain the pinned BGE
snapshot before training.  Inputs and fixed parameters are recorded in
`benchmark/phase3/phase3b_full/{dataset_summary.json,full_training_config.json,initialization.json}`:
3,466 usable train queries, dataset hash
`142c9f0d404531cf2a8c59ddfabdc80a2858dddd4656a1becc1fb5c0ff292a05`, seed
`20260909`, 3 epochs, batch 4, gradient accumulation 2, 1,302 optimizer steps,
learning rate `2e-5`, warmup 131, temperature `0.02`, two BM25 hard negatives and one
random negative per query.  The original run selected epoch 1 and reported 202.4
seconds of training on an RTX 3070.

## MODEL2507280001 results

Scope remained identical to the earlier review: 541 extracted reactions, of which 257
transport, 1 biomass, and 3 model pseudo-reactions are no-calls; 16 internal reactions
have insufficient metabolite mapping; 264 internal reactions are adequately mapped.
Reaction parsing/extraction succeeded for all 541.  Full-catalog candidate generation
succeeded for every reaction, although out-of-scope rankings are not annotations.

Transformation metrics use the same independent review as the BM25 report.  The held-
out supplement is joined only after rankings are frozen.  Water (`C00001`) and proton
(`C00080`) are ignored as bookkeeping, direction is accepted, and raw equations are
retained for manual stoichiometric review.  Other cofactor or donor substitutions are
not automatically accepted.

| Metric on 264 mapped internal reactions | BM25 only | Selected Phase 3B fusion | Change |
|---|---:|---:|---:|
| Strict transformation match @1 | 147 (55.7%) | 182 (68.9%) | +35 (+13.3 pp) |
| Strict transformation recall @3 | 169 (64.0%) | 211 (79.9%) | +42 (+15.9 pp) |
| Strict transformation recall @5 | 180 (68.2%) | 212 (80.3%) | +32 (+12.1 pp) |
| Strict transformation recall @10 | 194 (73.5%) | 215 (81.4%) | +21 (+8.0 pp) |
| Top-1 supported, including EC-backed partial match | 167 (63.3%) | 206 (78.0%) | +39 (+14.8 pp) |

At Top-1, the broader review rubric records 41 BM25 failures corrected by Phase 3B and
2 previously supported BM25 predictions degraded, leaving 165 correct under both and
56 unsupported under both.  Under the strict transformation threshold alone, the
transition is 37 corrected, 2 degraded, 145 correct under both, and 80 incorrect under
both.  Both degraded transformations remain strong candidates inside the fusion
Top-10, so fusion caused no loss of strict Top-10 coverage.

| Phase 3B outcome/stage | Count |
|---|---:|
| Top-1 strict transformation supported | 182 |
| Top-1 EC-backed partial plausible | 24 |
| Candidate-ranking failures (strong candidate in Top-10) | 33 |
| Candidate-retrieval failures by review rubric | 25 |
| No strong transformation candidate in Top-10 | 49 |
| No even weak/plausible transformation candidate in Top-10 | 9 |
| Final-adjudication failures | Not measurable; Phase 3C was not called |

The 49 without a strict Top-10 transformation comprise 24 EC-backed partial Top-1
calls, 16 weak-candidate-only cases, and 9 wrong/retrieval-miss cases.  This preserves
the distinction between imperfect database representation and a genuinely implausible
candidate set.

## Biological/chemical review

Representative corrections are based on participant transformations, not name alone:

- `R_G6PDH2r`: fusion selected KEGG `R00835`, whose G6P + NADP transformation to
  6-phosphogluconolactone + NADPH matches the modeled reaction (strict score 1.0).
- `R_3OAR100`: fusion replaced generic BM25 `R00119` with chain-specific `R04534`;
  the 3-oxo-decanoyl-ACP/NADPH reduction matches the model (strict score 1.0).
- `R_CYTBO3_4pp`: `R11325` matches the ubiquinol/oxygen redox transformation.  Proton
  compartment and multiplicity bookkeeping do not invalidate this call.

The two Top-1 degradations are genuine ranking regressions rather than retrieval loss:

- `R_3OAS160`: fusion Top-1 `R02768` is only partial (0.4167); correct `R04968` remains
  in the Top-10 with score 1.0.
- `R_APSR`: fusion Top-1 `R02021` is only partial (0.4167); correct `R07176` remains in
  the Top-10 with score 1.0.

Representative remaining failures:

- `R_NADK`: Top-1 `R00137` does not model NAD phosphorylation; best Top-10 score is
  only 0.3333.  This is a retrieval failure.
- `R_IPDPS`: `R05884` is biologically related, but KEGG uses reduced ferredoxin whereas
  this model uses NADH.  That donor change is material, so it is not counted as a mere
  proton/water representation difference.
- `R_ADSL2r`: Top-1 `R04640` is a different one-to-one transformation; no plausible
  Top-10 candidate was retrieved.

## Phase 3C preflight

The zero-call preflight materialized 264 requests and their exact Top-10 evidence.
It uses `gpt-5.6-terra`, low reasoning, 2,048 maximum output tokens, no tools, no
storage, and no retries.  Conservative input estimation is 1,730,485 tokens; the
worst-case cost at the committed 2026-09-03 price table is $9.949034.  API calls made:
zero.  To run after securely providing the credential:

```powershell
python tests\results\MODEL2507280001_reaction_review\run_phase3c_model_review.py --execute --confirm-live --max-cost-usd 10.00
```

The explicit live confirmation and cap are operational gates only; they do not change
the selected prompt, evidence, response schema, model, or ranking methodology.  Do not
put the key on the command line; provide `OPENAI_API_KEY` through the environment or a
protected gitignored `.env`.

## Reproduction commands used

```powershell
python -m benchmark.scripts.phase3b_release verify-archive --archive C:\Users\janis\Documents\software_projects\janisshin\AAAIM\benchmark\dist\aaaim-phase3b-selected-epoch1-inference.zip

C:\Users\janis\Documents\software_projects\janisshin\AAAIM\benchmark\phase3\_phase3b_env\Scripts\python.exe -m benchmark.scripts.phase3b_release restore --archive C:\Users\janis\Documents\software_projects\janisshin\AAAIM\benchmark\dist\aaaim-phase3b-selected-epoch1-inference.zip --batch 64

C:\Users\janis\Documents\software_projects\janisshin\AAAIM\benchmark\phase3\_phase3b_env\Scripts\python.exe tests\results\MODEL2507280001_reaction_review\run_phase3b_full_review.py --archive C:\Users\janis\Documents\software_projects\janisshin\AAAIM\benchmark\dist\aaaim-phase3b-selected-epoch1-inference.zip --batch 64

python tests\results\MODEL2507280001_reaction_review\run_phase3c_model_review.py

python -m pytest tests\test_phase3_retrieval.py tests\test_phase3c_grounded.py -q --basetemp tests\results\MODEL2507280001_reaction_review\_pytest_phase3_full
```

Archive restoration reproduced the frozen validation ranking exactly; the targeted test
run passed 32/32 tests.
