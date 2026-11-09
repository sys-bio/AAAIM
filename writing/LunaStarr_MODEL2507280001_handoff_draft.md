# Draft handoff: MODEL2507280001 reaction workflow review

Prepared from the committed case-study branch, current upstream main, and a local mocked public-workflow integration check; not sent to LunaStarr. LunaStarr's request was: “I’m working on several fixes based on the use case results and hope our functions are well synced. I’m currently working on these four use cases from non-curated BioModels in https://github.com/sys-bio/AAAIM/tree/main/tests/useCase. Among them there is one chemical reaction model: MODEL2507280001, could you please test and review the reaction prediction of it?”

**Status: partially fulfilled, not yet fully fulfilled.** The separate 541-reaction research review already exists and the current public API path has now been exercised end to end with mocked retrieval and provider functions. A full current public reaction run has not completed; no genuine current-workflow reaction predictions or provider responses were produced. The difference matters because the earlier reaction review used a separate Phase 3 retrieval implementation, not the public species-to-reaction annotation path Luna asked about.

## What was completed

- Parsed all **541/541** model reactions and generated full-catalog retrieval candidates for all 541.
- Archived a BM25 baseline and the selected Phase 3B fusion review: BM25 depth 100 plus the validation-selected epoch-1 trained bi-encoder, equal-weight RRF (`k=60`), Top-10 evidence review.
- Prepared **264** Phase 3C grounded-selector requests and a zero-call preflight. **No Phase 3C provider calls were made**; the archived report records that the API key was unavailable.
- No reaction model or annotation method was changed for this review.
- Read the current public API documentation and code. `annotate="reactions"` accepts a species recommendations DataFrame/CSV or reads ChEBI annotations from SBML; the available `full_species.csv` supplies 552 ChEBI recommendations from the existing public species-use-case run.
- The current SBML contains **zero species ChEBI annotations**, so reaction annotation must use an explicit recommendation input or rerun species annotation. The cached recommendation CSV is the appropriate available input; choosing `annotate="both"` would unnecessarily rerun species and is outside the requested reaction-only check.
- Exercised the public `annotate_model(..., annotate="reactions")` entry point on the actual SBML and 552-row species CSV, mocking only the local candidate-retrieval return and provider responses. It preserved reaction ID `R_UAAGDS`, returned the mocked KEGG candidate, populated both result tables, and serialized/reloaded the reaction CSV. This validates plumbing, not prediction quality.
- Historical note (superseded by the 2026-11-07 update below): the first local full-path dry run was stopped before candidate-generation completion. The bounded profile and later full local candidate run now provide exact coverage, runtime, request manifest, and cost estimates.
- Found and fixed a narrow API pass-through defect in the isolated local worktree: `max_completion_tokens` was accepted by `annotate_model` but dropped before reaction ranking. It now reaches both the model-meaningfulness prompt and reaction-ranking prompts. Regression test coverage verifies the ranking call receives the requested ceiling; an integration smoke test verified the public entry point sends the ceiling to both mocked requests.

## Inputs and provenance

| Input | Repository path | Bytes | SHA-256 |
|---|---|---:|---|
| SBML source (`iLG335`) | `tests/useCase/MODEL2507280001.xml` | 1,043,229 | `d4312d941b997ef5097f58da324a397a7f468e6f170a82d99fa24e7a03689ee3` |
| Publication supplement | `tests/useCase/context/publications/MODEL2507280001/supplement.xlsx` | 90,576 | `76aca5055f0e737cd1c0c037a8b93a800126c9948d497f4c07002d42bd9ef8d9` |
| Existing public species recommendations used in the mocked path check | `tests/useCase/results/MODEL2507280001/full_species.csv` | 131,440 | `94133c0e356b99f1d5a4a6790866a54ee3d1182ce80851eeb35a2864cf7a1417` |

The BioModels accession is recorded as `MODEL2507280001`; neither this checkout nor the archived report states a BioModels revision label. The current checkout's CRLF working-tree representation differs bytewise from the historical Git blob (`223d70f28000b979019125563cee4b190d0002f1d437f42fcb1b93de9538c444`, 1,022,825 bytes), but the files are identical after normalizing CRLF to LF and their parsed model contents match (593 species, 541 reaction IDs, no parsed species/reaction differences). The current raw working-tree checksum above identifies the file used for this task. The publication supplement was used only after rankings were frozen to support metabolite-reference review.

The source branch is `test/MODEL2507280001-reaction-review`, archived at `27cfe4f243d658bb5fc9391a95ada90c11e42bbf`. The review report identifies Phase 3C branch/code revision `debc529786035032fe8cc25a1c9e1e4b3070df836`; the selected Phase 3B release commit is `ca9a0f69ee23063eff2788485ddf19680f5d4395`. Portable selected inference archive SHA-256: `3412a3fa546347d8209ab62ca7ef55fb490e148b5f90c25cf33616328a0d0f53`.

The pinned inference archive is published as the GitHub release asset [aaaim-phase3b-selected-epoch1-inference.zip](https://github.com/sys-bio/AAAIM/releases/tag/model2507280001-reaction-review). The 399 MB optimizer/resume checkpoint is not needed for inference and was intentionally omitted.

## Scope and descriptive findings

The model has 593 species and 541 reactions. There were no existing KEGG reaction IDs and 265 EC annotations. The review classified **257 transport reactions, one biomass reaction and three model pseudo-reactions** as outside ordinary KEGG reaction-annotation scope. Of 280 remaining internal transformations, **16** did not have enough mapped metabolites for confident transformation review. Thus the transformation assessment covers **264** adequately mapped internal reactions.

| Transformation review on 264 assessable reactions | BM25 | Selected Phase 3B fusion |
|---|---:|---:|
| Strict transformation match at Top-1 | 147 (55.7%) | 182 (68.9%) |
| Strict transformation present by rank 3 | 169 (64.0%) | 211 (79.9%) |
| Strict transformation present by rank 5 | 180 (68.2%) | 212 (80.3%) |
| Strict transformation present by rank 10 | 194 (73.5%) | 215 (81.4%) |
| Top-1 supported including EC-backed partial representation | 167 (63.3%) | 206 (78.0%) |

These are rubric-based transformation support counts, not accuracy against an independently established reaction-level gold standard. The supplement provides metabolite references, not independent gold labels for every reaction. Water and proton bookkeeping were ignored in the strict review; other donor/cofactor substitutions were not automatically accepted. Raw equations were retained for stoichiometric inspection.

Within the selected fusion Top-10, 33 strong candidates were ranked below Top-1. Twenty-five cases were classified as retrieval failures under the review rubric. Forty-nine cases lacked a strict Top-10 transformation: 24 had an EC-backed partial Top-1, 16 had weak-candidate-only evidence, and nine had no plausible Top-10 transformation candidate. Fusion corrected many generic BM25 rankings, while two previous strict Top-1 transformations were degraded; in both cases the supported transformation remained in the fusion Top-10.

No-call and limitation categories should stay separate: **261/541** reactions were out of ordinary KEGG reaction scope; **16/280** internal reactions were under-mapped; and the remaining internal cases were assessed by transformation evidence. Candidate generation success for all 541 is mechanical coverage, not evidence that every candidate is an annotation. There was no grounded-model abstention result because Phase 3C was not executed.

## Limitations and independence

- This was not an independent external evaluation: `MODEL2507280001` already appeared in the repository's species-use-case inputs, reports and results during workflow development.
- The selected trained model and inference archive were available from the prior local checkout and verified against the committed archive digest; the current checkout does not contain the archive by default.
- Transformation support was assessed through the recorded chemical rubric, not an independent reaction gold standard. Do not label the percentages “accuracy.”
- Phase 3C prepared 264 requests, but no paid calls, grounded selections, abstentions or final annotations were produced.
- Runtime recorded for the full selected fusion review was 32.635 seconds on CUDA, batch size 64. BM25 candidate generation took about 9.2 seconds in the earlier review. No new provider cost was incurred for this handoff; costs of prior species-side work are not available in these reaction reports.
- This case study is separate from the sealed held-out benchmark; it was not selected as an independent test case and does not unseal or replace that evaluation.
- The earlier support counts (147/264 BM25 and 182/264 Phase 3B fusion strict Top-1; 264 is the assessable subset) are from the existing human rubric review, not accuracy against independent reaction labels. The repository report does not establish an independent expert reviewer or ground truth. Do not infer biological judgments from the mocked public-workflow output.
- The archived report describes manual stoichiometric review under a stated chemical rubric and says its assessment matches an earlier BM25 review; it does not name the human reviewer or establish that reviewer as independent from system development. Phase 3C was only preflighted, with zero grounded selections, abstentions, or predictions.
- Representative existing rubric findings (inherited labels from the archived review; no new biological judgments):

| Existing case | Archived status | Meaning for this request |
|---|---|---|
| `R_UAAGDS` / `R02788` | Fusion Top-1 strict transformation supported; warning: 1/7 participants lack AAAIM ChEBI evidence | Supported-looking prior research-pipeline result; not a public-workflow prediction |
| `R_3OAS160` | Top-1 degraded to partial `R02768`; supported `R04968` remained in Top-10 | Ambiguous/ranking regression to inspect if public output is later produced |
| `R_I2FE2SS2fdx` | Insufficient mapping, 2/8 participant occurrences | Mapping-limited; not evidence of a ranking failure |
| `R_COBALT2tps` | Transport no-call | Outside ordinary KEGG transformation review scope |
| `R_BIOMASS` | Biomass no-call | Special objective reaction, not a pipeline failure |
| `R_AA_providing_adult` | Model pseudo-reaction no-call | Special model construct, not a pipeline failure |

No genuine public-workflow failure case can yet be sampled because current public candidate generation did not complete. These are inherited report classifications, not new judgments.
- Historical note (superseded by the update below): the prompt count and cost were unknown before candidate generation completed. GPT-6.1 Terra is an explicit override of the prior species run model. The completion-token ceiling now reaches both reaction request stages.

## Files and reproduction

Committed outputs on `test/MODEL2507280001-reaction-review`:

- `tests/results/MODEL2507280001_reaction_review/PHASE3_FULL_REPORT.md` — selected Phase 3B fusion results, chemistry assessment and preflight record.
- `tests/results/MODEL2507280001_reaction_review/phase3b_fusion_reaction_review.csv` — per-reaction results.
- `tests/results/MODEL2507280001_reaction_review/phase3b_fusion_top10_candidates.csv` — Top-10 evidence rows.
- `tests/results/MODEL2507280001_reaction_review/phase3b_full_summary.json` — machine-readable aggregate, representative cases and runtime.
- `tests/results/MODEL2507280001_reaction_review/phase3b_artifact_provenance.json` — release archive digest and source checkpoint provenance.
- `tests/results/MODEL2507280001_reaction_review/phase3c_preflight_summary.json` and `phase3c_preflight_requests.jsonl` — prepared requests only; no responses.

Local verification on the isolated `codex/luna-public-workflow` worktree at upstream commit `866be205d45eb7c3f7eb1ab3be05e97af1dfb311`:

```powershell
python -m pytest tests/test_pr13_integration.py tests/test_llm_usage.py -q
```

Historical verification at the time: **5 passed**. A separate inline smoke test used the actual model and species CSV with mocked local retrieval and provider methods. The later 2026-11-07 update records the completed 53-test run and full local candidate-generation result.

The public workflow command to reproduce after local candidate-generation performance is resolved and a provider run is separately authorized is:

```python
from core.annotation_workflow import annotate_model

result = annotate_model(
    model_file="tests/useCase/MODEL2507280001.xml",
    annotate="reactions",
    species_recommendations_df=(
        "tests/useCase/results/MODEL2507280001/full_species.csv"
    ),
    llm_model="gpt-6.1-terra",  # explicit override of public species-run model
    method="rulebased",
    n_return=3,  # public default; evaluate_candidates remains False
    max_completion_tokens=512,
    save_to="tests/useCase/results/MODEL2507280001/public_reaction_gpt61",
)
```

Historical estimate text (superseded by the request manifest and cost proposal below): it was written before complete prompt payloads were generated. Identical payloads also cannot eliminate hosted-model version/temporal behavior variation. [Model/API support](https://developers.openai.com/api/docs/models/gpt-6.1-terra) · [API pricing](https://developers.openai.com/api/docs/pricing).

From a clone, switch to the case-study branch, install the pinned Phase 3B environment, download the release archive, and verify it before inference:

```powershell
git switch test/MODEL2507280001-reaction-review
gh release download model2507280001-reaction-review --repo sys-bio/AAAIM --pattern aaaim-phase3b-selected-epoch1-inference.zip --dir benchmark/dist
python -m benchmark.scripts.phase3b_release verify-archive --archive benchmark/dist/aaaim-phase3b-selected-epoch1-inference.zip
benchmark\phase3\_phase3b_env\Scripts\python.exe tests\results\MODEL2507280001_reaction_review\run_phase3b_full_review.py --archive benchmark\dist\aaaim-phase3b-selected-epoch1-inference.zip --batch 64
python tests\results\MODEL2507280001_reaction_review\run_phase3c_model_review.py
```

The last command is a zero-call preflight by default. To finish Luna's public-workflow request, first profile/complete the local ChEBI-to-KEGG candidate-generation stage and resolve the doc/code call-count discrepancy; then dry-run and archive the exact GPT-6.1 Terra request manifest and estimate using the fixed output ceiling. No paid inference is authorized here. After separate authorization, run the public workflow and review its actual candidate membership, mapped participants, outputs, and no-calls against the existing rubric evidence. Do not substitute the 264 Phase 3C preflight or call its responses public-workflow results.

The case-study archive reports 32/32 targeted tests passed under its recorded environment. These are historical results from the case-study branch, not a fresh run in the integration worktree.

## Update: completed local candidate generation and request dry run — 2026-11-07

The initial slow run was profiled and then completed locally in the isolated current-upstream worktree at `benchmark/integration_worktrees/luna_public` (branch `codex/luna-public-workflow`, base commit `866be205d45eb7c3f7eb1ab3be05e97af1dfb311`). The full stage took 150.451 seconds, with 138.691 seconds in catalog filtering/ontology expansion; it performed 1,797 filter calls and 21,321 unique cached side expansions, peaking near 448 MB RSS. The justified optimization reuses equivalent ontology expansions within the run. Focused comparisons preserved candidate membership, ordering, scores, and metadata over empty, ambiguous, and large fixtures. The isolated code also fixes positional reaction-ID misalignment when SBML extraction omits a reaction and forwards `max_completion_tokens` to both ranking stages.

The recommendation table has 3,789 rows, but 296 are blank placeholder rows, not candidates. There are 3,493 nonempty candidate rows for 222 reactions. Of 541 model reactions, 518 were extracted; 23 lacked mapped participant species. Another 296 extracted reactions had no nonempty candidate. Thus 319 have zero candidates. Per-reaction candidate-count distribution (including zeros) is: 0:319, 1:146, 2:32, 3:2, 4:1, 5:3, 6:1, 7:4, 8:3, 9:6, 10:2, 11:3, 12:1, 13:3, 15:5, 18:1, 19:2, 28:3, 112:1, 443:1, 1,136:2. These are mechanical candidate counts, not biological judgments.

The actual ranker prompt-construction path yielded 223 requests per conditional branch: one model-meaningfulness prompt and 222 reaction rankings. The TRUE branch includes the complete model context in reaction prompts; the FALSE branch omits it. Both were captured with a no-call recorder; synthetic TRUE/FALSE returns only constructed mutually exclusive prompt plans and are not real outputs. Complete payload JSONL files and per-payload SHA-256 digests are retained with ignored local artifacts in `benchmark/integration_worktrees/luna_public/tests/useCase/results/MODEL2507280001/public_reaction_audit/`.

With 512 maximum completion tokens, conservative input upper bound (UTF-8 text bytes plus 256 framing tokens per request), GPT-6.1 Terra Standard pricing, and the code's maximum six attempts/request, the TRUE branch is estimated at $46.13 uncached or $110.19 under 12-hour cache-write pricing. Even its one-attempt 12-hour cache-write bound is $18.37. The FALSE branch's six-attempt bounds are $7.94 uncached and $14.71 under 12-hour cache writes. These estimates pessimistically reserve failed/uncertain attempts. The approximately $20 balance was not account-verified; no paid call was made. The default retry policy is not suitable for a safely bounded TRUE-branch execution. Any execution needs verified available credit and an approved hard cap plus a guarded zero-retry or regenerated compact-context plan.

**Current fulfillment status:** the local candidate-generation and request-planning phases are complete. The actual GPT-6.1 Terra inference and biological review remain unperformed; the public-workflow request is therefore still partially fulfilled. No held-out data or external biological resources were accessed. The current targeted test suite passed 53 tests. See the ignored audit directory's README, candidate report, request manifest, and cost proposal for reproduction and detailed provenance.

## Correction: selected research retrieval analysis — 2026-11-07

This section supersedes earlier statements in this draft that local candidate generation or selected-retrieval preparation remained outstanding. Both paths are now documented separately below.

The preceding public-workflow diagnostics use ChEBI-to-KEGG rule-based candidate generation. Their 222 candidate-bearing / 319 zero-candidate counts describe that workflow only. They are not exclusions or coverage counts for the selected research retriever.

The separate selected-method analysis is now prepared in the isolated `codex/luna-public-workflow` worktree at `tests/useCase/results/MODEL2507280001/selected_rrf_analysis/` (ignored local artifacts). It uses the frozen Phase 3B BM25 plus validation-selected epoch-1 trained bi-encoder, equal-weight reciprocal-rank fusion (`k=60`), full 12,312-reaction catalog, and Top-10 KEGG evidence. It rebuilt queries for all 541 reactions from the historical archived species recommendation input and current SBML. The SBML is identical to the historical source after CRLF normalization; all 541 rebuilt queries match the archived ranking queries exactly. Component depth, model/tokenizer/checkpoint provenance, full RRF ranking order, and evidence order/scores are checked. Existing compatible retrieval outputs were reused; GPU inference was not rerun.

All 541 cases received Top-10 evidence and are eligible for a grounded request; no rule-based no-candidate cases were dropped. The prior single-team review's 264 assessable internal transformations, including descriptive support counts of 182/264 strict Top-1 and 215/264 strict Top-10, remain historical reviewer judgments rather than independent gold accuracy. Public-vs-RRF candidate overlap is a method comparison only (Top-1 ID agreement 81/222 among the public candidate-bearing cases), not a correctness result.

No GPT predictions or abstentions have been generated. The no-call request plan uses GPT-6.1 Terra, Responses API, low reasoning, strict `GroundedAnnotation` schema, maximum output 512 tokens, no tools, no storage, and default service tier. It requires 541 new calls if separately authorized; no compatible response cache exists. The pessimistic estimated reservation is USD 10.855272500, proposed cap USD 12.00, conditional on the account dashboard verifying available funds. The approximately USD 20 user-reported credit was not verified. No authorization or paid calls are implied. The plan has persistent unit-specific cache keys, one attempt per case, no automatic retries, reservation-before-send accounting, and retains reservations after failed or uncertain attempts; only verified billing settles to actual cost. Hosted model/version drift remains possible across time.

The selected retrieval and request-preparation stage is complete, but Luna's review request is not fully fulfilled until any authorized grounded predictions are produced and their biological outputs are reviewed. The no-call artifacts include `REPORT.md`, `validation_report.json`, `retrieval_method.json`, `coverage_541.csv`, `top10_evidence.jsonl`, `requests.jsonl`, `request_manifest.jsonl`, `provenance_sidecars.jsonl`, `dry_run_report.json`, and `REPRODUCE.md`. Public-rule-based findings remain separately documented above.

## Output-budget reassessment — 2026-11-07

The current proposal prefers **2,048 total output tokens** for GPT-6.1 Terra. This supersedes the earlier 512-token execution recommendation while preserving its original request plan and results. Reasoning, structured answer text, and non-visible formatting all consume this limit. Official documentation states that exhaustion can return `status=incomplete` and `incomplete_details.reason=max_output_tokens`, possibly with no visible answer and with billable usage.

The 163 unique GPT-5.6 Terra grounded validation responses used a 2,048 limit. Their total output tokens (already including reasoning) were: minimum 81, median 116, P90 227.6, P95 279.7, P99 361.12, maximum 386. None reached 450 or exceeded 512. Reasoning alone had median 0, P95 145.7, and maximum 262. There were 161 normalized successful records and two evidence-compliance errors; all 163 passed structural schema validation and had nonempty answers. No incomplete/truncated/empty/schema-invalid response is recorded, but the historical artifacts omit original provider status/incomplete details. These observations do not validate a reduced limit for GPT-6.1 Terra; no relevant measured grounded responses from that model are available.

Both options use the same frozen 541 units, retrieval/evidence, GPT-6.1 Terra, low reasoning, strict GroundedAnnotation schema, default Standard service tier, no tools, no storage, and zero automatic retries. Complete 512-token provider requests remain identical to the prior plan; every 2,048-token request changes only `max_output_tokens` and receives a new unit cache identity.

| Output limit | Heuristic expected cost | Conservative retained reservation | Proposed hard cap |
|---|---:|---:|---:|
| 512 | $2.378835–$2.875583 | $10.855272500 | $12 |
| 2,048 (preferred) | $2.378835–$2.875583 | $15.010828750 | $16 |

Expected cost transfers the historical GPT-5.6 input-token/complete-request-byte ratio and mean total output (144.8589 tokens) to this different model/case, with ordinary through all-30-minute-cache-write input billing and no cache-read discount. It is a scenario, not measured GPT-6.1 usage. Both output limits have the same expected-input/output assumption; maximum output is charged only when generated. Reservation per request is `(complete canonical provider body UTF-8 bytes + 4,096 framing tokens) × $1.25/M + maximum total output × $5/M`, rounded upward per request. This is a conservative token-bound assumption, not a provider-certified exact input count. All bounded requests are short context; no full-context-capacity reservation is used. Default 30-minute cache writes replace ordinary input billing. Pricing assumes the global Standard endpoint and excludes regional/FedRAMP uplifts, taxes, foreign exchange, unrelated account use, tools, extended cache retention, and alternative tiers. Reprice/re-authorize if these assumptions change.

The user reports **$19.97** current credit; no connected read-only billing mechanism can independently verify it. A $16 cap would leave **$3.97 unallocated for later work** if that balance is still current. Phase 5 remains paused and receives no allocation. Both caps cover all 541 attempts even with every reservation retained. Failed, invalid, truncated, or uncertain attempts retain their full reservation; only valid completed outputs with separately verified billing can settle downward. Failure statuses remain separate from biological abstentions, and all reactions remain in coverage. The durable journal blocks resends after interruption and reconstructs derived response caches when needed.

The current ignored proposal is at `benchmark/integration_worktrees/luna_public/tests/useCase/results/MODEL2507280001/selected_rrf_output_budget_20261107/EXECUTION_PROPOSAL.md`; the exact old plan is copied into `prior_512_plan_v1/`. Proposed authorization identities:

- Preferred 2,048-token / $16-cap plan SHA-256: `ce8ea533e1b435627eb2302add49c74a9e16ce33d5c8b3e53e84bf48049e7d4f`.
- 512-token / $12-cap comparison plan SHA-256: `c5773ca60ea0b17cb916adae3060c06f82623899bb1e81b5cbbfdf3f3915d59b`.

Neither plan is authorized. No inference, paid calls, retrieval reruns, held-out access, commits, or pushes occurred during this reassessment. Hosted-model version/time drift remains a limitation. See official [reasoning/output limits](https://developers.openai.com/api/docs/guides/reasoning), [output usage](https://developers.openai.com/api/docs/guides/token-counting), [pricing](https://developers.openai.com/api/docs/pricing), and [cache billing](https://developers.openai.com/api/docs/guides/prompt-caching).

Verification completed: 110 relevant tests passed, including synthetic incomplete/empty/schema-invalid handling, valid abstention separation, denial before authorization, one-attempt behavior, interruption/resume, cache reconstruction, and both actual 541-unit plans with all reservations retained. The prior-plan, Phase 3C smoke/validation, Phase 3B fusion and inference-archive verifiers passed; original source/cache checksums remain unchanged. Five byte-identical historical smoke-cache entries were staged in the isolated worktree solely to satisfy existing test prerequisites. `git diff --check` passed. Full results are in the ignored proposal directory's `verification_report.json`.

## Branch consolidation — 2026-11-07

The canonical task branch is now `feature/luna-model2507280001`, renamed from `codex/luna-public-workflow`. The physical `benchmark/integration_worktrees/luna_public` worktree and its ignored reports, requests, caches, plan digests and budget proposals are unchanged. The 2,048-token / USD 16 proposal still requires separate authorization. Canonical research phases remain `benchmark/phase-3c`, `benchmark/phase-4` and `benchmark/phase-5`; six local integration duplicates were archived after comparison. Published upstream review branches remain untouched. See `benchmark/BRANCHES.md`.
