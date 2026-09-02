# Phase 2 results: candidate retrieval and baseline ranking

Frozen candidate table, retrieval ceiling and baseline rankers over the Phase 1 corpus.

- **Source commit**: see `benchmark/PHASE2_MANIFEST.json` (`source_commit`)
- **Generation config**: `config_id = 86938b48ab88`
- **Phase 1 input**: `benchmark-phase1-v1`, `reactions.csv` = `f11ebc6f…6004f1`
- **Audit**: 14/14 invariants pass, zero pipeline failures (`benchmark/data/phase2_audit.json`)

## Headline

| Quantity | Value | Population |
| --- | --- | --- |
| Reactions with zero candidates | **59.44%** (3,457/5,816) | all evaluable |
| Exact recall at any rank | **34.71%** (2,019/5,816) | all evaluable |
| Overall exact Top-1 (heuristic) | **33.25%** (1,934/5,816) | all evaluable |
| Overall exact retrieval failure | **65.29%** (3,797/5,816) | all evaluable |
| Conditional heuristic Top-1 | **81.98%** (1,934/2,359) | nonempty candidate sets |
| Conditional oracle Top-1 | **85.59%** (2,019/2,359) | nonempty candidate sets |
| **Max gain from perfect reranking** | **+1.46 pp** (85/5,816) | all evaluable |

Retrieval, not ranking, is the bottleneck. The shipped heuristic already places the
correct answer first in 95.8% of the cases where it is reachable at all. Replacing it
with a perfect reranker would move overall Top-1 from 33.25% to 34.71%.

> **Read the denominators.** The conditional 81.98% is accuracy over the 40.6% of
> reactions that have a candidate set. It is *not* benchmark performance. The
> corresponding overall figure is 33.25%.

## 1. Corpus and status counts

| | Count |
| --- | --- |
| Models | 74 |
| Clusters (Phase 1 conservative) | 54 |
| Ground-truth reactions | 5,838 |
| Excluded as exchange/sink/source (SSX) | 22 |
| **Evaluable reactions** | **5,816** |
| Candidate rows | 91,802 |
| Distinct KEGG reaction ids proposed | 7,761 |

| Status | Reactions | Share | Candidate rows |
| --- | --- | --- | --- |
| `ok` | 2,359 | 40.56% | 91,802 |
| `no_candidates` | 2,646 | 45.50% | 0 |
| `unconstrained_candidate_set` | 811 | 13.94% | 0 |
| `generation_failed` / `absent_from_generator_output` | 0 | 0% | — |

`unconstrained_candidate_set` marks reactions where no participant mapped to a KEGG
compound. `filter_kegg_reactions` keeps a KEGG reaction when the model's mapped compound
keys are a *subset* of the KEGG reaction's keys, and the empty set is a subset of
everything, so all 12,312 KEGG reactions "match". That is the absence of retrieval rather
than a candidate set, so no rows are stored. Counting them as retrieval failures (which
the overall metrics do) is the honest treatment.

## 2. Unconditional retrieval metrics (all 5,816 evaluable reactions)

Exact matching:

| Metric | Reaction-micro | Model-macro | Cluster-macro |
| --- | --- | --- | --- |
| Recall @ any rank | 0.3471 | 0.4375 | 0.4244 |
| Recall @ 1 | 0.3325 | 0.3549 | 0.3586 |
| Recall @ 3 | 0.3428 | 0.4251 | 0.4122 |
| Recall @ 5 | 0.3437 | 0.4300 | 0.4163 |
| Recall @ 10 | 0.3437 | 0.4300 | 0.4163 |

Model-macro and cluster-macro exceed micro by roughly 9 pp because the seven genome-scale
models carry 78.3% of the reactions and retrieve worst. Micro is the pessimistic view and
the one to quote for "how well does this work on a reaction drawn from this corpus".

### Exact versus equivalence-aware (reaction-micro, all evaluable)

| Criterion | Recall @ any rank | Recall @ 1 |
| --- | --- | --- |
| Exact | 0.3471 | 0.3325 |
| EC number | 0.3504 | 0.3423 |
| KO | 0.3496 | 0.3401 |
| BRITE orthology | 0.3514 | 0.3427 |
| RCLASS | 0.3765 | 0.3556 |

Equivalence-awareness adds very little at any rank (+0.4 pp for BRITE, +2.9 pp for the
looser RCLASS criterion). This is expected: when the candidate set is empty, no matching
criterion can help. Equivalence mainly converts near-misses into hits *within* nonempty
sets, where BRITE lifts Top-1 from 0.8198 to 0.8448.

## 3. Conditional ranking metrics (2,359 reactions with candidates)

| Ranker | Top-1 exact (micro) | Top-1 exact (model-macro) | Top-1 exact (cluster-macro) | Top-1 BRITE (micro) |
| --- | --- | --- | --- | --- |
| Oracle (upper bound) | 0.8559 | 0.8619 | 0.8476 | 0.8639 |
| **Heuristic (shipped)** | **0.8198** | 0.7158 | 0.7369 | 0.8448 |
| Lexical (TF-IDF) | 0.8164 | 0.7259 | 0.7235 | 0.8457 |
| Random | 0.7813 | 0.6771 | 0.6884 | 0.8262 |
| Embedding (MiniLM) | *not run* | | | see §6 |
| LLM rerank | *not run* | | | see §6 |

Random reaching 0.7813 is the tell: the median nonempty candidate set has **one** element,
so most reactions are ranked correctly no matter what the ranker does. The whole spread
from random to oracle is 7.5 pp conditional, i.e. 3.0 pp overall.

## 4. Retrieval versus reranking decomposition

All rates below are in `baseline_summary.json` under
`headline.<ranker>.failure_decomposition`, each carrying its own numerator, denominator
and population string.

| Rate | Heuristic | Denominator | Population |
| --- | --- | --- | --- |
| `zero_candidate_rate` | 59.44% | 5,816 | all evaluable |
| `overall_retrieval_failure_rate` | 65.29% | 5,816 | all evaluable |
| `conditional_retrieval_failure_rate_nonempty` | 14.41% | 2,359 | nonempty sets |
| `conditional_reranking_failure_rate_retrievable` | 4.21% | 2,019 | retrievable answers |
| `overall_top1_accuracy` | 33.25% | 5,816 | all evaluable |
| `overall_top1_failure_rate` | 66.75% | 5,816 | all evaluable |

Reading the corpus of 5,816 evaluable reactions end to end with the heuristic ranker:

| Outcome | Reactions | Share |
| --- | --- | --- |
| Zero candidates generated | 3,457 | 59.44% |
| Candidates generated but exact answer absent | 340 | 5.85% |
| Answer retrievable but not ranked first | 85 | 1.46% |
| **Correct at rank 1** | **1,934** | **33.25%** |

The 3,797 retrieval failures (3,457 + 340) are 97.8% of all failures. The 85 reranking
failures are 2.2%. **A reranker-only Phase 3 is capped at +1.46 pp.**

Earlier logs reported a single `retrieval_fail=14.4%`, which was the conditional rate over
nonempty sets. That figure is real but is not the benchmark's retrieval-failure rate, which
is 65.29%. The reporting has been changed so every rate names its population; see §7.

### Failure stratification (all evaluable, exact criterion)

| Stratum | n | Zero-cand % | Recall any rank | Top-1 |
| --- | --- | --- | --- | --- |
| Genome-scale | 4,554 | 65.44% | 0.3052 | 0.3011 |
| Not genome-scale | 1,262 | 37.80% | 0.4984 | 0.4461 |
| Complexity 1–2 participants | 565 | 32.39% | 0.6106 | 0.5327 |
| Complexity 3–4 | 2,678 | 47.39% | 0.4675 | 0.4544 |
| Complexity 5–6 | 2,295 | 76.51% | 0.1747 | 0.1721 |
| Complexity 7+ | 278 | 89.57% | 0.0755 | 0.0755 |
| No relaxation needed | 4,060 | 49.51% | 0.4567 | 0.4377 |
| Relaxation required | 1,756 | 82.40% | 0.0940 | 0.0894 |
| Species source `kegg_compound` | 1,482 | 51.08% | 0.4811 | 0.4771 |
| Species source `chebi+kegg_compound` | 4,260 | 61.95% | 0.3035 | 0.2850 |
| Species source `chebi` only | 67 | 80.60% | 0.1940 | 0.1940 |
| Species source `none` | 7 | 100.00% | 0.0000 | 0.0000 |

Two clear gradients. Retrieval degrades monotonically with reaction size: 61% recall at
1–2 participants down to 7.6% at 7+, because every additional participant is another
compound that must map correctly for subset containment to hold. And when ontology
relaxation was needed at all, recall collapses to 9.4% — relaxation is currently a marker
of a failing match rather than a successful recovery mechanism.

## 5. Candidate-set size diagnostics

Machine-readable: `benchmark/data/candidate_diagnostics.json`,
`candidate_size_by_stratum.csv`, `candidate_largest_sets.csv`.

| | All evaluable (n=5,816) | Nonempty only (n=2,359) |
| --- | --- | --- |
| Mean | 15.78 | 38.92 |
| Median | 0 | **1** |
| p75 | 1 | 1 |
| p90 | 1 | 7 |
| p95 | 4 | 115 |
| p99 | 423.65 | 1,446 |
| Max | 2,353 | 2,353 |

| Threshold | Reactions above |
| --- | --- |
| > 15 candidates | 188 |
| > 50 | 150 |
| > 100 | 131 |
| > 1,000 | 45 |
| > 10,000 | 0 |

The distribution is extremely bimodal: the typical nonempty set holds exactly one
candidate, while a 131-reaction tail holds hundreds to thousands.

### Concentration

| Group | Candidate rows | Share of all rows |
| --- | --- | --- |
| Largest 1 model (`BIOMD0000001063`) | 77,073 | 83.96% |
| Largest 5 models | 87,036 | 94.81% |
| Largest 10 models | 89,623 | 97.63% |
| Largest 1 reaction | 2,353 | 2.56% |
| Largest 10 reactions | 16,819 | 18.32% |
| Largest 100 reactions | 82,409 | 89.77% |

### Degeneracy: large sets are noise, not ambiguity

Splitting the 2,359 `ok` reactions at 100 candidates (a reporting boundary only — nothing
is capped or discarded anywhere in the pipeline):

| | Reactions | Candidate rows | Exact retrievable | Exact Top-1 |
| --- | --- | --- | --- | --- |
| ≤ 100 candidates | 2,228 | 5,489 | 89.81% | 86.80% |
| > 100 candidates | 131 | 86,313 (94.02%) | 13.74% | **0.00%** |

The 131 large sets consume 94% of the storage, contain the right answer only 13.7% of the
time, and **never** rank it first. They are pure cost.

### Root cause: set size is governed by mapped-participant count

Across all `ok` reactions, grouped by `filtered_species_count` (participants that mapped
to a KEGG compound):

| Mapped participants | Reactions | Mean size | Median | Exact retrievable | Exact Top-1 |
| --- | --- | --- | --- | --- | --- |
| 1 | 129 | 549.14 | 166 | 10.08% | 5.43% |
| 2 | 1,037 | 17.80 | 1 | 86.11% | 80.71% |
| 3 | 492 | 2.44 | 1 | 93.29% | 89.43% |
| 4 | 528 | 1.62 | 1 | 93.56% | 93.18% |
| 5 | 90 | 2.49 | 1 | 93.33% | 93.33% |
| 6–9 | 80 | ~1.2 | 1 | 85–100% | 84–100% |

The relationship is monotonic and mechanical. Subset containment means each mapped
participant is a filter; with one constraint — typically a ubiquitous cofactor such as
water, ATP or a proton — thousands of KEGG reactions satisfy it. Two or more constraints
collapse the set to a median of one. The 129 single-constraint reactions produce 70,839 of
the 91,802 rows (77%) and are almost never retrievable.

This is the same defect as `unconstrained_candidate_set`, one step milder: zero
constraints matches all of KEGG and is dropped, one constraint matches thousands and is
kept as `ok`. Both are consequences of under-annotated species, not of ambiguity in the
biology.

## 6. `BIOMD0000001063` investigation

`BIOMD0000001063` (Yeast8-family genome-scale model, 874 evaluable reactions) contributes
77,073 of 91,802 candidate rows — 83.96%.

**Which reactions.** 467 of its 874 reactions are `ok`; 87 of those have > 100 candidates
and account for 75,454 rows, i.e. 97.9% of the model's rows and 82.2% of the entire
corpus. The largest are `r_2027` and `r_0537` at 2,353 each. Every one of the top sets has
`filtered_species_count = 1` while having 3–6 actual participants.

**Cause: expected behaviour of the retrieval filter under sparse annotation.** Ruling out
the alternatives:

- *Duplicated candidates*: no. Zero duplicate `(reaction_id, candidate_kegg)` pairs in
  this model or anywhere in the table; the audit asserts uniqueness corpus-wide.
- *Ontology relaxation*: no. 75,650 of the model's 77,073 rows are `relaxation_level = 0`
  (`relaxation_direction = exact`); only 1,423 rows involved any relaxation. The largest
  sets are all level 0. Relaxation is not the multiplier.
- *Software defect*: no evidence. Ranks are unique and consecutive per reaction, KEGG ids
  are well formed, `num_candidates` matches stored rows exactly, and reassembly from the
  per-model caches is byte-identical.
- *Genuine scientific ambiguity*: partly, but not in a useful sense. The sets are large
  because the query is nearly unconstrained, not because these reactions have thousands of
  plausible KEGG counterparts. A set of 2,353 that omits the answer 86% of the time is not
  a meaningful ambiguity statement.

The mechanism is the one in §5: this model's species annotations map only one participant
per reaction for these 87 reactions, so subset containment degenerates. It is the
`filtered_species_count = 1` pathology at scale, and it is a *retrieval-quality* problem
in the generator's constraint construction, not a bookkeeping bug.

**Effect on metrics.** Storage and runtime only, to three decimal places:

| | Including `BIOMD0000001063` | Excluding it |
| --- | --- | --- |
| Corpus recall any rank (exact, micro) | 0.3471 | 0.3493 |
| Corpus Top-1 (exact, micro) | 0.3325 | 0.3341 |

Its own recall (0.3352 any rank, 0.3238 Top-1) is close to the corpus average, so removing
it would move the headline by ~0.2 pp. Model-macro and cluster-macro averaging already
bound its influence to 1/74 and 1/54 respectively. Per-reaction accuracy is unaffected:
because these sets never rank the answer first, they neither inflate nor deflate Top-1.

**No action taken.** Nothing is capped or discarded. Capping large sets would change
retrieval semantics and is a Phase 3 design decision requiring approval; the honest record
of what the current generator produces is what belongs in the freeze.

## 7. Reporting corrections made during this audit

`retrieval_failure_pct = 14.41` previously appeared in `baseline_summary.json` and in the
run log with no stated denominator, inviting the reading that Phase 2 retrieval fails on
14% of the benchmark when the true figure is 65%. Changes:

- `failure_decomposition` replaces the bare percentages, reporting `zero_candidate_rate`,
  `overall_retrieval_failure_rate`, `conditional_retrieval_failure_rate_nonempty`,
  `conditional_reranking_failure_rate_retrievable`, `overall_top1_accuracy`,
  `overall_top1_failure_rate` and `conditional_top1_accuracy_nonempty`. Each is an object
  with `rate`, `pct`, `numerator`, `denominator` and `population`.
- `baseline_summary.json` gains a top-level `populations` block defining all three
  populations and their sizes.
- Headline keys are suffixed `_scored` where they average over ranked reactions only.
- `failure_stratification.csv` columns are renamed to state their denominator
  (`conditional_retrieval_failure_pct_nonempty`, `n_reactions_nonempty`, and so on) and a
  `conditional_reranking_failure_pct_retrievable` column with `n_retrievable` is added.
- The run log prints both denominators on every ranker line.

No numerator, denominator or metric definition changed; the underlying values are
identical to the original run. Backward compatibility was deliberately *not* preserved for
the ambiguous key names.

## 8. Limitations

- **Embedding baseline outstanding.** `chromadb` 1.0.11 and `onnxruntime` are installed
  and the MiniLM asset is SHA-256 pinned (`913d7300…16ec3`), but it is not in the local
  cache, so running it requires an ~80 MB download from
  `chroma-onnx-models.s3.amazonaws.com`. Not fetched, per instruction not to download
  without asking. The freeze is not blocked on it: the lexical baseline already lands
  within 0.3 pp of the heuristic and 4 pp of the oracle, so an embedding ranker cannot
  change the retrieval-first conclusion — it is bounded by the same +1.46 pp.
- **LLM baseline not run.** No API calls were made. Bounded by the same oracle ceiling.
- **Conditional macro averages are noisy.** Model-macro Top-1 over nonempty sets covers
  only the 65 models and 47 clusters that have any candidates, so it is not comparable
  term-by-term with the 74-model unconditional averages.
- **`num_participants` counts model participants, not mapped ones.** Complexity strata
  therefore mix well- and poorly-annotated reactions; `filtered_species_count` in
  `candidate_status.csv` is the annotation-quality variable and is the stronger predictor.
- **Three reactions with `filtered_species_count = 47`** produce 46 candidates each and 0%
  recall. These look like biomass/pseudo-reactions where participant mapping is
  meaningless. Too few to affect any headline; flagged for Phase 3 triage.
- **Equivalence classes depend on KEGG coverage**: 9,617/12,312 KEGG reactions carry an
  EC number and 7,181 a KO, so equivalence-aware recall is a lower bound.
- **Single generation configuration.** No sensitivity analysis over `max_relax_level`,
  cofactor handling or `top_k`; a 3-day pass per configuration makes sweeps expensive.
- **Cluster-macro uses the conservative Phase 1 clustering** that keeps the seven yeast
  models together. The provenance-based sensitivity analysis promised in Phase 1 is still
  outstanding.

## 9. Why Phase 3 should be retrieval-first

The decomposition leaves little room for interpretation:

1. **59.44% of evaluable reactions never enter the ranking problem at all.** No reranker,
   LLM or otherwise, can score a candidate set that does not exist.
2. **Perfect reranking of the current sets is worth +1.46 pp** overall (33.25% → 34.71%).
   The heuristic already captures 95.8% of the reachable answers.
3. **The failure is concentrated in constraint construction.** Reactions with ≥ 2 mapped
   participants are retrievable 86–100% of the time; those with 1 are retrievable 10% of
   the time and generate 77% of the rows. Retrieval quality tracks species-annotation
   coverage almost perfectly.
4. **Subset containment is the wrong retrieval operator under sparse annotation.** It is
   simultaneously too strict (2,646 `no_candidates` when one participant fails to map) and
   too loose (all of KEGG when none map). Both failure modes come from the same
   all-or-nothing set predicate.

This argues for open-set recovery and learned full-database retrieval rather than a
reranker:

- **Soft/partial matching** instead of strict subset containment, scoring partial
  participant overlap so a single unmapped cofactor does not zero out a reaction.
- **Learned retrieval over all 12,312 KEGG reactions**, embedding the reaction's text and
  participants and retrieving top-k directly, which removes the dependency on every
  participant mapping correctly and gives every reaction a nonempty set.
- **Improved species grounding**, since annotation coverage is the dominant predictor;
  recovering compound identity from names for unannotated species should convert
  `no_candidates` and single-constraint reactions into normally-constrained ones.
- **Report retrieval and ranking separately from the start**, over all evaluable reactions,
  so a conditional improvement can never be mistaken for a benchmark improvement.

A reranker-only Phase 3 would be optimising the 2.2% of failures that are reranking
failures while leaving the 97.8% that are retrieval failures untouched.

## Reproducing

```bash
# Analysis only; assumes benchmark/data/_candidates_cache/ is populated.
python benchmark/scripts/generate_candidates.py --assemble-only
python benchmark/scripts/analyze_retrieval.py
python benchmark/scripts/rank_baselines.py --rankers heuristic lexical random oracle
python benchmark/scripts/candidate_diagnostics.py
python benchmark/scripts/audit_phase2.py \
    --expect-config-id 86938b48ab88 --expect-models 74 --expect-reactions 5816 \
    --check-reassembly
```

Full generation (~3 days wall clock, resumable) is documented in `benchmark/PHASE2.md`.
