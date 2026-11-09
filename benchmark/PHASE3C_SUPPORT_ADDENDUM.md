# Phase 3C addendum: evidence support is distinct from correctness

This clarification supplements the frozen Phase 3C report and conclusions. It does not rewrite historical outputs, metric definitions, reports or manifests, and it supplies no biological judgment.

In `phase3c_validation.py`, `direct_unsupported` marks a direct-mode answer with an in-catalog identifier. Direct mode received no retrieved evidence and allowed internal model knowledge. The count **130/163** therefore means in-catalog direct answers made without retrieved evidence. It includes **49 exact-reference-correct** and **81 exact-reference-incorrect** answers. It is not a count of biological errors, hallucinations, fabricated identifiers or unsupported factual claims; reference mismatch alone does not establish biological hallucination.

`grounded_unsupported` instead marks an answered grounded prediction whose identifier is outside its supplied Top-10 evidence. The observed **0/163** is out-of-evidence identifier compliance for a candidate-constrained task. Direct and grounded modes offer different opportunities for evidence support, so these indicators do not share a biological hallucination definition.

The grounded method answered **124/163** and abstained on **39/163**. It made **22 in-evidence, exact-reference-incorrect** selections. Full evidence compliance was **161/163**; two abstentions cited evidence incorrectly despite zero out-of-evidence predictions. Thus zero out-of-evidence predictions does not mean zero errors or perfect compliance with every evidence rule.

The independently measured paired accuracy remains **grounded 102/163 versus fusion 93/163**, a net nine exact matches. The reported cluster-bootstrap interval `[0.01, 0.141304]` is an accuracy interval. The historical bootstrap for `grounded_minus_phase3a_unsupported_output_rate` compares differently defined support indicators; its interval does **not** establish reduced biological hallucination. A biological hallucination comparison would require a separately specified, consistently applied endpoint and appropriate assessment; none is established here.

The completed scoped single-reviewer audit remains complete. This clarification does not reopen it or imply independent adjudication.

The new counts are generated from the frozen 163-row validation join by:

```powershell
python -m benchmark.scripts.phase3c_support_clarification
```

The count artifact is stored under `benchmark/phase3/phase3c_support_addendum/`; all original Phase 3C artifacts remain unchanged.
