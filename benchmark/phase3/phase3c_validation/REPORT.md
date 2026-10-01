# Phase 3C 163-reaction paired validation pilot

This method-development experiment uses exactly the frozen Phase 3A 163-reaction validation pilot. It does not evaluate the held-out test set or all 969 validation reactions.

## Results

Grounded exact Top-1: 102/163; fusion Top-1: 93/163; frozen Phase 3A target-only: 49/163.
Grounded coverage was 124/163; abstentions were 39/163; evidence-compliant outputs were 161/163.
Unsupported grounded outputs: 0; incorrect in-evidence selections: 22; schema-invalid outputs: 0.

## Retrieval-state behavior

Truth at fused rank 1 (n=93): preserved 86, incorrect replacement 1, abstained 6; net harm 7.
Truth at ranks 2-10 (n=40): promoted 16, incorrect selection 9, abstained 15; net recovery 16.
Truth absent from Top 10 (n=30): abstained 18, unsupported/fabricated 0, in-evidence incorrect 12. Phase 3A made 21 unsupported in-catalog predictions (15 incorrect) in this state.

## Paired inference and uncertainty

Against fusion: 86 both correct, 16 grounded-only, 7 fusion-only, 54 neither.
Against Phase 3A target-only: 35 both correct, 67 grounded-only, 14 Phase-3A-only, 47 neither.
Grounded-minus-fusion exact delta: 0.055215 with 95% cluster-bootstrap interval [0.01, 0.141304]. No superiority claim is made when an interval includes zero.

## Decision

1. Yes. Grounding reduced unsupported output from 130/163 Phase 3A in-catalog guesses to 0/163 grounded outputs; the paired unsupported-rate interval excludes zero.
2. Yes. It recovered 16/40 truths available at fused ranks 2-10.
3. It harmed 7/93 already-correct fusion Top-1 cases: 1 incorrect replacement and 6 abstentions.
4. Yes at the observed cost: nine net exact recoveries over fusion, a strictly positive cluster-bootstrap interval, and zero unsupported IDs justify $2.108151 for this pilot. Production economics remain application-specific.
5. Recommended final core: `grounded_llm_on_every_reaction`. No selective-routing rule was prespecified, so any routing idea generated from these outcomes is post hoc and unvalidated. Fusion-plus-explanation remains an interface option, not an accuracy improvement established here.
6. No. An all-969 validation run is not scientifically necessary unless a separate prespecified follow-up question cannot be answered from this paired pilot.
7. Yes. The method is ready to freeze before one held-out test run; this milestone does not run or inspect that test.

## Cost and provenance

The validation run attempted 158 new calls, with 158 successes and 0 failures. New-call cost was $2.039463; complete-pilot recorded cost including reused smoke calls was $2.108151 under the $3.50 cap.
Paid responses used native Python orchestration: local frozen retrieval first, then a stateless OpenAI Responses request. `tools=[]`; the provider model did not call a tool. LangChain was exercised only in a zero-cost synthetic tool-message parity demonstration and is an integration layer, not the retriever or evaluator.
Confidence is self-reported descriptive metadata, not a calibrated probability; no threshold was selected.
