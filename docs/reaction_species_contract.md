# Species evidence contract for `rrf_grounded` (v1)

This contract was defined against upstream `866be205` before runtime implementation.
Upstream `identity_support.py` assigns `supported`, `unresolved`, or
`not_applicable`; qualifier prediction is a later, separate decision. Tied
accessions can occupy multiple rows for one identity. Identifier presence is
not evidence of biological correctness.

* A nonempty `identity_status=supported` permits a compound identifier when its
  qualifier is `is` (or blank, the upstream default). `unresolved` never supplies
  a trusted compound mapping. Unknown/non-applicable statuses are retained in
  the evidence audit but are not promoted to supported.
* Missing `identity_status` **column** means legacy evidence. Valid ChEBI and
  KEGG compound identifiers with `is`/blank qualifiers remain usable under the
  explicit legacy-unassessed policy, recorded as such; their support is unknown.
  A present column with an empty value means missing status, not legacy.
* Existing SBML `bqbiol:is` compound annotations are source assertions, treated
  as legacy-unassessed, not as independently validated identities. All other
  qualifiers are retained in the audit. `hasPart` and nonempty `component_id`
  describe components, never whole-species identities. `isVersionOf` and other
  relationships are not substituted for exact compound identity.
* A supplied recommendation table is authoritative for species it contains;
  SBML annotations fill only species absent from that table. This prevents an
  unresolved recommendation from being bypassed through an older source URI.
* Identifiers are normalized and deduplicated per species before constructing
  text. Multiple supported/legacy identities remain alternatives, sorted by
  identifier; they do not gain weight through duplicated tied rows. Only
  **whole-species identity assertions** (`is`/blank qualifier, no component ID)
  participate in identity conflicts. An explicit unresolved/missing/unknown or
  non-applicable identity status, or a qualifier abstention on such an assertion,
  conservatively vetoes that identifier. `conflicting_identifiers` records the
  intersection of eligible identities and these identity vetoes. Multiple
  supported alternatives remain alternatives; no candidate is adjudicated.
* The same URI may legitimately occur under `is` and `hasPart` or another
  relationship. Non-identity qualifiers and component rows neither supply nor
  veto whole-species identity evidence, regardless of their support status.
  They remain in provenance, including when no eligible identity exists.
* Names, participant IDs, stoichiometry and equation direction survive when
  identifiers cannot be used. Reaction identifiers/labels are not answers in
  retrieval text. KEGG reaction accessions are redacted. Direct KEGG compound
  IDs (`C` plus five digits) are allowed; reaction IDs are not species evidence.

The query uses the selected target-local template. Detailed identity status,
qualifier, source and conflict records are sidecar provenance, not repeated
identifier votes. Species uncertainty remains separate from a model's reaction
abstention. The legacy policy is explicit compatibility behavior, not a claim
that old recommendations have passed the new identity-support stage.

Row-level provenance reports `assertion_scope`, `eligible` (before conflict
resolution), `identity_veto`, `included` (after conflict resolution), and
`exclusion_reason`. `usable` now explicitly aliases final `included`, rather
than pre-conflict eligibility. Thus a supported row vetoed by an unresolved
whole-species assertion remains `eligible=true` but has `included=false`,
`usable=false`, and `exclusion_reason="identity_conflict"`. Source status and
qualifier values remain recorded; no biological support status is invented.
This corrects qualifier handling within the v1 contract; it changes sidecar
provenance fields, not retrieval defaults or provider settings.
