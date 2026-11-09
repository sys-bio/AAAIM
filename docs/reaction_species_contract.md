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
  identifier; they do not gain weight through duplicated tied rows. Any explicit
  unresolved/missing/unknown assertion for the same identifier conservatively
  vetoes that identifier, with a conflict recorded. No candidate is adjudicated.
* Names, participant IDs, stoichiometry and equation direction survive when
  identifiers cannot be used. Reaction identifiers/labels are not answers in
  retrieval text. KEGG reaction accessions are redacted. Direct KEGG compound
  IDs (`C` plus five digits) are allowed; reaction IDs are not species evidence.

The query uses the selected target-local template. Detailed identity status,
qualifier, source and conflict records are sidecar provenance, not repeated
identifier votes. Species uncertainty remains separate from a model's reaction
abstention. The legacy policy is explicit compatibility behavior, not a claim
that old recommendations have passed the new identity-support stage.
