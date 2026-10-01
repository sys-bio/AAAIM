# Offline reaction-label reviewer

Open `index.html` first, normally by double-clicking it. No server, installation, login, network connection, or remote asset is required.

1. Enter a stable reviewer identifier in Formal Pass 1.
2. Review the source reaction and existing BioModels label. Drafts autosave in browser local storage under a namespace containing schema `aaaim-human-review-v1` and dataset digest `64a7eb4de27024da4c94b0b3c8b3ffab5e47a335fb1ee6e5656e280f279c2b7f`.
3. Export JSON regularly; JSON is authoritative and CSV is a readable convenience copy. Browser storage is not a durable scientific archive.
4. Put exports in `benchmark/phase3/error_audit/review_work/`, which is intentionally ignored by Git. Suggested names are `pass1_review_<reviewer>.json`, `pass1_review_<reviewer>.csv`, `pass2_review_<reviewer>.json`, `pass2_review_<reviewer>.csv`, and `exploratory_review_<reviewer>.json`.
5. Pass 1 must precede Pass 2 to prevent system suggestions from influencing the source-label judgment. Pass 2 stays locked until it validates a complete 60-case Pass 1 JSON export.
6. Do not open browse-all until Formal Pass 1 is complete; browse-all exposes system outputs for overlapping cases.
7. Send the authoritative exported JSON file to LunaStarr using your normal approved project file-sharing channel. Do not paste reviews into source CSVs or commit them.
8. Two independent reviews will be compared and adjudicated in a later milestone. Nothing entered here is automatically treated as a corrected label.

The supplemental page is a separate training-label review for `BIOMD0000000013/E12`; it is not part of validation or prevalence estimates. Browse-all is an error-enriched exploratory queue and also cannot estimate prevalence.

If direct double-clicking is restricted by a particular browser policy, run `python -m http.server 8000 --directory benchmark/phase3/error_audit/reviewer` and open `http://127.0.0.1:8000/`. This fallback still uses only local files.
