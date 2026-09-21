# Source audit

Actual HEAD: `892ffc9aca25b275d861aee424559b90592d632f`. Requested reference: `b73e52ae4a787e33c5773673694464b8f2743e56`. The reference was fetched read-only; its tracked tree is identical to HEAD (two merge commits differ). No reset, checkout, or historical writer was run. No applicable AGENTS.md was found in the repository or workspace ancestors. Pre-existing local edits are hashed in the manifest and preserved.

`data/source_manifest.json` distinguishes local payload existence, Git tracking, and metadata-only/missing states. In particular, locally available ignored NPY caches are not claimed to be distributed by Git. Array shapes, dtypes, exact hashes, model revisions, precision and implementation metadata are recorded. Ordered query/catalog IDs are exported separately. Raw caches come from the historical provenance audit; no re-encoding or canonical rerun is needed.

Primary supports: WANDS 384 requested / 308 eligible / 21,299 Exact pairs / 42,994 products; ESCI 500 requested / 499 eligible / 4,434 E pairs / 10,076 products. The 96 WANDS development queries are excluded. ESCI is the existing evaluation sample. Hq includes absent-in-both products. Query macro and pair micro remain separate.

All 18 canonical conditions are reused through `canonical_sources.json` and their source records. Pure ascending is `canonical_raw`; M2 framing and historical `rule_appearance` are excluded. WANDS pipe entries and ESCI brand/color/whole bullet-field entries remain intact. Duplicates and repeated views remain.

`shared_query_cache_audit.csv` documents nonzero differences between historical Phase III Set-Mean queries and the common Original query matrix. This package uses Phase IV Set-Mean ranks/scores, which use the common queries; Phase III Set-Mean ranks are not substituted.

The supplied chapter package is preserved byte-for-byte under `inputs/chapter56_revision/`; updated copies will be separate. Its embedded execution prompt is supporting material, not an override of the user's initial scope. PDF version (6) is available. Complete LaTeX matching that PDF has not been supplied, so full-manuscript page count/float placement remains author integration work.

Missing analysis inputs: none among the inventoried required artifacts. Numerical and score reconstruction compatibility are checked separately in `data/hybrid_reconstruction_audit.csv`; presence alone is not treated as compatibility.
