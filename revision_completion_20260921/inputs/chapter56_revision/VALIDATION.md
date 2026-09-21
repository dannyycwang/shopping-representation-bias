# Validation scope

Checked on 2026-09-21 against manuscript PDF version (6) and repository snapshot `b73e52ae4a787e33c5773673694464b8f2743e56`.

## Numerical and scope checks

- Verified all 18 VI@20 values quoted in Section 6.1 against primary-population, query-macro rows for catalog-wide, target-only full, and target-only fully-fitting conditions.
- Checked persistent inclusion + VI + persistent omission = 1 on the same supports.
- Checked paired gain - loss = net Recall change and gain + loss = membership change in the source-order comparison data, including the WANDS/BGE reversal example.
- Retained all 36 pure-canonical dataset/encoder/rule/cutoff rows. Historical M2 rows are excluded from the pure-canonical tables.
- Built 28 shared-setting method Recall@100 estimates. M=2 centroid/max values were derived from saved query-level files after applying the fixed WANDS held-out IDs. Query counts are exactly 308 for WANDS and 499 for ESCI.
- Confirmed the method-dependent direction of Set-Mean and M=2 full-record aggregation changes; the prose does not call either a universal improvement or degradation.
- The code-defined ESCI entry unit is brand, color, and the complete bullet-point field, not individually split bullet sentences.
- Source-order hybrid Recall is not reported as seven-schedule hybrid robustness.

These checks validate transcription, aggregation, and scope of existing results. They do not constitute rerunning encoders or independent replication of raw retrieval.

## LaTeX and visual checks

The two replacement chapters, five tables, and optional appendix material compile together under a standard two-column article proof using the existing figure PDFs. Final compilation has no overfull/underfull boxes, undefined references/citations, or LaTeX warnings. Rendered pages were checked for clipping, table alignment, and legibility.

The local proof uses a standard article class because `acmart` is not installed. Its first three pages contain Sections 5--6; this is not a validation of the full paper's ACM page count. Appendix floats and bibliography in the isolated proof do not represent final manuscript placement. The proof's reference entries are validation stubs, so the proof PDF is intentionally not delivered as a paper draft; use the manuscript's existing bibliography.

The supplied Fig. 4 still carries its original legend wording. Its caption and surrounding prose explicitly define newly included/omitted shares; the requested legend revision is assigned to Codex. No data points or uncertainty intervals were changed.

## Remaining work

P1/P2/P3 in `CODEX_EXECUTION_PROMPT.md` are specifications, not completed experiments. In particular, the joint strategy table still needs matched membership/uncertainty analysis and schedule-level hybrid results. The supplied Section 6 is an evidence-grounded working draft, not a claim that those analyses have already run.

No existing manuscript PDF, repository source, historical experiment output, or Chapters 1--4 was modified.
