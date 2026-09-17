# Revised Section 3 / RQ2 evidence

Start with [EVIDENCE_STATUS.md](EVIDENCE_STATUS.md) for verified findings, evaluation populations, source preservation and limitations. [RQ2_MANUSCRIPT_TEXT.md](RQ2_MANUSCRIPT_TEXT.md) contains suggested results prose; the original manuscript is not edited.

- [Figures, captions and accessible descriptions](CAPTIONS_AND_ACCESSIBILITY.md); vector PDF/SVG outputs are in [figures/](figures/).
- [Per-query cancellation evidence](CANCELLATION_TABLE.md).
- [All frozen canonical controls](CANONICAL_CONTROLS.md), with C0 and raw seven-order-mean references kept separate.
- [Reproduction instructions](REPRODUCE.md), [prospective protocol](PROSPECTIVE_PROTOCOL.json), [BGE execution addendum](BGE_EXECUTION_ADDENDUM.json), and [validation results](data/validation_report.json).
- [Portable evidence package](evidence_package.zip), with file hashes in [DELIVERY_MANIFEST.json](DELIVERY_MANIFEST.json).

The Git snapshot includes analysis data, newly computed exact ranks, source hashes, embedding metadata, scripts and vector figures. Large `.npy` GPU embedding caches and temporary visual proofs stay local; they are not needed to reproduce the plots and statistics from the saved ranks. Historical input artifacts remain at their original repository paths, with the existing backup mechanism documented in the repository's `GITHUB_BACKUP_README.md`.

The Git-status text files are preservation snapshots taken during validation, before the delivery commit; they are not a live repository-status display.

The optional 16/32-schedule expansion is outside this delivery. Finite-family results do not establish exhaustive permutation robustness, and confidence intervals including zero do not establish equivalence.
