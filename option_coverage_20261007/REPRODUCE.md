# Reproduce the bounded option-coverage analysis

Run from the `shopping-representation-bias` repository root. The local run used Python 3.10, NumPy 1.26.4, pandas 2.3.3, PyArrow and Matplotlib; exact versions are recorded in `data/input_manifest.json` and `qa/validation.json`. No Torch, model download, inference, retrieval or training is required. All analysis inputs are local. Some saved experiment payloads are not distributed by Git; hashes identify the exact required copies.

```powershell
$env:PYTHONIOENCODING = 'utf-8'
python -m unittest discover -s option_coverage_20261007 -p test_core.py -v
python option_coverage_20261007/prepare.py
python option_coverage_20261007/analyze.py verify
python option_coverage_20261007/audit_serializations.py
python option_coverage_20261007/analyze.py analyze
python option_coverage_20261007/cases.py
python option_coverage_20261007/present.py
python option_coverage_20261007/validate.py
pdftoppm -scale-to 1800 -png -singlefile option_coverage_20261007/figures/option_coverage.pdf option_coverage_20261007/qa/option_coverage_pdf_render
```

`prepare.py` reads the entire catalog, independent of membership changes, and verifies every processed attribute against the original raw pipe-separated entry sequence. The version 1 extraction specification and `data/attribute_audit_review.json` preserve the original review of 90 hash-selected records. When reproducing unchanged source inputs, this is a replay of that frozen specification, not a new data-dependent mapping exercise. If inputs or extraction rules change, use a new version and repeat the catalog-only review before computing effects.

`analyze.py verify` reconstructs eligible query support from the saved held-out split and cleaned Exact judgments, validates catalog/query axes, and checks all 11 saved conditions against their full-catalog judged-rank files and previously saved Recall/nDCG records. It saves every query's original membership, then reports the 308/73 anchor check. Those numbers do not select or construct a population. Top-20 input lists must be complete and unique; unknown IDs, qrel mismatches and rank discrepancies raise errors. Optional top-list/rank pairs that are absent are listed by exact path and their contrasts skipped; core population/provenance dependencies must be restored from the manifest rather than substituted.

`audit_serializations.py` regenerates **text strings only**, checks raw stored representation text and row hashes for C0/C1/C2s1–C2s5, checks both canonical text hashes, and verifies query text provenance. It never loads a model or recomputes a retrieval score. The canonical batch-ceiling mismatch (48 versus 16) is documented in `data/provenance_checks.json`; there is no silent switch to a harmonized experiment.

`analyze.py analyze` checks frozen hashes, then writes strict per-field sets, eligibility, exclusions and 10,000-query-bootstrap summaries. Bootstrap seed is 20261007 as specified in the task; the historical packages' distinct analysis seeds are not repository-wide policy. Field/subgroup bootstrap samples contain only eligible queries. The cross-contrast sensitivity uses a fixed common complete cancellation support, resampling query vectors jointly. Undefined strict results remain JSON null/CSV empty; they are never zero-filled.

`cases.py` audits query 231 and selects at most two additional distinct query IDs by the frozen SHA-256 rule. Each selected category is field-specific; no completeness for unknown fields is implied. `present.py` renders every numerical table and figure from saved result CSVs. It creates an inputtable `booktabs` LaTeX table fragment (not a standalone document) without touching the manuscript. `validate.py` checks input preservation, all summary denominators and set identities, source-derived case selection and cases, bootstrap determinism, and table/figure data agreement. It writes a final output manifest; the manifest itself is excluded from its own hashes.

## Main outputs

- `config.json`, `core.py`, `prepare.py`, `analyze.py`, `audit_serializations.py`, `cases.py`, `present.py`, `validate.py`, `test_core.py`: reproducible implementation and checks.
- `data/input_manifest.json`, `serialization_audit.json`, `provenance_checks.json`, `available_runs.json`, `preexisting_state.json`: exact input paths/hashes, support and execution provenance.
- `extraction_spec_v1.json`, `data/extraction_freeze.json`, `raw_key_census.csv`, `catalog_coverage.csv`, `attribute_audit.jsonl`, `attribute_audit_review.json`: extraction design, full-catalog coverage and source review.
- `data/product_fields.jsonl.gz`: 128,982 product-field records, including original entries, keys, values, positions, status and normalized values.
- `data/validated_top20.jsonl`: 3,388 full Top-20 query-run lists and recomputed scores.
- `data/membership.jsonl`: 2,464 query-contrast records, including original relevant IDs and integer product transitions.
- `data/query_fields.jsonl`, `exclusions.jsonl`: 7,392 query-field-contrast records and every exclusion, observed-only arrays and strict option sets (null if ineligible).
- `data/membership_summary.csv`, `option_summary.csv`, `exclusion_summary.csv`, `coverage_summary.csv`, `joint_query_bootstrap.csv`: field-wise/subgroup summaries, eligibility, both directions, zero results and unadjusted conditional uncertainty.
- `data/cases.jsonl`, `case_selection.json`, `anti_fatigue_remaining_suppliers.json`, `CASE_AUDIT.md`: complete verifiable examples.
- `REPORT.md`, `table_option_coverage.tex`, `figures/option_coverage.pdf`, `figures/option_coverage.png`, `figures/option_coverage_plot_data.csv`: findings, candidate paragraph, table and figure.
- `qa/tests.log`, `qa/validation.json`, `qa/visual_review.json`, `qa/option_coverage_pdf_render.png`, `output_manifest.json`: numerical and visual verification.

All output writes are confined to this dated directory. Pre-existing manuscript files and working changes are protected by hashes. No manuscript integration, commit or push is performed.
