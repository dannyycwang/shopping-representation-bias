# Figure revision package

This presentation revision reuses `revision_evidence_20260917` estimates and intervals. All outputs stay in this directory. It does not change manuscript text, original figures, training, model selection, or the seven-order family.

| Deliverable | PDF / SVG stem | Final width |
|---|---|---:|
| Introduction Figure 1 candidate | `figure1_teal_chair_candidate` | 3.45 in |
| Section 3 Figure 2 schematic | `figure2_intervention_and_states` | 7.10 in |
| Results / RQ1 main, Top-20 VI and CIs | `rq1_vi_top20_main` | 7.10 in |
| RQ1 appendix, both cutoffs and all states | `rq1_states_top20_top100_appendix` | 7.10 in |
| Results / RQ2 main, Top-20 | `rq2_membership_top20_main` | 7.10 in |
| RQ2 appendix, Top-100 | `rq2_membership_top100_appendix` | 7.10 in |
| Results / RQ2 cancellation table | `rq2_cancellation_top20_table` | 3.45 in |

Each PDF in `figures/` has exactly one page. Each SVG contains vector paths and an accessible title/description. Minimum figure font size is 8 pt at the stated width. Do not shrink these figures below the supplied widths without checking readability.

- `FIGURE_REVISION_NOTES.md`: changes, sources, and case findings.
- `CAPTIONS_AND_ACCESSIBILITY.md` and `data/captions.json`: short/full captions, ACM Description text, and placement advice.
- `LATEX_INSERTIONS.tex`: short-caption figure insertions. These assume the shared setup in `SETUP_SNIPPET.tex`; full standalone captions are supplied separately.
- `CANCELLATION_MAIN_TABLE.tex`: native LaTeX alternative to the small vector table. Use one version, not both.
- `CANCELLATION_APPENDIX.{md,tex}` and `data/cancellation_appendix_all_comparisons.csv`: all 72 comparisons with raw counts and both denominators.
- `data/figure1_case_inputs.json`: original record, all seven complete inputs, entry identities and positions, and the displayed excerpt.
- `data/figure1_case_audit.json`: pinned model/tokenizer provenance, all seven token counts, rank reconstruction, and source hashes.
- `data/preserved_files.json`: initial hashes of the prior evidence package and manuscript state.
- `qa/`: machine validation and visual inspection record. Local 150-dpi PNG proofs can be regenerated; they are not committed or zipped.
- `DELIVERY_MANIFEST.json`: SHA-256 manifest. `figure_revision.zip` contains this package without intermediate proofs or itself.

## Reproduce

Run from the repository root. Python 3.10 with numpy, pandas, matplotlib, pypdf, pyarrow, and threadpoolctl is used for the plotting/validation pipeline; exact plotting versions are in `qa/runtime.json`. Font files are included. Poppler supplies `pdftoppm` and `pdffonts`. No LaTeX installation is needed to draw the figures.

```powershell
$env:PYTHONIOENCODING = 'utf-8'
python revision_evidence_20260917/figure_revision/scripts/prepare_revision_data.py
```

This checks the preserved-file snapshot and copies the three old plotting CSVs byte-for-byte. It creates Top-20/100 subsets preserving every original decimal string. Plotting never runs the old figure script.

The existing case audit is included. To repeat the full audit, use the historical environment with transformers 4.55.4, the pinned GTE tokenizer/model configuration cached locally, and the existing Phase II/III data and embeddings:

```powershell
./phase2/.venv/Scripts/python.exe revision_evidence_20260917/figure_revision/scripts/audit_teal_case.py
```

This tokenizes only the seven requested inputs and reconstructs one query's ranks using saved vectors and scores. It performs no model inference, training, new permutation generation, or case search. The token cap is checked against the pinned model configuration and the actual encoder's explicit `max_length`, not the tokenizer's unset-length sentinel.

```powershell
python revision_evidence_20260917/figure_revision/scripts/write_revision_docs.py
python revision_evidence_20260917/figure_revision/scripts/plot_revision.py
Get-ChildItem revision_evidence_20260917/figure_revision/figures/*.pdf | ForEach-Object {
    pdftoppm -r 150 -singlefile -png $_.FullName (Join-Path 'revision_evidence_20260917/figure_revision/qa' $_.BaseName)
}
python revision_evidence_20260917/figure_revision/scripts/validate_revision.py
```

With MiKTeX installed, `python revision_evidence_20260917/figure_revision/scripts/check_latex.py` additionally compiles all insertion snippets, both native tables, and the shared setup in an isolated proof at 7.1-inch text width / 3.45-inch column width. It writes only to this package's `qa/` folder, and does not compile or edit the manuscript. The delivered snippets passed two compilation passes without overfull boxes, oversized floats, or missing glyphs.

Inspect the seven PNG proofs at their supplied 3.45-inch / 7.1-inch widths after any visual change. Refresh `qa/VISUAL_QA.md`, then run:

```powershell
python revision_evidence_20260917/figure_revision/scripts/package_revision.py
```

To redraw from the portable ZIP without the full research repository, use the supplied `data/` and fonts and run only `write_revision_docs.py` and `plot_revision.py`. Full provenance revalidation requires the original repository files. SVG metadata or library-version changes may change file hashes on regeneration even when plotted values are unchanged.
