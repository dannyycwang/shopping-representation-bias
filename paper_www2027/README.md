# WWW 2027 paper package

## Build

From this directory, run:

```powershell
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

The class line is `\documentclass[sigconf,anonymous,review]{acmart}`. The draft contains no author identities or placeholder DOI. The official research-track CFP specifies 8 main pages and at most 12 total including references and appendix in one PDF: https://www2027.thewebconf.org/research-track-papers/ . The prior 10-body-page reading draft is archived at `../phase4/archive/paper_before_retrieval_extension/`.

## Structure

- `main.tex`: metadata, abstract, and section assembly
- `sections/`: paper body
- `sections/appendix.tex`: current integrated appendix; top-level `appendix.tex` and `supplementary.pdf` are historical files
- `references.bib`: bibliography
- `figures/`: current scientific figures and historical figures; only figures referenced by the current sections belong to this revision
- `generated/`: tables generated from the audited Phase IV result artifacts
- `RESULT_PROVENANCE.md`: mapping from claims to source CSV fields
- `main.pdf`: compiled review copy

## Regenerate figures and validate results

Use the full reproduction chain in `../phase4/README.md`. Do not regenerate the current paper using old Phase III effectiveness plotting scripts: they retain the historical ESCI gain convention. The current claim-to-source mapping is in `RESULT_PROVENANCE.md`; model revisions and historical artifacts remain intact. Unreferenced historical Pareto/effectiveness figures are not corrected current results.
