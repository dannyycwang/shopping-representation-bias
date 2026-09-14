# WWW 2027 paper package

## Build

From this directory, run:

```powershell
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

The class line is `\documentclass[sigconf,anonymous,review]{acmart}`. The draft contains no author identities. Conference metadata is provisional until the official WWW 2027 submission template and rights block are published.

## Structure

- `main.tex`: metadata, abstract, and section assembly
- `sections/`: paper body
- `appendix.tex`: additional controls and provenance note
- `references.bib`: bibliography
- `figures/`: vector figures copied from the Phase III result directory
- `RESULT_PROVENANCE.md`: mapping from claims to source CSV fields
- `main.pdf`: compiled review copy

## Regenerate figures and validate results

```powershell
& ../phase2/.venv/Scripts/python.exe ../phase3/scripts/make_phase3_figures.py
& ../phase2/.venv/Scripts/python.exe -m pytest ../phase3/tests -q
```

Figure source CSVs are in `../phase3/results/phase3_figures/`. Model revisions and experimental configurations remain in the Phase II and Phase III result metadata.

