# Reproduction

Run from the repository root with the local Python 3.10 runtime and NumPy 1.26.4, pandas 2.3.3, PyArrow 24.0.0, Matplotlib 3.10.9, threadpoolctl, pypdf, SciPy and fontTools. Exact versions used for core execution are in `qa/runtime.json`; inherited pinned embedding implementations are in the source manifest. Four CPU BLAS threads are fixed. No GPU inference, training, network model download or paid API is used.

```powershell
$env:PYTHONIOENCODING='utf-8'
python revision_completion_20260921/scripts/audit_sources.py
python revision_completion_20260921/scripts/hybrid.py
python revision_completion_20260921/scripts/analyze.py joint
python revision_completion_20260921/scripts/analyze.py diagnostics
python revision_completion_20260921/scripts/controls_and_cost.py
python revision_completion_20260921/scripts/figures.py
python revision_completion_20260921/scripts/report.py
python revision_completion_20260921/scripts/latex_qa.py
python revision_completion_20260921/scripts/validate.py
python revision_completion_20260921/scripts/finalize.py
```

The audit accepts `--chapter-package PATH` and `--reference-pdf PATH` if the supplied external inputs moved. The initial request attachment is also hashed at its original local path. Do not replace frozen historical artifacts with different model revisions or current downloads. Locally present NPY caches are often represented only by their JSON metadata in Git; consult `data/source_manifest.json` before attempting full numerical reproduction from a fresh clone. Missing payloads must be restored by hash; highest-label ranks or top1000 alone cannot reconstruct full-ranking RRF. No canonical inference is rerun: all 18 rank conditions are read through their September 17 source records.

The delivery ZIP contains the newly derived pair/query tables, bootstrap multiplicity matrices and their sorted query IDs, all plot CSVs, ranks and top1000 catalog indices, reports, scripts and the supplied chapter inputs. It does not duplicate all historical multi-GB embedding caches. Array top1000 indices map to `data/{dataset}_catalog_ordering.csv`; query IDs are saved with each array. Every new pair artifact includes complete Hq, including absent-in-both pairs.

`figures.py` reads inherited September 17 figure data/fonts plus the new diagnostic tables. It uses Matplotlib, not generative imagery. All figure sources, captions, accessible descriptions and commands are in `figures/`. The copied Fig. 3 is byte-identical. Fig. 4 numeric plot CSVs are byte-identical to their historical source CSVs. The two cutoff intervention groups and severity/ECDF plots remain optional appendix/artifact figures; no new main Fig. 5 is imposed.

`latex_qa.py` requires local MiKTeX/acmart, BibTeX and Poppler; automatic package installation and shell escape are disabled. It uses real WANDS/ESCI bibliography entries from the repository, not validation stubs. The proof uses ACM two-column geometry, natural font sizes, and `emergencystretch=3em` to permit clean line breaks; consider that preamble setting when integrating. Model names and dataset/encoder slashes have explicit allowable breaks. The proof's page breaks are local test boundaries, not proposed whole-paper float placements. Updated chapters use `tables/` and `figures/` relative to the manuscript root. Preserve author-controlled Chapters 1--4.

After rendering, visually inspect all seven figures and all pages of the chapter and table proofs. The delivered record is `qa/VISUAL_QA.md`. `validate.py` verifies historical source hashes, package input hashes, primary supports, all K-membership/rank matches, integer membership identities, state partitions, fixed fitting support, monotonic persistent states, vector output and LaTeX diagnostics. `finalize.py` refreshes output hashes and packages results, excluding scratch PNGs/logs and the ZIP itself. No historical `write_results.py` or `write_manuscript.py` is called.
