# Reproduction

Run from the repository root in the recorded Python 3.10 environment (NumPy 1.26.4, pandas 2.3.3, SciPy, PyArrow 24.0.0, threadpoolctl, Matplotlib and Jinja2). Exact runtime is in `qa/runtime.json`. Four BLAS threads are used. No model download, inference or training is needed.

The dated protocol is already frozen. Do not rerun `prepare.py` over it: the script deliberately refuses to overwrite the design. That script records the initial preparation and may be used in a separately named copy of the package before scores are viewed.

```powershell
$env:PYTHONIOENCODING='utf-8'
python -m unittest discover -s revision_graded_20260922/scripts -p test_metrics.py -v
python revision_graded_20260922/scripts/audit.py
python revision_graded_20260922/scripts/evaluate.py
python revision_graded_20260922/scripts/analyze.py
python revision_graded_20260922/scripts/state_contrasts.py
python revision_graded_20260922/scripts/report.py
python revision_graded_20260922/scripts/validate.py
python revision_graded_20260922/scripts/finalize.py
```

`data/input_manifest.json` identifies exact required payloads, sizes, hashes, array shapes and Git tracking state. All 445 main inputs were available locally. Git omits some large NPY payloads; restore the author's exact saved artifacts and verify SHA-256 before executing. `data/supplemental_audit_inputs.json` adds raw-label and historical-consumer/cost inputs, without changing the frozen design. Missing inputs cause a failure; there is no fallback to re-encoding.

`evaluate.py` reuses historical all-judged ranks and top lists. It reconstructs full-catalog hybrid rankings from frozen vectors/scores only where those all-judged hybrid ranks were not saved. It also constructs the 12 new canonical hybrids, requiring exact saved canonical judged-rank agreement. Every raw-hybrid highest rank and top-1000 list must match the previous completion. All outputs remain in this dated package.

`analyze.py` creates the shared query bootstrap weight matrices once per dataset and keeps each query's entire method/schedule cluster together. It averages matched schedule contrasts within a query before bootstrapping seven-schedule comparisons. Output tables use decimal 0–1 units (multiply by 100 for percentage points). No primary results use target-only ranks.

`validate.py` checks all supports and source hashes, reproduces 80 historical Recall cells and 42 old-gain raw cNDCG conditions, independently recalculates graded metrics on 366 query/condition records and checks bootstrap/membership logic. Tests and validation outcomes are under `qa/`.
