# Reproduce this scoped revision

Run from the repository root in PowerShell. The reference environment and exact packages are in `qa/environment.json`; each harmonized model's full profile pins CUDA/GPU, libraries, model/tokenizer commits, attention, precision, deterministic settings, padding and batching. Python 3.10, numpy, pandas, pyarrow, scipy/threadpoolctl, matplotlib, PyTorch/Transformers and the pinned local Hugging Face snapshots are required. Poppler and MiKTeX/acmart provide PDF QA/proof. No paid API is used.

```powershell
# Preserve the existing P0 snapshot; do not overwrite the audit baseline.
python revision_section6_20260924/scripts/verify_sources.py
python revision_section6_20260924/scripts/trace_rank_sources.py
python revision_section6_20260924/scripts/analyze_saved.py
python revision_section6_20260924/scripts/ledger.py
python revision_section6_20260924/scripts/figures.py
python revision_section6_20260924/scripts/harmonize.py --model minilm
python revision_section6_20260924/scripts/harmonize.py --model bge_base
python revision_section6_20260924/scripts/summarize_harmonized.py
python revision_section6_20260924/scripts/ledger.py
python revision_section6_20260924/scripts/reports.py
Push-Location revision_section6_20260924
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=qa qa/section6_acm_proof.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=qa qa/section6_acm_proof.tex
pdftoppm -r 144 -png qa/section6_acm_proof.pdf qa/proof
Pop-Location
# Inspect all current figure PDFs/PNGs and every proof page, then record
# that review in qa/visual_review.json before the final validation.
python revision_section6_20260924/scripts/validate_final.py
python revision_section6_20260924/scripts/package_review.py
```

On the audited Windows host, the exact interpreter is `C:\Users\ycw\AppData\Local\Programs\Python\Python310\python.exe`; use that executable in place of `python` if another environment is active. Stop on any command failure instead of consuming stale downstream outputs. The initial snapshot in `qa/starting_state.json` belongs to the original audit; preserve it with the package when making a separate reproduction copy. `p0_audit.py` created that snapshot initially and is not a resume command. Source verification accepts the audited commit or descendant delivery commits confined to this revision directory, while continuing to require all original source and pre-existing edit hashes. A fresh clone needs those local historical artifacts restored before it can reproduce the complete local audit.

Run GPU models sequentially. CPU saved-output analysis may run while encoding. `harmonize.py` verifies the immutable profile, planned aliases and every completed vector block before reuse; it recomputes only unfinished blocks. It refuses a changed profile in the same directory. A new scientific profile needs a new output version. Do not relabel newly generated vectors as historical payloads. If the device sleeps, completed blocks remain saved; block elapsed times may include the pause. Do not interpret them as benchmark latency.

The initial 82 rank/label conditions are enumerated in `revision_graded_20260922/data/conditions.json`; the authoritative `data/per_query.parquet` in that historical revision also records 40 later raw/canonical-hybrid conditions, for 122 total. The new provenance table merges both sources and checks every rank payload against the earlier input or output hash manifest. Original representations are `phase2/data/representations/<dataset>/C*.jsonl.gz`; native target-only ranks are in `phase3/results/target_only_permutations/`. Pinned expansion plan and cohort are under `revision_final_strengthening_20260922/data/`. `data/source_inventory.csv` and `data/historical_execution_matrix.csv` contain exact source/array hashes and explicit unavailable metadata. Large NPY payloads/model snapshots may be local/ignored; Git presence is not a guarantee of payload availability. Restore original missing artifacts using their hashes, never regenerate and label them original.

`data/method_rank_provenance.csv` maps all 122 saved method/schedule conditions to their exact rank hashes and available embedding sidecars, with historical input-manifest agreement and unavailable runtime fields distinguished. It supplements the raw seven-schedule execution matrix instead of treating a shared model name as complete execution provenance.

Some historical control sidecars lack exact per-array library versions. Common query support and checked output hashes do not establish bitwise matched execution across every historical method. Those comparisons retain their historical provenance; only the separately harmonized order-expansion supplement establishes the new uniform execution profile.

All generated evidence is confined to this revision directory. Original files, manuscript sources, historical results, prior failure records and dirty working-tree edits must remain byte-identical. Existing equivalent analyses are reused; the extra native implementation is limited to the harmonized supplement. No training, model sweep or attribution repair is a hidden dependency.

To resume after the graded stage only (same verified files):

```powershell
python revision_section6_20260924/scripts/analyze_saved.py --resume-controls
```

For the local ACM proof, run pdflatex twice from this revision directory with `-interaction=nonstopmode -halt-on-error -output-directory=qa qa/section6_acm_proof.tex`. This validates the supplied section and assets, not the unavailable full matching manuscript. PDF/PNG figures are emitted directly by matplotlib at seven-inch width with vector text/marks. Exact plotted records are `data/figure3_plotted.csv`, `data/figure4_plotted.csv`, `data/table2_plotted.csv` and `tables/main_table_candidate.csv`.
