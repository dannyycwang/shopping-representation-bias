# Retrieval-first extension

All commands run from the research repository root using `phase2/.venv/Scripts/python.exe`. Do not rerun old result writers with the legacy ESCI gains and mix their output with this extension.

Frozen inputs: `config/EXPERIMENT_PROTOCOL.frozen.md`, `config/splits.json`, `config/protocol.sha256`, `config/official_gains.json`, `config/AUDIT_AMENDMENTS.md`. `freeze.py` intentionally refuses to overwrite split IDs. The initial manuscript is archived in `archive/paper_before_retrieval_extension/`.

Execution sequence:

```powershell
phase2/.venv/Scripts/python.exe phase4/scripts/freeze.py
phase2/.venv/Scripts/python.exe -m pytest phase4/tests -q
phase2/.venv/Scripts/python.exe phase4/scripts/audit_and_diagnose.py
phase2/.venv/Scripts/python.exe phase4/scripts/repair_mitigation_summary.py
phase2/.venv/Scripts/python.exe phase4/scripts/verify_legacy_controls.py
phase2/.venv/Scripts/python.exe phase4/scripts/audit_legacy_candidate_ties.py
phase2/.venv/Scripts/python.exe phase4/scripts/run_methods.py --screen
phase2/.venv/Scripts/python.exe phase4/scripts/run_extension_queue.py
phase2/.venv/Scripts/python.exe phase4/scripts/audit_shared_query_cache.py
phase2/.venv/Scripts/python.exe phase4/scripts/complete_evidence.py
phase2/.venv/Scripts/python.exe phase4/scripts/build_presentation.py
phase2/.venv/Scripts/python.exe phase4/scripts/write_manuscript.py
phase2/.venv/Scripts/python.exe phase4/scripts/write_results.py
phase2/.venv/Scripts/python.exe phase4/scripts/write_appendix.py
```

`freeze.py` is only for a fresh reconstruction; the delivered repository already has frozen IDs and intentionally refuses to run it again. For the delivered snapshot, start with the audit/evaluation steps or reuse existing artifacts. After generation, inspect the outcome interpretation, build `paper_www2027/main.tex` with pdflatex → bibtex → pdflatex twice, visually verify the PDF and save `results/pdf_validation.json`, then run `write_report.py` and `finalize_provenance.py` to produce the final Chinese synthesis and inventory. The manuscript's abstract, metadata and reviewed interpretation are maintained in `main.tex`; generation does not invent a positive outcome.

The queue waits for `selection.json`, then runs fixed evaluation cells, m=7 saturation, new-query preparation/evaluation, canonical reranking and timing sequentially. Resume failed stages from the command recorded in the queue script. Model text caches are content-addressed; old result artifacts are read-only. Selection JSON cannot be changed by rerunning development selection with a different winner.

## Result layout

- `results/wands_dev/`: screening labels and metrics only for the frozen 96 development queries.
- `results/{wands,esci,esci_new}/`: per-query metrics, every judged pair's rank/score, full exact score arrays, and unique product indices through Top-1000. WANDS evaluation summaries filter the 384 held-out queries, independently for each metric's eligibility.
- `results/*_primary_contrasts.csv`: direct query-paired bootstrap differences; no p-value-based winner selection.
- `results/*_transitions.csv`: per-query rescued/newly missed/net counts and fractions at six K values.
- `results/*_visibility_states.csv`: original seven-view always/sometimes/never retrieval.
- `results/*_diagnostic_pairs.parquet`: original rank/score margin, tokenizer lengths, attribute/query features, and crossing indicator.
- `results/esci_gain_repair*`: legacy versus benchmark-standard effectiveness from unchanged rankings.
- `results/repaired_legacy_mitigation.csv`: Original-relative and direct set-mean–canonical comparisons, with the repaired ESCI gains.
- `results/*_canonical_reranking*`: common-text reranking, candidate-union inference cache, final Top-20 and metrics. Returned-list condensed NDCG normalizes against all judged gains; it must not be confused with full-catalog cNDCG.
- `results/*_exact_timing.csv`, `*_query_encoding_timing.csv`: repeated local timing; sparse-posting/dense-vector array bytes exclude vocabulary/text metadata and ANN overhead.
- `results/*_unique_views_m*.parquet`: requested versus unique text views and exact distinct encoded vectors.
- `data/catalog_manifest.json`: separate common catalog for frozen new ESCI queries. These comparisons do not establish product-disjoint generalization.

No model training or paid API usage is involved. Batch sizes were reduced for local memory efficiency; raw fields, encoder revisions, precision, token budgets and selection criteria remain fixed. Reports must retain this post-hoc study history and all negative results.

Additional evidence: `shared_query_cache_audit.csv` explains the separate historical set-mean query cache; `saturation_summary.csv` and `encoding_costs.csv` retain all fixed view counts and cache construction times; `selected_rule_coverage.csv` quantifies literal rule coverage; `recall_equivalent_candidate_budget.csv` is a discrete recall match; `recall_cost_frontier.csv` combines measured retrieval costs with explicitly estimated amortized candidate inference. Full m=2/4/7 exact timing is retained even when the selected strategy is a single-vector rule. Historical cached build times are not represented as matched cold-build measurements.
