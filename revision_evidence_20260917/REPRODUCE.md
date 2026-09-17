# Reproduction

Run from the repository root. All new outputs stay in `revision_evidence_20260917/`; historical sources are read-only. Do not rerun `freeze_design.py`: the design is already frozen, and its SHA-256 is in `PROSPECTIVE_PROTOCOL.sha256`.

The historical model runtime is `phase2/.venv/Scripts/python.exe`. The analysis/plotting runtime used here is the system Python 3.10 with NumPy, pandas, PyArrow, Matplotlib, threadpoolctl, fontTools and pypdf. See `experiments/runtime.json` for exact inference versions and device details. Poppler supplies `pdftoppm` for render verification. No network access is needed when pinned tokenizers and model weights are cached.

```powershell
python revision_evidence_20260917/scripts/analyze_membership.py
./phase2/.venv/Scripts/python.exe revision_evidence_20260917/scripts/audit_fitting.py
python revision_evidence_20260917/scripts/analyze_states.py
python revision_evidence_20260917/scripts/verify_target_rank_sources.py
python revision_evidence_20260917/scripts/audit_precision_differences.py
python revision_evidence_20260917/scripts/audit_embedding_provenance.py
python revision_evidence_20260917/scripts/complete_support_audit.py
./phase2/.venv/Scripts/python.exe revision_evidence_20260917/scripts/run_canonical_controls.py --models gte_modernbert minilm
./phase2/.venv/Scripts/python.exe revision_evidence_20260917/scripts/run_canonical_controls.py --models bge_base --batch-size-cap 16
python revision_evidence_20260917/scripts/analyze_canonical.py
python revision_evidence_20260917/scripts/write_canonical_table.py
python revision_evidence_20260917/scripts/plot_figures.py
python revision_evidence_20260917/scripts/write_evidence_report.py
python revision_evidence_20260917/scripts/validate_outputs.py
```

The converted manuscript fonts are supplied in `assets/fonts/`, so plotting does not require a local TeX installation. `scripts/prepare_manuscript_fonts.py` is an optional rebuild utility for the original MiKTeX Linux Libertine OTF files at the source path recorded in that script; it is not required when using the supplied fonts.

The control runner resumes completed conditions using their recorded hashes and only encodes missing conditions. Four inherited `canonical_raw` controls can be registered without new encoding using `--models minilm bge_base --reuse-only`. The raw RQ2 membership and target-state analyses never run retrieval. All 42 raw schedule rank files are local and match the documented backup; there is no reason to recover or regenerate them.

The delivered runner includes UTF-8 reads, a `--reuse-only` registration option and the inherited-baseline numerical audit added while the original GPU batch was active. That process retained its already-loaded code; these edits did not change serialization, encoder settings, inference or exact ranking. The frozen protocol hashes the shared representation and encoder sources, and each condition records its text hash, rank hash, query-vector hash and embedding metadata. Treat the delivered runner as the reproducible final script, not a byte-for-byte snapshot of the process at launch.

The four BGE conditions were subsequently run in a fresh process using the final runner and a maximum batch size of 16 after a Windows sleep interruption and GPU-memory pressure at batch size 48. `BGE_EXECUTION_ADDENDUM.json` records this change before any new BGE condition completed. It changes execution batching, not the model, native pooling, tokenizer cap, precision, canonical rules or evaluation design; independent floating-point forward passes need not be bitwise identical. See `experiments/EXECUTION_NOTES.md`; no partial ranks from interrupted attempts enter the analysis.

Both historical-full and primary populations are exported. For the new manuscript comparison filter `population=primary_existing_evaluation`; for the main membership plot also filter `aggregation=query_macro`, `reference=C0`. All-21-pair results remain in the same full CSV, explicitly identified. Do not combine full and held-out denominators or treat duplicated ESCI population labels as independent samples.

For a smaller package, share the figures, data, reports, source scripts, protocol and manifests. Cached new catalog embeddings in `experiments/embeddings/` are useful for computational reproduction but are not required to reproduce the figures or statistics once exact ranks exist.
