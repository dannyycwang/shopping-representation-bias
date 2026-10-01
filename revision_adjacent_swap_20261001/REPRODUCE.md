# Reproduce the adjacent-swap experiment

Run from the repository root. Preserve the frozen files; do not rerun `freeze.py` into an already frozen directory. No manuscript files or historical experiments are modified. The original freeze commit precedes all new swap inference. These commands use Python 3.10 and the exact CUDA environment recorded in `execution_profile.json`.

## Recompute analysis and figures from the committed results

```powershell
python revision_adjacent_swap_20261001/scripts/test_core.py
python revision_adjacent_swap_20261001/scripts/analyze.py
python revision_adjacent_swap_20261001/scripts/figures.py
python revision_adjacent_swap_20261001/scripts/report.py
python revision_adjacent_swap_20261001/scripts/validate.py
```

The analysis consumes the two completed `data/*_variants.parquet` files and numerical reports, never a partial sweep. Figures use matplotlib vector PDF output and Poppler `pdftoppm` for independent PNG rendering. Inspect the rendered PDFs in `qa/rendered/`; record visual review in `qa/visual_review.json`. Captions and the manuscript paragraph are English LaTeX fragments, not a modification of the paper.

On the original checkout, use `python revision_adjacent_swap_20261001/scripts/validate.py --check-original-worktree` to additionally verify that the user's pre-existing manuscript/figure edits remain byte-identical. A fresh reproduction clone does not need those unrelated uncommitted edits. Re-rendered PDF timestamps can change hashes, so update the visual-review record only after actually inspecting the new PDF renders.

## Resume or repeat inference using the verified historical reference

```powershell
python -u revision_adjacent_swap_20261001/scripts/infer.py --model minilm
python -u revision_adjacent_swap_20261001/scripts/infer.py --model bge_base
```

Run the models sequentially on the GPU. The code validates source file hashes, exact model revisions, device, versions, tokenizer inputs, reference-vector hashes, and all completed new vector blocks. It does not silently change precision, batch size or device after an error. Query and C0 cache reuse is by native model-input SHA-256 with explicit aliases. Repeated forward checks are additional audit computation, not additional sample selection. The encoding block times include host delays and are not benchmark latency.

Both encoders use eager attention, disabled TF32, deterministic algorithms, FP16 model inference and FP32 pooling/normalization/scoring. MiniLM uses masked mean pooling and limit 256; BGE uses CLS and limit 512. Empty query instructions match the established experiment. Fixed bucket sizes and batch sizes are in the execution profile. A different software/hardware profile is a new experiment: regenerate a consistent C0 reference and create a separate, explicitly labeled freeze/output package rather than overwriting this one.

The original interpreter is `C:\Users\ycw\AppData\Local\Programs\Python\Python310\python.exe`. All installed versions and the GPU driver are recorded in `execution_profile.json`; `requirements-experiment.txt` is exported from that metadata. Install the CUDA 12.8 build of PyTorch 2.11.0 using the PyTorch wheel index and the remaining versions with pip. The model snapshots are:

- `sentence-transformers/all-MiniLM-L6-v2` at `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`.
- `BAAI/bge-base-en-v1.5` at `a5beb1e3e68b9ab74eb54cfd186867f64f240e1a`.

Download those pinned snapshots with Hugging Face `snapshot_download` before running (the experiment itself sets `HF_HUB_OFFLINE=1`). Snapshot file hashes are recorded. Complete processed catalogs, queries and labels live under `phase2/data/processed/`; the held-out eligibility definition is `revision_graded_20260922/PROTOCOL.json`. `freeze_manifest.json` lists exact source hashes. Missing repository data can be restored from the existing `research-backup-2026-09-14` release using `GITHUB_BACKUP_README.md` and `GITHUB_BACKUP_MANIFEST.json`. In particular, the original 18,232,389-byte WANDS catalog is in `research-data-004.zip` with SHA-256 `a3fb53329b9ecb9d0c774641a767e582a6fee1056620fc3c1be3f7c0b145ca67`. Restore the original payloads; regenerated gzip files can have different container hashes even when their decompressed text agrees. Large historical harmonized vectors are local, ignored payloads, not guaranteed to be in Git.

## Rebuild only C0 if historical vector arrays are unavailable

In a fresh reproduction checkout with no existing `revision_adjacent_swap_20261001/cache/<model>/` run, the following generates a clearly labeled reference under this revision. It needs the processed inputs and pinned model snapshots, but no earlier schedule vectors. All competitors and queries use the same frozen profile. Do not use these flags midway through an existing run: its cache manifest rejects a changed reference.

```powershell
python revision_adjacent_swap_20261001/scripts/rebuild_reference.py --model minilm
python revision_adjacent_swap_20261001/scripts/infer.py --model minilm --reference-dir revision_adjacent_swap_20261001/cache/rebuilt_reference_minilm
python revision_adjacent_swap_20261001/scripts/rebuild_reference.py --model bge_base
python revision_adjacent_swap_20261001/scripts/infer.py --model bge_base --reference-dir revision_adjacent_swap_20261001/cache/rebuilt_reference_bge_base
```

The optional rebuild route is supplied for reproduction; the delivered experiment used the verified original harmonized C0 cache. New references are never labeled as historical artifacts. Run the analysis/figure/report/validation commands afterward. Preserve original numerical evidence when comparing a reproduction.

## Generate the cohort in a new output copy

`scripts/freeze.py` constructs the complete eligibility inventory and deterministically selects the cohort. It refuses to overwrite `freeze_manifest.json`. Its output is based only on the processed highest-label judgments, held-out IDs, serialization, and tokenization. C0 inputs already exceeding a context limit are rejected immediately; `all_variants_tokenized_<model>=false` marks that short circuit, and `max_tokens_<model>` then records the observed C0 length rather than a full-family maximum. Every selected product has a complete enumeration and tokenization. The untouched protocol and hashes are the authoritative original freeze.

All scientific outputs, full selected serializations, audits, and compact summaries are tracked. New vectors, historical reference arrays, repeat-vector payloads, logs, and redundant PDF render previews stay local with SHA-256 inventories. No simulated retrieval outcomes are used; synthetic data appear only in unit tests for ranking and preservation invariants.
