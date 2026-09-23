# Reproduce and resume

Run from the repository root in PowerShell with Python 3.10. Exact installed package versions are in qa/runtime_packages.json; torch 2.11.0+cu128, transformers 4.46.2, CUDA on an RTX 4060 Laptop GPU were used. Pinned model snapshots, empty prefixes, caps and pooling are in PROTOCOL.json. HF_HUB_OFFLINE=1 is set inside encoding/XAI scripts; models must already be available locally. Original libraries are not assumed numerically identical: forward/precision audits accompany new computation.

## Payload restoration

INPUT_MANIFEST.json lists 185 required input files with sizes/hashes and Git availability, including exact original embeddings and pinned model snapshots. The supplemental source manifest covers inherited report-only references. New vector/score arrays are ignored by Git; CACHE_MANIFEST.json inventories local payload hashes. Metadata under cache/coverage and complete pair/schedule result parquets are tracked.

Restore missing original arrays from the author's audited payloads. Do not regenerate them under another revision and label them original. To rebuild new vectors, use the exact frozen plan plus the native encoding command below; runtime-specific numerical differences must be reported. All seeds, supports and view families are fixed.

## Frozen plan and stage commands

The following stages were executed in this package; computation logs and QA records are retained. freeze.py and coverage_prepare.py refuse to replace existing plans. A fresh scientific rerun requires a separate package directory with HERE adjusted to that directory, preserving this run. For the published run, reuse the committed plan and begin with validation.

1. Initial one-time plan/selection: python revision_final_strengthening_20260922/scripts/freeze.py
2. Existing-score boundaries: python revision_final_strengthening_20260922/scripts/boundary.py
3. Control identities: python revision_final_strengthening_20260922/scripts/transitions.py
4. Coverage text/token planning: python revision_final_strengthening_20260922/scripts/coverage_prepare.py
5. Validate all planned serializations: python revision_final_strengthening_20260922/scripts/coverage_plan_validate.py
6. Query-conditioned attribution: python revision_final_strengthening_20260922/scripts/xai.py
7. New native targets, sequentially:
   - python revision_final_strengthening_20260922/scripts/coverage_encode.py --model minilm
   - python revision_final_strengthening_20260922/scripts/coverage_encode.py --model bge_base
8. Score against fixed cached competitors: python revision_final_strengthening_20260922/scripts/coverage_score.py
9. Numerical sensitivity: python revision_final_strengthening_20260922/scripts/coverage_precision.py
10. Attribution accounting/compact table: python revision_final_strengthening_20260922/scripts/supplement_quality.py --xai
11. Figures: python revision_final_strengthening_20260922/scripts/figures.py
12. Result-based reports: python revision_final_strengthening_20260922/scripts/reports.py
13. Final validation/manifests: python revision_final_strengthening_20260922/scripts/validate.py

Run GPU stages sequentially, with XAI finished before coverage encoding. CPU analysis can run independently. Native encoding keeps batch-size/OOM splitting from the inherited encoder; cached queries, model revisions and the full C0 competitor catalog remain fixed. No production latency benchmark is claimed.

BGE's initial batch of 48 reached the 8 GiB GPU memory limit and slowed markedly. After 50 complete WANDS blocks (51,200 inputs), execution resumed with a maximum native microbatch of 24 and unused CUDA allocator memory released between blocks. ENCODING_MEMORY_ADDENDUM.json records this resource-driven adjustment; completed vectors were verified/reused. No input, model, seed or support changed. Earlier metadata without effective_batch_size use native MiniLM128/BGE48; subsequent block metadata explicitly records the microbatch size. Successful-block times exclude interrupted partial work and model reloads.

Scoring was split by completed model while the other model encoded: coverage_score.py --model minilm and coverage_score.py --model bge_base. The default coverage_score.py command combines verified completed model parts when both exist, or executes the full score stage otherwise. All exported final estimates use both models.

## Resume command

Rerun the same encoding command for the interrupted model, for example:

    python revision_final_strengthening_20260922/scripts/coverage_encode.py --model minilm

Each completed block is reused only after verifying its exact input list, encoder-profile fingerprint and payload hash. Missing/incomplete blocks alone are encoded. This cache-only resume was tested after completion; qa/coverage_resume_minilm.log records cache reuse. The original block times remain the measured compute inventory. Do not modify seeds or support after outcomes.

XAI also resumes from individual variant/baseline QA files. It retains failed completeness records; it does not replace a failed case or automatically expand past 256 nodes. Current selected-case results contain 98 passes and two completeness failures, all completed and accounted for.

## Outputs and validation

The boundary stage initially encountered a manifest-construction keyword collision only after all six numerical outputs had been written; the corrected boundary.py --finish-only resumed the final 50-variant manifest. The first coverage planner encountered a path-concatenation error before any plan/output scoring and was corrected before its successful full run. The baseline comparison postprocessor had a pandas attribute-name collision, corrected without recalculating IG. These implementation recoveries did not change scientific selections, seeds, scores or caps. Successful final commands are reproducible from the corrected scripts; scientific failures remain in their QA records.

validate.py checks frozen input hashes and all pre-existing dirty files; exact boundary supports and the inherited discrepancy; all selected XAI runs and accounting; 48 identity partitions and included-column Recall; common coverage supports, original-seven states and per-pair monotonicity; required reports and figure formats. It emits OUTPUT_MANIFEST.json and CACHE_MANIFEST.json. The manifests and validation files exclude themselves to avoid recursive hashes; logs named final_validation.log are excluded while written.

All inferential intervals use 10,000 paired query-cluster percentile draws, seed 2026091701, uncorrected. Machine-readable metrics remain 0–1. Source C0, seven-schedule means, query macro, pair micro and target-only frequencies have distinct meanings.

The repository preserves file bytes with -text. Frozen Windows CRLF files retain their exact hashes; the package's whitespace attribute treats CR as a line-ending component. Generated SVG path formatting has trailing horizontal whitespace removed without changing plotted content.

No optional unexecuted stage is counted as a pass. Training, GTE attribution/new encoding, PI-FT comparisons, joint-catalog exhaustive permutations and ANN benchmarks were not part of this run.
