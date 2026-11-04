# Confirmatory full-catalog visibility and cross-encoder transfer

This independent experiment preserves the completed `value_swap_visibility` pilot. The user explicitly confirmed **GTE + Harrier**. It studies Top-20 exposure for Exact WANDS products when values are swapped (C), with whole-entry reordering (O) as a presentation-order control. It does not infer a pure internal binding mechanism from C−O.

Completed results are in [report.md](report.md); detailed numerical execution and timing history are in [method_notes.md](method_notes.md). Both encoders have all 9,636 expected rows. The final numerical QA checks 19,272 rows, 10,512 source choices and 188 bootstrap estimates/contrasts, and preserves 1,755 existing files. The paper artifacts were rendered and visually reviewed; the review and exact artifact hashes are recorded in `qa/visual_review.json`.

## Frozen design

- WANDS held-out eligible queries minus the pilot's 100 development query IDs: **208 confirmation queries**. Query IDs are disjoint; products need not be.
- Seed `20261103`; SHA256 of canonical JSON `[str(seed), "confirmatory_visibility:" + purpose, *str(ids)]`. Product exchange sets use the `attribute_pairs` hash as the Python Random seed, sampling five distinct valid entry-index pairs without replacement. Their returned order defines j=1…5. Query-target sampling uses independent `target` hashes. Smoke selection uses `smoke`.
- Same conservative missing-value, unique selected-key and first-colon parsing rules as the pilot. Other duplicated keys do not exclude a product. No semantic/type whitelist or query mention filter.
- Select five pairs per product before token auditing. Require all eleven full versions to fit both profiles. Then select up to five Exact targets per confirmation query by hash. Never select by score/rank or substitute a different exchange set to rescue length.
- Final support: **876 query-product pairs, 871 products**; 8 of the 12,161 confirmation-pool products have fewer than five valid pairs, and no five-pair-eligible product fails length. Five products appear under two confirmation queries each (10 pairs); three products overlap the development sample (three pairs). This is not a disjoint-product generalization experiment.
- A is the source-order record. C_j swaps only entire raw value substrings; O_j swaps the same two entire entries. All other content is unchanged. Each product has the same fixed set for every query. Entry indices are zero-based.

All selection, mappings, full inputs, model details and tokenizer audits were frozen before new effectiveness scores. `sample_manifest.json` hashes them and their inherited source files. `data/token_audit_{gte,harrier}.csv` separately records actual untruncated token counts and hashes of token IDs, including model preprocessing/special tokens. Query prefixes and actual query inputs are in `data/query_inputs_*.csv`.

## Existing profiles and compatible baselines

GTE reads the pinned profile and unchanged `phase2/src/encoding.py:DenseEncoder` used by the pilot: `Alibaba-NLP/gte-modernbert-base`, revision `e7f32e3c00f91d699e8c43b53106206bcc72bb22`, empty prefixes, CLS, FP16 model, FP32 pooling/L2 normalization, native 8192 cap, SDPA. The pilot's rebuilt source baseline is reused only after bitwise fresh A/query replay.

Harrier reads `revision_recent_models_20261016/profiles/harrier_20261019/PROTOCOL.json` and its unchanged native adapter: `microsoft/harrier-oss-v1-0.6b`, revision `f9b9dc8d367d443f2479d27aa5d8d2850c0774ee`, last nonpadding token, BF16 native normalization followed by FP32 unit normalization, SDPA, left padding, 32768 cap. The existing web-search query prefix is preserved; document prefix is empty. Its cache holds seven historical representations. Only the first **source-order catalog slice** is used for ranking; other representation rows are never catalog competitors. The original deduplication aliases and batch companions are recovered for replay.

`cache_plan.py` verifies model weights, source-text hashes, catalog/query order, software/profile descriptors and output checksums. `cache_manifest.json`, `cache_plan_*.json`, `model_profiles.json` and `qa/execution_profile_*.json` document the exact execution. No cache with a different numerical profile is mixed in. A replay mismatch stops execution; only a necessary source-order rebuild may resolve it, never a profile substitution.

## Reproduction commands

Run from the repository root with the two existing environments. No package upgrade or model replacement is required. Run one GPU worker at a time. Every command must succeed before the next dependent command.

The following preparation sequence was executed **once**. Do not overwrite the frozen manifest in the delivered experiment; make a new experiment copy to change the design.

```powershell
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONDONTWRITEBYTECODE = '1'
& phase2/.venv/Scripts/python.exe experiments/attribute_value_binding_pilot/confirmatory_visibility_20261103/prepare.py select
& phase2/.venv/Scripts/python.exe experiments/attribute_value_binding_pilot/confirmatory_visibility_20261103/prepare.py lengths --model gte
& revision_recent_models_20261016/.venv/Scripts/python.exe experiments/attribute_value_binding_pilot/confirmatory_visibility_20261103/prepare.py lengths --model harrier
& phase2/.venv/Scripts/python.exe experiments/attribute_value_binding_pilot/confirmatory_visibility_20261103/prepare.py freeze
& phase2/.venv/Scripts/python.exe experiments/attribute_value_binding_pilot/confirmatory_visibility_20261103/cache_plan.py
```

The delivered artifact already contains the packing validation gate described below. Re-run the frozen experiment with:

```powershell
& phase2/.venv/Scripts/python.exe experiments/attribute_value_binding_pilot/confirmatory_visibility_20261103/tests.py
& phase2/.venv/Scripts/python.exe experiments/attribute_value_binding_pilot/confirmatory_visibility_20261103/run.py --stage smoke --model gte
& revision_recent_models_20261016/.venv/Scripts/python.exe experiments/attribute_value_binding_pilot/confirmatory_visibility_20261103/run.py --stage smoke --model harrier
& phase2/.venv/Scripts/python.exe experiments/attribute_value_binding_pilot/confirmatory_visibility_20261103/analysis_smoke.py
& phase2/.venv/Scripts/python.exe experiments/attribute_value_binding_pilot/confirmatory_visibility_20261103/run.py --stage formal --model gte
& revision_recent_models_20261016/.venv/Scripts/python.exe experiments/attribute_value_binding_pilot/confirmatory_visibility_20261103/run.py --stage formal --model harrier
& phase2/.venv/Scripts/python.exe experiments/attribute_value_binding_pilot/confirmatory_visibility_20261103/analyze.py
& phase2/.venv/Scripts/python.exe experiments/attribute_value_binding_pilot/confirmatory_visibility_20261103/paper_artifacts.py
& phase2/.venv/Scripts/python.exe experiments/attribute_value_binding_pilot/confirmatory_visibility_20261103/report.py
& phase2/.venv/Scripts/python.exe experiments/attribute_value_binding_pilot/confirmatory_visibility_20261103/verify.py
```

Completed forwards are resumed only when their full text/profile/context descriptors and vector checksums match. Model checkpoints, WANDS processed inputs and inherited cache artifacts are local prerequisites identified by checksums, not bundled downloads. This preserves the repository's prepared research environment; a bare clone alone is insufficient.

For a durable continuation after an interrupted Harrier run, `finish_pipeline.py` runs missing Harrier inference, the frozen analysis, artifact generation and numerical verification in sequence. It stops on any failure, records `qa/continuation_status.json` and a dated log, and temporarily inhibits Windows automatic sleep for its lifetime. Manual sleep/shutdown can still interrupt it. Do not start it concurrently with another GPU worker; `--wait-pid PID` is available when attaching to an existing worker. Visual review of the rendered artifacts remains a separate final step.

```powershell
& phase2/.venv/Scripts/python.exe experiments/attribute_value_binding_pilot/confirmatory_visibility_20261103/finish_pipeline.py
```

The interrupted November 3 Harrier attempt has no observed termination time. `recovery_audit.py` accounts only for its completed checkpoint timings, including identifiable sleep; it does not label the overnight gap as active work. Its audit is stored separately without rewriting the original attempt record.

To regenerate tables, figures and the report from the delivered raw scores, start at `analyze.py`; no new model inference is needed. `verify.py` additionally reads the local source baselines to check full-catalog A ranks and every observed crossing. `provenance/source_snapshot_manifest.json` maps exact copies of the inherited model/parser/serialization code to their original paths, including the local Harrier profile, for audit.

## Numerical checks and efficiency amendment

The first smoke used exact original batch companions for all variants. It verified original A/query vectors bitwise, lossless intervention strings, complete encoding, unchanged competitors and all 55 full-array sorts per model. `qa/smoke_original/` and `results/smoke_original/` retain this run. A subsequent audit caught that Harrier's original numerical profile requires the full 308-query matrix, rather than one query row. The correction was made before any formal effectiveness scores, and smoke was repeated. Its original-companion vectors are independently re-scored under the correct full-matrix profile; the historical Harrier row-score CSV is not combined with formal results. See `qa/scoring_profile_correction.json`.

Before formal execution, same-shape packing was tested to avoid repeatedly encoding unchanged companions. It retains the original batch size B and padded length L, using an original longest-text guard row. Every usable row position must reproduce A bitwise. Every packed smoke C/O vector must equal the original-companion vector bitwise, with no tolerance. Both gates passed. This is recorded in `execution_amendment.json` and `qa/packing_*`; it changes neither sampled text nor encoder precision/pooling. The updated smoke ranks and scores must equal those from original-companion vectors under each model's inherited scoring profile exactly.

During formal execution, each product's packed A positions are checked again. B<3, unequal variant token lengths, larger padded length, or an A mismatch triggers original-companion encoding for that product. No product or result is excluded by this optimization. Packing multiple independent encoder input rows does **not** combine retrieval interventions: each ranked counterfactual replaces exactly one target embedding in a fresh source baseline.

Reference scoring uses the inherited FP32 operations: GTE uses the prior pilot's query-row × complete-catalog product; Harrier's reference operation uses its entire 308-query matrix × complete-catalog product for A and each replacement. The final Harrier numerical optimization is described below. The relevant confirmation query row is extracted only afterward; development-query rows are discarded and never used in selection or analysis. Every competitor score for the evaluated query stays at baseline. The inherited stable catalog-index tie break is retained. Each rank is independently checked by counting higher-scored competitors plus equal-score predecessors; smoke also verifies complete-array sorting. Catalog size remains 42,994.

The final Harrier execution additionally uses the certified computation in `column_scoring.py`. Its **baseline and ranking arrays remain the entire catalog**, and its query matrix remains all 308 rows. For a target score only, a 256-column source block can be used when every A score in that block is bitwise equal to full-catalog multiplication. All 13,242,152 baseline cells were checked: 164/168 blocks qualify; the other four blocks (23 selected products) use original full-catalog multiplication. All 50 fixed smoke C/O versions also match the full operation for all 308 queries. Each edit still replaces exactly one target row; no multi-target catalog is ranked. Every observed crossing from a certified block receives an additional original full-matrix score check. This numerical optimization is separate from sampling and never removes a product.

Harrier also releases unused GPU allocator cache after a completed forward has returned CPU vectors. This changes no model/input/arithmetic setting. All 55 fresh original-companion smoke batches (A plus C/O) matched every original vector bitwise; GPU reserved memory after cleanup was 1,192 MiB in the check. `execution_efficiency_amendment.json`, `qa/column_scoring_*` and `qa/allocator_*` preserve the gates. The first completed GTE formal output and prior full-matrix smoke are retained under `*_full_matrix_history`; final GTE output must match its first complete score CSV exactly. The initial Harrier A replay was interrupted to activate these validated execution adjustments; its measured checkpoint timings are recorded separately.

For a fresh reconstruction without the delivered packing gates, run `packing.py --model gte` in the GTE environment and `packing.py --model harrier` in the Harrier environment after `cache_plan.py` and before smoke. These validate original-companion and packed embeddings directly. The retained historical smoke CSVs document this execution history; they are not needed as additional training or evaluation data.

To reconstruct the two final Harrier efficiency gates, run `column_scoring.py` in the GTE Python environment (CPU only) and `allocator_check.py` in the Harrier environment after original-companion smoke vectors exist and before the final smoke/formal commands. Both require exact equality; a failed gate must not be used to alter sample membership or silently substitute an encoder/profile.

## Estimands and uncertainty

**RQ1:** Within each pair, average all five C edits or all five O edits. Pool pair means with equal pair weight. Entry conditions on A outside Top-20; exit on A inside; crossing uses all pairs. Raw event counts and trial counts are also reported. In the primary analysis, raw trials = 5 × eligible pairs, but these trials are not independent observations. `score_delta` and `rank_delta` are secondary; summaries use per-pair averages over changes.

The pre-specified yes/no split means an entire selected raw value, after matching-only whitespace/case normalization, equals `yes` or `no`. For each subtype, average its available changes within each pair and pool only pairs containing that subtype. Counts of eligible pairs and raw subtype trials are both supplied; with unequal subtype counts, the pair-weighted rate need not equal raw events/raw trials. These strata describe different eligible subsets and do not establish causal moderation.

**RQ2:** For b=1,3,5, the source encoder selects the highest score among A and the first b members of one family. A wins exact ties, then the smallest j. Apply the selected variant ID to the other encoder without re-selection. Every pair remains in the denominator, including retained A. Report source and target results separately, actual modification rates, entry/exit and ΔInclusion@20 in percentage points. Source gains are selection-affected; transfer concerns the same known query, not unseen-query selection.

**Bootstrap:** 10,000 paired query-cluster draws over the fixed 208 confirmation queries. Preserve all targets, changes and model conditions of each drawn query. Ratio estimands are recomputed from pooled numerator/denominator sums; this retains equal pair weighting despite varying targets per query. The same index matrix is used for every model, C/O, budget and stratum. C−O intervals use paired differences of bootstrap draws. Report percentile 95% intervals and any zero-denominator draws. Budget and stratum comparisons are exploratory, with no multiple-testing correction. Repeated products across queries remain a dependence limitation.

## Deliverables

- `data/`: query partition, eligible/excluded pool, fixed five-pair mappings, selected pairs, product deduplication/overlap map, complete A/C/O text and tokenizer audits.
- `results/formal/*_scores.csv`: all 11 versions per pair per encoder, full-catalog ranks, scores and A-relative deltas.
- `results/rq1_*.csv`: pair averages, conditional rates, raw events/trials, C−O intervals and query concentration.
- `results/rq2_choices.csv`: all source choices and target outcomes for every family and budget; `rq2_summary.csv` and `rq2_C_minus_O.csv` contain estimates and intervals.
- `results/bootstrap_indices.npy` / `bootstrap_manifest.json`: common reproducible cluster draws.
- `paper/table1.{csv,md,tex}`, `table2.{csv,md,tex}`, `figure1.{png,pdf,svg}` and `figure1_data.csv`: standalone four-page-paper artifacts, without modifying the manuscript. LaTeX tables use `booktabs`.
- `qa/`: source replay, original and packed smoke, intervention/length/rank checks, execution details, preservation checks and timing.
- `report.md`: findings, supported/unsupported claims and limitations.

Timers use `time.monotonic()` and Windows `QueryUnbiasedInterruptTime`. Awake elapsed excludes OS sleep/hibernation but includes CPU, I/O, scheduling and GPU wait. It is not pure GPU-kernel time. Wall/monotonic differences and possible suspended intervals are recorded separately. Interrupted runs preserve completed per-forward checkpoints; timing summaries distinguish resumed work from original checkpoint creation.

Exact qrels identify originally relevant products; they do not re-certify corrupted text. Exposure gain does not make a product more relevant and does not automatically demonstrate worse search quality.
