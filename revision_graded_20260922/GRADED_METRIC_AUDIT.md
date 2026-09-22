# Graded metric audit — 22 September 2026

P0 and P1 are complete. The package contains 122 conditions and 49,227 per-query records. No encoding, model update, training, or manuscript integration was performed. P2's optional 16/32-schedule expansion was deferred under a budget of zero new encodings.

## Repository and provenance

Current branch: `codex/chapter56-completion-20260921`, HEAD `edf4b3258677f5bc58c1957508fda0c1313285e3`. It is the direct parent of reference merge `32b24160afcaa229a8b7057c04ecf638c7fae488`; their tracked trees are identical. The reference was fetched without switching branches. No applicable AGENTS.md was found in the repository or workspace ancestors. Existing working changes and historical outputs were preserved and hashed. All new work is under this dated directory.

The frozen protocol SHA-256 is `0810361666810b8ec8aca294cda26e9ad23fc50ea381e8ebbd686c26bab2747e`. It was written before new score evaluation. `data/input_manifest.json` inventories 445 available files, including ignored local payloads and Git tracking status; `data/supplemental_audit_inputs.json` hashes 24 read-only label/cost/consumer/score audit inputs. The four authoritative ascending-canonical full-score hashes also match their expected hashes in the frozen historical manifest. No required payload is missing. Git alone does not distribute every embedding or full-score payload. Restore the exact listed artifacts when reproducing elsewhere; never regenerate them under a different model revision. Model revisions, profiles and source paths remain in the hashed historical metadata and `encoder_profiles_and_rank_sources.csv`.

## Actual label and grade trace

The [ESCI paper, Section 3.1](https://arxiv.org/pdf/2206.06588) assigns direct gains E=1, S=.1, C=.01, I=0. Phase II's `phase2.json` instead declares E=1, C=.1, S=.01, I=0. `prepare_data.py` maps the original `esci_label` into the stored `grade` column, and `evaluation.py` uses that numeric column directly. This is an actual data-path issue, not only a documentation error.

`qa/raw_label_trace.json` verifies original ESCI example labels against every processed query/product judgment. On the frozen eligible support there are 4,434 E, 3,530 S, 505 C and 1,650 I judgments. All stored S grades are .01 and C grades .1. WANDS has 21,299 Exact, 96,566 Partial and 44,353 Irrelevant judgments; its 3/1/0 gains remain unchanged. The WANDS raw-label audit also reproduces the historical conflict-removal and deduplication rule.

The correction was already implemented in Phase IV's `pair_metrics`, and a historical audit recomputed 79 ESCI rank artifacts. The new follow-up aligns full-ranking nDCG and condensed cNDCG on the exact 308/499 query populations. It does not claim that all historical outputs were uncorrected. `tables/historical_metric_consumers.csv` and `data/historical_gain_repair_inventory.csv` identify consumers and earlier repairs.

Affected historical outputs include Phase II cNDCG, gain/rank Spearman, SAR, paired graded mitigation contrasts and their derived success/frontier classifications; Phase III stored-grade audits, official judged-pool NDCG, frozen Marqo and invariant-method graded reports, and downstream Pareto summaries. Grade>0 support is unchanged by swapping two positive gains. Phase III alpha selection used WANDS development cNDCG; Phase IV rule/view selection used WANDS Recall@100 with corrected cNDCG as a tiebreaker. Those selections are unaffected by the ESCI gain swap. The reused pretrained encoders, query vectors, canonical construction and BM25/RRF scores do not train on these gains. Separate later WANDS training studies use WANDS labels/gains and are outside this follow-up; there is no evidence that this ESCI reporting correction changes the frozen retrieval training pipeline.

`GAIN_MAPPING.json` preserves old and corrected mappings. New gains are always computed from labels. The `old_*` fields are counterfactual evaluation of the same rankings under the old mapping; for newly constructed hybrids they are not historical published measurements.

## Evaluation scope and limitations

WANDS: 308 eligible queries, 21,299 Exact pairs, all 162,218 judgments for those queries, full 42,994-product catalog. ESCI: 499 eligible queries, 4,434 E pairs, all 10,119 judgments, fixed 10,076-product pooled catalog. Exact IDs and ordered catalog/query axes are exported under `data/`. The remaining historically requested queries are excluded only to hold the stated highest-label-eligible population fixed. This differs from older positive-IDCG-only population summaries and must not be compared as though the denominators matched.

nDCG@10/@20 discounts at original full catalog ranks. Unjudged items have zero gain for this computation only; their true relevance is unknown. IDCG sorts the same query's available judged direct gains, without an exponential transform. cNDCG@10/@20 removes unjudged items before assigning positions and is always named as condensed. JudgedCoverage uses original top-K IDs, counting every judged label including zero-gain ones. Target-only counterfactual ranks are never combined into a query ranking. Full-ranking nDCG is a secondary measure under incomplete judgments, not a claim of complete relevance assessment; coverage and condensed results are reported alongside it.

Raw retrieval includes all 42 dataset/encoder/schedule configurations. All 18 pure canonical conditions are reused (GTE is secondary). MiniLM/BGE controls include Set-Mean, centroid M=2, multi-vector M=2, BM25 and all seven raw hybrids. BGE M=4/7 centroid/multi-vector conditions remain secondary. C0 and the seven-schedule query mean are distinct. Every raw hybrid is paired with its corresponding raw schedule before query-level averaging. The common historical query embeddings, frozen tokenization/model metadata and fixed catalog-index tie rule are preserved.

All 18 reconstructed canonical judged rankings match their saved ranks exactly on the frozen support. Reconstructed raw scores differ at six non-highest judged ranks: WANDS BGE at 442/6,384/4,340/5,737 and ESCI BGE at 7,545/7,548, each by one rank. The exact records are in `qa/rank_precision_differences.csv`; saved raw ranks remain authoritative. Highest-label ranks match exactly. Every reconstructed raw hybrid matches its authoritative highest-label ranks and top-1000 IDs; all C0 hybrid judged ranks and full scores also match. No truncated candidate fusion was used.

The evaluator explicitly verifies identical highest-label sets and inclusion IDs under old/new gains for every query, including Recall@20/@100. Validation additionally reproduces 42 historical old-gain raw cNDCG conditions and 80 historical Recall summary cells. Eight small tests ran before reevaluation; an independent dense gain-vector implementation checks 366 query/condition records. All 469 source-hash records and pre-existing edits are rechecked.

## Correction magnitude on raw C0

| model | metric | old_mean | corrected_mean | delta |
| --- | --- | --- | --- | --- |
| bge_base | nDCG@20 | 0.652517 | 0.662921 | 0.010404 |
| bge_base | cNDCG@20 | 0.790147 | 0.813336 | 0.023189 |
| gte_modernbert | nDCG@20 | 0.637028 | 0.643482 | 0.006454 |
| gte_modernbert | cNDCG@20 | 0.790749 | 0.81231 | 0.021561 |
| minilm | nDCG@20 | 0.5935 | 0.603631 | 0.010131 |
| minilm | cNDCG@20 | 0.771138 | 0.796114 | 0.024976 |

All WANDS gain-correction differences are exactly zero. ESCI graded values change while rank and highest-label inclusion do not. `tables/gain_correction.csv` retains all conditions and paired intervals for the correction.

## Membership change and nDCG

Across the six source-order contrasts per encoder, the following ranges retain every contrast and use each contrast's changed-query denominator. Full records and separate nDCG@10 diagnostics are provided; K10 conditioned on K20 membership is explicitly a different-cutoff diagnostic.

| Dataset | Encoder | All queries | Changed queries | Median |delta nDCG20| | 95th percentile | Changed with |delta| > .01 | Numerical zero, changed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| WANDS | MiniLM | 308 | 195–211 | 0.0330–0.0399 | 0.1208–0.1914 | 74.0–79.2% | 18–25 |
| WANDS | BGE | 308 | 179–187 | 0.0226–0.0278 | 0.0876–0.1149 | 64.1–69.8% | 21–27 |
| WANDS | GTE | 308 | 189–198 | 0.0250–0.0288 | 0.0957–0.1248 | 66.2–73.3% | 25–29 |
| ESCI | MiniLM | 499 | 60–92 | 0.0410–0.0475 | 0.1191–0.1333 | 81.1–96.2% | 0–1 |
| ESCI | BGE | 499 | 55–78 | 0.0401–0.0469 | 0.0828–0.1271 | 83.3–91.0% | 0–2 |
| ESCI | GTE | 499 | 92–123 | 0.0388–0.0482 | 0.1103–0.1452 | 83.7–92.3% | 0–0 |

Among changed queries, |delta nDCG@20| exceeds .01 in 64.1–79.2% of WANDS cases and 81.1–96.2% of ESCI cases across dataset/encoder/schedule cells. Thus small mean shifts cannot be generalized into negligible query-level graded change. Raw mean nDCG@20 deltas span -0.01101 to +0.00151 on WANDS and -0.00393 to +0.00092 on ESCI. Five of the 36 raw schedule contrasts have uncorrected intervals wholly below zero; the other 31 include zero. This does not establish equivalence.

Numerical zero with changed membership is observed: WANDS has 18–29 such query records per raw schedule contrast; ESCI has 0–2. All observed examples are retained in `data/observed_numerical_zero_membership_records.csv`, including gains/losses and integer Recall cancellation. Numerical equality uses absolute tolerance 1e-12, relative tolerance 0; it is distinct from exact integer gains=losses. These examples support a limited non-identification statement, not a stable-aggregate claim. Thresholds .001, .005 and .01 were frozen and are descriptive sensitivity values, not tests of equivalence.

## Canonical–lexical hybrid

Twelve fixed indexes combine BM25 and one of the three pure canonical dense rankings. Each branch ranks the entire catalog; fusion is float32 1/(60+r_BM25)+1/(60+r_dense), with stable original catalog-index ties. No new weight or rule is selected. All complete entries and duplicate multiplicities are retained. Canonical input equality is verified over all seven incoming schedules, and intrinsic sorting plus deterministic branch ranking makes VI=0 structural. One fixed index suffices; seven encodings were not performed.

| Dataset | Encoder | Rule | nDCG20 vs dense | nDCG20 vs BM25 | nDCG20 vs raw hybrid mean | Recall20 vs raw hybrid mean |
| --- | --- | --- | --- | --- | --- | --- |
| WANDS | MiniLM | Ascending | +0.0696 [+0.0558, +0.0837] | +0.0412 [+0.0274, +0.0555] | -0.0023 [-0.0065, +0.0020] | +0.0045 [-0.0032, +0.0133] |
| WANDS | MiniLM | Descending | +0.0638 [+0.0493, +0.0786] | +0.0431 [+0.0288, +0.0576] | -0.0004 [-0.0050, +0.0043] | +0.0017 [-0.0050, +0.0090] |
| WANDS | MiniLM | Type/style | +0.0676 [+0.0540, +0.0812] | +0.0441 [+0.0298, +0.0589] | +0.0005 [-0.0040, +0.0052] | +0.0058 [-0.0010, +0.0127] |
| WANDS | BGE | Ascending | +0.0259 [+0.0142, +0.0378] | +0.0691 [+0.0554, +0.0832] | -0.0014 [-0.0040, +0.0011] | -0.0062 [-0.0137, -0.0008] |
| WANDS | BGE | Descending | +0.0249 [+0.0129, +0.0366] | +0.0724 [+0.0582, +0.0869] | +0.0018 [-0.0009, +0.0045] | +0.0030 [-0.0016, +0.0082] |
| WANDS | BGE | Type/style | +0.0232 [+0.0116, +0.0349] | +0.0714 [+0.0572, +0.0860] | +0.0008 [-0.0018, +0.0035] | -0.0045 [-0.0117, +0.0004] |
| ESCI | MiniLM | Ascending | +0.0377 [+0.0231, +0.0518] | +0.0241 [+0.0123, +0.0358] | -0.0012 [-0.0039, +0.0010] | -0.0001 [-0.0026, +0.0022] |
| ESCI | MiniLM | Descending | +0.0397 [+0.0254, +0.0533] | +0.0254 [+0.0139, +0.0369] | +0.0001 [-0.0019, +0.0019] | +0.0006 [-0.0023, +0.0037] |
| ESCI | MiniLM | Type/style | +0.0378 [+0.0234, +0.0519] | +0.0241 [+0.0123, +0.0358] | -0.0013 [-0.0040, +0.0010] | +0.0000 [-0.0025, +0.0023] |
| ESCI | BGE | Ascending | +0.0078 [-0.0064, +0.0215] | +0.0559 [+0.0454, +0.0663] | +0.0020 [+0.0002, +0.0043] | +0.0017 [-0.0003, +0.0038] |
| ESCI | BGE | Descending | +0.0078 [-0.0059, +0.0213] | +0.0539 [+0.0433, +0.0646] | +0.0000 [-0.0019, +0.0023] | -0.0008 [-0.0033, +0.0017] |
| ESCI | BGE | Type/style | +0.0080 [-0.0063, +0.0217] | +0.0558 [+0.0453, +0.0663] | +0.0020 [+0.0001, +0.0042] | +0.0019 [+0.0000, +0.0040] |

Values are signed differences on the 0–1 metric scale, with uncorrected paired 95% query-cluster bootstrap intervals (10,000 draws, seed 2026091701). Canonical hybrids improve nDCG@20 over BM25 in all 12 cells and over their corresponding dense branch in nine; the three ESCI/BGE dense-branch comparisons are inconclusive. Against the mean of seven raw-hybrid references, ten nDCG@20 contrasts are inconclusive, while ESCI/BGE ascending and type/style show small positive differences under uncorrected intervals. All 12 nDCG@10 and Recall@100 comparisons against raw-hybrid means are inconclusive. WANDS/BGE ascending reduces Recall@20 by 0.00624 (CI -0.01370 to -0.00084); ESCI/BGE type/style increases it by 0.00195 (CI +0.00001 to +0.00402), also uncorrected. These metric-specific results prevent a claim of universal preservation or dominance.

Persistent inclusion/omission, mean/source Recall and VI are in `tables/persistent_states.csv` with query-macro and pair-micro results separated. `tables/canonical_hybrid_state_contrasts.csv` supplies paired intervals for each persistent-state difference against the own dense branch, BM25 and the seven-schedule raw hybrid. Every P1 membership contrast against its dense branch, BM25 and all seven raw-hybrid references is retained, plus matched query-level means. The graded nDCG analysis is query-macro; it does not silently substitute pair weighting.

Each canonical hybrid stores one dense vector per product plus the fixed sparse BM25 index. Dense+sparse array bytes are 118,698,552/184,737,336 for WANDS MiniLM/BGE and 24,049,044/39,525,780 for ESCI MiniLM/BGE. Exact inference scores one dense catalog and one lexical branch and performs two branch sorts plus a fusion sort. `tables/canonical_hybrid_costs.csv` separates historical dense encoding time, zero new encoding, measured local batch work and unmeasured online latency. It excludes model weights, vocabulary/metadata overhead and research result files. Inherited costs retain M=2/4/7 construction budgets, actual vector counts and duplicate views; centroid scoring uses one vector after construction, while multi-vector scoring retains M vectors. No ANN or cold deployment latency is claimed.

## Deliverables

`PROTOCOL.json`, `GAIN_MAPPING.json`, manifests, exact supports, all-query metrics, paired differences, membership/ECDF records, CSV/LaTeX tables, figure sources, `MANUSCRIPT_INSERTIONS.md`, and `CLAIM_VERDICTS.md` are included. `REPRODUCE.md` gives commands. Historical manuscripts are untouched; the English insertion is for review. Optional finite-permutation expansion and universal robustness claims are not part of this execution.
