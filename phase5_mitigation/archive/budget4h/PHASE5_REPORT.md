# PHASE5_REPORT — 預算限定的 Phase V 篩選

**執行狀態：已達凍結的 4 小時上限，完整實驗矩陣未完成。這是已完成部分的正式交付，不是整個 Phase V 已完成的宣告。**

## Executive summary

**A. KEEP CURRENT PAPER STRUCTURE。** 目前已完成的兩個 learned trials 沒有支持明顯更好的 robustness–effectiveness trade-off；不足以升級為 diagnosis + solution。這是依現有證據的保守建議，不是否定尚未完成的 M3 或尚未訓練的 M4。

- Perm-FT_s42: Recall@100 62.771%，VI@20 10.487%，相對 VI 降幅 12.11%；Robust@100 57.744%，Never@100 32.345%。
- Perm-FT-cons0.05_s42: Recall@100 63.085%，VI@20 12.737%，相對 VI 降幅 -6.75%；Robust@100 57.547%，Never@100 31.431%。

兩個 learned trials 均只有 seed 42。Consistency 的 .05 是固定執行順序中的第一個 trial，**不是 best lambda**。λ=.1 僅完成訓練與 C0 embedding；λ=.5 未訓練；M4 未訓練，因此不能提供其 ≥100 商品的 trained-model invariance test。已有合成輸入架構單元測試不替代這項實驗。

## Exact setup / Method definitions / Training / Hyperparameters

# Learned mitigation screening

This directory implements the newly requested Phase V. `phase5/` contains the
earlier rewriting audit and is not overwritten. The manuscript is not edited.

## Audit and prospective decisions

The repository has inference/evaluation infrastructure and a frozen 96-query
development / 384-query historical held-out split, but no WANDS BGE training
pipeline or training loss. Consequently, these are adapted small-budget training
baselines, not reproductions of an existing fine-tuning procedure. The 96 old
development queries are hash-partitioned into 72 training and 24 validation
queries. The old held-out set remains unchanged; it is not pristine confirmation.

Training uses a shared query/product BGE encoder and pairwise softmax loss over
one Exact product and one explicitly Irrelevant product from the same training
query, temperature 0.05. Partial and unjudged items are never sampled as negatives.
At most 16 Exact products per training query, chosen by hash independently of
ranks. One epoch, microbatch 2, accumulation 8, AdamW lr 1e-5, weight decay .01,
gradient norm cap 1, full encoder fine-tuning with activation checkpointing and
FP16 autocast. M3 adds consistency between two positive permutations; its second
positive never enters the negative pool. M2 and all three M3 trials use seed 42
and identical supervised examples. Both positive and negative products receive
complete-atom permutations. No dropout of fields. The retrieval loss is identical
across M2/M3/M4, but there is no unaugmented-training control; improvements cannot
be attributed uniquely to augmentation rather than domain fine-tuning.

M4 freezes BGE and uses a 768->128->1 tanh attention scorer over independent
attribute embeddings, without positional inputs. Weighted attribute means are
normalized before the existing equal-weight nonattribute/attribute interpolation
and final normalization. Empty attribute sets contribute zero, as in Set-Mean.
One epoch over the same training triples, batch 16, AdamW lr 1e-3. No architecture
search or query tuning. Its 98,561 parameters are the only trainable parameters.

## Frozen selection and budget

Run M2 seed 42, M3 lambda .05/.1/.5 seed 42, then M4 seed 42. Select M3 on the 24
validation queries only: candidates within 1 absolute percentage point of Original
seven-order mean Recall@100, then lowest query-macro VI@20, then highest robust
coverage@100, then smaller lambda. If none passes the recall screen, apply the
latter ordering among all and explicitly report failure. This threshold is an
engineering screening margin, not a statistical equivalence test.

Additional seeds 43/44 are allowed only after the first screening if a method has
at least 25% lower VI@20 and loses at most 1 recall point on validation, or is
invariant and improves over Set-Mean without losing more than 1 point to Original.
They must fit the total 4-hour GPU-task wall-time cap. The cap includes inference
and is checked between batches; interrupted runs remain incomplete, not failures.
No other datasets/models, text generation, paid API, or large sweeps.

All 42,994 catalog products remain candidates. Evaluate the exact seven original
serializations via the existing `build` function, CLS pooling, 512 tokens, unit
normalization, FP32 dot products, and stable catalog-index ties. Cache identity
includes checkpoint identity and actual text hashes. Reuse baseline ranks and
embeddings. Canonical means pure raw-atom sorting, not the M2 field template.

Recall is query-macro, averaged across seven catalog schedules (also retain C0
and individual schedules). VI/robust/never are first computed over seven ranks
per Exact pair, then query-macro; retain pair-micro as supplementary output.
Target-only uses the existing `rank_target` function with each model's own C0
competitors and separately inserts each target. Its average inclusion is not a
single-index recall. Bootstrap paired query deltas with 10,000 draws, seed 20260963.
Report the common fully-fitting target subset without shrinking the catalog;
tokenizer and inputs are shared by Original/M2/M3. Seven-order worst observed is
not an all-permutations bound. Zero VI alone is not success.

Commands (from repository root):

```
phase2/.venv/Scripts/python.exe phase5_mitigation/screen.py freeze
phase2/.venv/Scripts/python.exe phase5_mitigation/screen.py run
phase2/.venv/Scripts/python.exe phase5_mitigation/screen.py report
```

The machine-readable frozen config, IDs, hashes, checkpoints, training logs,
rank-level files, validation selection and figure are kept under this directory.


### 實際資料與監督範圍

| split      |   queries |   eligible_exact_queries |   highest_pairs |   unique_highest_products |   highest_pairs_with_training_product |
|:-----------|----------:|-------------------------:|----------------:|--------------------------:|--------------------------------------:|
| train      |        72 |                       54 |            3635 |                      3613 |                                   575 |
| validation |        24 |                       17 |             536 |                       534 |                                     5 |
| test       |       384 |                      308 |           21299 |                     18010 |                                   413 |
72 個訓練查詢中 54 個有 Exact，產生 546 組 supervised triples；24 validation 查詢中 17 個有 Exact；384 歷史 held-out 查詢中 308 個有 Exact。產品目錄跨 split 共用，表內保留訓練產品與 held-out targets 的 overlap，不把 product-disjoint generalization 當成本輪結果。

## Table V1 — catalog-wide quality and robustness

數字為百分比；Delta 為百分點。破折號表示缺結果，不表示零。選定 consistency 的必要 row 保留為未完成，另列已完成但未選定的 .05 trial。

| Method                                  | Status        | Recall@20   | Recall@100   | VI@20   | VI@100   | Robust@100   | Never@100   | Delta Recall@100   |
|:----------------------------------------|:--------------|:------------|:-------------|:--------|:---------|:-------------|:------------|:-------------------|
| Original                                | complete      | 36.038      | 63.620       | 11.932  | 12.305   | 57.180       | 30.515      | 0.000              |
| Canonical                               | complete      | 36.072      | 63.237       | 0.000   | 0.000    | 63.237       | 36.763      | -0.383             |
| Set-Mean                                | complete      | 35.301      | 61.820       | 0.000   | 0.000    | 61.820       | 38.180      | -1.800             |
| Perm-FT_s42                             | complete      | 34.915      | 62.771       | 10.487  | 9.911    | 57.744       | 32.345      | -0.850             |
| Perm-FT + Consistency (selected)        | not available | —           | —            | —       | —        | —            | —           | —                  |
| Set-Attention_s42                       | not available | —           | —            | —       | —        | —            | —           | —                  |
| Perm-FT-cons0.05_s42 (unselected trial) | complete      | 34.777      | 63.085       | 12.737  | 11.022   | 57.547       | 31.431      | -0.535             |

## Table V2 — target-only inclusion

| Method                                  | Status        | VI@20   | VI@100   | Robust@100   | Never@100   | Mean Inclusion@100   |
|:----------------------------------------|:--------------|:--------|:---------|:-------------|:------------|:---------------------|
| Original                                | complete      | 10.832  | 12.112   | 56.953       | 30.935      | 63.206               |
| Perm-FT_s42                             | complete      | 9.528   | 10.057   | 56.757       | 33.186      | 61.918               |
| Perm-FT + Consistency (selected)        | not available | —       | —        | —            | —           | —                    |
| Set-Attention_s42                       | not available | —       | —        | —            | —           | —                    |
| Perm-FT-cons0.05_s42 (unselected trial) | complete      | 11.705  | 11.751   | 55.638       | 32.611      | 61.594               |
每個 target 是獨立反事實，competitors 固定為同一模型的 C0。Mean Inclusion 不是單一 index 的 Recall。

## Statistical uncertainty

| method               | mode    | metric     |   delta_pp |   ci_low_pp |   ci_high_pp |
|:---------------------|:--------|:-----------|-----------:|------------:|-------------:|
| Original             | catalog | Recall@100 |      0.000 |       0.000 |        0.000 |
| Original             | catalog | VI@20      |      0.000 |       0.000 |        0.000 |
| Original             | catalog | VI@100     |      0.000 |       0.000 |        0.000 |
| Original             | catalog | Robust@100 |      0.000 |       0.000 |        0.000 |
| Original             | catalog | Never@100  |      0.000 |       0.000 |        0.000 |
| Canonical            | catalog | Recall@100 |     -0.383 |      -1.399 |        0.540 |
| Canonical            | catalog | VI@20      |    -11.932 |     -13.720 |      -10.292 |
| Canonical            | catalog | VI@100     |    -12.305 |     -14.292 |      -10.482 |
| Canonical            | catalog | Robust@100 |      6.057 |       4.896 |        7.435 |
| Canonical            | catalog | Never@100  |      6.248 |       5.008 |        7.705 |
| Set-Mean             | catalog | Recall@100 |     -1.800 |      -3.343 |       -0.294 |
| Set-Mean             | catalog | VI@20      |    -11.932 |     -13.720 |      -10.292 |
| Set-Mean             | catalog | VI@100     |    -12.305 |     -14.292 |      -10.482 |
| Set-Mean             | catalog | Robust@100 |      4.640 |       2.981 |        6.417 |
| Set-Mean             | catalog | Never@100  |      7.665 |       6.032 |        9.406 |
| Perm-FT-cons0.05_s42 | catalog | Recall@100 |     -0.535 |      -1.972 |        0.928 |
| Perm-FT-cons0.05_s42 | catalog | VI@20      |      0.806 |      -1.172 |        2.786 |
| Perm-FT-cons0.05_s42 | catalog | VI@100     |     -1.283 |      -3.277 |        0.664 |
| Perm-FT-cons0.05_s42 | catalog | Robust@100 |      0.367 |      -1.426 |        2.336 |
| Perm-FT-cons0.05_s42 | catalog | Never@100  |      0.916 |      -0.513 |        2.248 |
| Perm-FT_s42          | catalog | Recall@100 |     -0.850 |      -2.246 |        0.550 |
| Perm-FT_s42          | catalog | VI@20      |     -1.445 |      -3.293 |        0.350 |
| Perm-FT_s42          | catalog | VI@100     |     -2.394 |      -4.224 |       -0.670 |
| Perm-FT_s42          | catalog | Robust@100 |      0.563 |      -1.126 |        2.363 |
| Perm-FT_s42          | catalog | Never@100  |      1.831 |       0.630 |        3.023 |
上述為百分點的 paired query bootstrap、10,000 次、seed 20260963；區間未作多重比較校正，作描述性 screening uncertainty，不能逐格挑顯著性。CI 包含零不證明等效。未執行 seeds 43/44，沒有 training-seed mean±std，不能捏造為零變異。

## Fully-fitting subset

| method               |   queries |   pairs |   Recall@100 |   VI@20 |   Robust@100 |
|:---------------------|----------:|--------:|-------------:|--------:|-------------:|
| Original             |       239 |    6797 |      0.56516 | 0.08385 |      0.50511 |
| Canonical            |       239 |    6797 |      0.56395 | 0.00000 |      0.56395 |
| Set-Mean             |       239 |    6797 |      0.59757 | 0.00000 |      0.59757 |
| Perm-FT-cons0.05_s42 |       239 |    6797 |      0.60579 | 0.13076 |      0.55462 |
| Perm-FT_s42          |       239 |    6797 |      0.60590 | 0.11160 |      0.56487 |
此表為比例值。只限制 targets 為原七順序都 ≤512 tokens 的共同商品，完整目錄不縮減。M2/M3 tokenizer 與文本完全一致。競爭商品仍可能截斷，所以不是全目錄無截斷實驗。M4 field-truncation 與實測 invariance 結果未取得。

## Failure cases / Interpretation

目前不能把 training loss 下降解讀為有效解法。M2 的 VI 下降有限，Recall 點估計較低，Never 增加；stable exclusion 是重要反例。對照 Canonical 與 Set-Mean，應同時看品質與穩定性，不能用單一 VI 宣告成功。
單 epoch、546 對 supervision、無 unaugmented FT control、17 個 eligible validation queries、歷史 test exposure、單 seed 都限制推論。未完成矩陣是執行限制，不能寫成「consistency 或 learned aggregation 全面失敗」。本輪沒有支持 C，也不把簡單已知 training 方法包裝為新穎架構。

## Compute / Stop / Reproducibility

計入成本合計 4.00011 小時；上限判斷在 batch 邊界觸發。最後中止 stage 以檔案時間估計，可能包含數秒 transition overhead。此為 task wall time，含 CPU／等待與未知中斷的保守計帳，不是純 GPU kernel 小時。

| name                                        |   seconds |
|:--------------------------------------------|----------:|
| Perm-FT_s42_train                           |   150.797 |
| Perm-FT_s42_encode_evaluate                 | 11133.609 |
| Perm-FT-cons0.05_s42_train                  |   106.032 |
| Perm-FT-cons0.05_s42_interrupted_evaluation |  1050.000 |
| Perm-FT-cons0.05_s42_encode_evaluate        |  1012.531 |
| Perm-FT-cons0.1_s42_train                   |   186.484 |
| Perm-FT-cons0.1_s42_cap_stopped_evaluation  |   760.952 |
詳見 INTERRUPTIONS.md、run_initial_interrupted.log、run.log、resume_errors.log。M2 大部分花費在 inference；最初執行時間估計不足，4 小時上限未覆蓋全矩陣。沒有私自延長預算或挑選較好試驗。
新增檔案清單及 SHA256 見 phase5_mitigation/delivery_manifest.json；checkpoint metadata 含 seed 與權重 hash。原稿與既有結果來源全部通過 source_integrity_check.json 的 unchanged 檢查。
重現已完成報告：`phase2/.venv/Scripts/python.exe phase5_mitigation/finalize_bounded.py`。完整實驗命令仍在 README，但目前 cap 已用盡，直接 run 不會重啟計算。任何後續預算必須另記 amendment，不可覆蓋本輪 freeze。報告依賴新增 tabulate==0.9.0。

## Implication for WWW Paper

1. 現有結果增加了一項有用的負面控制：訓練 loss 改善不保證 relevance–visibility trade-off 改善；但不實質支持方法論文定位。
2. 目前不建議新增主文 solution claim。可將 M2 與未選定 .05 trial 作附錄 screening evidence，明示矩陣不完整。
3. 若未來完整確認有必要進主文，可縮短 appearance-first transfer 與重複逐模型數字；保留 target-only、fully-fitting 和候選 loss 的核心證據。
4. 選擇 A：維持 diagnosis paper。沒有足夠證據選 B 或 C；M4 的潛力尚未被本輪測定。
5. 最強可支持句子：**In this budget-limited WANDS screening, permutation fine-tuning and one unselected consistency trial did not establish a better relevance–robustness trade-off; reduced instability must be interpreted alongside stable exclusion.**

原稿未修改。後續 Phase VI 跨資料／模型確認目前不建議啟動；應先完成尚缺的 WANDS screening，且須有獨立的計算預算記錄。
