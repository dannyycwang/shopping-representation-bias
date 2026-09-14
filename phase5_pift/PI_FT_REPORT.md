# PI-FT WANDS 適配實驗報告

**這是 WANDS/BGE 的 source-aligned adaptation，不是 DevDataBench 原樣重現。只保護 title，不推測 facet。**

## 設定與來源

作者原文與 pinned code、差異、資料與超參數見 AUDIT_AND_PROTOCOL.md；使用者批准的 title-only 修訂見 TITLE_ONLY_AMENDMENT.md。407 組訓練 pairs、每組三個明確 Irrelevant negatives，五 epochs、seed42、cached contrastive batch128。因近鄰 guard 等條件排除 139 個舊 training positives，清單完整保留。沒有依 test 挑選 checkpoint。
共同 labeled serializer 保留 native attribute keys，添加 title/class/category/description labels；七個 test schedules 只換 attributes 順序，scalar sections 固定；訓練的全欄位 permutation 範圍更廣。test 不做 dropout。

## Catalog-wide（百分比）

| method           |   Recall@20 |   Recall@100 |   VI@20 |   VI@100 |   Robust@100 |   Never@100 |
|:-----------------|------------:|-------------:|--------:|---------:|-------------:|------------:|
| Labeled-ZeroShot |      36.372 |       63.848 |  10.590 |    9.975 |       58.842 |      31.183 |
| Standard-FT      |      34.399 |       62.394 |   6.667 |    6.446 |       59.369 |      34.185 |
| Adapted-PI-FT    |      33.974 |       62.272 |  11.565 |   11.168 |       56.686 |      32.146 |

## Target-only（百分比）

| method           |   Inclusion@20 |   Inclusion@100 |   VI@20 |   VI@100 |   Robust@100 |   Never@100 |
|:-----------------|---------------:|----------------:|--------:|---------:|-------------:|------------:|
| Labeled-ZeroShot |         36.080 |          63.028 |   9.951 |   10.152 |       58.238 |      31.609 |
| Standard-FT      |         33.713 |          61.490 |   6.218 |    6.499 |       58.378 |      35.123 |
| Adapted-PI-FT    |         33.085 |          61.100 |  10.828 |   10.932 |       55.455 |      33.612 |
每個 target 獨立替換，competitors 固定為各模型 C0。Inclusion 不是單一 index recall。Robust=Always；VI=Sometimes；Never=所有七順序都漏失。皆先 pair 內計算，再在 308 個 Exact-eligible 歷史 held-out queries 做 macro。

## Paired uncertainty（百分點）

| comparison                           | mode                  | metric     |   delta_pp |   ci_low_pp |   ci_high_pp |
|:-------------------------------------|:----------------------|:-----------|-----------:|------------:|-------------:|
| Adapted-PI-FT minus Standard-FT      | catalog               | Recall@20  |     -0.425 |      -1.242 |        0.331 |
| Adapted-PI-FT minus Standard-FT      | catalog_fully_fitting | Recall@20  |      1.162 |      -0.188 |        2.654 |
| Adapted-PI-FT minus Standard-FT      | catalog               | Recall@100 |     -0.122 |      -1.086 |        0.829 |
| Adapted-PI-FT minus Standard-FT      | catalog_fully_fitting | Recall@100 |      2.024 |       0.018 |        4.207 |
| Adapted-PI-FT minus Standard-FT      | catalog               | VI@20      |      4.899 |       3.527 |        6.370 |
| Adapted-PI-FT minus Standard-FT      | catalog_fully_fitting | VI@20      |      4.348 |       1.855 |        6.841 |
| Adapted-PI-FT minus Standard-FT      | catalog               | VI@100     |      4.721 |       3.197 |        6.358 |
| Adapted-PI-FT minus Standard-FT      | catalog_fully_fitting | VI@100     |      4.426 |       1.907 |        7.016 |
| Adapted-PI-FT minus Standard-FT      | catalog               | Robust@100 |     -2.683 |      -4.138 |       -1.439 |
| Adapted-PI-FT minus Standard-FT      | catalog_fully_fitting | Robust@100 |     -0.023 |      -2.126 |        2.213 |
| Adapted-PI-FT minus Standard-FT      | catalog               | Never@100  |     -2.038 |      -3.438 |       -0.677 |
| Adapted-PI-FT minus Standard-FT      | catalog_fully_fitting | Never@100  |     -4.403 |      -7.149 |       -1.851 |
| Standard-FT minus Labeled-ZeroShot   | catalog               | Recall@20  |     -1.973 |      -3.609 |       -0.388 |
| Standard-FT minus Labeled-ZeroShot   | catalog_fully_fitting | Recall@20  |      2.334 |       0.090 |        4.650 |
| Standard-FT minus Labeled-ZeroShot   | catalog               | Recall@100 |     -1.454 |      -3.272 |        0.369 |
| Standard-FT minus Labeled-ZeroShot   | catalog_fully_fitting | Recall@100 |      3.355 |       0.365 |        6.258 |
| Standard-FT minus Labeled-ZeroShot   | catalog               | VI@20      |     -3.923 |      -5.395 |       -2.577 |
| Standard-FT minus Labeled-ZeroShot   | catalog_fully_fitting | VI@20      |     -2.970 |      -5.774 |       -0.208 |
| Standard-FT minus Labeled-ZeroShot   | catalog               | VI@100     |     -3.528 |      -4.870 |       -2.257 |
| Standard-FT minus Labeled-ZeroShot   | catalog_fully_fitting | VI@100     |     -2.964 |      -5.780 |       -0.245 |
| Standard-FT minus Labeled-ZeroShot   | catalog               | Robust@100 |      0.527 |      -1.426 |        2.511 |
| Standard-FT minus Labeled-ZeroShot   | catalog_fully_fitting | Robust@100 |      5.065 |       1.704 |        8.469 |
| Standard-FT minus Labeled-ZeroShot   | catalog               | Never@100  |      3.001 |       1.133 |        4.886 |
| Standard-FT minus Labeled-ZeroShot   | catalog_fully_fitting | Never@100  |     -2.100 |      -5.270 |        1.083 |
| Adapted-PI-FT minus Labeled-ZeroShot | catalog               | Recall@20  |     -2.398 |      -3.982 |       -0.915 |
| Adapted-PI-FT minus Labeled-ZeroShot | catalog_fully_fitting | Recall@20  |      3.496 |       1.042 |        5.947 |
| Adapted-PI-FT minus Labeled-ZeroShot | catalog               | Recall@100 |     -1.576 |      -3.429 |        0.255 |
| Adapted-PI-FT minus Labeled-ZeroShot | catalog_fully_fitting | Recall@100 |      5.379 |       2.290 |        8.526 |
| Adapted-PI-FT minus Labeled-ZeroShot | catalog               | VI@20      |      0.975 |      -0.851 |        2.790 |
| Adapted-PI-FT minus Labeled-ZeroShot | catalog_fully_fitting | VI@20      |      1.378 |      -1.415 |        4.026 |
| Adapted-PI-FT minus Labeled-ZeroShot | catalog               | VI@100     |      1.193 |      -0.527 |        2.873 |
| Adapted-PI-FT minus Labeled-ZeroShot | catalog_fully_fitting | VI@100     |      1.462 |      -1.235 |        4.169 |
| Adapted-PI-FT minus Labeled-ZeroShot | catalog               | Robust@100 |     -2.156 |      -4.086 |       -0.263 |
| Adapted-PI-FT minus Labeled-ZeroShot | catalog_fully_fitting | Robust@100 |      5.042 |       1.918 |        8.262 |
| Adapted-PI-FT minus Labeled-ZeroShot | catalog               | Never@100  |      0.963 |      -1.199 |        3.122 |
| Adapted-PI-FT minus Labeled-ZeroShot | catalog_fully_fitting | Never@100  |     -6.503 |      -9.803 |       -3.302 |
| Adapted-PI-FT minus Standard-FT      | target                | Recall@20  |     -0.628 |      -1.731 |        0.371 |
| Adapted-PI-FT minus Standard-FT      | target_fully_fitting  | Recall@20  |      0.463 |      -0.816 |        1.767 |
| Adapted-PI-FT minus Standard-FT      | target                | Recall@100 |     -0.390 |      -1.400 |        0.587 |
| Adapted-PI-FT minus Standard-FT      | target_fully_fitting  | Recall@100 |      2.481 |       0.406 |        4.731 |
| Adapted-PI-FT minus Standard-FT      | target                | VI@20      |      4.610 |       3.176 |        6.209 |
| Adapted-PI-FT minus Standard-FT      | target_fully_fitting  | VI@20      |      4.067 |       1.915 |        6.269 |
| Adapted-PI-FT minus Standard-FT      | target                | VI@100     |      4.434 |       2.944 |        6.052 |
| Adapted-PI-FT minus Standard-FT      | target_fully_fitting  | VI@100     |      4.111 |       1.590 |        6.697 |
| Adapted-PI-FT minus Standard-FT      | target                | Robust@100 |     -2.923 |      -4.402 |       -1.627 |
| Adapted-PI-FT minus Standard-FT      | target_fully_fitting  | Robust@100 |     -0.101 |      -2.257 |        2.161 |
| Adapted-PI-FT minus Standard-FT      | target                | Never@100  |     -1.511 |      -2.730 |       -0.304 |
| Adapted-PI-FT minus Standard-FT      | target_fully_fitting  | Never@100  |     -4.010 |      -6.721 |       -1.443 |
| Standard-FT minus Labeled-ZeroShot   | target                | Recall@20  |     -2.367 |      -4.063 |       -0.717 |
| Standard-FT minus Labeled-ZeroShot   | target_fully_fitting  | Recall@20  |      2.144 |      -0.070 |        4.396 |
| Standard-FT minus Labeled-ZeroShot   | target                | Recall@100 |     -1.538 |      -3.326 |        0.223 |
| Standard-FT minus Labeled-ZeroShot   | target_fully_fitting  | Recall@100 |      2.486 |      -0.284 |        5.205 |
| Standard-FT minus Labeled-ZeroShot   | target                | VI@20      |     -3.733 |      -5.129 |       -2.454 |
| Standard-FT minus Labeled-ZeroShot   | target_fully_fitting  | VI@20      |     -3.027 |      -5.569 |       -0.598 |
| Standard-FT minus Labeled-ZeroShot   | target                | VI@100     |     -3.654 |      -5.279 |       -2.182 |
| Standard-FT minus Labeled-ZeroShot   | target_fully_fitting  | VI@100     |     -4.031 |      -6.581 |       -1.529 |
| Standard-FT minus Labeled-ZeroShot   | target                | Robust@100 |      0.140 |      -1.775 |        2.063 |
| Standard-FT minus Labeled-ZeroShot   | target_fully_fitting  | Robust@100 |      4.898 |       1.723 |        8.076 |
| Standard-FT minus Labeled-ZeroShot   | target                | Never@100  |      3.514 |       1.701 |        5.367 |
| Standard-FT minus Labeled-ZeroShot   | target_fully_fitting  | Never@100  |     -0.867 |      -3.773 |        2.073 |
| Adapted-PI-FT minus Labeled-ZeroShot | target                | Recall@20  |     -2.995 |      -4.636 |       -1.459 |
| Adapted-PI-FT minus Labeled-ZeroShot | target_fully_fitting  | Recall@20  |      2.607 |       0.221 |        4.980 |
| Adapted-PI-FT minus Labeled-ZeroShot | target                | Recall@100 |     -1.928 |      -3.710 |       -0.173 |
| Adapted-PI-FT minus Labeled-ZeroShot | target_fully_fitting  | Recall@100 |      4.967 |       1.991 |        7.993 |
| Adapted-PI-FT minus Labeled-ZeroShot | target                | VI@20      |      0.877 |      -1.011 |        2.807 |
| Adapted-PI-FT minus Labeled-ZeroShot | target_fully_fitting  | VI@20      |      1.040 |      -1.437 |        3.405 |
| Adapted-PI-FT minus Labeled-ZeroShot | target                | VI@100     |      0.780 |      -0.846 |        2.350 |
| Adapted-PI-FT minus Labeled-ZeroShot | target_fully_fitting  | VI@100     |      0.080 |      -2.615 |        2.754 |
| Adapted-PI-FT minus Labeled-ZeroShot | target                | Robust@100 |     -2.783 |      -4.675 |       -0.959 |
| Adapted-PI-FT minus Labeled-ZeroShot | target_fully_fitting  | Robust@100 |      4.797 |       1.906 |        7.748 |
| Adapted-PI-FT minus Labeled-ZeroShot | target                | Never@100  |      2.003 |       0.111 |        3.966 |
| Adapted-PI-FT minus Labeled-ZeroShot | target_fully_fitting  | Never@100  |     -4.877 |      -8.103 |       -1.746 |
10,000 次 paired query bootstrap、seed20260963，描述性未校正 CI，不逐格挑顯著性；單 training seed，CI 不包含 training-seed uncertainty。CI 包含零不等於等效。

## Ranking quality 與 cutoff crossing 分開解讀

| method           | metric                |   eligible_queries |      C0 |   seven_order_mean |   mean_worst_observed |   mean_spread |
|:-----------------|:----------------------|-------------------:|--------:|-------------------:|----------------------:|--------------:|
| Labeled-ZeroShot | nDCG@10_unjudged_zero |                383 | 0.70794 |            0.70629 |               0.66626 |       0.07933 |
| Labeled-ZeroShot | cNDCG@10              |                383 | 0.81823 |            0.81725 |               0.79093 |       0.05020 |
| Standard-FT      | nDCG@10_unjudged_zero |                383 | 0.71728 |            0.71634 |               0.68938 |       0.05264 |
| Standard-FT      | cNDCG@10              |                383 | 0.80892 |            0.80791 |               0.79020 |       0.03488 |
| Adapted-PI-FT    | nDCG@10_unjudged_zero |                383 | 0.70508 |            0.70433 |               0.66208 |       0.08101 |
| Adapted-PI-FT    | cNDCG@10              |                383 | 0.81079 |            0.80983 |               0.78305 |       0.05112 |
nDCG@10_unjudged_zero 使用 WANDS Exact/Partial/Irrelevant gains 3/1/0，unjudged 當零只是計算慣例，不是已知 irrelevant。cNDCG@10 把未判斷商品移除。兩者在具有 positive IDCG 的查詢計算，分母可能不同於最高相關商品 Recall。各順序相對 C0 的 paired delta 見 ndcg_order_deltas.csv；這不是原文單正例 benchmark 的可直接比較分數。平均 quality 改變不能替代 VI。

## 截斷與共同 fully-fitting targets

| method           | mode    |   pairs |   queries |   Recall@100 |   VI@20 |   Robust@100 |
|:-----------------|:--------|--------:|----------:|-------------:|--------:|-------------:|
| Labeled-ZeroShot | catalog |    6473 |       232 |      0.57506 | 0.09248 |      0.52462 |
| Standard-FT      | catalog |    6473 |       232 |      0.60861 | 0.06278 |      0.57527 |
| Adapted-PI-FT    | catalog |    6473 |       232 |      0.62885 | 0.10626 |      0.57504 |
| Labeled-ZeroShot | target  |    6473 |       232 |      0.56789 | 0.08224 |      0.51582 |
| Standard-FT      | target  |    6473 |       232 |      0.59275 | 0.05197 |      0.56480 |
| Adapted-PI-FT    | target  |    6473 |       232 |      0.61756 | 0.09264 |      0.56379 |
七順序都 ≤512 tokens 的商品共 15870/42994。所有模型使用同一 support，只限制 targets、不縮減 competitors。每個商品／順序的長度見 input_lengths.csv。沒有作者的 schema-specific character budgeting；不能把截斷差異與 absolute-position mechanism 混為一談。

## Dropout 與實際輸入檢查

| method        |   document_presentations |   eligible_field_presentations |   dropped_fields |   observed_dropout |   title_protection_failures |   unexpected_PI_protection_sets |
|:--------------|-------------------------:|-------------------------------:|-----------------:|-------------------:|----------------------------:|--------------------------------:|
| Standard-FT   |                     8140 |                         475160 |                0 |           0.000000 |                           0 |                               0 |
| Adapted-PI-FT |                     8140 |                         475160 |            71277 |           0.150006 |                           0 |                               0 |
檢查 retained segment indices，包括重複 atoms，title 保護違規必須為零。PI-FT positive protect 必須只含 title；negative 由 renderer 的全域 title 保護。欄位完整保留於 test，但 training dropout 可能移除相關證據，不宣稱 relevance-preserving augmentation。

## 限制與論文解讀

全體 catalog 結果不支持本次 adapted PI-FT 同時改善品質與排列穩健性。相對 matched Standard-FT，Recall@100 差 −0.122 百分點（CI 包含零），VI@20 增加 4.899 百分點，Robust@100 降低 2.683 百分點。Standard-FT 的 VI 較低，但相對 labeled zero-shot 的 Recall 點估計也較低；不能只依 VI 宣稱成功。
共同 fully-fitting 子集必須保留：PI-FT 的 Recall@100 為 62.885%，Standard 為 60.861%，labeled zero-shot 為 57.506%。PI 對 Standard 的增幅約 2.024 百分點，描述性 CI 下界僅略高於零；然而 PI 的 VI@20 仍較高（10.626% 對 6.278%）。這是 6,473 pairs、232 queries 的條件性結果，不是全體成功，也不是截斷的因果效果分解。
沒有 facet 標註、407 pairs、單 seed、歷史 test、不同 miner、known-label negative mask、未縮短 schema 與不同 test intervention 都限制外推。結果不能推翻作者在原 benchmark 的結論。Standard-FT 是相同 supervised contrastive training 關閉 permutation/dropout 的 matched control，並非新的演算法。
Set-attention 是獨立必要實驗，使用原先 raw-family／546 triples 與 Set-Mean 對照；不可將兩種 representation track 不加區分地排名或相加。

## 成本、修復、重現

| name                              |   seconds |
|:----------------------------------|----------:|
| Standard-FT_train                 |   873.234 |
| Standard-FT_encode_evaluate       |  3666.05  |
| Adapted-PI-FT_train               |   406.859 |
| interrupted_unaccounted_wall_time |    43.548 |
| Adapted-PI-FT_encode_evaluate     |  4713.81  |
| Labeled-ZeroShot_encode_evaluate  |  3199.11  |
模型載入的網路中斷已計帳，之後離線載入，詳見 EXECUTION_REPAIRS.md。8 小時 wall-time cap 未重設。執行命令：`phase2/.venv/Scripts/python.exe phase5_pift/run.py run`；分析：`... phase5_pift/analyze.py`。完成 checkpoints/ranks 重用，不重新訓練。
程式、config、source hashes、augmentation traces、checkpoint hashes、完整 ranking 與 CI 均保存在本資料夾。論文未修改。
