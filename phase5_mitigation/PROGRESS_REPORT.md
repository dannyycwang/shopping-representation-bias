# Phase V 進度報告（非最終結論）

更新：2026-09-09T16:05:52.734557+00:00

僅彙整已完成的完整七順序產物。M3 lambda grid 尚未完整時，不選 best lambda；表中單一 trial 不是選定方法。所有 test 數據來自既有歷史 held-out，不稱 pristine confirmation。

- Perm-FT_s42: 完整七順序結果已保存
- Perm-FT-cons0.05_s42: 完整七順序結果已保存
- Perm-FT-cons0.1_s42: 訓練 checkpoint 已保存；完整評估未完成
- Perm-FT-cons0.5_s42: 尚未完成訓練
- Set-Attention_s42: 尚未完成訓練

## Catalog-wide（百分比）

| method               |   Recall@100 |   VI@20 |   VI@100 |   Robust@100 |   Never@100 |
|:---------------------|-------------:|--------:|---------:|-------------:|------------:|
| Original             |       63.620 |  11.932 |   12.305 |       57.180 |      30.515 |
| Canonical            |       63.237 |   0.000 |    0.000 |       63.237 |      36.763 |
| Set-Mean             |       61.820 |   0.000 |    0.000 |       61.820 |      38.180 |
| Perm-FT-cons0.05_s42 |       63.085 |  12.737 |   11.022 |       57.547 |      31.431 |
| Perm-FT_s42          |       62.771 |  10.487 |    9.911 |       57.744 |      32.345 |

Recall 是七順序平均 query-macro；Robust 為七順序都入選，Never 為都不入選。Canonical／Set-Mean 的零 VI 為結構性結果，不代表沒有漏失。

## Target-only（百分比）

| method               |   Recall@100 |   VI@20 |   VI@100 |   Robust@100 |   Never@100 |
|:---------------------|-------------:|--------:|---------:|-------------:|------------:|
| Original             |       63.206 |  10.832 |   12.112 |       56.953 |      30.935 |
| Canonical            |       63.237 |   0.000 |    0.000 |       63.237 |      36.763 |
| Set-Mean             |       61.820 |   0.000 |    0.000 |       61.820 |      38.180 |
| Perm-FT-cons0.05_s42 |       61.594 |  11.705 |   11.751 |       55.638 |      32.611 |
| Perm-FT_s42          |       61.918 |   9.528 |   10.057 |       56.757 |      33.186 |

此表 Recall@100 欄實際意義為 mean target-intervention inclusion@100，不是單一 index recall。每個模型以自己的 C0 competitors 固定，獨立替換每個 target。

## 相對 Original 的 paired 95% CI（百分點）

| method               | mode    | metric     |   delta_pp |   ci_low_pp |   ci_high_pp |
|:---------------------|:--------|:-----------|-----------:|------------:|-------------:|
| Original             | catalog | Recall@100 |      0.000 |       0.000 |        0.000 |
| Original             | catalog | VI@20      |      0.000 |       0.000 |        0.000 |
| Original             | catalog | Robust@100 |      0.000 |       0.000 |        0.000 |
| Canonical            | catalog | Recall@100 |     -0.383 |      -1.399 |        0.540 |
| Canonical            | catalog | VI@20      |    -11.932 |     -13.720 |      -10.292 |
| Canonical            | catalog | Robust@100 |      6.057 |       4.896 |        7.435 |
| Set-Mean             | catalog | Recall@100 |     -1.800 |      -3.343 |       -0.294 |
| Set-Mean             | catalog | VI@20      |    -11.932 |     -13.720 |      -10.292 |
| Set-Mean             | catalog | Robust@100 |      4.640 |       2.981 |        6.417 |
| Perm-FT-cons0.05_s42 | catalog | Recall@100 |     -0.535 |      -1.972 |        0.928 |
| Perm-FT-cons0.05_s42 | catalog | VI@20      |      0.806 |      -1.172 |        2.786 |
| Perm-FT-cons0.05_s42 | catalog | Robust@100 |      0.367 |      -1.426 |        2.336 |
| Perm-FT_s42          | catalog | Recall@100 |     -0.850 |      -2.246 |        0.550 |
| Perm-FT_s42          | catalog | VI@20      |     -1.445 |      -3.293 |        0.350 |
| Perm-FT_s42          | catalog | Robust@100 |      0.563 |      -1.126 |        2.363 |

308 個具有 Exact 標籤的歷史 held-out 查詢；query-cluster paired bootstrap 10,000 次。只有 seed 42，未估計訓練 seed 變異；CI 包含零不證明等效。

## Fully-fitting targets（比例值）

| method               |   queries |   pairs |   Recall@100 |   VI@20 |   Robust@100 |
|:---------------------|----------:|--------:|-------------:|--------:|-------------:|
| Original             |       239 |    6797 |      0.56516 | 0.08385 |      0.50511 |
| Canonical            |       239 |    6797 |      0.56395 | 0.00000 |      0.56395 |
| Set-Mean             |       239 |    6797 |      0.59757 | 0.00000 |      0.59757 |
| Perm-FT-cons0.05_s42 |       239 |    6797 |      0.60579 | 0.13076 |      0.55462 |
| Perm-FT_s42          |       239 |    6797 |      0.60590 | 0.11160 |      0.56487 |

沿用既有 product_features 的七順序 ≤512-token support；完整 competitors 不移除。僅篩選 target，不能聲稱整個候選目錄都未截斷。

完整 setup 見 README.md/config.json，執行中斷見 INTERRUPTIONS.md。4 小時計算上限未變，未完成項目不當成方法失敗。原稿未修改。
