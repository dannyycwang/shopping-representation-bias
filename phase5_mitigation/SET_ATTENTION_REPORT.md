# Set-attention 最終篩選報告



**結論：排列不變性成立，但目前沒有足夠證據支持新增 learned solution 主張。**

這是原 raw serializer 的獨立 track；不能與 labeled PI-FT track 視為完全 matched 比較。保留全 42,994 商品、歷史 held-out queries，單 seed42。

## 全體 catalog（百分比）



| method            |   pairs |   queries |   Recall@100 |   VI@20 |   Robust@100 |   Never@100 |   worst_schedule@100 |
|:------------------|--------:|----------:|-------------:|--------:|-------------:|------------:|---------------------:|
| Original          |   21299 |       308 |       63.620 |  11.932 |       57.180 |      30.515 |               60.274 |
| Canonical         |   21299 |       308 |       63.237 |   0.000 |       63.237 |      36.763 |               63.237 |
| Set-Mean          |   21299 |       308 |       61.820 |   0.000 |       61.820 |      38.180 |               61.820 |
| Set-Attention_s42 |   21299 |       308 |       62.173 |   0.000 |       62.173 |      37.827 |               62.173 |

## Target-only（百分比）



| method            |   pairs |   queries |   Inclusion@100 |   VI@20 |   Robust@100 |   Never@100 |   worst_schedule@100 |
|:------------------|--------:|----------:|----------------:|--------:|-------------:|------------:|---------------------:|
| Original          |   21299 |       308 |          63.206 |  10.832 |       56.953 |      30.935 |               59.390 |
| Canonical         |   21299 |       308 |          63.237 |   0.000 |       63.237 |      36.763 |               63.237 |
| Set-Mean          |   21299 |       308 |          61.820 |   0.000 |       61.820 |      38.180 |               61.820 |
| Set-Attention_s42 |   21299 |       308 |          62.173 |   0.000 |       62.173 |      37.827 |               62.173 |

Target competitors 為各方法自己的 C0。Invariant 方法的七版本相同，其 target 表重用固定表示索引排名；這不是只更換 target、所有方法都使用 Original competitors 的跨方法介入。Inclusion 不能稱為一個混合索引的 Recall。

## Paired query bootstrap（百分點）



| comparison                    | mode                  | metric     |   delta_pp |   ci_low_pp |   ci_high_pp |
|:------------------------------|:----------------------|:-----------|-----------:|------------:|-------------:|
| Set-Attention minus Original  | catalog               | Recall@100 |     -1.447 |      -3.014 |        0.137 |
| Set-Attention minus Original  | catalog               | Recall@20  |     -0.795 |      -2.052 |        0.403 |
| Set-Attention minus Original  | catalog               | VI@20      |    -11.932 |     -13.720 |      -10.292 |
| Set-Attention minus Original  | catalog               | VI@100     |    -12.305 |     -14.292 |      -10.482 |
| Set-Attention minus Original  | catalog               | Robust@100 |      4.993 |       3.290 |        6.843 |
| Set-Attention minus Original  | catalog               | Never@100  |      7.312 |       5.671 |        9.072 |
| Set-Attention minus Canonical | catalog               | Recall@100 |     -1.064 |      -2.643 |        0.577 |
| Set-Attention minus Canonical | catalog               | Recall@20  |     -0.830 |      -2.071 |        0.340 |
| Set-Attention minus Canonical | catalog               | VI@20      |      0.000 |       0.000 |        0.000 |
| Set-Attention minus Canonical | catalog               | VI@100     |      0.000 |       0.000 |        0.000 |
| Set-Attention minus Canonical | catalog               | Robust@100 |     -1.064 |      -2.643 |        0.577 |
| Set-Attention minus Canonical | catalog               | Never@100  |      1.064 |      -0.577 |        2.643 |
| Set-Attention minus Set-Mean  | catalog               | Recall@100 |      0.353 |       0.066 |        0.779 |
| Set-Attention minus Set-Mean  | catalog               | Recall@20  |     -0.058 |      -0.550 |        0.360 |
| Set-Attention minus Set-Mean  | catalog               | VI@20      |      0.000 |       0.000 |        0.000 |
| Set-Attention minus Set-Mean  | catalog               | VI@100     |      0.000 |       0.000 |        0.000 |
| Set-Attention minus Set-Mean  | catalog               | Robust@100 |      0.353 |       0.066 |        0.779 |
| Set-Attention minus Set-Mean  | catalog               | Never@100  |     -0.353 |      -0.779 |       -0.066 |
| Set-Attention minus Original  | catalog_fully_fitting | Recall@100 |      3.826 |       1.105 |        6.656 |
| Set-Attention minus Original  | catalog_fully_fitting | Recall@20  |      5.128 |       2.824 |        7.604 |
| Set-Attention minus Original  | catalog_fully_fitting | VI@20      |     -8.385 |     -10.214 |       -6.724 |
| Set-Attention minus Original  | catalog_fully_fitting | VI@100     |    -11.216 |     -13.791 |       -8.848 |
| Set-Attention minus Original  | catalog_fully_fitting | Robust@100 |      9.831 |       6.799 |       13.052 |
| Set-Attention minus Original  | catalog_fully_fitting | Never@100  |      1.384 |      -1.330 |        3.985 |
| Set-Attention minus Canonical | catalog_fully_fitting | Recall@100 |      3.948 |       1.158 |        6.867 |
| Set-Attention minus Canonical | catalog_fully_fitting | Recall@20  |      4.950 |       2.779 |        7.309 |
| Set-Attention minus Canonical | catalog_fully_fitting | VI@20      |      0.000 |       0.000 |        0.000 |
| Set-Attention minus Canonical | catalog_fully_fitting | VI@100     |      0.000 |       0.000 |        0.000 |
| Set-Attention minus Canonical | catalog_fully_fitting | Robust@100 |      3.948 |       1.158 |        6.867 |
| Set-Attention minus Canonical | catalog_fully_fitting | Never@100  |     -3.948 |      -6.867 |       -1.158 |
| Set-Attention minus Set-Mean  | catalog_fully_fitting | Recall@100 |      0.585 |      -0.032 |        1.600 |
| Set-Attention minus Set-Mean  | catalog_fully_fitting | Recall@20  |     -0.275 |      -1.440 |        0.574 |
| Set-Attention minus Set-Mean  | catalog_fully_fitting | VI@20      |      0.000 |       0.000 |        0.000 |
| Set-Attention minus Set-Mean  | catalog_fully_fitting | VI@100     |      0.000 |       0.000 |        0.000 |
| Set-Attention minus Set-Mean  | catalog_fully_fitting | Robust@100 |      0.585 |      -0.032 |        1.600 |
| Set-Attention minus Set-Mean  | catalog_fully_fitting | Never@100  |     -0.585 |      -1.600 |        0.032 |
| Set-Attention minus Original  | target                | Recall@100 |     -1.033 |      -2.590 |        0.577 |
| Set-Attention minus Original  | target                | Recall@20  |     -0.394 |      -1.673 |        0.834 |
| Set-Attention minus Original  | target                | VI@20      |    -10.832 |     -12.497 |       -9.268 |
| Set-Attention minus Original  | target                | VI@100     |    -12.112 |     -14.082 |      -10.324 |
| Set-Attention minus Original  | target                | Robust@100 |      5.220 |       3.501 |        7.084 |
| Set-Attention minus Original  | target                | Never@100  |      6.892 |       5.247 |        8.653 |
| Set-Attention minus Canonical | target                | Recall@100 |     -1.064 |      -2.643 |        0.577 |
| Set-Attention minus Canonical | target                | Recall@20  |     -0.830 |      -2.071 |        0.340 |
| Set-Attention minus Canonical | target                | VI@20      |      0.000 |       0.000 |        0.000 |
| Set-Attention minus Canonical | target                | VI@100     |      0.000 |       0.000 |        0.000 |
| Set-Attention minus Canonical | target                | Robust@100 |     -1.064 |      -2.643 |        0.577 |
| Set-Attention minus Canonical | target                | Never@100  |      1.064 |      -0.577 |        2.643 |
| Set-Attention minus Set-Mean  | target                | Recall@100 |      0.353 |       0.066 |        0.779 |
| Set-Attention minus Set-Mean  | target                | Recall@20  |     -0.058 |      -0.550 |        0.360 |
| Set-Attention minus Set-Mean  | target                | VI@20      |      0.000 |       0.000 |        0.000 |
| Set-Attention minus Set-Mean  | target                | VI@100     |      0.000 |       0.000 |        0.000 |
| Set-Attention minus Set-Mean  | target                | Robust@100 |      0.353 |       0.066 |        0.779 |
| Set-Attention minus Set-Mean  | target                | Never@100  |     -0.353 |      -0.779 |       -0.066 |
| Set-Attention minus Original  | target_fully_fitting  | Recall@100 |      4.598 |       1.796 |        7.505 |
| Set-Attention minus Original  | target_fully_fitting  | Recall@20  |      5.431 |       3.116 |        7.913 |
| Set-Attention minus Original  | target_fully_fitting  | VI@20      |     -7.160 |      -8.712 |       -5.739 |
| Set-Attention minus Original  | target_fully_fitting  | VI@100     |    -10.883 |     -13.389 |       -8.572 |
| Set-Attention minus Original  | target_fully_fitting  | Robust@100 |     10.166 |       7.081 |       13.416 |
| Set-Attention minus Original  | target_fully_fitting  | Never@100  |      0.717 |      -2.140 |        3.429 |
| Set-Attention minus Canonical | target_fully_fitting  | Recall@100 |      3.948 |       1.158 |        6.867 |
| Set-Attention minus Canonical | target_fully_fitting  | Recall@20  |      4.950 |       2.779 |        7.309 |
| Set-Attention minus Canonical | target_fully_fitting  | VI@20      |      0.000 |       0.000 |        0.000 |
| Set-Attention minus Canonical | target_fully_fitting  | VI@100     |      0.000 |       0.000 |        0.000 |
| Set-Attention minus Canonical | target_fully_fitting  | Robust@100 |      3.948 |       1.158 |        6.867 |
| Set-Attention minus Canonical | target_fully_fitting  | Never@100  |     -3.948 |      -6.867 |       -1.158 |
| Set-Attention minus Set-Mean  | target_fully_fitting  | Recall@100 |      0.585 |      -0.032 |        1.600 |
| Set-Attention minus Set-Mean  | target_fully_fitting  | Recall@20  |     -0.275 |      -1.440 |        0.574 |
| Set-Attention minus Set-Mean  | target_fully_fitting  | VI@20      |      0.000 |       0.000 |        0.000 |
| Set-Attention minus Set-Mean  | target_fully_fitting  | VI@100     |      0.000 |       0.000 |        0.000 |
| Set-Attention minus Set-Mean  | target_fully_fitting  | Robust@100 |      0.585 |      -0.032 |        1.600 |
| Set-Attention minus Set-Mean  | target_fully_fitting  | Never@100  |     -0.585 |      -1.600 |        0.032 |

10,000 draws，seed20260963，描述性未校正 95% CI，不含 training-seed uncertainty。CI 包含零不證明等效。

## 共同 fully-fitting raw targets



| method            |   pairs |   queries |   Recall@100 |   VI@20 |   Robust@100 |   Never@100 |   worst_schedule@100 |
|:------------------|--------:|----------:|-------------:|--------:|-------------:|------------:|---------------------:|
| Original          |    6797 |       239 |       56.516 |   8.385 |       50.511 |      38.273 |               52.338 |
| Canonical         |    6797 |       239 |       56.395 |   0.000 |       56.395 |      43.605 |               56.395 |
| Set-Mean          |    6797 |       239 |       59.757 |   0.000 |       59.757 |      40.243 |               59.757 |
| Set-Attention_s42 |    6797 |       239 |       60.342 |   0.000 |       60.342 |      39.658 |               60.342 |



| method            |   pairs |   queries |   Inclusion@100 |   VI@20 |   Robust@100 |   Never@100 |   worst_schedule@100 |
|:------------------|--------:|----------:|----------------:|--------:|-------------:|------------:|---------------------:|
| Original          |    6797 |       239 |          55.745 |   7.160 |       50.177 |      38.941 |               51.757 |
| Canonical         |    6797 |       239 |          56.395 |   0.000 |       56.395 |      43.605 |               56.395 |
| Set-Mean          |    6797 |       239 |          59.757 |   0.000 |       59.757 |      40.243 |               59.757 |
| Set-Attention_s42 |    6797 |       239 |          60.342 |   0.000 |       60.342 |      39.658 |               60.342 |

使用 phase4/results/wands_product_features.csv 的七原始順序 fully_fits 定義；所有方法限制相同 targets，competitors 仍完整。這個條件子集不保證各 set field 完全不截斷，也不提供截斷的因果分解。

## 訓練、不變性與 seed gate



Frozen BGE；768→128→1 tanh scorer，98,561 trainable parameters；無 positional input。獨立 attribute embeddings 加權平均後先 normalize，再與非屬性向量各半混合並 normalize。546 triples，一 epoch，batch16，AdamW lr0.001；沒有 architecture search。

實際 100 商品 × 7 次重排：max L2=1.22205392e-07，max cosine difference=2.38418579e-07。七 schedules 共用同一 cached vector，因此 VI=0 是架構/cache 的結果；浮點微差仍可能影響極近 ties。

Validation 的 Original/Set-Mean/Set-attention Recall@100 分別為 77.821%/79.815%/79.584%。Set-attention 未超過 Set-Mean，不符合原凍結的額外 seed gate，故不執行 seeds43/44。這個 gate 已先在 development 檢查，未依 test 決定。

## 成本與限制



本次 Set-attention 訓練及評估計帳 16.391 秒，重用已快取的 frozen field embeddings；不含先前產生 embeddings 的成本或 torch import。不能據此宣稱完整系統只需 16 秒。既有 unique attribute cache 的 field 截斷率約 0.0085%（129,151 unique fields）。

Set-attention 全體 Recall 點估計只略高於 Set-Mean，低於 Original 與 canonical sorting。穩定漏失仍存在；零 VI 本身不是品質改善。Validation 未通過追加 seed gate；保留單 seed、historical test 與 query-bootstrap 的限制。

重現分析：`phase2/.venv/Scripts/python.exe phase5_mitigation/report_set_attention.py`。排名/checkpoint/invariance.json 保存在 results/Set-Attention_s42 與 checkpoints/Set-Attention_s42。未修改論文。

