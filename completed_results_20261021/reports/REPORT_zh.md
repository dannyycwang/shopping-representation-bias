# 新模型補充實驗：合併報告

更新時間（UTC）：2026-10-21T17:59:03.142602+00:00。完成狀態只採用通過檔案雜湊驗證的完整實驗。

目前執行 Gemma 與 Harrier；Qwen 4B 由使用者暫緩，不列入完成條件。Gemma 的既有 profile 保持不變。Harrier 在已知 Gemma／ESCI 結果後，依使用者明確要求加入獨立 profile；此擴充不是事前啟用第三模型，也不是整篇研究的 preregistration。

已完成 RQ1/RQ2 與 boundary 的 active cells：4/4。全部 active-model 實驗含 canonical 完成：False。

| Model | Dataset | Stage | Status |
|---|---|---|---|
| embeddinggemma2 | esci | smoke | completed |
| embeddinggemma2 | esci | encode | completed |
| embeddinggemma2 | esci | evaluate | completed |
| embeddinggemma2 | esci | boundary | completed |
| embeddinggemma2 | esci | canonical | completed |
| embeddinggemma2 | wands | smoke | completed |
| embeddinggemma2 | wands | encode | completed |
| embeddinggemma2 | wands | evaluate | completed |
| embeddinggemma2 | wands | boundary | completed |
| embeddinggemma2 | wands | canonical | running |
| harrier06b | esci | smoke | completed |
| harrier06b | esci | encode | completed |
| harrier06b | esci | evaluate | completed |
| harrier06b | esci | boundary | completed |
| harrier06b | esci | canonical | not_run |
| harrier06b | wands | smoke | completed |
| harrier06b | wands | encode | completed |
| harrier06b | wands | evaluate | completed |
| harrier06b | wands | boundary | completed |
| harrier06b | wands | canonical | not_run |
| qwen3_4b | esci | smoke | not_run (user-deferred) |
| qwen3_4b | esci | encode | not_run (user-deferred) |
| qwen3_4b | esci | evaluate | not_run (user-deferred) |
| qwen3_4b | esci | boundary | not_run (user-deferred) |
| qwen3_4b | esci | canonical | not_run (user-deferred) |
| qwen3_4b | wands | smoke | not_run (user-deferred) |
| qwen3_4b | wands | encode | not_run (user-deferred) |
| qwen3_4b | wands | evaluate | not_run (user-deferred) |
| qwen3_4b | wands | boundary | not_run (user-deferred) |
| qwen3_4b | wands | canonical | not_run (user-deferred) |

## 已完成的主要結果

| Model | Dataset | C0 Recall@20 | Raw mean Recall@20 | Target VI@20 [95% CI] | Catalog VI@20 [95% CI] |
|---|---|---|---|---|---|
| embeddinggemma2 | wands | 41.88% | 41.77% | 10.36% [9.12, 11.68] | 10.34% [9.10, 11.68] |
| embeddinggemma2 | esci | 81.86% | 81.97% | 1.70% [1.21, 2.32] | 2.39% [1.80, 3.07] |
| harrier06b | wands | 38.82% | 39.04% | 8.49% [7.11, 10.06] | 8.08% [6.99, 9.21] |
| harrier06b | esci | 78.79% | 78.77% | 1.81% [1.38, 2.29] | 3.19% [2.41, 4.09] |

## 已完成的 canonical 比較

尚無完成的 canonical 比較；不宣稱已實測 VI=0 或品質提升。

所有比較採相同完整商品庫與 query support；模型設定與來源 hashes 見各 profile。此圖只比較 full support，未混入舊研究不同母體的結果。模型差異是描述性的，不能單獨歸因於模型大小或架構。CI 包含零不代表等效，canonical 固定輸入的不變性也不代表品質更高。

未執行、排隊、執行中、受阻及使用者暫緩均不填成零結果。各階段完整 logs、有效 checkpoint、token audit、pair ranks、query metrics、membership 與 boundary evidence 保留在各 profile。執行與恢復命令見各 profile 的 RUN_COMMANDS.md；目前進度以程序與 checkpoint 為準。
