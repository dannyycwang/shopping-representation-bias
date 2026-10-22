# 新模型補充實驗執行報告

更新時間（UTC）：2026-10-21T20:30:49.629407+00:00。本次是看到既有研究結果後制定的新模型 follow-up，不是整篇研究的 preregistration。

目前執行範圍：embeddinggemma2。暫緩模型：qwen3_4b；暫緩是使用者指示，不是模型失敗，恢復前須有使用者明確指示。

## 已完成的資料與實作核對

- WANDS：完整 42,994 products；96 development／384 held-out；308 eligible queries，21,299 Exact pairs。
- ESCI：既有 500-query sample 的完整 10,076-product pool；499 eligible queries，4,434 E pairs。
- 七種 raw schedules 的全部商品文字與歷史保存字串逐筆相等；attribute multiplicity、固定區塊與三個 canonical rules 的輸入不變性均已核對。
- 新增專用 adapter；舊 MiniLM/BGE/GTE 程式及既有未提交修改的 hashes 保持不變。
- 修正後 direct gains、query-macro、exact rank insertion、tie、自我移除、membership cancellation 與 checkpoint integrity 已有驗收測試。

## 模型與階段狀態

| Model | Dataset | Stage | Status |
|---|---|---|---|
| qwen3_4b | wands | smoke | not_run |
| qwen3_4b | wands | encode | not_run |
| qwen3_4b | wands | evaluate | not_run |
| qwen3_4b | wands | boundary | not_run |
| qwen3_4b | wands | canonical | not_run |
| qwen3_4b | esci | smoke | not_run |
| qwen3_4b | esci | encode | not_run |
| qwen3_4b | esci | evaluate | not_run |
| qwen3_4b | esci | boundary | not_run |
| qwen3_4b | esci | canonical | not_run |
| embeddinggemma2 | wands | smoke | completed |
| embeddinggemma2 | wands | encode | completed |
| embeddinggemma2 | wands | evaluate | completed |
| embeddinggemma2 | wands | boundary | completed |
| embeddinggemma2 | wands | canonical | completed |
| embeddinggemma2 | esci | smoke | completed |
| embeddinggemma2 | esci | encode | completed |
| embeddinggemma2 | esci | evaluate | completed |
| embeddinggemma2 | esci | boundary | completed |
| embeddinggemma2 | esci | canonical | completed |

## 正式結果

- embeddinggemma2 / wands: C0 Recall@20=0.418759; raw mean=0.417668; target-only VI@20=0.103577 [0.091204, 0.116842]; catalog-wide VI@20=0.103387 [0.090999, 0.116816].
- embeddinggemma2 / esci: C0 Recall@20=0.818551; raw mean=0.819736; target-only VI@20=0.017020 [0.012107, 0.023218]; catalog-wide VI@20=0.023868 [0.018014, 0.030675].

## 技術驗證、阻礙與限制

- embeddinggemma2: 官方 reference 比對通過；三次固定 profile repeated-input score range=0.0; rank changes=0。Batch-size 差異另存，不能混合執行 profiles。

未完成模型不替換、不縮小維度、不量化。尚未執行的 canonical 階段不宣稱 VI=0 的實測結果。所有結果僅適用於既定模型、catalog、query support、七種 schedules 與 K；不同模型 fitting supports 的差異不是同一母體比較，full/fitting 差異不是 truncation 因果效果。CI 含零不表示等效。

重現命令見 RUN_COMMANDS.md；query-level evidence、pair ranks、full scores 與 token audit 只在對應階段完成後產生。
