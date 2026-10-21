# Gemma2 與 Harrier：已完成結果快照（2026-10-21）

這份快照先交付已驗證完成的實驗。兩模型在 ESCI、WANDS 的 RQ1／RQ2 與 boundary audit 均已完成（4／4）；Gemma2／ESCI 的三項 canonical 規則也已完成。**整套 canonical 實驗尚未完成。**

| 模型 | ESCI 主實驗及 boundary | WANDS 主實驗及 boundary | ESCI canonical | WANDS canonical |
|---|---|---|---|---|
| Gemma2 (`embeddinggemma2`) | 已完成 | 已完成 | 三項規則均完成 | 執行中，未納入 |
| Harrier (`harrier06b`) | 已完成 | 已完成 | 等待，未納入 | 等待，未納入 |

Jina 已由使用者取消；Qwen3-Embedding-4B 仍由使用者暫緩。這份上傳沒有啟動、重跑或變更任何實驗，原本的 Gemma2／WANDS 程序繼續執行。後續完成的結果不會自動出現在本快照。

## 已完成的主要結果

下列 VI@20 為 full-support query-macro 指標，數值單位為百分比。

| 模型 | 資料集 | C0 Recall@20 | 七種 raw 順序平均 Recall@20 | Target VI@20（95% CI） | Catalog VI@20（95% CI） |
|---|---|---:|---:|---:|---:|
| Gemma2 | ESCI | 81.86 | 81.97 | 1.70 [1.21, 2.32] | 2.39 [1.80, 3.07] |
| Gemma2 | WANDS | 41.88 | 41.77 | 10.36 [9.12, 11.68] | 10.34 [9.10, 11.68] |
| Harrier | ESCI | 78.79 | 78.77 | 1.81 [1.38, 2.29] | 3.19 [2.41, 4.09] |
| Harrier | WANDS | 38.82 | 39.04 | 8.49 [7.11, 10.06] | 8.08 [6.99, 9.21] |

兩個模型均呈現 WANDS 比 ESCI 更高的順序敏感性；幅度並非完全相同。平均 Recall 的變化也不能取代對相關商品進出 top-K 的檢查。這是描述性比較，不能單獨歸因於模型架構或大小；信賴區間包含零也不代表等效。

- ESCI：499 queries、4,434 highest-label pairs、10,076 商品。
- WANDS：308 queries、21,299 highest-label pairs、42,994 商品。
- 保持相同母體、資料切分與七種 raw 排序；不要混入舊研究不同 support 的結果，或混用 pair-micro 與 query-macro 指標。
- Boundary 結論只適用於實際抽選並重複檢驗的 pairs，不能推廣成每一個 pair 都無數值歧義。
- Harrier 在已知部分 Gemma 結果後依使用者要求加入；這不是整篇研究的事前註冊。

## Canonical 的交付範圍

Gemma2／ESCI 於 2026-10-21 17:53 UTC 完成 `lexical_ascending`、`lexical_descending`、`field_priority_type` 三項規則的編碼與完整商品庫評估。已附上完成憑證、pair ranks、query metrics 與 transitions。

原有報告的 canonical 效能差值與 CI 表會在完整 canonical phase 收尾時重新產生；本快照保留當時報告原始位元組，因此報告可能同時顯示 ESCI canonical 已完成、但尚無彙總的 canonical 比較表。**不能把規則固定或程序完成解讀為已證明 VI=0 或品質提升。**

## 檔案與驗證

- [合併報告](reports/REPORT_zh.md)、[主要結果表](reports/tables/main_candidates.csv)、[完整摘要與 CI](reports/tables/summary.csv)、[論文補充文字](reports/MANUSCRIPT_ADDITIONS.tex)。
- `models/embeddinggemma2/`、`models/harrier06b/`：凍結 protocol、原始科學程式、profile 統計表、完成階段憑證、逐查詢／逐 pair 結果與 boundary 證據。
- `provenance/`：主實驗與 Gemma2／ESCI canonical 的驗證紀錄，以及範圍、執行順序的修改證據。
- [SNAPSHOT_MANIFEST.json](SNAPSHOT_MANIFEST.json)：上傳檔案的大小、SHA-256、原始相對位置，以及原完成憑證引用但未上傳的大型輸出清單。

使用標準 Python 即可在任意位置驗證，不需安裝模型套件或啟動 GPU：

```console
python completed_results_20261021/verify_snapshot.py
```

這是約 15 MB 的**結果快照**，不是完整訓練環境或所有 checkpoint 的備份。模型權重、原始資料集、向量、分數矩陣、編碼 cache、未完成的 canonical 輸出與 venv 留在原工作目錄。`omitted_receipt_outputs` 明列未上傳檔案；其雜湊來自原完成憑證，分數矩陣也在匯出時重新核對。Portable verifier 驗證本快照與憑證的對應關係，不宣稱能驗證未隨快照交付的大型檔案。

原 protocol 與原 receipt 的內容保持不變，因此其中保留原 Windows 絕對路徑和歷史狀態。`source/` 用於記錄產生結果的凍結程式；此快照不提供可直接啟動原操作佇列的完整環境。原模型、資料及依賴套件的授權仍以各自來源為準。
