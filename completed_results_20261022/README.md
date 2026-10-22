# Gemma2 與 Harrier：目前已驗證結果（2026-10-22）

**Gemma2 已完整完成；Harrier 的 WANDS canonical 仍在執行。** 本快照包含目前全部已完成的結果與可攜式驗證資料，原本的實驗程序繼續運行。

| 模型 | ESCI RQ1/RQ2＋boundary | WANDS RQ1/RQ2＋boundary | ESCI canonical | WANDS canonical |
|---|---|---|---|---|
| Gemma2 | 完成 | 完成 | 三項規則完成 | 三項規則完成 |
| Harrier | 完成 | 完成 | 三項規則完成 | 未完成，未納入結果 |

三項固定規則為 `lexical_ascending`、`lexical_descending`、`field_priority_type`。Jina 已取消，Qwen3-Embedding-4B 仍暫緩。

## 結果入口

- [合併中文報告](reports/REPORT_zh.md)、[主要結果表](reports/tables/main_candidates.csv)、[完整指標與信賴區間](reports/tables/summary.csv)、[論文補充文字](reports/MANUSCRIPT_ADDITIONS.tex)。
- [Gemma2 完整報告](models/embeddinggemma2/REPORT_zh.md)與[完整摘要表](models/embeddinggemma2/tables/summary.csv)。
- [Harrier ESCI canonical 逐查詢指標](models/harrier06b/esci/canonical/query_metrics.parquet)與[完成驗證](provenance/harrier_esci_canonical_verification.json)。
- [資料、評估條件與歷史結果的對照](provenance/SOURCE_MAP.md)。

兩模型在 ESCI、WANDS 的 RQ1/RQ2 與 boundary 均完成，共四個 primary cells。Canonical 已完成三個 model/dataset cells。Harrier 的 canonical 彙總差值與 CI 會由原程序於 WANDS 完成後產生；目前附上的 Harrier profile 報告與部分表格仍是 primary phase 的版本，不能視為全模型的最終報告。合併報告更新時間為 2026-10-21 21:29 UTC，與已完成階段對應；快照時間見 manifest。

## 主要發現與限制

下列 VI@20 是 full-support query-macro 百分比：

| 模型 | 資料集 | Target VI@20 | Catalog VI@20 |
|---|---|---:|---:|
| Gemma2 | ESCI | 1.70% | 2.39% |
| Gemma2 | WANDS | 10.36% | 10.34% |
| Harrier | ESCI | 1.81% | 3.19% |
| Harrier | WANDS | 8.49% | 8.08% |

兩模型均呈現 WANDS 較高的順序敏感性；這是描述性比較，不能單獨歸因於架構或模型大小。ESCI 使用 499 queries、4,434 highest-label pairs、10,076 商品；WANDS 使用 308 queries、21,299 pairs、42,994 商品。不要混入不同 support、早期錯置的 ESCI gains，或混用 pair-micro 和 query-macro 指標。

Recall、標準 nDCG、judged-condensed cNDCG 等指標均有計算，包含 K=20、100。Gemma2 六項 full-support canonical-minus-seven-raw-mean 的 Recall@20 與標準 nDCG@20 差值，其 95% query-bootstrap CI 均包含零；這不代表等效，也不能推論所有品質指標均無差異。WANDS `field_priority_type` 的 cNDCG@20 差值為 +0.002432，未調整多重比較的 95% CI 為 [0.000175, 0.004722]；須保留多重比較限制。

資料、切分、評估母體、gains、raw schedules、seeds、ranking 與 bootstrap 與修正後的歷史協定對齊；模型各自使用原生 prefix、pooling、維度、長度與批次設定。Boundary 結論僅涵蓋抽選且重複檢驗的 pairs。保留原生截斷、fitting-population 與數值歧義限制；不能由 canonical 輸入固定推論實測 VI=0 或品質提升。加入後續模型不是整篇研究的事前註冊。

## 驗證與封存範圍

```console
python completed_results_20261022/verify_snapshot.py
```

[SNAPSHOT_MANIFEST.json](SNAPSHOT_MANIFEST.json) 列出全部上傳檔案的大小、SHA-256、原始位置、完成憑證與明確省略的大型輸出。匯出時重新核對 frozen sources、合併報告、Gemma2 完整 manifest，以及所附完成憑證引用的輸出雜湊；portable verifier 僅驗證隨快照交付的檔案及其憑證對應。

`models/` 包含凍結 protocol／科學程式、統計表、逐 query／pair 的 parquet 結果與 boundary 證據。`provenance/` 保存驗證與範圍歷史。模型權重、原始資料集、embedding／score matrices、cache／checkpoint、venv 與未完成的 Harrier WANDS canonical 輸出保留在本機，沒有宣稱已隨此快照備份。

原 protocol、receipt、報告與科學程式的位元組保持不變，可能含 Windows 絕對路徑或歷史狀態。此資料夾是結果快照，不是可直接啟動 GPU 佇列的完整環境。前次 [2026-10-21 快照](../completed_results_20261021/README.md) 保留原狀；本次上傳亦未改動實驗參數或重跑推論。
