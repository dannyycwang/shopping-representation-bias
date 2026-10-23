# Gemma2 與 Harrier：兩模型完整結果（2026-10-23）

**兩個模型在 ESCI、WANDS 的 RQ1、RQ2、RQ3 均已完成並驗證。** 本快照補齊 Harrier WANDS 的三種 canonical 結果及最終分析、信賴區間與報告。

| 模型 | ESCI RQ1/RQ2＋boundary | WANDS RQ1/RQ2＋boundary | ESCI canonical | WANDS canonical |
|---|---|---|---|---|
| Gemma2 | 完成 | 完成 | 三項規則完成 | 三項規則完成 |
| Harrier | 完成 | 完成 | 三項規則完成 | 三項規則完成 |

三項規則為 `lexical_ascending`、`lexical_descending`、`field_priority_type`。Jina 已由使用者取消，Qwen3-Embedding-4B 仍暫緩，均不納入完成範圍。

## 結果與證據

- [合併中文報告](reports/REPORT_zh.md)、[主要結果](reports/tables/main_candidates.csv)、[所有指標與信賴區間](reports/tables/summary.csv)、[論文補充文字](reports/MANUSCRIPT_ADDITIONS.tex)。
- [Gemma2 最終報告](models/embeddinggemma2/REPORT_zh.md)、[Harrier 最終報告](models/harrier06b/REPORT_zh.md)。
- [Harrier WANDS canonical 逐查詢結果](models/harrier06b/wands/canonical/query_metrics.parquet)、[逐 pair 排名](models/harrier06b/wands/canonical/pair_ranks.parquet)。
- [完整範圍驗證](provenance/final/verification.json)、[驗證檔案雜湊](provenance/final/verified_files.json)、[資料與歷史協定對照](provenance/SOURCE_MAP.md)。

Harrier WANDS encoding 在 2026-10-22 23:51:51 UTC 完成，評估於 23:52:18 完成，最終分析與報告於 23:54:24 完成（柏林時間 2026-10-23 01:54）。其 206 個 checkpoint、26,306 batches、128,525 unique inputs 與 128,982 列最終向量已核對原始輸入 fingerprints、planned indices、輸出 hashes、FP32 維度與有限值，並逐塊對照最終向量。原先 151 個 checkpoint 與失敗紀錄仍保留。

兩模型的四個 primary cells、四個 canonical cells、兩份 profile manifests、十個合併報告檔案及完成憑證均通過驗證。所有實驗程序已退出，定時監看已停止。這次上傳未重跑推論或改變科學設定。

## 主要發現與解讀範圍

VI@20 為 full-support query-macro 百分比：

| 模型 | 資料集 | Target VI@20 | Catalog VI@20 |
|---|---|---:|---:|
| Gemma2 | ESCI | 1.70% | 2.39% |
| Gemma2 | WANDS | 10.36% | 10.34% |
| Harrier | ESCI | 1.81% | 3.19% |
| Harrier | WANDS | 8.49% | 8.08% |

ESCI 使用 499 queries、4,434 highest-label pairs、10,076 商品；WANDS 使用 308 queries、21,299 pairs、42,994 商品。兩模型均呈現 WANDS 較高的順序敏感性，屬描述性比較，不能單獨歸因於架構或模型大小。不要混入舊 support、早期錯置的 ESCI gains，或混用 pair-micro 與 query-macro 指標。

兩模型、兩資料集、三種 canonical 的十二項 full-support Recall@20 相對七種 raw 平均差值，其 95% query-bootstrap CI 均包含零。這未建立明確的 Recall@20 改善或等效，也不能推論所有品質指標均無差異。標準 nDCG、judged-condensed cNDCG 與 judged coverage 均附 K=20、100 結果；例如 Gemma2 WANDS `field_priority_type` 的 cNDCG@20 差值為 +0.002432，未調整多重比較的 95% CI 為 [0.000175, 0.004722]，須保留多重比較限制。

資料、切分、母體、gains、raw schedules、seeds、ranking 與 bootstrap 與修正後歷史協定對齊；模型各自使用原生 prefix、pooling、維度、長度與批次設定。Boundary 結論僅涵蓋抽選且重複檢驗的 pairs。不能由 canonical 輸入固定推論實測 VI=0 或品質提升；後續新增模型不是整篇研究的事前註冊。

## 可攜式驗證與保存範圍

```console
python completed_results_20261023/verify_snapshot.py
```

[SNAPSHOT_MANIFEST.json](SNAPSHOT_MANIFEST.json) 列出全部上傳檔案的大小、SHA-256、來源位置、完成憑證與明確省略的大型輸出。匯出時重新核對完整驗證紀錄涵蓋的來源與輸出 hashes。可攜式 verifier 驗證本快照內檔案、憑證映射與全部 canonical 統計列；它不宣稱重新驗證未上傳的 bulk artifacts，也不重新執行 GPU 實驗。

模型權重、原始資料集、embedding／score matrices、cache／checkpoint 與 venv 保留本機，未隨此快照備份。`models/` 包含凍結協定、科學程式、統計表、逐 query／pair parquet 與 boundary 證據。原 protocol／receipt／科學程式的位元組保持不變，可能含 Windows 絕對路徑；`provenance/` 的早期紀錄保留當時狀態，現在的完成證據以 `provenance/final/` 為準。

[2026-10-22 快照](../completed_results_20261022/README.md)與[2026-10-21 快照](../completed_results_20261021/README.md)保留原狀，供對照進度與歷史結果。
