# Query constraint case analysis (2026-10-07)

從 repository root 執行一個命令：

```powershell
python query_constraint_cases_20261007/run.py
```

依序保留/驗證 frozen protocol → semantic tests → 固定 12 contrasts → 固定 3 cases 的 token/cache audit → independent verification → report/table/draft → output hashes。只寫本目錄，不訓練、不 download、不改 query/模型/support、不改 manuscript 或舊 option analysis。使用者後續已明確授權完成後 commit/push；runner 本身不執行 Git 寫入。

初始來源 HEAD `be9f8ab795a09f3da106526b4cf6bee80565dd2f`；已核對 historical reference `fb78cc9f2a2b03399b71931f50da16684aac3af5` 的 tree 相同。後續 checkout commit 會變，但 frozen input SHA 必須仍匹配。`prepare.py` 不會根據結果重建 protocol。selection 同樣不可在修正後改選；已完整稽核 367、292、300，停止擴張。

環境 Python 3.10.0；套件精確版本見 `requirements.txt` / `input_manifest.json`。不需 GPU；cache rescoring 用 CPU numpy FP32。需已有本地 pinned Hugging Face tokenizer snapshots；使用 local_files_only 且 HF_HUB_OFFLINE。tokenizer 會提示 uncapped 序列超過 max length，這是稽核刻意計數，沒有將過長序列送入 model forward。

必要本地 artifacts：

- 原始 `data/raw/product.csv`, `query.csv`, `label.csv`；processed WANDS products/queries/judgments。
- `phase4/config/splits.json`、`revision_graded_20260922/PROTOCOL.json` 與 per-query metrics；已有 option_coverage provenance / validated lists。
- 兩模型 × C0/C1/五 shuffles 的 top50 與 pair-rank parquet、metadata；selected C0/C1 product/query `.npy` payloads。
- 保存的 C0/C1 `phase2/data/representations/wands/*.jsonl.gz`；BGE/MiniLM pinned tokenizers。

完整確切路徑、bytes、SHA 位於 `input_manifest.json`、`qa/cache_tokenizer_manifest.json`。大 cache/data 在原 repo 的忽略範圍，這次不重新 push 原始大型資料；fresh clone 若沒有這些既有本地 artifacts，不能僅憑新分析目錄重跑來源核對。可直接讀已保存 CSV/JSONL/報告。缺資料時 runner fail-fast，不改用別的設定。當前本機所需 artifacts 全在；缺失的 per-batch/GPU/forward telemetry 與不可靠 numeric units 在 REPORT 明列。

實際最終執行命令、exit code 與 stdout/stderr 會保存 `qa/execution_log.json` / `qa/run_logs/`。準備/修正過程見 `qa/semantic_corrections.json` 與 `design_history/`，不是 preregistration。起始 dirty status 与 165 個保護檔 hash 見 `qa/preexisting_state.json`；任何既有資料或稿件變化都會使驗證失敗，不會還原/覆寫使用者檔案。

交付重點：

- `REPORT.md`：中文結論、正向/零/相反結果、分母、缺值與限制。
- `wooden_chair_outdoor_audit.md` / `top_cases/*/AUDIT.md`：來源逐件審閱、歷史與 matched-cache 分表、全進出與 token。
- `query_constraints.csv` / `protocol.json`：308 queries 與凍結條件；`query_comparison_summary.csv` 保留每組各方向。
- `candidate_evidence.jsonl`：4493 份 query-product-clause evidence，含 scope、qrel、rank、原始對應欄位與完整描述；`catalog_evidence.jsonl` 去重存原始全欄位，以 product_id 連結。
- `candidate_ranking.csv` / `candidate_selection.csv`：全部候選及未稽核原因。screen grade 不等於所有案例人工驗證。
- `missingness_summary.csv`：所有集合的 M/C/U/A 分母。
- `paper_case_table.tex`：表格 fragment；`RQ2_illustrative_case.txt`：106 words，未置入論文。
- `qa/independent_verification.json` / `qa/output_manifest.json`：檢核結果與輸出 hash。

本任務不要求新增大型 figure；以可逐項核對的表格呈現，沒有修改或新增論文圖。
