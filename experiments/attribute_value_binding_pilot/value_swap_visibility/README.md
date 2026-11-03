# Value-swap visibility pilot

只測量完整商品記錄的 C−A：交換兩個原始 values，保留 product ID、key、entry 順序、title、description 與其他 attributes。每次只替換一個 target，對完整 WANDS catalog 排名。第一輪及已取消的 `round2_controlled` 保留原狀；不執行其他實驗方向。

## 執行

從 repository root，以既有 `phase2/.venv` 執行；所有結果僅寫入本目錄。不要安裝或升級套件來繞過相容性檢查。精確模型、revision、runtime、prefix、pooling、normalization 與數值設定見 `config.json`。

已凍結資料的重跑命令（PowerShell）：

```powershell
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONDONTWRITEBYTECODE = '1'
& phase2/.venv/Scripts/python.exe experiments/attribute_value_binding_pilot/value_swap_visibility/test_integrity.py
& phase2/.venv/Scripts/python.exe experiments/attribute_value_binding_pilot/value_swap_visibility/run.py --stage smoke
& phase2/.venv/Scripts/python.exe experiments/attribute_value_binding_pilot/value_swap_visibility/run.py --stage pilot
& phase2/.venv/Scripts/python.exe experiments/attribute_value_binding_pilot/value_swap_visibility/analyze.py
& phase2/.venv/Scripts/python.exe experiments/attribute_value_binding_pilot/value_swap_visibility/verify.py
```

每一步必須成功才執行下一步。`pilot` 會檢查 smoke 的 manifest 與程式 checksum。`run.py` 使用具有精確文字、profile、來源 batch 及 checksum 的本地 forward cache；不是把其他條件的 embeddings 當成這次結果。驗證及分析不需要 GPU。

資料準備程式為 `prepare.py`，建立抽樣、完整 A/C 及凍結 manifest。在新實驗副本且尚未有結果時執行：

```powershell
& phase2/.venv/Scripts/python.exe experiments/attribute_value_binding_pilot/value_swap_visibility/prepare.py
```

交付目錄已凍結，**不要再次執行 prepare 覆寫它**；它會重建 manifest 的時間與 checksum，使既有 baseline / smoke 簽章失效。要重做抽樣應另存新的實驗目錄，保留現有資料。

## Baseline 相容性及必要重建

歷史 cache 的來源及 profile 通過檢查，但第一次 fresh replay 的 5 個 A 與 5 個 query 向量皆未 bitwise 相同。精確歷史數值 runtime 無法由 sidecar 完整還原；失敗不是 C−A 結果。`qa/historical_cache_replay/` 保留該次檢查，當時尚未計算 C。

先確認目前環境重複執行的 10 個 batches bitwise 相同，再按原有 `DenseEncoder._encode_native` 只重建 source-order 全庫及原始 query cache。重建成本估計、範圍與 profile 在 `baseline_rebuild_plan.json`，完成時間及新 array hashes 在 `baseline_manifest.json`。新的 baseline 綁定原凍結 selection manifest，沒有重新選樣，歷史 cache 未覆寫。`config.json.cache` 是初次檢查的歷史來源；**本次評分實際使用 baseline_manifest.json 的 arrays**。

若要重算此 source-only baseline（需 GPU，歷史估計約 25 分鐘），使用：

```powershell
& phase2/.venv/Scripts/python.exe experiments/attribute_value_binding_pilot/value_swap_visibility/rebuild_baseline.py
```

之後重新跑 smoke → pilot → analyze → verify。新 baseline 必須通過所有抽中 A/query 的 fresh replay gate；不能放寬 gate 混用不一致向量。此程式只重建一個模型的一個 baseline，不啟動任何矩陣。

## 抽樣與計算規則

- 用既有 parser 的 `|` entries 與第一個 `:`，透過從 0 起算的 entry index 對調完整原始 value。無全域 replace，raw strings 包括空白皆保留。選定 key 的身份須唯一；未選 key 重複不整件排除。
- 缺值清單、value 相同及 key 身份使用 casefold/空白正規化判斷，實際文字不正規化。沒有顏色、尺寸、類別、同類型或 query 提及欄位的限制。
- seed `20261030`；SHA256 canonical JSON `[str(seed), "value_swap_visibility:" + purpose, *str(ids)]`。商品交換以 `attribute_pair` hash 當 Python Random seed 在 valid pairs 均勻抽一組；query / target / smoke / ordinary_case 使用各自 purpose，依 hash 排序。每商品僅一組交換，先選組、再檢查 A/C 長度、最後選 query/target；不根據分數或排名改選。
- 沿用 held-out evaluation split 與 Exact qrels 清理。最多 100 queries、每 query 5 targets；少於上限就使用可用數量，不以其他 query 補足每 query 的缺額。
- GTE native max_length 8192；實際完整 A/C 包含空 prefix 與 special tokens。全庫與變體 token 數分別記錄，沒有假設交換前後必然相同。
- 以原 source length bucket 的同伴及 padding 重算 A/query；C 在同一 source batch 只替換 target。重算同伴只為數值條件匹配，排名時競爭商品仍使用 source baseline 向量。
- 對每個 query 以 FP32 `query_row @ catalog.T` 得到完整分數。各 target 獨立替換自己的 embedding 後以同形 matmul 計算 C 分數，核對其他 42,993 個分數逐 bit 不變，再還原。使用繼承的 target-only ranking 函式與 stable catalog-index tie break（等同升序 numeric product ID）。
- `rank_delta = rank_C - rank_A`；負數為改善。Smoke 以完整 score array 排序核對；全部 pairs 另以獨立計數公式核對。
- Summary 的 entry 分母是原本 Top-20 外 pairs；exit 是原本內 pairs；crossing 是全部 pairs。零分母 rate CSV 留空，報告列 NA。四分位數為 NumPy 線性插值。沒有顯著性檢定。
- 主要結果為全部 pairs；原始 rank 區間的結果僅供描述。所有 Top-20 events、最大改善／下降與 5 個不按結果篩選的固定 hash 普通案例均保留。

## 檔案索引

| 檔案 | 內容 |
|---|---|
| `report.md` | 結果、比例、案例及解讀限制 |
| `config.json`, `selection_manifest.json` | 精確設定、來源 checksum、跑分前凍結的輸入 |
| `baseline_rebuild_plan.json`, `baseline_manifest.json` | 必要重建理由、範圍、實際評分 baseline hashes |
| `data/selected_pairs.csv`, `data/smoke_pairs.csv` | query、Exact 標籤、商品、entry indices、raw keys/values、token counts、hash |
| `data/full_inputs.csv`, `.parquet` | 每個選中商品的完整原始／錯填字串及實際模型輸入 |
| `data/preselected_product_swaps.csv` | 全商品池先選定的唯一交換 |
| `data/eligible_products.csv`, `excluded_products.csv`, `entry_exclusion_counts.csv` | 合格及排除理由；entry 排除不等於商品排除 |
| `data/query_selection.csv`, `eligible_pair_disposition.csv` | 全 eligible query/pair 的入選／hash cap 原因 |
| `data/catalog_token_counts.csv`, `query_token_counts.csv`, `length_distribution.csv` | 實際全文 token 數及 truncation 檢查 |
| `data/sample_counts.json`, `baseline_batch_plan.json` | 母體／樣本數及原始 batch schedule |
| `results/smoke/pairs.csv` | 先執行的 5 pairs |
| `results/pilot/pairs.csv` | 全部逐筆分數、rank、Top-20、delta；每列是獨立 target-only 實驗 |
| `results/pilot/summary.csv`, `baseline_rank_distribution.csv` | 主摘要及原始排名分布；含 numerator/denominator |
| `results/pilot/cases.csv`, `case_full_inputs.csv` | 完整案例及原始／錯填全文 |
| `qa/` | 舊 cache mismatch、重算一致性、交換、排序、smoke/pilot gates、原工作保存驗證 |
| `cache/` | 本次實際 baseline 及可追溯的 forward vectors；不可與其他 profile 混用 |
| `delivery_manifest.json` | 最終輸出 hashes |

原始 qrels 只表示同一商品原本的相關性，不認證錯填內容的語意仍正確。這不是對模型內部理解能力的二選一測試；結果是完整 catalog 下的 target-only 能見度變化。
