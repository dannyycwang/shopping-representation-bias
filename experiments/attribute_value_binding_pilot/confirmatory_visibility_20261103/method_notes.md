# 設計、相容性、依賴與計時限制

使用 seed=20261103 及獨立 purpose hash；8 個池內商品不足五組有效配對、長度排除 0 個。沒有按新分數改選樣本。五組交換及所有實際輸入均先凍結。5 件商品跨確認 queries 重複（10 pairs）；3 件與開發資料商品重疊（3 pairs）。query bootstrap 保留每 query 全部觀察，但未完全處理跨 query 重複商品及其他資料相關性。

使用 10,000 次 paired query bootstrap，同一索引矩陣保留 targets、全部交換與模型條件；以分子／分母比重新估計，不把五組交換視為五倍樣本。報告 95% percentile CI，重點是效果量與事件數，不追加抽樣追求顯著。

GTE 與 Harrier 都沿用 pinned revision、prefix、pooling、normalization、precision 及 native cap。正式 A/query fresh replay 必須逐 bit 相同；競爭商品使用同一 source baseline，所有反事實獨立替換一個 target。Harrier 的歷史多表示 cache 僅取 C0 catalog slice。source/model/cache 詳細 checksum 在 manifest 與 cache_plan 中。

數值審查發現 Harrier 的舊 profile 指定完整 308-query 矩陣，而初版 smoke 逐 query 計分會出現浮點差異。在任何正式效果分數計算前已修正，並用原批次向量獨立重算、重新通過 smoke。早期不同計分的 Harrier smoke 保留為歷史 QA，不併入正式結果；樣本與變體沒有更動。GTE 保持前輪 pilot 的 query-row 計分方式。

最終 Harrier 另以原始全庫 baseline 的全部 13,242,152 個分數逐 bit 認證 256-column 計算區塊，保留完整 308-query 矩陣。164/168 區塊可用，另四區塊中的 23 件選中商品仍以原全庫矩陣計分；五個 smoke pairs 的全部 C/O 對所有 queries 也完全吻合。每次仍只替換一個 target，排名永遠使用完整 42,994 項 baseline array，絕非區塊／子庫排名。由此方法觀察到的每次 crossing 再以原完整矩陣驗分。

另在 native forward 回傳 CPU 向量後釋放未使用的 GPU allocator cache；沒有更動參數、精度、輸入或 batch。55 個原始 smoke 批次的所有向量逐 bit 一致。兩項執行調整與 gate 證據記錄在 execution_efficiency_amendment.json，並保留調整前 GTE 正式結果作完整 CSV 一致性核對。

| 模型 | 不同商品版本數 | 最短 tokens | 最長 tokens | native cap | 版本 token 數不同的商品數 |
|---|---|---|---|---|---|
| gte | 9581 | 113 | 1499 | 8192 | 0 |
| harrier | 9581 | 103 | 1483 | 32768 | 4 |

Token 數包含實際預處理與 special tokens，逐版本記錄，沒有假設交換必然保持 token 數相同。Harrier 為保留舊計分矩陣形狀而計算的開發 query rows 全數丟棄；選擇與分析只使用 208 個確認 queries。

| 模型 | 通過 same-shape packing 的商品 | 全部商品 | baseline native truncation 商品數 |
|---|---|---|---|
| gte | 837 | 871 | 0 |
| harrier | 827 | 871 | 0 |

效率調整保持來源 batch 大小及 padded length，以原最長輸入作 guard；所有可用位置的 A 與原 baseline 必須逐 bit 相同。Smoke 的每個 packed C/O 向量亦須逐 bit 等於原方法。不同 token 長度、小 batch 或 A 不吻合時回到原 companions，沒有排除商品。這項計算調整在正式評分前獨立記錄於 execution_amendment.json。

| 模型 | 執行段 | monotonic 秒 | 排除休眠的 awake 秒 | 可能休眠秒 | wall-clock jump 秒 |
|---|---|---|---|---|---|
| gte | first_completed_formal | 1650.9 | 1650.9 | 0.0 | 0.0 |
| gte | final_replay_and_rescore | 594.1 | 594.1 | 0.0 | 0.0 |
| harrier | resumed_formal | 7234.7 | 7234.7 | 0.0 | 0.2 |

計時使用 monotonic 與 Windows unbiased interrupt time；awake 時間包含 CPU、I/O、排程及 GPU 等待，不是純 GPU kernel 耗時。每個 forward checkpoint 亦保留建立時的計時及階段；中斷後的 cache 重用不重算為本次新 GPU 工作。

上表是完成的正式執行段。數值審查期間曾中斷初次 GTE 段；其 839 個已完成 forward checkpoints 有實測 awake 489.5 秒，後續直接重用。該早期中斷段缺少完整 end-to-end 結束計時，不能把這個 checkpoint 合計冒充該段全部活躍時間；未記錄的 setup/I/O overhead 未納入。之後各次執行均保留開始與完成／失敗狀態計時。準備、tokenization、smoke 與統計的獨立計時在 qa/，不與包含它們的上層段重複加總。

啟用通過檢查的效率調整時，也中斷過 Harrier 的 A replay：保留 676 個 forward checkpoints，實測 awake 合計 1271.1 秒。此數字同樣只涵蓋已完成 checkpoint 的計時，未冒充完整中斷段耗時。GTE 表中的 first_completed_formal 與 final_replay_and_rescore 是兩段實際工作；後者重用向量並核對同一批結果，不能把第一次耗時隱去、只報快速重跑時間。

11 月 4 日接續時，前一天的 Harrier 程序已不在執行；終止時刻與原因沒有被捕捉。該段另保存 209 個 forward checkpoints，實測 awake 合計 440.4 秒、checkpoint 內可辨識休眠約 19281.1 秒。這仍是已完成 forward 的下限，不能把整夜間隔算成活躍運算；逐筆計時與範圍見 qa/interrupted_20261103_final_attempt.json。接續程序僅在工作期間抑制 Windows 自動休眠，手動休眠或關機仍可能中斷。

重跑命令、凍結設計與檔案索引見 README.md。paper/ 提供兩張表的 CSV/Markdown/LaTeX，以及圖 1 的 PNG/PDF/SVG；這些是獨立論文素材，未寫入既有 manuscript。
