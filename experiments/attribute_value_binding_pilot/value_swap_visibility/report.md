# 商品欄位值錯填對能見度的影響

研究對象是同一件實體商品的記錄遭到刻意錯填；不假定文字語意不變。唯一處理是交換兩個完整原始 values（C），對照原記錄（A）。未做最小描述二選一、反事實 query、換序條件、reranking、訓練或論文修改。先前 round2_controlled 已依使用者要求取消，留存檔案未用於此實驗。

## 1. 是否觀察到排名改善或下降？

觀察到排名改善 153/422 (36.26%)、下降 129/422 (30.57%)、不變 140/422 (33.18%)。這是本次抽樣的描述性比例；有些改善不能推論錯填通常有利。

| 指標 | Q1 | 中位數 | Q3 |
|---|---|---|---|
| score_C − score_A | -0.0008394 | 0.0001655 | 0.0011916 |
| rank_C − rank_A | -2 | 0 | 1 |

rank_delta < 0 代表排名改善，> 0 代表下降；正 score_delta 代表分數提高。不同 query 的分數差僅在各自 A/C 內比較後作描述性彙整。

## 2–3. Top-20 進出及本次抽樣比例

觀察到 Top-20 entry 4/295 (1.36%)、exit 2/127 (1.57%)；crossing 為 6/422 (1.42%)。

Entry 分母只計原本 Top-20 外的 pairs；exit 分母只計原本 Top-20 內的 pairs；crossing 分母為全部 pairs。零分母為 NA。

| 事件 | numerator/denominator |
|---|---|
| 排名改善 | 153/422 (36.26%) |
| 排名下降 | 129/422 (30.57%) |
| 排名不變 | 140/422 (33.18%) |
| Top-20 entry | 4/295 (1.36%) |
| Top-20 exit | 2/127 (1.57%) |
| Top-20 crossing | 6/422 (1.42%) |

## 4. 抽樣、原始排名及限制

實際核對 WANDS catalog 為 42,994 件商品、480 個來源 queries；既有 held-out split 與有 Exact qrels 的交集為 308 個 queries、21,299 個 Exact pairs。清理方式沿用既有程式：重複同標籤合併、14 個衝突標籤 pairs 排除。

先對每個商品以 seed=20261030 及獨立 purpose 字串選一組欄位，再檢查全文 A/C 長度，最後 hash 抽取最多 100 queries、每 query 最多 5 個 Exact targets。實際抽中 100 queries、422 pairs、422 個不同商品；不足 500 是部分已抽中 query 的 eligible Exact targets 少於 5，未以其他 query 補滿。相同商品的 C 向量按商品 ID/輸入與 profile 重用；不以新分數、原始排名或是否接近 Top-20 選樣。

42,973 件商品通過通用解析與值條件；排除 21 件（21/42994）。可用 Exact pairs 為 21,290。欄位沒有顏色／尺寸或同類型白名單；query 不必提及交換欄位。缺值占位符清單在 config.json；yes/no/0 均保留。選定 key 以 casefold/空白正規化判定身份唯一，其他未選 key 的重複不排除整件商品。索引與原始字串均保留。

長度排除 0/42973；GTE cap=8192，實際全文含 special tokens，prefix 為空。全庫原始輸入最長 3,910 tokens，baseline native truncation 實測 0/42994。本次抽中 A/C 為 153–1354 tokens，各自實測後長度不同的 pairs 為 0/422，未事先假定 token 數相同。沒有縮短 title/description 或轉成最小描述。

| 原始 rank_A | pairs / 全部 |
|---|---|
| 1-20 | 127/422 (30.09%) |
| 21-100 | 105/422 (24.88%) |
| 101-1000 | 154/422 (36.49%) |
| 1001+ | 36/422 (8.53%) |

原本有 127/422 pairs 在 Top-20，另有 105/422 在 21–100，並非幾乎全部遠離 Top-20。跨越門檻事件仍僅 6/422；曝光進出與一般排名變動應分開看。極端名次變動也可能發生在很後面的順位，不能等同於取得 Top-20 曝光。

主要結果包含全部抽樣 pairs。summary.csv 的 rank 區間列僅為原始排名分層描述，未取代整體結果。每 query 可有多個相關 targets，且資料中的商品／類別可能相關；本輪不做把 pairs 視為獨立觀察的顯著性檢定。只用一個 encoder、一次固定欄位交換，不能推論其他模型或多次交換。

原始 Exact qrels 的用途是追蹤原本相關的同一商品，因錯填記錄而得到或失去曝光；它不認證錯填文字仍符合 query 語意。排名下降不一律等於模型失敗，排名改善也不等於商品更相關。

## 5. 是否值得擴大？

本次能提供真實完整 catalog 上的可量測變化，適合下一步以預先固定的新 seed／交換組重複，並在資源允許時增加另一個相容 encoder。應分開估計商品、query 與交換組的變異，保留全部樣本及 Top-20 正反向事件。擴大前不應把本次比例解讀為錯填的一般方向或商業效果。

## 可核對案例

cases.csv 收錄全部實際 Top-20 entry/exit、最大排名改善與最大下降（明確標為極端案例），以及全部 pairs 中 hash 最小的 5 個普通案例。極值同分以固定 hash 決定；普通案例完全不按結果篩選。下表在 entry/exit 各呈現固定 hash 的一例，其餘事件均可於 CSV 查閱。

| 案例類型 | query_id / product_id | 完整 query | 交換欄位與原值（C 對調） | score A→C | rank A→C |
|---|---|---|---|---|---|
| top20_entry | 263 / 36945 | small curtain rods | piecesincluded:1 rod , 2 finials and 2 brackets ↔ materialdetails : steel | 0.6925548 → 0.7038016 | 42 → 13 |
| top20_exit | 247 / 25951 | beaded curtains | color : green/beige ↔ canadaproductrestriction : no | 0.7070722 → 0.7000418 | 19 → 27 |
| extreme_rank_improvement | 153 / 17845 | bohemian | additionalmaterials : wooden stick ↔ overallproductweight:0.7 | 0.4493961 → 0.4558954 | 3430 → 2660 |
| extreme_rank_decline | 159 / 5849 | meditation | chairframematerial : cotton ↔ mountingequipmentincluded : no | 0.3815669 → 0.3799567 | 13950 → 14688 |
| ordinary_hash_sample | 453 / 39866 | midcentury tv unit | tvstandwidth-sidetoside:60 ↔ removablebackpanel : no | 0.6827257 → 0.6833414 | 4 → 4 |
| ordinary_hash_sample | 421 / 38917 | dining room tables | overalldepth-fronttoback:31.5 ↔ tablebasetype : pedestal | 0.6802432 → 0.6812873 | 543 → 525 |
| ordinary_hash_sample | 426 / 22638 | outdoor storage | primarymaterial : plastic ↔ topweightcapacity:660 | 0.6441068 → 0.6422288 | 287 → 294 |
| ordinary_hash_sample | 138 / 34162 | glass doors for bath | doorhanding : reversible ↔ frametype : framed | 0.7227887 → 0.7201496 | 11 → 12 |
| ordinary_hash_sample | 165 / 16013 | kitchen storage shelves | legheight-toptobottom:2 ↔ bins : no | 0.7309782 → 0.7297769 | 16 → 18 |

未出現的案例類型：無；上述類型均有實例。

## 模型、cache 與檢查

模型：Alibaba-NLP/gte-modernbert-base，revision `e7f32e3c00f91d699e8c43b53106206bcc72bb22`。沿用 phase2 DenseEncoder：空 query/document prefix、CLS、FP16 model、FP32 pooling/L2 normalization、SDPA、native cap 8192。使用既有 phase2/.venv（transformers 4.55.4、torch 2.11.0+cu128），完整 runtime 與模型 snapshot checksum 在 config.json。

歷史 cache 的來源 checksum、順序、模型及 profile 符合，但最初 smoke 的 5 個 A 與 5 個 query 向量均未 bitwise 重現，最大分量差 0.0009531081。sidecar 缺完整套件版本，不能把差異確定歸因於某一項 runtime 設定。重複執行目前環境的 10 個 batches 全部 bitwise 相同後，僅以既有 DenseEncoder 重建 source-order 全庫與原始 queries，耗時 29.89 分鐘。舊 cache 保留，但未用於主要評分；新 manifest 綁定原凍結抽樣與精確 profile。

主要評分前，以原始 length-sorted batch 及 companions fresh replay 所有抽中 A 和 queries，對使用中的 baseline 最大向量差 0，全部 bitwise 相同。C 也由原 batch 僅替換該 target 的文字編碼，未混用其他設定 embeddings。

每個 pair 都獨立把 target embedding 放回同一完整 catalog index，使用相同 shape 的 FP32 matmul；所有競爭商品分數 bitwise 不變，再以沿用的 target-only 公式排名。Smoke 先跑 5 pairs 並以完整 score array 排序核對 A/C；pilot 全部另以獨立計數公式核對。相同 query 下的 targets 從未同時錯填；結果不是一次 catalog-wide intervention。

下列與歷史結果的差異同時包含 cache 重建的數值差異及 matmul shape 差異，屬相容性紀錄，不是錯填效應。歷史結果採全 query-matrix matmul，本次為一個 query × full catalog 的同形 A/C matmul。與歷史逐筆 A scores 的最大差為 0.00164；抽中 A ranks 差異 169/422。詳見 qa/pilot/historical_rank_comparison.csv，A/C 主要比較使用完全一致的 baseline 及運算形式。

時間紀錄限制：首次正式 pilot 的 time.time() 累計跨越 2026-10-30 至 2026-11-03，stdout 在 C 編碼期間出現大幅牆鐘時間跳躍。該次 complete.json 的 seconds 包含跨日間隔，不能當成純 GPU 計算耗時或成本估計；實際活躍 GPU 時間未獨立量測。原始時間觀察保存於 qa/timing_observation.json。

重跑命令與所有檔案索引見 README.md。selection_manifest.json 在計算新分數前凍結；完整實際 A/C 在 data/full_inputs.csv/.parquet。data/ 包含母體、排除、長度與抽樣清單；results/pilot/ 包含逐筆、summary、rank distribution 與案例；qa/ 保留 replay、排名及第一輪／論文未變檢查。
