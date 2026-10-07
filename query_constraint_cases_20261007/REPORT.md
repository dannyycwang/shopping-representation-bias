# Query-explicit constraint case search — 2026-10-07

**三個完整稽核案例中，沒有找到比 query 367 更清楚的替代例子。** 木椅可作 **A 級、frame-material 子條件** 的探索性 illustration：完整 Top20 的確認木框架數 8→6，Exact 子集 5→4；相同商品的不確定性抵消後淨差分別固定為 −2、−1。Recall 完全相同，但 nDCG 下降。白色抱枕為 C；MiniLM 羊毛地毯最初看似 A，完整描述揭露混紡變體後降為 B，沒有重選第四例。

重要限制：木椅 8 件進出商品的 16 個實際序列全部超過 512 tokens。因此可以用於 native catalog-wide pipeline 的案例，**不能作 fully-fitting 或已排除截斷的證據**。本次只證實既有向量在同一計分實作重現；未重新執行模型 forward。

| Query | Model | 最終等級 | Exact Recall | 完整 T MATCH | 抵消共有項 Δ 界限 | Exact R MATCH | Δ nDCG | 進出序列完整輸入 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 367 | bge_base | A（子條件） | 7/42 → 7/42 | 8→6 | [-2, -2] | 5→4 | -0.005347114988415 | 0/16 |
| 292 | bge_base | C（子條件） | 0/12 → 0/12 | 4→2 | [-6, 4] | 0→0 | 0.022983306442240 | 2/28 |
| 300 | minilm | B（子條件） | 0/2 → 0/2 | 17→16 | [-2, 0] | 0→0 | 0.000000000000000 | 2/20 |

A/B/C 等級針對明確標出的子條件。白色案例產品類型及混色歧義較多；羊毛案例的 0/2 Recall 及多重未核定限制，不宜用來取代木椅。其餘篩檢 A 級（如 q275/q417 的 secondary contrasts）沒有在三例上限之外做完整來源/token/重現稽核，**不能聲稱搜尋空間不存在更佳案例**。

## 設定、來源與有效分母

實際起點 `be9f8ab795a09f3da106526b4cf6bee80565dd2f`，歷史參考 `fb78cc9f2a2b03399b71931f50da16684aac3af5`。兩者 tree 相同，參考是 merge commit；來源檔逐一 SHA 保存，沒有靜默拼接不同版本。主分析 WANDS/BGE native C0→C1，K=20，所有 42,994 競爭商品都套用 attribute ordering。它不是 target-only/fixed-competitor。

原始 480 queries；既有 heldout 384；其中 76 沒有 Exact，排除後 eligible 308，Exact query-product pairs 21299。H_q 固定為原 processed qrels 的 Exact；Recall=|Top20∩H_q|/|H_q|，gain 為 3/1/0，unjudged 評分 0、但不等於屬性 CONTRADICTION。獨立重核 raw labels 清理：14 conflicting pairs、1559 duplicate rows；清理結果與 231859 筆 processed qrels 完全一致。原始 query 逐字相同。

依 query/schema 預先保存 62 queries、68 個可執行屬性子條件；其餘 246 queries 也有排除/延後理由。numeric dimension 缺可靠單位/方向而延後；直接記錄的 nominal bedding sizes 有納入。所有多條件原文與其他限制保留，不由單條件 MATCH 推論整句滿足。

14 個 schedule runs、12 個預定 contrasts 全可用；總 3696 query-contrast、816 assessed clause-contrast，另有未執行 clause 的 placeholders，合計 3768 summary rows。744 個 attribute-query-contrast 其實是同 62 queries 重複 12 次，816 clauses 亦非獨立樣本。76 candidate clause-contrast，來自 28 個 distinct queries；同 query 不同模型/seed 重複保留。主分析 73 nonzero Exact-cancellation、194 equal-count Recall queries，均由整數集合重算，不用先前常數強制結果。

## 全範圍結果：包括相反方向與零結果

| Comparison | Query contrasts | 有條件 queries / clauses | Exact cancellation | Equal Recall | Candidates | Screen A/B/C | 證實下降/上升 | 未定下降/上升 | confirmed 不變 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bge_base_C0_C1 | 308 | 62/68 | 73 | 194 | 7 | 1/4/2 | 1/1 | 11/14 | 41 |
| bge_base_C0_C2s1 | 308 | 62/68 | 78 | 207 | 2 | 1/1/0 | 1/1 | 11/15 | 40 |
| bge_base_C0_C2s2 | 308 | 62/68 | 78 | 202 | 4 | 0/3/1 | 0/2 | 9/11 | 46 |
| bge_base_C0_C2s3 | 308 | 62/68 | 68 | 195 | 3 | 0/3/0 | 2/1 | 19/6 | 40 |
| bge_base_C0_C2s4 | 308 | 62/68 | 69 | 195 | 4 | 1/2/1 | 2/0 | 13/14 | 39 |
| bge_base_C0_C2s5 | 308 | 62/68 | 67 | 193 | 3 | 0/2/1 | 0/1 | 14/14 | 39 |
| minilm_C0_C1 | 308 | 62/68 | 57 | 154 | 12 | 0/3/9 | 3/0 | 19/14 | 32 |
| minilm_C0_C2s1 | 308 | 62/68 | 52 | 153 | 7 | 1/2/4 | 2/0 | 11/16 | 39 |
| minilm_C0_C2s2 | 308 | 62/68 | 60 | 170 | 9 | 0/5/4 | 1/0 | 15/12 | 40 |
| minilm_C0_C2s3 | 308 | 62/68 | 59 | 172 | 7 | 1/0/6 | 3/0 | 12/15 | 38 |
| minilm_C0_C2s4 | 308 | 62/68 | 64 | 176 | 9 | 0/3/6 | 0/0 | 16/15 | 37 |
| minilm_C0_C2s5 | 308 | 62/68 | 67 | 173 | 9 | 1/4/4 | 3/0 | 15/9 | 41 |

表中 A/B/C 是保守自動 screen 的級別，不能視為全部已人工核實的案例。12 組共 18 個 clause-contrast 有 tightened bound 支持下降、6 個支持上升、165 個確認 MATCH 下降但淨差不定、155 個確認 MATCH 上升但淨差不定、472 個確認 MATCH 不變。最後一類也可能仍有不確定性，並非證明真正符合數完全不變。沒有 post-selection 顯著性檢定，這些不是一般發生率估計。

主分析 68 個 clauses 中：1 下降（木椅）、1 上升（q440 `wood bar stools`，MATCH 9→12、tight [1,3]，Recall 10/255→12/255，故不滿足 equal-Recall candidate）。另有 11 未定下降、14 未定上升、41 confirmed 不變；其中 24 個前後皆零確認 MATCH。全範圍通過完整證據的 MATCH 消失數為 0，不能用零確認 MATCH 代替消失證明。

## 缺值與歧義

| 主分析所有 68 子條件×Top20（非獨立商品） | MATCH | CONTRADICTION | UNKNOWN | AMBIGUOUS | 總數 |
| --- | --- | --- | --- | --- | --- |
| before | 347 | 79 | 402 | 532 | 1360 |
| after | 351 | 89 | 409 | 511 | 1360 |

`missingness_summary.csv` 列出全部 12 comparisons 的 full/Exact/joint、前後四狀態與分母。這些計數是 product-clause instances，可重複同商品；不偽裝成獨立樣本。每個 query 的具體缺值、變體和 scoped uncertainty 皆在完整 summary/evidence。

木椅最終 full 為 8/3/1/8→6/5/1/8；loose [-11,7]，抵消前後相同商品後 tight [-2,-2]。Exact 為 5/0/1/1→4/1/1/1；loose [-3,1]、tight [-1,-1]。共同的 1 UNKNOWN、8 AMBIGUOUS 使絕對符合數不確定，但都未進出，不能任意在兩邊換真值，因此不影響淨差。這依賴屬性/商品身分固定；全原始紀錄重數與非排序文字均核對。

白色 full 4/1/2/13→2/1/2/15，tight [-6,4]；羊毛 full 17/0/0/3→16/1/0/3，tight [-2,0]。二者都不能證實完整候選的真實淨損失；Exact 子集則各為空集合。三例 joint-MATCH 均不能證明全部 query 選項消失。

## 木椅線索的核對與更正

- H=42、Exact 7→7、Recall 7/42 完全重現。Joetta 26196 是 6→23；Vandergriff 36317 是 12→69；Jettie 26195 是 30→20；Lewallen 32126 是 38→17；四件都為 Exact，完整涵蓋 Exact 進出。
- nDCG 0.434359699004102→0.429012584015687，Δ -0.005347114988415；不能寫兩個 effectiveness metrics 不變。
- 全 Top20 另有 1817/3604 退出、35827/4355 進入；納入後木框架淨差仍 −2。新增 4355 是 Partial，其餘三件 unjudged。
- 完整描述審閱使共同的三件木欄位/鋁底座商品變成 AMBIGUOUS，所以初次 screen 的 11→9 修正為 **8→6**，而不是忽略冲突。Exact 的 5→4 不變。所有進出者對該框架子條件均已判清，淨差不因共同歧義而反轉。
- 名稱 wood 與塑膠框架不能互相代換；wood grain 是外觀，座面材質仍未核定。Lewallen 仍為官方 Exact，不改標籤、不宣稱是假標。
- 全部 8 進出商品的 16 序列為 **555–870 tokens**，包含 [CLS]/[SEP]，沒有一個 ≤512。不可使用 fully-fitting 說法。

完整逐件來源、全部留存者、所有進出者的 token 數及正反表請見 [wooden_chair_outdoor_audit.md](wooden_chair_outdoor_audit.md)。

## 設計凍結、修正與選擇透明度

這是 exploratory、非 preregistered；367/292 都是已知 seed。v1.0 到 v1.1 的 nominal-size / bamboo 修訂在任何本輪排名方向計算前，原檔位於 `design_history/`。已凍結 protocol/constraints/source SHA 持續檢查。

兩次執行語意修正發生在結果可见後，完整列於 `qa/semantic_corrections.json`：先修正類型分類不能把未命中 class 字串當成矛盾；再於三例完整描述審閱後，補足 collection 纖維變體及 structural-base 衝突的保守檢查。都全量重跑原 12 comparisons，沒有改動 query、support、K、field maps 或 schedule family。不能把最終所有 adjudication 稱為排名前完全固定。初次 screen 與修正前 ranking 有留存，可重建修訂歷程。

在第二次修正前，固定兩個 seed 與最高排序的另一 query 300，沒有取最大 rank drop。full-source audit 將 300 從 A 降 B 後，仍保留三例，不補換 q275/q417。`candidate_selection.csv` 保存全部 76 records 的排序欄位與未稽核理由；`selection.json` 鎖定三例。

## 模型、重現與尚缺的 artifacts

BGE name `BAAI/bge-base-en-v1.5`，revision `a5beb1e3e68b9ab74eb54cfd186867f64f240e1a`，CLS、512 tokens、configured batch 48。MiniLM `sentence-transformers/all-MiniLM-L6-v2` revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`，attention-mask mean、256、configured batch 128。tokenizer 與 model revision 相同，BertTokenizerFast；empty query prefix；所有 cached execution metadata 同為 phase2-encoder-v1、CUDA fp16 model/FP32 pooling、L2 normalize、FP32 dot product、stable original-catalog tie。當前 encoding.py 的 v3 實作與 cache 的 v1 不混稱為同次 forward。

實際 template：依原 section_order 非空段落以 newline 串接 title/class/category/description/pipe-joined raw attributes；C1 只反轉 entries。保存的 representation 檔案雜湊、解壓文字 hash、全部 42,994 IDs 次序與入選模板均核對。三例各 C0/C1 的 full-catalog cache 計分都重現 Top20 順序，歷史/new matched cached scoring 分表；沒有只在聯集重新排序。

所需排名、原始商品/query/qrels、14 product caches、兩個 query caches 與 pinned local tokenizers 都存在，沒有因此跳過任何預定 contrast。仍缺：歷史每個 batch 的實際分組與 padding/OOM subdivision、每 artifact GPU 型號及完整 forward telemetry；無新 matched encoder forward，所以這些不能補成已知。缺 unit/axis schema 的 numeric dimension 未做；部分來源本身少描述、少映射欄位或有變體冲突，明列 U/A。入選聯集 1 件有至少一邊無保存的 exact outside rank，歷史欄只寫 >20；新 cache 計分提供分開的 exact ranks。未另行標註原始圖片、未用外部網頁補資料。没有預測 satisfaction/sales/exposure/agent success 的資料。

## 可支持的稿件措辭

建議主文若需一例，使用木椅的 **native pipeline / recorded frame clause**，並保留截斷限制；若段落主張完整輸入的純位置效應，這例不適用。它支持：在 catalog-wide attribute reversal 下，benchmark-label Recall 相同可以伴隨有明確記錄的木框架候選減少；這個例子的 nDCG 有小幅下降。它不支持 preference/utility/satisfaction 已下降，也不支持一個商品直接因果排擠另一個、官方 qrel 已被證偽或兩項指標皆無變化。

`paper_case_table.tex` 列出所有四件 relevant 進出與兩種集合的完整四狀態分母；`RQ2_illustrative_case.txt` 提供 80–120-word 英文草稿。都留在本分析目錄，沒有編輯 manuscript。

## 驗證與交付

獨立於 core 的 `verify.py` 已重算 3696 組集合/指標、816 組子條件 full/Exact/joint bounds，驗證原始 qrels/query、來源 SHA、三例重現與 token 一致性。14 個語意測試涵蓋缺值、複合/變體、共同未知抵消、full vs Exact、scope、>20 與描述衝突。165 個既有檔案（含 manuscript、舊 analysis 和已修改檔）SHA 未變。算術驗證不是第二位人工 annotator；保守字面映射可能漏掉可由更深入理解支持的 MATCH。

一命令與環境依賴見 [README.md](README.md)；資料/執行設定見 `input_manifest.json`、`protocol.json`、`qa/cache_tokenizer_manifest.json`、`qa/reproduction_runtime.json`；可核對的最終數據在 `query_comparison_summary.csv`、`comparison_summary.csv`、`candidate_evidence.jsonl`、`catalog_evidence.jsonl` 與 `top_cases/`。沒有新 canonical、hybrid、ESCI、模型、K 或額外 schedule；完成既定搜尋與三例後停止。
