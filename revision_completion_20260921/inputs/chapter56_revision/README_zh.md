# 第五、六章修訂與執行導覽

依據：作者提供的第 (6) 版 PDF，以及 2026-09-21 讀取的 GitHub main `b73e52ae4a787e33c5773673694464b8f2743e56`。前四章保持原檔不動；新文件替換第五、六章。文內使用連續段落，不新增 `\paragraph`。

## 已交付的文字

- `chapter5_experimental_setup.tex`：完整第五章；資料與評估母體、encoder 與七種 schedules、共同評估與選擇歷史。
- `chapter6_results.tex`：以已核對證據寫成的第六章；6.1 順序如何改變候選資格；6.2 Recall 如何掩蓋商品身分變化；6.3 固定表示之後的效果、商品身分與成本。
- `tables/`：兩張主表，以及完整 canonical CI、fully-fitting 支持數與 cancellation 附表。表格直接從 checked CSV 生成，不手抄挑選勝出條件。
- `figures/`：沿用已存在的 RQ1/RQ2 主圖與兩張附錄圖。這次沒有改動來源估計或 CI。
- `appendix_evaluation_details.tex`：可併入現有附錄的重現細節，包含第四章承諾的 canonical matching/fallback 規則；不是額外的完整附錄重寫。
- `CODEX_EXECUTION_PROMPT.md`：交給本機 Codex 的完整執行規格。

第五章引用沿用現有 bibliography keys `chen2022wands`、`reddy2022shopping`；模型使用精確名稱與 pinned protocol，不引入未核對的新模型文獻。

## 這版能用到哪裡

這是一份有已驗證結果的可編譯工作稿。6.1 與 6.2 的主結論已有支撐；6.3 的 canonical 比較已完整，aggregation 的共同 Recall 表也已從既有結果整理好。6.3 尚未完成「全部策略共同的 paired membership/uncertainty 表」與「hybrid 七種順序的 VI/三狀態」。因此末段明確限於已存在的 source-order hybrid 效果，沒有宣稱 hybrid 提高或降低 VI。LaTeX 註解標明補完後要替換的位置，正文沒有假數字或 TBD 結果。

完成 P1/P2 後，第六章可自然收束到：不同策略如何影響效果、持續涵蓋與候選商品身分。不要為了配合預期 story 挑選一種策略或排列。

## 圖表放置

| 位置 | 圖表 | 狀態與動作 |
|---|---|---|
| Introduction | 作者的 Fig. 1，11 與 939 個案 | 保留；不是典型位移幅度的估計 |
| Section 3 | 作者的 Fig. 2 | 保留 |
| 6.1，target-only/fully-fitting 段落附近 | `figures/rq1_vi_top20_main.pdf`，預期 Fig. 3 | 已有；target-only，不得改標 catalog-wide；資料集橫軸範圍不同 |
| 6.2，gains/losses 段落附近 | `figures/rq2_membership_top20_main.pdf`，預期 Fig. 4 | 已有；Codex 需將圖例改成 Newly included / Newly omitted；本包先沿用原 PDF，正文與 caption 已明確解釋 |
| 6.3，canonical 段落 | `tables/canonical_top20.tex` | 已生成；六個 dataset/encoder rows，三種固定規則的 Delta Recall；完整 paired CI 表另附，節省主文空間 |
| 6.3，aggregation/hybrid 段落 | `tables/platform_recall100.tex` | 已生成的共同 source-order Recall 點估計；P1/P2 後以 `strategy_joint_main.tex` 替換為 mean Recall/VI 聯合表 |
| 附錄 | `tables/cancellation_top20.tex` | 主文已交代全部六種 dataset/encoder 的範圍；小表優先放附錄，節省篇幅 |
| 附錄 | `tables/fitting_support.tex` | 完整 query/pair 支持數 |
| 附錄或 artifact | `rq1_states_top20_top100_appendix.pdf`、`rq2_membership_top100_appendix.pdf` | 已有；只挑主文理解所需的內容放有限 PDF 附錄，其餘保留 artifact |
| 附錄，必要時提升 6.1 | 新的 cutoff/severity 圖 | P3 由既有 ranks 產生；先規劃為附圖，避免主文再加大型圖 |

Fig. 4 的 intervals 屬於 net Recall，並不是左右 bars 的 CI。左右份額的分母為 query 的完整最高相關集合 Hq，而非 Top-20 列表長度。

## 已完成，不需要再跑

1. Raw 七種順序、兩個 interventions、2 datasets × 3 encoders 的主要敏感性分析。
2. 每個 encoder 都使用其自己的 tokenizer/token cap 的 fully-fitting 分析。
3. 六個 C0-relative contrasts 與全部 21 pairs 的 membership 分析；exact cancellation 由整數 gain/loss 判斷。
4. 三種純 canonical 規則 × 兩資料集 × 三模型，共 18 conditions；兩個 cutoffs 的結果、配對 CI、gains/losses 已齊。
5. BGE/MiniLM 的既有 Set-Mean、BM25、source-order hybrid、M=2 centroid/max；BGE 的 M=4/7 額外 sweep。

`canonical_complete_results.csv` 保留三種純規則的完整 36 rows；`platform_recall100_points.csv` 保留本次整理的 28 個方法/資料集/模型點估計與來源路徑。後者不代替新 joint comparison 的 paired CI。

## 優先完成的工作

| 優先序 | 工作 | 新編碼/訓練需求 | 補上哪個論證 |
|---|---|---|---|
| P0 | 檢查來源、support、命名與既有檔案 | 無 | 確保比較同母體，沒有把 M2 或不同 track 拼入 |
| P1 必做 | BGE/MiniLM × 兩資料集，全部七種 raw orders 的 full-ranking BM25+dense RRF | 優先重用分數/embedding；不訓練 | hybrid 的效果與順序穩定性是否一致 |
| P2 必做 | 主要策略共同的 Recall、三狀態、paired gains/losses、CI 與成本表 | 主要是重分析 | 第四章的比較承諾在第六章有共同衡量方式 |
| P3 建議完成 | cutoff sweep 與 rank-crossing severity | 既有完整相關商品 ranks 即可 | 是否只在 K 附近擺動；擴大候選數保住多少商品 |
| P4 可選 | 額外 schedules 或 graded metric 診斷 | 可需要額外編碼；不是本次必做 | 有限排列族與只看 Recall 的適用範圍 |

先完成便宜的 paired analysis 與 hybrid fusion；不需要新增 rewriting、Set Transformer 或訓練流程。

## 前四章只需作者確認的銜接問題

這次不直接修改前四章。以下是新版 PDF 中會影響後文的明確問題：

1. **Section 4.3 第一個句子**把 lexical 與 hybrid 都描述為 naturally indifferent to input ordering，但同節後面正確指出 hybrid 不保證 invariant。最小替換：`We include lexical and hybrid retrieval controls to examine the contribution of order-invariant lexical evidence.`
2. **Introduction 第二項 contribution**使用 stable aggregate effectiveness / significant turnover。現有 WANDS/MiniLM 有較明顯 Recall 下降，不能把全部設定概括成 aggregate stable；第六章已採用「net Recall can understate membership changes」。significant 若不是已定義的統計檢定結果，應避免這個詞。
3. **Introduction 第三項 contribution**預告平均效果提高可伴隨更高 instability。現有 source-order hybrid 表不足以支持它，固定 canonical/aggregation 的 structural VI=0 也不能支持。請先等待 P1/P2 的 claim verdict；若未支持，最終必須把承諾改成「jointly characterizing effectiveness and consistent inclusion」。不能挑另一個 rewriting track 來補這句。
4. Section 4.1 的 lowercase-normalized 可以在附錄精確交代為 `casefold()`；它不是只針對 ASCII 的 `lower()`。本包的附錄已有精確規則。

## 安裝與篇幅

將兩個章節檔替換論文中的 `sections/experiments.tex`、`sections/results.tex`，或修改 main 的兩個 `\input` 指向新檔。將 `tables/`、`figures/` 合併到 manuscript root，保留原 Fig. 1/2 與其他檔案。需要 `graphicx`、`booktabs`；ACM 的 `\Description` 正常使用。請勿用 repo 舊 main/intro/methodology 覆蓋作者 PDF 對應的最新前四章來源。

最新版前四章已延伸到第 5 頁，摘要仍未完成。第五章約 0.6--0.8 頁、第六章約 2--2.5 頁只是目標，實際取決於 ACM 浮動圖表。**這版需要在作者最新 Overleaf/ACM 稿件中合併確認，不能保證不調整篇幅就能進 8 頁。** 優先將完整 cancellation 表、完整 states 圖、view-budget/cost sweep 放附錄或 artifact；不要縮小主圖到讀不清楚。

WWW 2027 要求主文 8 頁、含 references/optional appendix 最多 12 頁，前 8 頁必須自足：[官方要求](https://www2027.thewebconf.org/research-track-papers/)。局部 LaTeX proof 不等於整稿 ACM 頁數通過。

## 主要來源

- [固定設計與 encoder 設定](https://github.com/dannyycwang/shopping-representation-bias/blob/b73e52ae4a787e33c5773673694464b8f2743e56/revision_evidence_20260917/PROSPECTIVE_PROTOCOL.json)
- [Raw 證據與母體](https://github.com/dannyycwang/shopping-representation-bias/blob/b73e52ae4a787e33c5773673694464b8f2743e56/revision_evidence_20260917/EVIDENCE_STATUS.md)
- [Canonical 全條件](https://github.com/dannyycwang/shopping-representation-bias/blob/b73e52ae4a787e33c5773673694464b8f2743e56/revision_evidence_20260917/CANONICAL_CONTROLS.md)
- [資料欄位建構，尤其 ESCI entry 定義](https://github.com/dannyycwang/shopping-representation-bias/blob/b73e52ae4a787e33c5773673694464b8f2743e56/phase2/scripts/prepare_data.py)
- [方法實作與 source-order hybrid 的現有範圍](https://github.com/dannyycwang/shopping-representation-bias/blob/b73e52ae4a787e33c5773673694464b8f2743e56/phase4/scripts/run_methods.py)
- [BGE view-budget sweep](https://github.com/dannyycwang/shopping-representation-bias/blob/b73e52ae4a787e33c5773673694464b8f2743e56/phase4/results/saturation_summary.csv)

## 驗證範圍

已核對主文數字與共同評估分母；table generator 的 MiniLM/BGE M=2 結果從每個 query 的檔案計算，WANDS 只保留固定 held-out IDs。附包數值檔保留原始精度。

局部 LaTeX 編譯及視覺檢查結果記於 `VALIDATION.md`。目前不宣稱 P1/P2/P3 已執行，也不宣稱前四章已被更動。
