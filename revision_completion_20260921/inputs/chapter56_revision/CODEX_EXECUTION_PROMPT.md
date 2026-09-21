# Codex 執行規格：完成第 5、6 章的共同評估與圖表

你正在處理 `dannyycwang/shopping-representation-bias`。本次目標是補齊論文第五、六章需要的證據，沿用既有研究定義與結果，完成可重現分析、圖表和可核對的英文結果文字。請先讀本 repo 的 AGENTS.md（若有）與相關 protocol，再執行以下 P0--P3。不要停在提出計畫；遇到缺失輸入則先完成所有不依賴它的工作，精確列出缺口。

## 研究定位與不可混淆的範圍

主線：完整 attribute entries 的排列可以改變相關商品的 candidate inclusion；Recall 的淨變動可能掩蓋商品更替；不同表示/檢索 controls 必須同時評估效果、持續涵蓋、商品身分與成本。

不要預設 mitigation 必須成功，或 invariant representation 必須犧牲 Recall。混合與負面結果都保留。不要為了配合 Introduction 的既有句子尋找 test-set winner。

作者最新前四章以第 (6) 版 PDF 對應的來源為準，repo 的舊 introduction/methodology 不一定是最新版。本次不要覆蓋、重建或重寫前四章。新工作集中於 `revision_completion_20260921/`；保留所有 Phase II--VI、`revision_evidence_20260917/` 的歷史輸出。不要重新執行會覆寫舊 manuscript 的歷史 `write_results.py` 或 `write_manuscript.py`。

本規格核對時的 main SHA 是 `b73e52ae4a787e33c5773673694464b8f2743e56`。若 HEAD 已更新，先讀相關 diff，重用已完成工作，記錄實際 HEAD；不要 reset 或覆蓋使用者變更。

## P0：一次完成來源與評估母體核對

閱讀：

- `revision_evidence_20260917/PROSPECTIVE_PROTOCOL.json`
- `revision_evidence_20260917/EVIDENCE_STATUS.md`
- `revision_evidence_20260917/CANONICAL_CONTROLS.md`
- `revision_evidence_20260917/data/evaluation_populations.json`
- `revision_evidence_20260917/data/{membership_changes_sources,inclusion_states_sources,canonical_sources}.json`
- `phase4/config/{EXPERIMENT_PROTOCOL.frozen.md,splits.json,selection.json}`
- `phase4/scripts/{common.py,run_methods.py}`
- `phase2/src/{representations.py,encoding.py,evaluation.py}`

固定 primary populations：

| Dataset | Requested IDs | Eligible queries | Highest-label pairs | Catalog |
|---|---:|---:|---:|---:|
| WANDS | existing 384 held-out | 308 | 21,299 Exact | 42,994 |
| ESCI | existing 500-query evaluation sample | 499 | 4,434 E | 10,076 |

WANDS 96 development queries 不進主結果；ESCI 不是新的 untouched held-out sample。所有指標先在每個 query 內用完整 Hq 計算，再 query-macro aggregation。Pair-micro 另存。Hq 包含兩次檢索都缺席的 highest-label products。

七種 raw schedules 固定為 C0、C1、C2s1--C2s5。C1 是 reverse；random seeds 是 20260911--20260915。保留 raw source-order Recall 與 mean-over-seven Recall 的不同名稱。Target-only inclusion 來自分開的 counterfactual indexes，不可稱一個共同 index 的 Recall。

Canonical 原始結果已完成 18/18 條件，不能再跑一遍當新實驗。`canonical_raw` 對應純 lexical ascending；`canonical`/M2 是另外改 framing 的歷史模板，不得拼進 pure sorting。歷史 `rule_appearance` 與新固定 `field_priority_type` 也不同。

屬性單位以現有 preprocessing 為準：WANDS split pipe；ESCI 最多三個 entries（brand、color、整段 bullet-point field），不重新拆句、不補 field label。保留重複 entries 與重複 views。

產出 `SOURCE_AUDIT.md` 與 machine-readable source manifest，記錄來源路徑、SHA256、model revision、query/product ordering、dtype、母體與可重用快取。區分「本機檔案存在」、「Git 中只追蹤 metadata」、「缺失」。不要宣稱未執行的檔案已完成。

## P1：補齊七種 schedule 的 hybrid 評估（必做）

固定 cells：WANDS/ESCI × MiniLM/BGE；GTE 暫只維持原始敏感性及 canonical 對照，不為湊矩陣新增 GTE aggregation。每 cell 跑七種 catalog-wide raw schedules。

1. 優先使用既有完整 raw dense scores、產品 embeddings 與 historical query vectors。Highest-label pair ranks 或 Top-1000 lists **不夠**重建 full-ranking fusion；先確認全 catalog scores/ranks 或 embeddings 可用。
2. BM25 固定使用完整 source-order product text，lowercase `[a-z0-9]+` token counts，k1=1.2、b=.75；沿用 `phase4/scripts/common.py` 的 query-token handling。每個 query 的 BM25 full ranking 在七種 schedules 間相同。
3. 對每個 schedule，以全 catalog 的 1-based ranks 計算 equal-weight RRF：`1/(60+rank_bm25) + 1/(60+rank_dense_s)`。保持既有 FP32 fusion semantics。不可只對兩邊 Top-K union 計算，不調權重或常數。
4. 使用固定 catalog index 解決 ties，保留 K unique product IDs。驗證 C0 hybrid 與既有 `phase4/results/{dataset}/{encoder}_hybrid_*` 一致。
5. 若完整 scores 缺失但 embeddings 存在，可分 query batches 做 exact dot product，無需重新編碼。先與 authoritative saved raw ranks 對齊。浮點差異需記錄；若影響相關商品 membership，建立明確標示的共同重建比較（raw/hybrid 都用同一重建分數），不要把新 hybrid 與不相容的舊 raw 當 matched comparison，也不要覆寫歷史 ranks。
6. 若 embeddings 也缺失，先完成其他分析，列出精確缺失檔案與可重建條件；不要默默改模型 revision/prefix/precision。只有現有 pinned 模型與原始資料可用時才重建缺少的快取；保持凍結設定並記錄新 execution provenance。無訓練、無付費 API。

在 K=20,100 輸出 per-query/per-pair ranks、七種 inclusion bits、mean Recall、source-order Recall、persistent inclusion/VI/persistent omission。對同一母體直接比較 raw-seven vs hybrid-seven 的 Delta mean Recall、Delta VI、Delta persistent inclusion；另做 hybrid C0 vs raw C0 的 paired membership changes。不可從 C0 單一排序推論七種 schedule 的 VI。

## P2：共同策略表與不確定性（必做）

主要共同表涵蓋兩資料集 × MiniLM/BGE：

- Raw seven schedules（另外保留 Raw C0 參考）
- 三種 pure canonical rules，均保留
- Set-Mean，固定 .5/.5 與既有 normalization
- Centroid M=2、multi-vector max M=2
- BM25
- P1 的 Hybrid seven schedules

BGE M=4,7 centroid/max 為固定 budget sweep；不替換 M=2 共同設定、不挑最大 held-out Recall。GTE canonical 三規則仍在獨立完整 canonical 表，不混淆覆蓋範圍。

來源優先使用 `phase4/results/{dataset}/{encoder}_{method}_pairs.parquet`、`*_per_query.csv`、`*_transitions.csv`，並透過 `canonical_sources.json` 找到 canonical ranks。原始 raw ranks 位於 `phase2/results/phase2_pair_ranks/{dataset}_{encoder}_native_{schedule}.parquet`。檢查 query embedding 與 catalog IDs 完全相容；尤其沿用 historical set-mean cache 時查核 `shared_query_cache_audit.csv`。不要只比 rounded summary。

每個 K/cell/method 輸出：

- source-order Recall、seven-schedule mean Recall（對 invariant controls 兩者相同）；
- persistent inclusion、VI、persistent omission；
- 相對 Raw C0 的 retained/newly included/newly omitted/absent-in-both 整數 counts 與各 query fractions；
- paired Delta Recall vs Raw C0；另列 vs raw-seven mean 的 Recall contrast；
- exact within-query cancellation：integer gains=losses>0；分母同時列 all eligible 與 any-change queries；
- queries、highest-label pairs、catalog size、aggregation、source IDs、view budget、distinct views、index vectors/bytes。

**Raw-seven mean 沒有一個共同候選集合。** 不要虛構「相對 mean list」的 gains/losses；membership 對 Raw C0 或各明確 schedule，平均 Recall 的 contrast 則另算。

Canonical、Set-Mean、固定 canonical-seeded views 與 BM25 的 incoming-order invariance 是結構性質。可用 deterministic fingerprints/inclusion checks 驗證，不需七次 GPU 重新編碼。若 structural VI 非零，先查 implementation 或數值重算；不要把 bug 當研究發現。所有 three states 的 denominator/aggregation 必須一致。

統計：10,000 paired query-cluster bootstrap draws，seed 2026091701，95% percentile CI。同一 cell/matched contrast 用相同 sorted query IDs 與 resample indices；每個被抽 query 的所有 products 和 schedules 一起保留。報告 effect size/CI、不以 uncorrected CI 挑 winner。CI 含零不等於 equivalence，也不宣稱沒有影響。若保留歷史 seed 20260907 的 interval，必須另標；共同主表一律重算。

成本沿用已有 measurements，明確區分 offline encoding、query encoding、exact scoring/sorting、dense/sparse array bytes。Centroid 多 views 增加 offline work但仍一個 index vector；max 存 M vectors。不要把 cache hit time 當 cold encoding cost，不把 exact-local latency 當 ANN 部署效能。若缺時間資料，明列未量測，先報可精確計算的向量數/bytes。

必要一致性檢查：三狀態和=1；retained+gain+loss+absent=|Hq|；gain-loss=Delta Recall；source C0 matches；catalog 完整且 product IDs unique；all highest-label pairs 包括 absent-in-both；K unique products。這些是研究結果可信度檢查，不需增加與本任務無關的測試套件。

## P3：cutoff 與退出幅度（已有 ranks 的便宜診斷）

使用 raw ranks，兩資料集 × 三模型，catalog-wide 與 target-only 分開，K=20,50,100,200,500,1000。完整母體與 fully-fitting targets 分開；fit mask 使用各 encoder 全七種序列的 token counts，包括 special tokens。Competitor catalog 永遠不縮小。

每個 K 報告 query-macro persistent inclusion、VI、persistent omission 與 paired query intervals。同一比較中使用相同 target support；不能改分母製造 curve。

對 target-only，定義 `best=min(rank_s)`、`worst=max(rank_s)`；crossing_K 為 `best<=K<worst`。在 K=20,100 報告：

1. all eligible pairs 中 crossing 且 worst>2K、worst>5K 的份額（query-macro 為主）；
2. crossing pairs 中超過 2K、5K 的條件比例，清楚列分母及 weighting；
3. crossing pairs 的 worst rank 分布/ECDF，與 rank range 分布；
4. 另列 source-order included、在至少一個 alternative order omitted 的定向分析，避免把任意 best-case inclusion 都稱為 source-order loss。

不要預期 VI 隨 K 單調下降：它可能先升再降。Persistent inclusion 隨 K 不減、persistent omission 隨 K 不增，可作一致性檢查。大幅 rank movement 不直接證明 attention 或 embedding 機制。

## 圖表交付

沿用 `revision_evidence_20260917/figure_revision/` 的 sources 與字型規格；PDF/SVG 使用 vector，圖中文字在論文尺寸至少 8pt。科學圖使用現有繪圖程式/Matplotlib，不用生成式圖像工具。保留每張圖的 machine-readable CSV、caption、ACM Description 和繪圖 command。

1. **Fig. 3 / Section 6.1**：沿用 `rq1_vi_top20_main.pdf`，保持 target-only、full/fully-fitting、query/pair counts、CI。不要把它標成 catalog-wide。保留資料集不同橫軸範圍的說明。
2. **Fig. 4 / Section 6.2**：保留所有六個 C0 contrasts 與全部六個 dataset/encoder panels；圖例改成 `Newly included` / `Newly omitted`。左側表示 omitted share、右側 included share；diamonds/CI 為 net Recall。任何版面修改都不能改數字。輸出新檔，不覆寫 historical figure。
3. **Canonical 主表 / Section 6.3**：2 datasets × 3 encoders，Raw C0 Recall@20 + 三種 canonical 的 paired Delta/95% CI；各規則 structural VI=0 在表註交代。若完整 CI 造成主文超頁，可採 compact Delta 主表、完整 paired CI 附表，兩者均交付。K=100、raw-seven reference、gains/losses 在完整附表保留。不從 held-out 選最佳 rule。
4. **Joint strategy 主表 / Section 6.3**：以 2 datasets × MiniLM/BGE 的四個欄組，方法為 rows，各格同時呈現 `mean Recall@100 / VI@100`。Raw/hybrid 的 mean 使用七種 schedules；invariant controls 每 schedule 相同。所有三種 canonical 均報告，或主表中 canonical 指向完整規則表並明確只取預先指定 ascending 作 compact reference。不可暗中選最好者。Always/Never、K=20、paired CI、source-order contrasts、gains/losses 和成本另給可查附表。替換作者工作稿的 source-order point-estimate table，不把 mean 與 C0 混在同欄。
5. **Cancellation 附表**：沿用 exact within-query cancellation 的全部條件，保留 all-query/changed-query 分母。主文已敘述範圍，不再新增一張重複的圖片。
6. **新 cutoff/severity 附圖**：兩個明確 panel groups；第一組 K 對 persistent inclusion/VI（資料集與模型可用分面），第二組 K=20/100 crossing 的 worst-rank ECDF 或 >2K/>5K 份額。可拆成兩張附圖以保持可讀；主文不預設新增大型 Fig. 5。

只在完整論文位置編譯後決定 float 位置，不把字縮得過小來塞版面。最新前四章已佔約 4.6 頁；主文上限 8 頁，含 references/appendix 上限 12。核心 RQ1/RQ2 與 RQ3 不能全部移到附錄。

## 結果文字與 claim 判斷

以附件 `chapter5_experimental_setup.tex`、`chapter6_results.tex` 作為工作稿，逐條對照新產物後才更新新副本。若尚未有這兩檔，先完成分析並另外輸出可插入 Section 5/6 的段落，不覆蓋 repo 舊稿。

產出 `CLAIM_VERDICTS.md`，逐項列 Supported / Descriptive only / Not supported / Incomplete，附精確來源與範圍：

- fully-fitting target-only order sensitivity 是否持續存在；
- 小 net Recall 是否伴隨較大的 membership changes；
- 同 query Recall exactly unchanged 的 substitutions；
- canonical 是否一律降低效果（不要預設是）；
- 是否有 matched raw-seven/hybrid-seven 條件同時 mean Recall 增加、VI 增加；point estimates 與 statistical evidence 分開；
- full-record aggregation 是否普遍改善（不要以部分 setting 外推）；
- crossing 是否主要限於 cutoff 附近，以及增加 K 的实际影響。

若沒有支持 `gains in average retrieval performance coexist with heightened inclusion instability`，明確標 Not supported，建議最終 contribution 改為中性的 joint evaluation。不要為證明這句改資料、挑 seeds、挑子群或引入另一個 rewriting track。前四章的建議修正另列給作者，不自動修改。

## 輸出、重現與完成條件

在 `revision_completion_20260921/` 交付：

- `SOURCE_AUDIT.md`、固定 `PROTOCOL.json`、source/output hashes；
- `data/` 的完整 per-query、per-pair、summary、paired intervals、denominator 表；
- `figures/` 的 PDF/SVG 和對應 plot CSV；
- `tables/` 的 LaTeX/CSV；
- `MANUSCRIPT_INSERTIONS.tex`（有證據才寫結果；維持 inline prose，不用 paragraph headings）；
- `CLAIM_VERDICTS.md`、`COMPLETION_STATUS.md`、`REPRODUCE.md`；
- 主文圖表縮到實際 ACM 尺寸後的視覺 QA、整稿 page count（若有最新版完整 LaTeX）。

逐項列 P0--P3 完成/未完成。缺失輸入不能用 dummy 值替代；沒有量測不能填 0。Cache 和舊結果保持完整，所有新執行與重用來源可追溯。

P4 僅作後續選項，本輪先不執行：若 P0--P3 完成且需要更多證據，再規劃固定 query subset、nested 7/16/32 schedules，或在相同 query support 上加入正確定義的 graded metric。不得把 condensed NDCG 叫作官方 nDCG，也不得用新增排列數同時改變 fitting 母體。
