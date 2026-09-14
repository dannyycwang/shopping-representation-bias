"""Readable Chinese synthesis of the completed experiment packet."""
from common import *
assert (OUT/'queue_completed.json').exists()
assert (OUT/'pdf_validation.json').exists(), 'Finish PDF verification before declaring the integrated revision complete'
selected=json.loads((P4/'config/selection.json').read_text())['method']
def read(name):return pd.read_csv(OUT/name)
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |',*['| '+' | '.join(map(str,row))+' |' for row in rows]])
def interval(r):return f'{100*r.delta:+.2f} [{100*r.ci_low:+.2f}, {100*r.ci_high:+.2f}]'
transfer=[]
for ds,model in [('wands','bge_base'),('esci','bge_base'),('wands','minilm'),('esci','minilm'),('esci_new','bge_base')]:
    f=read(f'{ds}_{model}_primary_contrasts.csv');r=f[(f.method_a==selected)&(f.method_b=='Original')&(f.metric=='Recall@100')].iloc[0];h=f[(f.method_a==selected)&(f.method_b=='hybrid')&(f.metric=='Recall@100')].iloc[0]
    transfer.append([ds,model,interval(r),interval(h)])
baseline=[]
for ds in ['wands','esci']:
    f=read(f'{ds}_bge_base_retrieval_summary.csv')
    for m in ['Original','BM25','hybrid','canonical_raw','canonical','set_mean',selected]:
        r=f[(f.method==m)&(f.metric=='Recall@100')].iloc[0];n=f[(f.method==m)&(f.metric=='cNDCG@10')].iloc[0]
        baseline.append([ds,m,f'{r["mean"]:.4f}',f'{n["mean"]:.4f}',int(r.queries),int(n.queries)])
sat=[]
for r in read('saturation_summary.csv').itertuples():
    if r.metric=='Recall@100':sat.append([r.dataset,r.method,f'{r.mean:.4f}',f'{100*r.delta_vs_Original:+.2f} [{100*r.ci_low:+.2f}, {100*r.ci_high:+.2f}]'])
cost=[]
for r in read('presentation_cost.csv').itertuples():cost.append([r.dataset,r.method,f'{r.index_arrays_MiB:.1f}',f'{r.score_sort_ms_median:.1f}',f'{r.score_sort_ms_p95:.1f}'])
target=[]
for ds in ['wands','esci']:
    r=read(f'{ds}_bge_base_target_only_summary.csv').query('K==100').iloc[0]
    target.append([ds,f'{100*r.net_macro_recall:+.2f} [{100*r.ci_low:+.2f},{100*r.ci_high:+.2f}]',f'{100*r.crossing_micro:.2f}'])
new=json.loads((P4/'data/catalog_manifest.json').read_text());decision=json.loads((OUT/'reranking_extension_decision.json').read_text())
dev=read('development_selection.csv');devrows=[[r.method,f'{r.recall:.4f}',f'{r.cndcg:.4f}',int(r.vectors)] for r in dev.itertuples()]
hybrid=read('descriptive_baseline_contrasts.csv');hybrid=hybrid[(hybrid.method_a=='hybrid')&(hybrid.method_b=='Original')&(hybrid.metric=='Recall@100')]
hybridrows=[[r.dataset,r.encoder,interval(r)] for r in hybrid.itertuples()]
trans=read('transition_summary.csv');trans=trans[(trans.encoder=='bge_base')&(trans.K==100)&trans.method.isin(['hybrid',selected])]
transrows=[[r.dataset,r.method,int(r.rescued_pairs),int(r.newly_missed_pairs),f'{100*r.macro_net:+.2f}'] for r in trans.itertuples()]
rerankrows=[]
for ds in ['wands','esci']:
    f=read(f'{ds}_canonical_reranking_summary.csv')
    for method in ['Original','hybrid',selected]:
        cells=[]
        for k in [50,100,200,500]:
            if k not in f.candidate_K.unique():cells.append('未執行');continue
            r=f[(f.method==method)&(f.candidate_K==k)&(f.metric=='Recall@20')].iloc[0];n=f[(f.method==method)&(f.candidate_K==k)&(f.metric=='returned_condensed_NDCG@10')].iloc[0];cells.append(f'{r["mean"]:.3f} / {n["mean"]:.3f}')
        rerankrows.append([ds,method,*cells])
lines=['# 研究修訂報告：從順序敏感到相關商品召回','',
'## 先說結論','',
'本次結果支持保留 **representation robustness／可見性不穩定的實證主線**，不支持把論文改寫成一個已能普遍提升召回的新方法。開發集選中的 appearance-first 規則在 BGE 保留測試集沒有確認的提升，轉到 MiniLM／WANDS 更下降；hybrid 是必須正面承認的強基準。多表示與 set mean 的好壞取決於資料、模型與評估指標，不能以 VI=0 當成成功。','',
'**有效的部分：** 原有 catalog-wide 與 target-only 控制仍顯示，商品事實不變時排序能改變候選資格；新增 fully-fitting 子集證據排除了「所有現象都只是截斷」的說法。簡單 hybrid 在部分主要條件取得更好的召回，完整差異見下表。','',
'**未獲支持的部分：** 開發集最佳固定規則未穩定泛化；更多 views 不是普遍更好；set mean 的 cNDCG 差異不顯著不能推出召回沒有下降，更不能推出等價。沒有利用本次 held-out 或新增查詢結果重新挑選規則。','',
'**論文定位：** 聚焦「相同事實如何影響 relevant-product inclusion，以及修補方法的邊界和成本」。保留原題，避免使用暗示已成功普遍改善召回的標題。這些資料增加可信度與判讀深度，但不構成 WWW 錄用保證。','',
'## 1. 固定策略的泛化','',
'以下皆為 query-macro highest-label Recall@100 的差異，單位為**百分點**，括號為 paired query bootstrap 95% CI。負值代表較差，區間跨零不代表等價。', '',
table(['資料','encoder','相對 Original：Δ [95% CI]','相對 hybrid：Δ [95% CI]'],transfer),'',
f'新查詢組包含 {new["queries"]} 個先凍結的 ESCI query IDs、{new["products"]:,} 件共同目錄商品、{new["judgments"]:,} 筆 judgments。它與舊 ESCI 目錄不同，只能在各自固定目錄內比較策略，不能把兩組絕對差異直接稱為泛化效果。這些 query IDs 不在既有處理過的測試集，但整個來源資料曾被存取，因此不聲稱整個研究是 pristine confirmation。','',
'### Hybrid 的直接對照','',table(['資料','encoder','hybrid − Original：Δ [95% CI]'],hybridrows),'',
'## 2. 必要基準與主要數據','',table(['資料','方法','Recall@100','cNDCG@10','Recall eligible q','cNDCG eligible q'],baseline),'',
'`canonical_raw` 僅排序完整 raw atoms；歷史 `canonical` 是 M2 模板，還會增加欄位 framing、移動 description 的段落位置。因此不能把 M2 的效果全部歸因於排序。新比較共享同一 query embedding matrix；歷史 set-mean 原本使用另一個 query cache，微小數值差異已獨立稽核，沒有覆寫舊結果。','',
'## 3. 所有開發集配置與飽和檢查','',
'只使用凍結的 96 個 WANDS development queries 選擇；Recall 無最高標籤的 query 排除，cNDCG 則獨立保留正 IDCG query。表中最佳新增方法仍可低於 hybrid。','',
table(['方法','dev Recall@100','dev cNDCG@10','index vectors/product'],devrows),'',
'以下 m=2/4/7 是固定的後續飽和檢查，**不能取代已凍結的 winner**。沒有為後來表現較好的配置重新做 test-driven 選擇。','',
table(['資料','方法','Recall@100','相對 Original：Δ pp [95% CI]'],sat),'',
'## 4. 單件介入與群體介入不同','',
'保留 Figure 1 的 primary C0/C1/C2 target-only 個案：query 162「turquoise chair」、product 34536；同一 Exact-relevant 商品在固定 C0 競爭者下為 Rank 5 與 1,527，C0 為 486。全部原始 title／attributes／七個 rank 皆可查。3 與 1,303 是另一個 catalog-wide 条件，不能混用。','',
'新增 selected-rule target-only 結果（BGE，K=100）：','',table(['資料','macro recall net Δ pp [95% CI]','pair crossing %'],target),'',
'跨界並不表示淨收益。商家只能改單件完整文本；多向量、centroid、set mean、hybrid 屬平台索引／打分方法。對單件相關商品的診斷不代表商家部署時知道測試查詢相關標籤，也不代表所有競爭者一起採用後仍有相同收益。','',
'## 5. 漏失、機制邊界與 reranking','',
'在全部歷史查詢（不是僅 held-out queries）的 BGE Top-100 下，WANDS 有 8,384 個最高相關 pairs 永遠入選、3,856 個有時入選、13,230 個從未入選；ESCI 為 3,995、44、395。這些是 query–product pairs，不是 unique products。永遠不被找回者也有 VI=0。','',
'控制 original-rank band 並限制在同 query 的共同支持後，WANDS 的 Exact-minus-Irrelevant 差異會變號；ESCI 沒有一致的正梯度。因此撤回普遍 relevance-specific／fairness 解讀。WANDS 完全放得進 BGE 的 Exact 子集仍有 macro VI@20=8.69%，CI [7.15%,10.39%]；與超長子集的差異是觀察性比較，不能宣稱已證明 attention 或 position 機制。','',
'Top-100 找回／新漏失的原始 pair 數：','',table(['資料','方法','rescued','newly missed','macro net pp'],transrows),'',
f'新 reranking 固定同一份 M2 canonical text，統一 final Top-20。K=50/100/200 已完成；K=500 延伸決策為 **{"執行" if decision["triggered"] else "未觸發"}**，依據 development Recall@200→500 增益 {100*decision["development_original_gain_200_to_500"]:.2f} pp 與實測 throughput 所估的額外成本，而非 held-out 效果。K=1000 只做第一階段曲線，未做 reranking。每方法／K 的 final Recall@20、returned-list condensed NDCG、paired CI 與 latency 估計保留於 structured artifacts 和附錄。','',
'舊 reranker 同時接收不同版本的商品文字，屬另一個條件。它的 common-pool VI 只針對每次都活下來的候選，不能與全體最高相關 pairs 的 VI 直接比較。','',
'下表每格為 **final Recall@20 / returned-list condensed NDCG@10**；後者不能與 full-catalog cNDCG 混為同一指標。','',table(['資料','first stage','K=50','K=100','K=200','K=500'],rerankrows),'',
'## 6. 成本','',
'以下是本機四 CPU threads、warm-up 後 5 次×50 queries 的 exact score/sort 中位數／p95；索引數字只含 dense vectors 與 hybrid sparse postings，不含文本、詞彙表與 ANN overhead。','',
table(['資料','方法','index arrays MiB','score/sort p50 ms','p95 ms'],cost),'',
'`recall_cost_frontier.csv` 的 processing cost = query encoding median + exact retrieval median + K×measured amortized cross-encoder pair time，屬**批次 throughput 推算**。圖上的 y 仍是第一階段 recall，不是 reranking 後 recall；不是商業系統或線上 request latency。`recall_equivalent_candidate_budget.csv` 提供 Original 在離散 K grid 上達到相近 recall 的最小 K，不能稱精確成本匹配。','',
'多表示 offline encoding seconds 見 `encoding_costs.csv`，actual unique text/vector views 見 `unique_view_summary.csv`。舊 cache 的建置時間與本次負載不同，沒有把它包裝成公平的 cold-build benchmark。','',
'## 7. 稽核與來源修正','',
'- ESCI 論文 Section 3.1 的標準為 E/S/C/I=1/.1/.01/0；舊稿把 S/C 顛倒。官方 repository 的 training 與論文一致，但 qrels＋eval shell 自身不一致。本次由 79 個舊 rank artifacts 重算 effectiveness，保留新舊對照；ranks、最高標籤 recall、VI 不受 gain swap 影響。','- set mean 先 normalize attribute mean，再與 non-attribute vector 內插；方程已修正。','- cNDCG 不再因沒有最高標籤而丟 query；WANDS held-out 有 384 queries，但只有 383 個正 IDCG。','- 舊 candidate tie／浮點 audit 發現部分排序不同，但全部 14 個 dataset×serialization 條件的 highest-label Top-100 membership 未改變。','- 已有的 target insertion／metric reconciliation 與新增 fact-equivalence／normalization／unique-product tests 合計 9 項通過。','',
'## 8. 已完成／待執行／證據不足','',
'**已完成：** 稽核與增益修正；固定 protocol、splits、rules、seeds；A0–A3 小範圍 development 篩選；選定方法／規則跨資料集和另一 encoder；m=7；固定新 query IDs；全目錄與 selected-rule target-only；K sweep；canonical reranking 與 conditional extension；條件診斷；local costs；英文論文整合及最終 PDF 檢查。','',
'**待執行：** 本次優先工作沒有用缺失實驗填入結論。K=1000 reranking、ANN、品類／商品監督策略、query-aware matched-position 介入未納入已完成項目，並非維持本稿主張所必需。','',
'**證據不足：** 不支持跨 Web 領域相同效應大小、unseen-product 泛化、普遍欄位語意轉移、fairness/disparate impact、click/sales/purchase/user satisfaction、agent 最終推薦改善、效果等價或商業 latency。','',
'## 9. 檔案與閱讀路線','',
'1. `paper_www2027/main.pdf`：整合的英文論文、references 與 appendix。官方 WWW 2027 研究軌為 8 頁主文、最多 12 頁合計；舊 10 頁內文版本已封存。本次 PDF 實際為 8 頁主文、合計 11 頁；排版檢查見 `phase4/results/pdf_validation.json`。','2. `EXPERIMENT_PROTOCOL.md` 與 `phase4/config/`：這次事後延伸的凍結規則，不是整個研究的預註冊。','3. `paper_www2027/RESULT_PROVENANCE.md`：逐項數值、圖、表的來源。','4. `phase4/results/`：per-query metrics、rank-level parquet、Top-1000 unique indices、full scores、paired contrasts、costs。','5. `phase4/README.md`：重現指令；`REVIEWER_RESPONSE_MAP.md`：每項 reviewer concern 對應的證據與剩餘限制。','6. `phase4/archive/paper_before_retrieval_extension/`：先前原稿、圖與 provenance，沒有刪除不利舊結果。','',
'來源：[WWW 2027 官方 CFP](https://www2027.thewebconf.org/research-track-papers/)、[ESCI benchmark 原論文](https://arxiv.org/pdf/2206.06588)、[tabular representational stability](https://arxiv.org/abs/2604.24040)、[CHARM](https://arxiv.org/abs/2501.18707)。']
(ROOT/'REVISION_REPORT.md').write_text('\n'.join(lines),encoding='utf8')
print('Wrote Chinese synthesis; final PDF validation file must be produced before delivery.')
