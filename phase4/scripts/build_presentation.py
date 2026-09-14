"""Generate tables/figures and Chinese report strictly from completed artifacts."""
from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
PAPER=ROOT/'paper_www2027';GEN=PAPER/'generated';GEN.mkdir(exist_ok=True)
assert (OUT/'queue_completed.json').exists(), 'Do not publish a completion report before the queue completes.'
choice=json.loads((P4/'config/selection.json').read_text())
LABEL={'Original':'Original','hybrid':'BM25 + dense','canonical':'Canonical template','canonical_raw':'Raw sort','set_mean':'Set mean','BM25':'BM25','rule_type':'Type-first','rule_appearance':'Appearance-first','rule_use':'Use-first','rule_dimension':'Dimension-first'}
for m in [2,4,7]:LABEL[f'centroid_{m}']=f'Centroid ({m})';LABEL[f'max_{m}']=f'Max views ({m})'
def label(x):return LABEL.get(x,x)
def markdown(df):
    return '\n'.join(['| '+' | '.join(df.columns)+' |','| '+' | '.join(['---']*len(df.columns))+' |',*['| '+' | '.join(map(str,row))+' |' for row in df.itertuples(index=False,name=None)]])
def tex_table(name,caption,cols,header,rows):
    text='\\begin{table}[t]\n\\caption{'+caption+'}\n\\label{tab:'+name+'}\n\\centering\\small\n\\begin{tabular}{'+cols+'}\\toprule\n'+' & '.join(header)+'\\\\\\midrule\n'
    text+='\n'.join(' & '.join(map(str,row))+'\\\\' for row in rows)+'\n\\bottomrule\\end{tabular}\n\\end{table}\n';(GEN/f'{name}.tex').write_text(text,encoding='utf8')
def metric(ds,encoder,method,key):
    f=pd.read_csv(OUT/f'{ds}_{encoder}_retrieval_summary.csv');return f[(f.method==method)&(f.metric==key)].iloc[0]

methods=list(dict.fromkeys(['Original','BM25','hybrid','canonical_raw','canonical','set_mean',choice['rule'],choice['method']]))
rows=[]
for m in methods:rows.append([label(m),*[f'{metric(ds,"bge_base",m,"Recall@100")["mean"]:.3f}' for ds in ['wands','esci']],*[f'{metric(ds,"bge_base",m,"cNDCG@10")["mean"]:.3f}' for ds in ['wands','esci']]])
tex_table('recall_main','Fixed BGE strategies. R100 is query-macro highest-label recall; cN is cNDCG@10. W/E denote held-out WANDS and the historical ESCI union. ESCI uses benchmark-standard gains.','lrrrr',['Method','R100 W','R100 E','cN W','cN E'],rows)

rows=[];report=[]
for ds,model in [('wands','bge_base'),('esci','bge_base'),('wands','minilm'),('esci','minilm'),('esci_new','bge_base')]:
    contrasts=pd.read_csv(OUT/f'{ds}_{model}_primary_contrasts.csv');r=contrasts[(contrasts.method_a==choice['method'])&(contrasts.method_b=='Original')&(contrasts.metric=='Recall@100')].iloc[0]
    h=contrasts[(contrasts.method_a==choice['method'])&(contrasts.method_b=='hybrid')&(contrasts.metric=='Recall@100')].iloc[0]
    rows.append([{'wands':'WANDS','esci':'ESCI old','esci_new':'ESCI new'}[ds],{'bge_base':'BGE','minilm':'MiniLM'}[model],f'{100*r.delta:+.2f}',f'[{100*r.ci_low:+.2f},{100*r.ci_high:+.2f}]'])
    report.append({'dataset':ds,'encoder':model,'method':choice['method'],'delta_recall100_vs_Original':r.delta,'low':r.ci_low,'high':r.ci_high,'delta_vs_hybrid':h.delta,'hybrid_low':h.ci_low,'hybrid_high':h.ci_high})
tex_table('transfer','Unchanged BGE-selected strategy transferred across evaluation cells. Entries are Recall@100 differences from Original in percentage points, with paired query-bootstrap 95\% CIs. Historical evaluations were previously inspected; new ESCI queries use a separate common catalog.','llrl',['Data','Encoder','$\\Delta$','95\% CI'],rows)
pd.DataFrame(report).to_csv(OUT/'presentation_generalization.csv',index=False)

plt.rcParams.update({'font.size':8,'axes.titlesize':9,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7})
# Render at native column size so the historical cross-model evidence stays legible.
sensitivity=pd.concat([pd.read_csv(P2/'results/phase2_tables/table1_representation_sensitivity_native.csv'),pd.read_csv(ROOT/'phase3/results/phase3_tables/ecommerce_retriever_sensitivity.csv')],ignore_index=True)
model_keys=['minilm','bge_base','gte_modernbert','marqo_ecommerce_b'];model_names=['MiniLM','BGE','GTE-ModernBERT','Marqo-Ecommerce-B']
fig,axes=plt.subplots(2,1,figsize=(3.35,3.25))
for ax,ds in zip(axes,['wands','esci']):
    f=sensitivity[sensitivity.dataset==ds].set_index('retriever').loc[model_keys]
    v=100*f['VI@20_query_macro'].to_numpy();lo=100*f['VI@20_macro_ci_low'].to_numpy();hi=100*f['VI@20_macro_ci_high'].to_numpy()
    ax.errorbar(v,np.arange(4),xerr=[v-lo,hi-v],fmt='o',ms=3,capsize=2,color='#0072B2');ax.axvline(1,color='#999',ls='--',lw=.7);ax.set_yticks(range(4),model_names);ax.invert_yaxis();ax.set_xlim(0,float(max(hi))*1.1);ax.set_title('WANDS' if ds=='wands' else 'ESCI union catalog',fontsize=8);ax.grid(axis='x',alpha=.2)
axes[1].set_xlabel('Query-macro VI@20 (%)',fontsize=8);fig.subplots_adjust(left=.36,right=.98,bottom=.13,top=.93,hspace=.48);fig.savefig(PAPER/'figures/figure2_cross_model_vi.pdf');plt.close(fig)
fig,axes=plt.subplots(2,2,figsize=(7,4.4));cost_frontier=[]
for col,ds in enumerate(['wands','esci']):
    ax=axes[0,col];costax=axes[1,col]
    timing=pd.read_csv(OUT/f'{ds}_exact_timing.csv');qencode=pd.read_csv(OUT/f'{ds}_query_encoding_timing.csv').single_query_encode_ms.median();perpair=json.loads((OUT/f'{ds}_canonical_reranker_timing.json').read_text())['seconds_per_pair_amortized']*1000
    for m,c in [('Original','#555555'),('hybrid','#D55E00'),(choice['method'],'#0072B2')]:
        f=pd.read_csv(OUT/f'{ds}_bge_base_retrieval_summary.csv');v=[float(f[(f.method==m)&(f.metric==f'Recall@{k}')]['mean'].iloc[0]) for k in KS];ax.plot(KS,v,'o-',color=c,label=label(m),ms=3)
        retrieval=timing[timing.method==m].retrieval_ms.median();estimated=[qencode+retrieval+k*perpair for k in KS];costax.plot(estimated,v,'o-',color=c,label=label(m),ms=3)
        cost_frontier.extend(dict(dataset=ds,method=m,K=k,first_stage_recall=r,estimated_processing_ms=t,query_encoding_ms=qencode,retrieval_ms=retrieval,estimated_reranker_ms=k*perpair,scope='batch-throughput estimate; y is first-stage recall, not final reranked recall') for k,r,t in zip(KS,v,estimated))
    ax.set_xscale('log');ax.set_xticks(KS,[str(k) for k in KS],rotation=25);ax.set_xlabel('Unique candidate budget K');ax.set_title('WANDS held-out' if ds=='wands' else 'ESCI old union');ax.grid(alpha=.2);ax.set_ylim(0,1)
    costax.set_xscale('log');costax.set_xlabel('Estimated processing cost (ms)');costax.grid(alpha=.2);costax.set_ylim(0,1)
axes[0,0].set_ylabel('First-stage macro recall');axes[1,0].set_ylabel('First-stage macro recall');axes[0,1].legend(fontsize=7,frameon=False);fig.tight_layout();fig.savefig(PAPER/'figures/recall_budget.pdf');plt.close(fig)
pd.DataFrame(cost_frontier).to_csv(OUT/'recall_cost_frontier.csv',index=False)

fig,axes=plt.subplots(1,2,figsize=(7,2.8),sharey=True)
bands=['1-10','11-20','21-50','51-100','101-500','501+']
for ax,ds in zip(axes,['wands','esci']):
    f=pd.read_csv(OUT/f'{ds}_rank_controlled_contrasts.csv').set_index('rank_band').loc[bands]
    v=100*f.delta_macro_vi20.to_numpy();lo=100*f.ci_low.to_numpy();hi=100*f.ci_high.to_numpy();ax.errorbar(v,np.arange(6),xerr=[v-lo,hi-v],fmt='o',capsize=3,color='#0072B2');ax.axvline(0,color='#555',ls='--',lw=.7);ax.set_title(ds.upper());ax.set_xlabel('Highest minus irrelevant VI@20 (pp)');ax.grid(axis='x',alpha=.2)
    for i,n in enumerate(f.common_queries):ax.annotate(f'n={n}',(v[i],i),xytext=(3,5),textcoords='offset points',fontsize=6)
axes[0].set_yticks(range(6),bands);axes[0].invert_yaxis();axes[0].set_ylabel('Original rank stratum');fig.tight_layout();fig.savefig(PAPER/'figures/rank_controlled.pdf');plt.close(fig)

rows=[];cost=[]
for ds in ['wands','esci']:
    f=pd.read_csv(OUT/f'{ds}_exact_timing.csv');qt=pd.read_csv(OUT/f'{ds}_query_encoding_timing.csv').single_query_encode_ms.median()
    for m,g in f.groupby('method'):
        size=(g.vector_bytes.iloc[0]+g.lexical_sparse_bytes.iloc[0])/1024**2;median=g.retrieval_ms.median();p95=g.retrieval_ms.quantile(.95)
        cost.append({'dataset':ds,'method':m,'index_arrays_MiB':size,'score_sort_ms_median':median,'score_sort_ms_p95':p95,'query_encode_ms_median':qt})
        if m in ['Original','hybrid',choice['method']]:rows.append([ds.upper(),label(m),f'{size:.1f}',f'{median:.1f}',f'{p95:.1f}'])
tex_table('cost','Local exact retrieval on four CPU threads. Array storage is dense vectors plus sparse postings for hybrid, excluding text/vocabulary metadata. Time includes scoring, fusion where applicable, and stable sorting; query encoding is reported separately. These are not commercial latency estimates.','llrrr',['Data','Method','MiB','ms p50','ms p95'],rows)
pd.DataFrame(cost).to_csv(OUT/'presentation_cost.csv',index=False)
rows=[];saturation=pd.read_csv(OUT/'saturation_summary.csv');costframe=pd.DataFrame(cost)
for m in [2,4,7]:
    for family in ['centroid','max']:
        method=f'{family}_{m}';v=[saturation[(saturation.dataset==ds)&(saturation.method==method)&(saturation.metric=='Recall@100')]['mean'].iloc[0] for ds in ['wands','esci']];c=costframe[(costframe.dataset=='wands')&(costframe.method==method)].iloc[0]
        rows.append([label(method),*[f'{x:.3f}' for x in v],f'{c.index_arrays_MiB:.1f}',f'{c.score_sort_ms_median:.1f}'])
tex_table('multiview_cost','Full-record view-count checks: held-out WANDS and historical ESCI Recall@100 and WANDS index/scoring cost. Centroids store one vector after multiple offline encodings; max stores all views. These fixed checks do not reopen strategy selection.','lrrrr',['Method','R100 W','R100 E','MiB W','ms W'],rows)

rows=[]
for ds in ['wands','esci']:
    f=pd.read_csv(OUT/f'{ds}_canonical_reranking_summary.csv')
    for m in ['Original','hybrid',choice['method']]:
        rs=[f[(f.method==m)&(f.candidate_K==k)&(f.metric=='Recall@20')]['mean'].iloc[0] for k in [50,100,200]];rows.append([ds.upper(),label(m),*[f'{x:.3f}' for x in rs]])
tex_table('reranking','Highest-label Recall@20 after reranking a fixed canonical product text. Each column changes only the first-stage unique candidate budget. Same cross-encoder and final Top-20 across methods.','llrrr',['Data','First stage','K=50','100','200'],rows)

# A reproducible numeric packet for manuscript prose and independent review.
packet={'selected':choice,'generalization':report,'cost':cost,'new_catalog':json.loads((P4/'data/catalog_manifest.json').read_text())}
(OUT/'paper_numeric_packet.json').write_text(json.dumps(packet,indent=2))
lines=['# Revision report / 修訂結果','',f'固定 development 選出的新增策略：**{label(choice["method"])}**；全域欄位規則：**{label(choice["rule"])}**。','', '## 泛化與主要效果','',markdown(pd.DataFrame(report)),'','## 成本','',markdown(pd.DataFrame(cost)),'','## 稽核修正','', '- ESCI 按 benchmark 論文改採 E/S/C/I=1/.1/.01/0。保留官方 repository 腳本內部不一致與舊增益結果。','- 舊 reranking 候選排序與本次精確排序部分不同，但所有七個條件的最高相關商品 Top-100 membership 均未變；新實驗统一候選與 tie 規則。','- cNDCG 獨立保留無最高標籤但有正 gain 的 queries；recall 排除沒有最高標籤的 queries。','', '## 已完成','', '- 原始排名增益重算、兩資料集條件診斷、固定 development 篩選、跨編碼器／資料集轉移、m=7 飽和檢查、固定新查詢測試、canonical-input reranking、成本曲線。','', '## 未執行與證據不足','', '- 未進行品類／商品監督策略；不主張 unseen-product 泛化。','- 未做 query-aware matched-atom 移動；不主張特定屬性位置或 attention 是已證明的內部機制。','- 未做 ANN、線上使用者或完整購物 agent；不主張購買／推薦／福利改善。','- 舊測試結果曾被閱讀；新查詢僅代表凍結後新增的查詢評估，目錄擴大後不可與舊絕對數字直接歸因比較。','- 未設定 non-inferiority margin；不顯著不代表等價。','', '## 重現','', '所有執行入口與步驟見 `phase4/README.md`；數據引用見 `paper_www2027/RESULT_PROVENANCE.md`。']
(ROOT/'REVISION_REPORT.md').write_text('\n'.join(lines),encoding='utf8')
print('Generated presentation tables, figures and evidence packet')
