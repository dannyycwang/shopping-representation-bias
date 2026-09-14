"""Analyze frozen complete PI-FT artifacts; never select a training checkpoint."""
import run as r
from run import *
METHODS=['Labeled-ZeroShot','Standard-FT','Adapted-PI-FT']
def lengths():
    dest=HERE/'input_lengths.csv'
    if dest.exists():return pd.read_csv(dest,dtype={'product_id':str})
    from transformers import AutoTokenizer
    p,*_=s.data();tok=AutoTokenizer.from_pretrained(SPEC['name'],revision=SPEC['revision'],local_files_only=True)
    frame=pd.DataFrame({'product_id':[str(x['product_id']) for x in p]})
    for stem,cond in zip(STEMS,s.CFG['primary_variant_family']):
        vals=[]
        for start in range(0,len(p),256):
            texts=[labeled_build(x,cond,s.CFG['attribute_permutation_seeds']) for x in p[start:start+256]]
            vals.extend(map(len,tok(texts,truncation=False,verbose=False)['input_ids']))
        frame[stem]=vals;print('lengths',stem,flush=True)
    frame['fully_fits']=frame[STEMS].max(axis=1)<=512;frame.to_csv(dest,index=False);return frame
def analyze(require_complete=True):
    c=config();available=[x for x in METHODS if (OUT/x/'catalog_queries.csv').exists()]
    if require_complete:assert available==METHODS,'Complete matched comparison required'
    fitting=lengths();fitids=set(fitting.loc[fitting.fully_fits,'product_id']);summary=[];unc=[];fits=[];ndcg=[];deltas=[]
    for mode in ['catalog','target']:
        frames={};fitframes={}
        for name in available:
            pq=pd.read_csv(OUT/name/f'{mode}_queries.csv').set_index('query_id');pq=pq.loc[pq.index.isin(c['test'])];frames[name]=pq
            pairs=pd.read_parquet(OUT/name/f'{mode}_pairs.parquet');pairs=pairs[pairs.query_id.isin(c['test'])]
            summary.append({'method':name,'mode':mode,'eligible_queries':len(pq),'highest_pairs':len(pairs),**pq.mean().to_dict()})
            fit=pairs[pairs.product_id.astype(str).isin(fitids)];_,fpq=s.metric_frame(fit);fitframes[name]=fpq;fits.append({'method':name,'mode':mode,'pairs':len(fit),'queries':len(fpq),**fpq.mean().to_dict()})
        for a,b in [('Adapted-PI-FT','Standard-FT'),('Standard-FT','Labeled-ZeroShot'),('Adapted-PI-FT','Labeled-ZeroShot')]:
            if not {a,b}.issubset(frames):continue
            assert set(frames[a].index)==set(frames[b].index)
            for key in ['Recall@20','Recall@100','VI@20','VI@100','Robust@100','Never@100']:
                mu,lo,hi=s.bootstrap(frames[a][key]-frames[b].loc[frames[a].index,key]);unc.append({'comparison':a+' minus '+b,'mode':mode,'metric':key,'delta_pp':100*mu,'ci_low_pp':100*lo,'ci_high_pp':100*hi})
                assert set(fitframes[a].index)==set(fitframes[b].index)
                mu,lo,hi=s.bootstrap(fitframes[a][key]-fitframes[b].loc[fitframes[a].index,key]);unc.append({'comparison':a+' minus '+b,'mode':mode+'_fully_fitting','metric':key,'delta_pp':100*mu,'ci_low_pp':100*lo,'ci_high_pp':100*hi})
    for name in available:
        n=pd.read_csv(OUT/name/'ndcg_per_query.csv');n=n[n.query_id.isin(c['test'])]
        for metric in ['nDCG@10_unjudged_zero','cNDCG@10']:
            w=n.pivot(index='query_id',columns='variant',values=metric).dropna(subset=STEMS)
            ndcg.append({'method':name,'metric':metric,'eligible_queries':len(w),'C0':w.C0.mean(),'seven_order_mean':w[STEMS].mean(axis=1).mean(),'mean_worst_observed':w[STEMS].min(axis=1).mean(),'mean_spread':(w[STEMS].max(axis=1)-w[STEMS].min(axis=1)).mean()})
            for stem in STEMS[1:]:
                mu,lo,hi=s.bootstrap(w[stem]-w.C0);deltas.append({'method':name,'metric':metric,'variant':stem,'delta_vs_C0_pp':100*mu,'ci_low_pp':100*lo,'ci_high_pp':100*hi})
    for file,rows in [('summary.csv',summary),('paired_ci.csv',unc),('fully_fitting.csv',fits),('ndcg_summary.csv',ndcg),('ndcg_order_deltas.csv',deltas)]:pd.DataFrame(rows).to_csv(OUT/file,index=False)
    p,q,j,pi,qi=s.data();audit=[]
    for name in ['Standard-FT','Adapted-PI-FT']:
        path=HERE/'checkpoints'/name/'augmentation.jsonl'
        if not path.exists():continue
        f=pd.read_json(path,lines=True,dtype={'product_id':str});dropped=0;eligible=0;title_fail=0;unexpected_protect=0
        for x in f.itertuples():
            seg=segments(p[pi[str(x.product_id)]]);titles={i for i,v in enumerate(seg) if v[0]=='title'};kept=set(x.retained_segment_indices)
            assert len(kept)==x.rendered_segments and len(seg)==x.source_segments and kept.issubset(set(range(len(seg))))
            title_fail+=int(not titles.issubset(kept));eligible+=len(seg)-len(titles);dropped+=len(seg)-len(kept)
            if name=='Adapted-PI-FT':unexpected_protect+=int(bool(set(x.protected)-{'title'}))
        assert title_fail==unexpected_protect==0
        if name=='Standard-FT':assert dropped==0
        audit.append({'method':name,'document_presentations':len(f),'eligible_field_presentations':eligible,'dropped_fields':dropped,'observed_dropout':dropped/eligible,'title_protection_failures':title_fail,'unexpected_PI_protection_sets':unexpected_protect})
    pd.DataFrame(audit).to_csv(OUT/'dropout_audit.csv',index=False)
    if require_complete:write_report()
def write_report():
    df=pd.read_csv(OUT/'summary.csv');ci=pd.read_csv(OUT/'paired_ci.csv');fits=pd.read_csv(OUT/'fully_fitting.csv');nd=pd.read_csv(OUT/'ndcg_summary.csv');la=pd.read_csv(HERE/'input_lengths.csv');drop=pd.read_csv(OUT/'dropout_audit.csv')
    def table(mode):
        x=df[df['mode']==mode][['method','Recall@20','Recall@100','VI@20','VI@100','Robust@100','Never@100']].copy()
        for col in x.columns[1:]:x[col]*=100
        if mode=='target':x=x.rename(columns={'Recall@20':'Inclusion@20','Recall@100':'Inclusion@100'})
        return x.to_markdown(index=False,floatfmt='.3f')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,4.5))
    for _,x in df[df['mode']=='catalog'].iterrows():
        ax.scatter(x['VI@20']*100,x['Recall@100']*100);ax.annotate(x['method'],(x['VI@20']*100,x['Recall@100']*100),xytext=(7,8),textcoords='offset points',fontsize=9)
    ax.margins(x=.4,y=.35);ax.set(xlabel='VI@20 (%) - lower is better',ylabel='Seven-order mean Recall@100 (%)',title='Labeled WANDS + BGE: matched training comparison');ax.grid(alpha=.2);fig.tight_layout();fig.savefig(OUT/'pift_comparison.png',dpi=180);fig.savefig(OUT/'pift_comparison.pdf');plt.close(fig)
    costs=[json.loads(x) for x in (HERE/'cost.jsonl').read_text().splitlines()]
    lines=['# PI-FT WANDS 適配實驗報告','',
    '**這是 WANDS/BGE 的 source-aligned adaptation，不是 DevDataBench 原樣重現。只保護 title，不推測 facet。**',
    '', '## 設定與來源','',
    '作者原文與 pinned code、差異、資料與超參數見 AUDIT_AND_PROTOCOL.md；使用者批准的 title-only 修訂見 TITLE_ONLY_AMENDMENT.md。407 組訓練 pairs、每組三個明確 Irrelevant negatives，五 epochs、seed42、cached contrastive batch128。因近鄰 guard 等條件排除 139 個舊 training positives，清單完整保留。沒有依 test 挑選 checkpoint。',
    '共同 labeled serializer 保留 native attribute keys，添加 title/class/category/description labels；七個 test schedules 只換 attributes 順序，scalar sections 固定；訓練的全欄位 permutation 範圍更廣。test 不做 dropout。',
    '', '## Catalog-wide（百分比）','',table('catalog'),
    '', '## Target-only（百分比）','',table('target'),
    '每個 target 獨立替換，competitors 固定為各模型 C0。Inclusion 不是單一 index recall。Robust=Always；VI=Sometimes；Never=所有七順序都漏失。皆先 pair 內計算，再在 308 個 Exact-eligible 歷史 held-out queries 做 macro。',
    '', '## Paired uncertainty（百分點）','',ci.to_markdown(index=False,floatfmt='.3f'),
    '10,000 次 paired query bootstrap、seed20260963，描述性未校正 CI，不逐格挑顯著性；單 training seed，CI 不包含 training-seed uncertainty。CI 包含零不等於等效。',
    '', '## Ranking quality 與 cutoff crossing 分開解讀','',nd.to_markdown(index=False,floatfmt='.5f'),
    'nDCG@10_unjudged_zero 使用 WANDS Exact/Partial/Irrelevant gains 3/1/0，unjudged 當零只是計算慣例，不是已知 irrelevant。cNDCG@10 把未判斷商品移除。兩者在具有 positive IDCG 的查詢計算，分母可能不同於最高相關商品 Recall。各順序相對 C0 的 paired delta 見 ndcg_order_deltas.csv；這不是原文單正例 benchmark 的可直接比較分數。平均 quality 改變不能替代 VI。',
    '', '## 截斷與共同 fully-fitting targets','',fits[['method','mode','pairs','queries','Recall@100','VI@20','Robust@100']].to_markdown(index=False,floatfmt='.5f'),
    f'七順序都 ≤512 tokens 的商品共 {int(la.fully_fits.sum())}/{len(la)}。所有模型使用同一 support，只限制 targets、不縮減 competitors。每個商品／順序的長度見 input_lengths.csv。沒有作者的 schema-specific character budgeting；不能把截斷差異與 absolute-position mechanism 混為一談。',
    '', '## Dropout 與實際輸入檢查','',drop.to_markdown(index=False,floatfmt='.6f'),
    '檢查 retained segment indices，包括重複 atoms，title 保護違規必須為零。PI-FT positive protect 必須只含 title；negative 由 renderer 的全域 title 保護。欄位完整保留於 test，但 training dropout 可能移除相關證據，不宣稱 relevance-preserving augmentation。',
    '', '## 限制與論文解讀','',
    '全體 catalog 結果不支持本次 adapted PI-FT 同時改善品質與排列穩健性。相對 matched Standard-FT，Recall@100 差 −0.122 百分點（CI 包含零），VI@20 增加 4.899 百分點，Robust@100 降低 2.683 百分點。Standard-FT 的 VI 較低，但相對 labeled zero-shot 的 Recall 點估計也較低；不能只依 VI 宣稱成功。',
    '共同 fully-fitting 子集必須保留：PI-FT 的 Recall@100 為 62.885%，Standard 為 60.861%，labeled zero-shot 為 57.506%。PI 對 Standard 的增幅約 2.024 百分點，描述性 CI 下界僅略高於零；然而 PI 的 VI@20 仍較高（10.626% 對 6.278%）。這是 6,473 pairs、232 queries 的條件性結果，不是全體成功，也不是截斷的因果效果分解。',
    '沒有 facet 標註、407 pairs、單 seed、歷史 test、不同 miner、known-label negative mask、未縮短 schema 與不同 test intervention 都限制外推。結果不能推翻作者在原 benchmark 的結論。Standard-FT 是相同 supervised contrastive training 關閉 permutation/dropout 的 matched control，並非新的演算法。',
    'Set-attention 是獨立必要實驗，使用原先 raw-family／546 triples 與 Set-Mean 對照；不可將兩種 representation track 不加區分地排名或相加。',
    '', '## 成本、修復、重現','',pd.DataFrame(costs).to_markdown(index=False),
    '模型載入的網路中斷已計帳，之後離線載入，詳見 EXECUTION_REPAIRS.md。8 小時 wall-time cap 未重設。執行命令：`phase2/.venv/Scripts/python.exe phase5_pift/run.py run`；分析：`... phase5_pift/analyze.py`。完成 checkpoints/ranks 重用，不重新訓練。',
    '程式、config、source hashes、augmentation traces、checkpoint hashes、完整 ranking 與 CI 均保存在本資料夾。論文未修改。','']
    (HERE/'PI_FT_REPORT.md').write_text('\n'.join(lines),encoding='utf8')
if __name__=='__main__':
    if '--lengths-only' in sys.argv:lengths()
    else:analyze()
