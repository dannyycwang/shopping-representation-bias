from common import *
from common import _plain
from transformers import AutoTokenizer
from threadpoolctl import threadpool_limits
threadpool_limits(4)

def gain_repair():
    _,q,j=load('esci');sources=list((P2/'results/phase2_pair_ranks').glob('esci_*_native_*.parquet'))
    sources+=list((P2/'results').glob('**/esci_*M*C*_pairs.parquet'))
    sources+=list((ROOT/'phase3/results/phase3_invariant_method').glob('esci_*_pairs.parquet'))
    # Discover other mitigation/Marqo rank artifacts by schema, retaining source paths.
    sources+=list((P2/'results/phase2_mitigation_pair_ranks').glob('esci*.parquet'))
    sources+=list((ROOT/'phase3/results').glob('**/esci*pair*.parquet'))
    records=[];qs=[]
    for path in sorted(set(sources)):
        frame=pd.read_parquet(path)
        if not {'query_id','product_id','label','rank'}.issubset(frame):continue
        if frame.duplicated(['query_id','product_id']).any():continue
        for qid,g in frame.groupby('query_id'):
            v=g.sort_values('rank',kind='stable').label
            for setting,gains in [('legacy',{'E':1,'C':.1,'S':.01,'I':0}),('official',{'E':1,'S':.1,'C':.01,'I':0})]:
                a=v.map(gains).to_numpy();n=min(10,len(a));disc=1/np.log2(np.arange(2,n+2));ideal=np.sort(a)[::-1][:n]@disc
                qs.append({'source':str(path.relative_to(ROOT)),'query_id':int(qid),'gain_setting':setting,'cNDCG@10':a[:n]@disc/ideal if ideal else np.nan})
    values=pd.DataFrame(qs);values.to_parquet(OUT/'esci_gain_repair_per_query.parquet',index=False)
    for name,g in values.groupby('source'):
        wide=g.pivot(index='query_id',columns='gain_setting',values='cNDCG@10');mu,lo,hi=boot(wide.official-wide.legacy)
        records.append({'source':name,'queries':len(wide),'legacy_cNDCG@10':wide.legacy.mean(),'official_cNDCG@10':wide.official.mean(),'delta':mu,'ci_low':lo,'ci_high':hi})
    pd.DataFrame(records).to_csv(OUT/'esci_gain_repair.csv',index=False)
    print('GAIN REPAIR',len(records),'rank artifacts',flush=True)

def diagnose(ds):
    p,q,j=load(ds);idmap={str(x['product_id']):i for i,x in enumerate(p)};highest='Exact' if ds=='wands' else 'E';irrelevant='Irrelevant' if ds=='wands' else 'I'
    tok=AutoTokenizer.from_pretrained('BAAI/bge-base-en-v1.5',revision=next(x['revision'] for x in CFG['models'] if x['key']=='bge_base'),local_files_only=True)
    maxlen=np.zeros(len(p),dtype=int)
    for stem in ['C0','C1','C2s1','C2s2','C2s3','C2s4','C2s5']:
        with gzip.open(P2/f'data/representations/{ds}/{stem}.jsonl.gz','rt',encoding='utf8') as f:rs=list(map(json.loads,f))
        for start in range(0,len(p),512):
            tt=[x['text'] for x in rs[start:start+512]];lens=[len(x) for x in tok(tt,truncation=False,verbose=False)['input_ids']];maxlen[start:start+len(tt)]=np.maximum(maxlen[start:start+len(tt)],lens)
    fs=[]
    for x,l in zip(p,maxlen):fs.append({'product_id':str(x['product_id']),'attribute_count':len(x['attributes']),'duplicate_atoms':len(x['attributes'])-len(set(x['attributes'])),'max_tokens_seven':int(l),'fully_fits':bool(l<=512)})
    features=pd.DataFrame(fs);features.to_csv(OUT/f'{ds}_product_features.csv',index=False)
    frames=[pd.read_parquet(P2/f'results/phase2_pair_ranks/{ds}_bge_base_native_{s}.parquet') for s in ['C0','C1','C2s1','C2s2','C2s3','C2s4','C2s5']]
    f=frames[0][['query_id','product_id','label','rank','score']].copy();rr=[]
    for g in frames:
        z=f[['query_id','product_id']].merge(g[['query_id','product_id','rank']],on=['query_id','product_id'],validate='one_to_one');rr.append(z['rank'].to_numpy())
    rr=np.stack(rr,axis=1);f['VI@20']=(rr.min(1)<=20)&(rr.max(1)>20)
    for k in KS:
        f[f'always@{k}']=(rr<=k).all(1);f[f'sometimes@{k}']=(rr<=k).any(1)&(rr>k).any(1);f[f'never@{k}']=(rr>k).all(1)
    f['product_id']=f.product_id.astype(str);f=f.merge(features,on='product_id',validate='many_to_one')
    qlookup=q.set_index('query_id')['query'].to_dict();qsets={qid:set(tokens(str(t))) for qid,t in qlookup.items()};aset={str(x['product_id']):set(tokens(' '.join(a.split(':',1)[-1] for a in x['attributes']))) for x in p}
    f['query_length']=[len(tokens(str(qlookup[x]))) for x in f.query_id];f['value_overlap']=[bool(qsets[qid]&aset[pid]) for qid,pid in zip(f.query_id,f.product_id)]
    f['rank_band']=pd.cut(f['rank'],[0,10,20,50,100,500,np.inf],labels=['1-10','11-20','21-50','51-100','101-500','501+']).astype(str)
    f['attribute_band']=pd.cut(f.attribute_count,[-1,0,3,10,30,np.inf],labels=['0','1-3','4-10','11-30','31+']).astype(str)
    qv,pv=original_cache(ds,'bge_base');score=qv@pv.T;threshold=np.sort(score,axis=1)[:,-20];ql={int(v):i for i,v in enumerate(q.query_id)}
    f['C0_margin20']=[s-threshold[ql[int(qid)]] for s,qid in zip(f.score,f.query_id)]
    f.to_parquet(OUT/f'{ds}_diagnostic_pairs.parquet',index=False)
    summary=[]
    for factor in ['rank_band','fully_fits','attribute_band','value_overlap']:
        for (level,label),g in f.groupby([factor,'label'],observed=True):
            mu,lo,hi=boot(g.groupby('query_id')['VI@20'].mean());summary.append({'dataset':ds,'factor':factor,'level':str(level),'label':label,'pairs':len(g),'queries':g.query_id.nunique(),'macro_vi20':mu,'ci_low':lo,'ci_high':hi})
    pd.DataFrame(summary).to_csv(OUT/f'{ds}_diagnostic_strata.csv',index=False)
    contrasts=[]
    for band,g in f.groupby('rank_band'):
        m=g[g.label.isin([highest,irrelevant])].groupby(['query_id','label'])['VI@20'].mean().unstack()
        if highest not in m or irrelevant not in m:continue
        m=m.dropna();mu,lo,hi=boot(m[highest]-m[irrelevant]);contrasts.append({'dataset':ds,'rank_band':band,'common_queries':len(m),'delta_macro_vi20':mu,'ci_low':lo,'ci_high':hi,'scope':'within-query same-rank-band contrast; residual margin confounding remains'})
    pd.DataFrame(contrasts).to_csv(OUT/f'{ds}_rank_controlled_contrasts.csv',index=False)
    cover=[];h=f[f.label.eq(highest)]
    for k in KS:
        for state in ['always','sometimes','never']:
            mu,lo,hi=boot(h.groupby('query_id')[f'{state}@{k}'].mean());cover.append({'dataset':ds,'K':k,'state':state,'pairs':int(h[f'{state}@{k}'].sum()),'macro':mu,'ci_low':lo,'ci_high':hi})
    pd.DataFrame(cover).to_csv(OUT/f'{ds}_visibility_states.csv',index=False)
    case=f[(f.query_id==162)&(f.product_id=='34536')]
    if len(case):
        product=p[idmap['34536']];(OUT/'illustrative_case.json').write_text(json.dumps({'query':qlookup[162],'query_id':162,'product':product,'source_rank_table':'phase3/results/target_only_permutations/wands_minilm_pairs.parquet','selection':'illustrative post-hoc extreme'},indent=2,ensure_ascii=False),encoding='utf8')
    print('DIAGNOSIS',ds,flush=True)

if __name__=='__main__':gain_repair();diagnose('wands');diagnose('esci')
