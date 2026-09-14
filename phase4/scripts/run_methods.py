from common import *
from common import _plain
import argparse, gc
from threadpoolctl import threadpool_limits
threadpool_limits(4)

def run(ds,model,screen=False):
    p,q,j=load(ds);qv,pv=original_cache(ds,model)
    if screen:
        idx=dev_mask(q);qv=qv[idx];q=q.loc[idx].reset_index(drop=True);j=j[j.query_id.isin(q.query_id)].copy()
    base=qv@pv.T;outds=ds+'_dev' if screen else ds
    folder=OUT/outds;folder.mkdir(exist_ok=True);timing=[]
    def save(name,scores,bytes_,vectors,seconds=0):
        name=f'{model}_{name}';path=folder/f'{name}_per_query.csv'
        if path.exists():return pd.read_csv(path)
        t=time.perf_counter();pq=evaluate(scores,p,q,j,ds,name,outds)
        timing.append({'dataset':ds,'model':model,'method':name,'index_vector_bytes':bytes_,'vectors_per_product':vectors,'encoding_seconds':seconds,'evaluation_seconds':time.perf_counter()-t})
        pd.DataFrame(timing).to_csv(folder/f'{model}_cost_build.csv',index=False)
        return pq
    original=save('Original',base,pv.nbytes,1)
    bmfile=folder/'BM25_scores.npy'
    if bmfile.exists():bm=np.load(bmfile);bbytes=json.loads((folder/'BM25_cost.json').read_text())['bytes']
    else:
        bm,bbytes=bm25([_plain(x) for x in p],q['query'].fillna('').tolist());np.save(bmfile,bm);(folder/'BM25_cost.json').write_text(json.dumps({'bytes':bbytes}));evaluate(bm,p,q,j,ds,'BM25',outds)
    rb=ranks(bm)[1];rd=ranks(base)[1];hybrid=(1/(60+rb.astype('f4'))+1/(60+rd.astype('f4')))
    save('hybrid',hybrid,pv.nbytes+bbytes,1)
    # Reuse historical canonical embeddings; its field framing differs from pure order.
    files=list((P2/'results/embeddings').glob(f'{ds}_{model}_native_M2C0_*.npy'))
    if not files: files=list((P2/'results/embeddings').glob(f'{ds}_{model}_M2C0_*.npy'))
    spec=next(s for s in CFG['models'] if s['key']==model);enc=None
    def get_encoder():
        nonlocal enc
        if enc is None:enc=DenseEncoder(spec,CFG,P4/'results/embeddings')
        return enc
    def em(text,key):
        t=time.perf_counter();v=encode(get_encoder(),text,f'{ds}_{model}_{key}');return v,time.perf_counter()-t
    if len(files)==1:can=np.load(files[0]);secs=0
    else:
        from src.mitigations import build_mitigation
        can,secs=em([build_mitigation(x,'M2_normalized_attributes','C0',CFG['attribute_permutation_seeds']) for x in p],'canonical')
    save('canonical',qv@can.T,can.nbytes,1,secs)
    # Audit-added comparator: pure sorting isolates priority rules from M2's framing changes.
    if not screen:
        raw,secs=em(texts(p,'canonical_raw'),'canonical_raw');save('canonical_raw',qv@raw.T,raw.nbytes,1,secs)
    # Reconstruct existing set-mean architecture (normalized attribute mean).
    from scipy.sparse import csr_matrix
    unique=sorted({a for x in p for a in x['attributes']});lookup={a:i for i,a in enumerate(unique)};rows=[];cols=[];weight=[]
    for i,x in enumerate(p):
        for a in x['attributes']:rows.append(i);cols.append(lookup[a]);weight.append(1/len(x['attributes']))
    inc=csr_matrix((np.asarray(weight,dtype='f4'),(rows,cols)),shape=(len(p),len(unique)))
    nt=['\n'.join(t for t in [x['title'],x.get('class',''),x.get('category',''),x.get('description','')] if t) for x in p]
    if model=='bge_base':
        old=ROOT/'phase3/results/phase3_invariant_method/embeddings';tf=[]
        for f in old.glob(f'{ds}_title_description_*.npy'):
            meta=json.loads(f.with_suffix('.json').read_text())
            if meta['source_sha256']==hashlib.sha256('\0'.join(nt).encode()).hexdigest():tf.append(f)
        af=list(old.glob(f'{ds}_unique_attributes_*.npy'))
    else:tf=[];af=[]
    t=time.perf_counter();tv=np.load(tf[0]) if len(tf)==1 else em(nt,'nonattributes')[0];av=np.load(af[0]) if len(af)==1 else em(unique,'atoms')[0];sv=norm(.5*tv+.5*norm(inc@av));save('set_mean',qv@sv.T,sv.nbytes,1,time.perf_counter()-t)
    if screen:methods=list(RULES);ms=[2,4]
    else:
        choice=json.loads((P4/'config/selection.json').read_text());methods=[choice['rule']];ms=sorted(set([int(choice['method'].split('_')[-1])] if choice['method'].startswith(('centroid_','max_')) else [2]))
    dev=dev_mask(q) if ds=='wands' else np.zeros(len(q),dtype=bool)
    selection=[]
    for method in methods:
        ts=texts(p,method)
        for x,t in zip(p,ts):assert Counter(tokens(t))==Counter(tokens(_plain(x)))
        v,secs=em(ts,method);pq=save(method,qv@v.T,v.nbytes,1,secs)
        if screen:selection.append({'method':method,'recall':pq.loc[dev,'Recall@100'].mean(),'cndcg':pq.loc[dev,'cNDCG@10'].mean(),'vectors':1,'m':1})
    views=[];elapsed=0
    for i in range(1,max(ms)+1):
        v,secs=em(texts(p,f'view{i}'),f'view{i}');views.append(v);elapsed+=secs
        if i in ms:
            centroid=norm(np.mean(views,axis=0));mx=np.maximum.reduce([qv@v.T for v in views])
            for method,score,nvec in [(f'centroid_{i}',qv@centroid.T,1),(f'max_{i}',mx,i)]:
                pq=save(method,score,pv.nbytes*nvec,nvec,elapsed)
                if screen:selection.append({'method':method,'recall':pq.loc[dev,'Recall@100'].mean(),'cndcg':pq.loc[dev,'cNDCG@10'].mean(),'vectors':nvec,'m':i})
    if screen:
        trials=pd.DataFrame(selection).sort_values(['recall','cndcg','vectors','m','method'],ascending=[False,False,True,True,True]);trials.to_csv(OUT/'development_selection.csv',index=False)
        rules=trials[trials.method.isin(RULES)];record={'method':trials.iloc[0].method,'rule':rules.iloc[0].method,'selection_metric':'WANDS development macro Recall@100 then cNDCG then simplicity','heldout_used_for_selection':False}
        dest=P4/'config/selection.json'
        if dest.exists():assert json.loads(dest.read_text())==record
        else:dest.write_text(json.dumps(record,indent=2))
        print('FROZEN CHOICE',record,flush=True)
    if enc:enc.close()
    print('DONE',ds,model,flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--dataset',default='wands');a.add_argument('--model',default='bge_base');a.add_argument('--screen',action='store_true');args=a.parse_args();run(args.dataset,args.model,args.screen)
