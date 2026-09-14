"""Bounded WANDS/BGE screening. No manuscript mutations."""
from pathlib import Path
import sys, json, hashlib, time, random, argparse, gc, math
from collections import Counter
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'phase4/scripts'))
from common import load, original_cache, ranks, norm, CFG, PROFILE, DenseEncoder
from common import _plain
from src.representations import build, _random_order
sys.path.insert(0, str(ROOT/'phase3/scripts'))
from analyze_target_only_permutations import rank_target
from run_reranker import bootstrap
from threadpoolctl import threadpool_limits
threadpool_limits(4)
STEMS = ['C0','C1','C2s1','C2s2','C2s3','C2s4','C2s5']
SPEC = next(s for s in CFG['models'] if s['key']=='bge_base')
OUT = HERE/'results'
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path, x):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(x, indent=2, ensure_ascii=False), encoding='utf8')
def hashkey(x): return hashlib.sha256(str(x).encode()).hexdigest()
def freeze():
    path=HERE/'config.json'
    if path.exists(): return
    p,q,j=load('wands'); old=json.loads((ROOT/'phase4/config/splits.json').read_text())
    ids=sorted(old['wands_dev'],key=lambda x:hashkey(f'mitigation42:{x}'))
    cfg={'frozen_utc':pd.Timestamp.now(tz='UTC').isoformat(),'train':ids[:72], 'validation':ids[72:], 'test':old['wands_heldout'],
         'seeds':[42,43,44], 'lambdas':[.05,.1,.5], 'epochs':1,'max_positives_per_query':16,'microbatch':2,'accumulation':8,
         'learning_rate':1e-5,'temperature':.05,'gpu_task_seconds_cap':14400,'readme_sha256':sha(HERE/'README.md'),
         'source_split_sha256':sha(ROOT/'phase4/config/splits.json'), 'model':SPEC}
    assert not (set(cfg['train'])&set(cfg['validation']) or set(ids)&set(cfg['test']))
    triples=[]
    for qid in cfg['train']:
        g=j[j.query_id.eq(qid)]
        pos=sorted(g[g.label.eq('Exact')].product_id.astype(str),key=lambda x:hashkey(f'positive42:{qid}:{x}'))[:16]
        neg=sorted(g[g.label.eq('Irrelevant')].product_id.astype(str),key=lambda x:hashkey(f'negative42:{qid}:{x}'))
        if not neg: continue
        for n,pid in enumerate(pos): triples.append({'query_id':int(qid),'positive':pid,'negative':neg[n%len(neg)]})
    dump(HERE/'triples.json',triples);cfg['triples_sha256']=sha(HERE/'triples.json');cfg['training_triples']=len(triples)
    dump(path,cfg);print('FROZEN',len(triples),'triples',flush=True)
def setup():
    c=json.loads((HERE/'config.json').read_text()); assert sha(HERE/'README.md')==c['readme_sha256']
    assert sha(HERE/'triples.json')==c['triples_sha256']
    amendment=HERE/'budget_amendment.json'
    if amendment.exists():
        extra=json.loads(amendment.read_text(encoding='utf8'))
        assert extra['original_config_sha256']==sha(HERE/'config.json')
        c['gpu_task_seconds_cap']=extra['total_task_seconds_cap']
    return c
def cost(name, seconds, **extra):
    f=HERE/'cost.jsonl'
    with f.open('a',encoding='utf8') as h:h.write(json.dumps({'name':name,'seconds':seconds,**extra})+'\n')
def spent():
    f=HERE/'cost.jsonl';return sum(json.loads(l)['seconds'] for l in f.read_text().splitlines()) if f.exists() else 0
def guard(start=0):
    if spent()+(time.monotonic()-start if start else 0)>setup()['gpu_task_seconds_cap']:raise RuntimeError('Frozen compute cap reached; keep incomplete outputs.')
def data():
    p,q,j=load('wands');pi={str(x['product_id']):i for i,x in enumerate(p)};qi={int(v):i for i,v in enumerate(q.query_id)}
    return p,q,j,pi,qi
def metric_frame(frame):
    rr=frame[STEMS].to_numpy(); d=frame[['query_id','product_id']].copy()
    for k in [20,100]:
        inc=rr<=k
        d[f'Recall@{k}']=inc.mean(1); d[f'VI@{k}']=inc.any(1)&~inc.all(1)
        d[f'Robust@{k}']=inc.all(1);d[f'Never@{k}']=~inc.any(1)
        for s in STEMS:d[f'{s}_Recall@{k}']=(frame[s]<=k).to_numpy()
    d['rank_range']=rr.max(1)-rr.min(1)
    pq=d.drop(columns='product_id').groupby('query_id').mean()
    for k in [20,100]:
        cols=[f'{s}_Recall@{k}' for s in STEMS]
        pq[f'worst_schedule@{k}']=pq[cols].min(axis=1);pq[f'schedule_spread@{k}']=pq[cols].max(axis=1)-pq[cols].min(axis=1)
    return d,pq
def save_metrics(name, f, target=False):
    folder=OUT/name;folder.mkdir(parents=True,exist_ok=True);stem='target' if target else 'catalog'
    f.to_parquet(folder/f'{stem}_pairs.parquet',index=False)
    d,pq=metric_frame(f);pq.to_csv(folder/f'{stem}_queries.csv')
    dump(folder/f'{stem}_micro.json',d.drop(columns=['query_id','product_id']).mean().to_dict())
def baselines():
    if (OUT/'Original/catalog_queries.csv').exists():return
    frames=[pd.read_parquet(ROOT/f'phase2/results/phase2_pair_ranks/wands_bge_base_native_{s}.parquet') for s in STEMS]
    f=frames[0].query("label == 'Exact'")[['query_id','product_id']].copy()
    for s,g in zip(STEMS,frames):f[s]=f.merge(g[['query_id','product_id','rank']],on=['query_id','product_id'],validate='one_to_one')['rank'].to_numpy()
    save_metrics('Original',f)
    t=pd.read_parquet(ROOT/'phase3/results/target_only_permutations/wands_bge_base_pairs.parquet');save_metrics('Original',t[['query_id','product_id']+STEMS],True)
    for name,stem in [('Canonical','canonical_raw'),('Set-Mean','set_mean')]:
        b=pd.read_parquet(ROOT/f'phase4/results/wands/bge_base_{stem}_pairs.parquet').query("label == 'Exact'")
        f=b[['query_id','product_id']].copy()
        for s in STEMS:f[s]=b['rank'].to_numpy()
        save_metrics(name,f);save_metrics(name,f,True)
def train(name, lam=0, seed=42):
    import torch
    from transformers import AutoModel,AutoTokenizer
    dest=HERE/'checkpoints'/name;ck=dest/'weights.pt'
    if ck.exists():return
    guard();start=time.monotonic();c=setup();torch.set_num_threads(4);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    p,q,j,pi,qi=data();examples=json.loads((HERE/'triples.json').read_text());random.Random(seed).shuffle(examples)
    tok=AutoTokenizer.from_pretrained(SPEC['name'],revision=SPEC['revision'],local_files_only=True)
    model=AutoModel.from_pretrained(SPEC['name'],revision=SPEC['revision'],local_files_only=True).cuda().train()
    model.gradient_checkpointing_enable();opt=torch.optim.AdamW(model.parameters(),lr=c['learning_rate'],weight_decay=.01)
    scaler=torch.amp.GradScaler('cuda');logs=[];opt.zero_grad(set_to_none=True)
    def enc(ts):
        inp=tok(ts,padding=True,truncation=True,max_length=512,return_tensors='pt').to('cuda')
        return torch.nn.functional.normalize(model(**inp).last_hidden_state[:,0].float(),dim=1)
    def rep(pid,view):
        x=p[pi[pid]];a=_random_order(x['attributes'],pid,seed*100+view)
        assert Counter(a)==Counter(x['attributes'])
        return _plain(x,a)
    batches=math.ceil(len(examples)/2)
    for bi,offset in enumerate(range(0,len(examples),2)):
        guard(start);g=examples[offset:offset+2];qt=[q.iloc[qi[x['query_id']]]['query'] for x in g]
        a=[rep(x['positive'],0) for x in g];n=[rep(x['negative'],0) for x in g]
        with torch.autocast('cuda',dtype=torch.float16):
            qv=enc(qt);av=enc(a);nv=enc(n)
            ret=torch.nn.functional.softplus(((qv*nv).sum(1)-(qv*av).sum(1))/.05).mean()
            cons=av.sum()*0
            if lam:cons=(1-(av*enc([rep(x['positive'],1) for x in g])).sum(1)).mean()
            loss=ret+lam*cons
        group_start=(bi//8)*8;div=min(8,batches-group_start)
        scaler.scale(loss/div).backward()
        if (bi+1)%8==0 or bi+1==batches:
            scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(model.parameters(),1)
            scaler.step(opt);scaler.update();opt.zero_grad(set_to_none=True)
        logs.append({'batch':bi,'retrieval_loss':ret.item(),'consistency':cons.item(),'loss':loss.item(),'seconds':time.monotonic()-start})
        if bi%20==0:print(name,'train',bi,batches,'loss',loss.item(),flush=True)
    dest.mkdir(parents=True,exist_ok=True);torch.save(model.cpu().state_dict(),ck)
    pd.DataFrame(logs).to_csv(dest/'training.csv',index=False)
    dump(dest/'metadata.json',{'seed':seed,'lambda':lam,'weights_sha256':sha(ck),'config_sha256':sha(HERE/'config.json'),'parameters':sum(x.numel() for x in model.parameters())})
    cost(name+'_train',time.monotonic()-start);del model,opt;gc.collect();torch.cuda.empty_cache()
def evaluate(name):
    if (OUT/name/'catalog_queries.csv').exists():return
    import torch
    guard();start=time.monotonic();p,q,j,pi,qi=data();cfg=json.loads(json.dumps(CFG));cfg['encoding']['batch_size_bge_base']=32
    enc=DenseEncoder(SPEC,cfg,HERE/'embeddings'/name)
    ck=HERE/'checkpoints'/name/'weights.pt';enc.model.load_state_dict(torch.load(ck,map_location='cpu',weights_only=True));enc.model.eval()
    # Revision in cache identity is augmented after loading the pinned base model.
    enc.spec=dict(SPEC,trained_weights_sha256=sha(ck))
    original_forward=enc._forward_native
    def bounded(*a,**kw):guard(start);return original_forward(*a,**kw)
    enc._forward_native=bounded
    def em(ts,key):return np.asarray(enc.encode(ts,key,PROFILE,hashlib.sha256('\0'.join(ts).encode()).hexdigest()),dtype='f4')
    qv=em(q['query'].fillna('').tolist(),'queries');hi=j.query("label == 'Exact'")[['query_id','product_id']].reset_index(drop=True)
    iq=np.array([qi[int(x)] for x in hi.query_id]);ip=np.array([pi[str(x)] for x in hi.product_id]);f=hi.copy();t=hi.copy();base=None
    for s,condition in zip(STEMS,CFG['primary_variant_family']):
        ts=[build(x,condition,CFG['attribute_permutation_seeds']) for x in p];pv=em(ts,s);scores=qv@pv.T
        if base is None:base=scores.copy()
        f[s]=ranks(scores)[1][iq,ip];tr=np.empty(len(hi),dtype='i4')
        for qid,g in hi.groupby('query_id',sort=False):tr[g.index]=rank_target(base[qi[int(qid)]],ip[g.index],scores[qi[int(qid)],ip[g.index]])
        t[s]=tr;print(name,s,'ranks done',flush=True)
    assert np.array_equal(f.C0,t.C0)
    save_metrics(name,f);save_metrics(name,t,True);cost(name+'_encode_evaluate',time.monotonic()-start)
    enc.close();gc.collect();torch.cuda.empty_cache()

def field_data():
    p,q,j,pi,qi=data();old=ROOT/'phase3/results/phase3_invariant_method/embeddings'
    nt=['\n'.join(t for t in [x['title'],x.get('class',''),x.get('category',''),x.get('description','')] if t) for x in p]
    sig=hashlib.sha256('\0'.join(nt).encode()).hexdigest()
    tf=[f for f in old.glob('wands_title_description_*.npy') if json.loads(f.with_suffix('.json').read_text())['source_sha256']==sig]
    unique=sorted({a for x in p for a in x['attributes']});sig=hashlib.sha256('\0'.join(unique).encode()).hexdigest()
    af=[f for f in old.glob('wands_unique_attributes_*.npy') if json.loads(f.with_suffix('.json').read_text())['source_sha256']==sig]
    assert len(tf)==len(af)==1
    lookup={a:i for i,a in enumerate(unique)};indices=[[lookup[a] for a in x['attributes']] for x in p]
    return np.load(tf[0]),np.load(af[0]),indices,unique
def set_class():
    import torch
    class SetAttention(torch.nn.Module):
        def __init__(self):
            super().__init__();self.att=torch.nn.Sequential(torch.nn.Linear(768,128),torch.nn.Tanh(),torch.nn.Linear(128,1))
            torch.nn.init.zeros_(self.att[-1].weight);torch.nn.init.zeros_(self.att[-1].bias)
        def forward(self,a,mask,non):
            logits=self.att(a).squeeze(-1).masked_fill(~mask,-1e9)
            w=logits.softmax(1)*mask;pooled=(a*w[:,:,None]).sum(1)
            return torch.nn.functional.normalize(.5*non+.5*torch.nn.functional.normalize(pooled,dim=1),dim=1)
    return SetAttention
def set_run(seed=42):
    import torch
    name=f'Set-Attention_s{seed}';dest=HERE/'checkpoints'/name;ck=dest/'weights.pt'
    if (OUT/name/'catalog_queries.csv').exists():return
    guard();start=time.monotonic();torch.set_num_threads(4);torch.manual_seed(seed)
    p,q,j,pi,qi=data();tv,av,idx,unique=field_data();qv,_=original_cache('wands','bge_base')
    at=torch.tensor(av,device='cuda');nt=torch.tensor(tv,device='cuda');qt=torch.tensor(qv,device='cuda');model=set_class()().cuda()
    def batch(ids,shuffle=None):
        lists=[list(idx[i]) for i in ids]
        if shuffle is not None:
            for v in lists:shuffle.shuffle(v)
        n=max(1,max(map(len,lists)));ii=np.zeros((len(ids),n),dtype='i8');mask=np.zeros_like(ii,dtype=bool)
        for r,v in enumerate(lists):ii[r,:len(v)]=v;mask[r,:len(v)]=True
        return at[torch.tensor(ii,device='cuda')],torch.tensor(mask,device='cuda'),nt[ids]
    if not ck.exists():
        examples=json.loads((HERE/'triples.json').read_text());random.Random(seed).shuffle(examples);opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=.01);logs=[]
        for offset in range(0,len(examples),16):
            guard(start);g=examples[offset:offset+16];a=model(*batch([pi[x['positive']] for x in g]));n=model(*batch([pi[x['negative']] for x in g]));qq=qt[[qi[x['query_id']] for x in g]]
            loss=torch.nn.functional.softplus(((qq*n).sum(1)-(qq*a).sum(1))/.05).mean();opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();logs.append({'step':offset//16,'loss':loss.item()})
        dest.mkdir(parents=True,exist_ok=True);torch.save(model.state_dict(),ck);pd.DataFrame(logs).to_csv(dest/'training.csv',index=False)
        dump(dest/'metadata.json',{'seed':seed,'parameters':sum(x.numel() for x in model.parameters()),'weights_sha256':sha(ck)})
    else:model.load_state_dict(torch.load(ck,weights_only=True))
    model.eval();ids=sorted(range(len(p)),key=lambda i:hashkey(f'invariance42:{p[i]["product_id"]}'))[:100];diff=[];cos=[]
    with torch.inference_mode():
        v=model(*batch(ids))
        for s in range(7):
            w=model(*batch(ids,random.Random(42+s)));diff.extend((v-w).norm(dim=1).tolist());cos.extend((1-torch.nn.functional.cosine_similarity(v,w)).abs().tolist())
        inv={'products':100,'shuffles':7,'l2_max':max(diff),'l2_mean':float(np.mean(diff)),'cosine_difference_max':max(cos),'cosine_difference_mean':float(np.mean(cos)),'product_ids':[str(p[i]['product_id']) for i in ids]}
        dump(OUT/name/'invariance.json',inv);assert max(diff)<1e-5
        pv=np.empty_like(tv)
        for off in range(0,len(p),64):guard(start);pv[off:off+64]=model(*batch(list(range(off,min(off+64,len(p)))))).cpu().numpy()
    scores=qv@pv.T;rr=ranks(scores)[1];f=j.query("label == 'Exact'")[['query_id','product_id']].copy()
    ar=rr[[qi[int(x)] for x in f.query_id],[pi[str(x)] for x in f.product_id]]
    for s in STEMS:f[s]=ar
    # Mathematical set invariance uses one cached product vector for all schedules.
    save_metrics(name,f);save_metrics(name,f,True);np.save(OUT/name/'products.npy',pv)
    cost(name+'_train_evaluate',time.monotonic()-start);del model,at,nt,qt;gc.collect();torch.cuda.empty_cache()
def mean_metrics(name,split='validation'):
    df=pd.read_csv(OUT/name/'catalog_queries.csv');return df[df.query_id.isin(setup()[split])].mean(numeric_only=True).to_dict()
def selection():
    path=HERE/'selection.json'
    if path.exists():return json.loads(path.read_text())
    base=mean_metrics('Original');rows=[]
    for lam in setup()['lambdas']:
        name=f'Perm-FT-cons{lam}_s42';m=mean_metrics(name);rows.append({'name':name,'lambda':lam,**m,'recall_pass':m['Recall@100']>=base['Recall@100']-.01})
    chosen=sorted(rows,key=lambda x:(not x['recall_pass'],x['VI@20'],-x['Robust@100'],x['lambda']))[0]
    rec={'selected':chosen['name'],'all_validation_trials':rows,'test_used':False};dump(path,rec);return rec
def run():
    setup();baselines();train('Perm-FT_s42');evaluate('Perm-FT_s42')
    for lam in setup()['lambdas']:
        name=f'Perm-FT-cons{lam}_s42';train(name,lam);evaluate(name)
    selected=selection();set_run()
    # Seeds are gated exclusively by development evidence and remaining fixed budget.
    base=mean_metrics('Original');sm=mean_metrics('Set-Mean');extra=[]
    for name in ['Perm-FT_s42',selected['selected'],'Set-Attention_s42']:
        m=mean_metrics(name);ok=m['Recall@100']>=base['Recall@100']-.01 and m['VI@20']<=.75*base['VI@20']
        if name.startswith('Set'):ok=ok and m['Recall@100']>sm['Recall@100']
        if not ok:continue
        for seed in [43,44]:
            # Reserve the measured seed-42 end-to-end duration plus 25% safety margin.
            costs=[json.loads(l) for l in (HERE/'cost.jsonl').read_text().splitlines()]
            required=1.25*sum(x['seconds'] for x in costs if x['name'].startswith(name))
            if spent()+required>setup()['gpu_task_seconds_cap']:break
            n=name.replace('s42',f's{seed}')
            if n.startswith('Set'):set_run(seed)
            else:
                lam=0 if '-cons' not in n else float(n.split('-cons')[1].split('_')[0]);train(n,lam,seed);evaluate(n)
            extra.append(n)
    dump(HERE/'run_complete.json',{'selected':selected['selected'],'additional_seeds':extra,'cost_seconds':spent(),'completed_utc':pd.Timestamp.now(tz='UTC').isoformat()})

if __name__=='__main__':
    cmd=argparse.ArgumentParser();cmd.add_argument('action',choices=['freeze','run','report']);a=cmd.parse_args()
    if a.action=='freeze':freeze()
    elif a.action=='run':
        started=time.monotonic();previous=spent()
        try:run()
        except BaseException:
            unaccounted=max(0,time.monotonic()-started-(spent()-previous))
            cost('interrupted_run_unaccounted_wall_time',unaccounted,accounting='Run wall time minus already recorded completed stages; includes cache and transition overhead.')
            raise
    else:
        from report import main
        main()
