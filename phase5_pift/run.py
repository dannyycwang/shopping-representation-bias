from pathlib import Path
import os
# Every required model is pinned and already cached; avoid network-only failures on resume.
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
import sys,json,random,time,hashlib,math,re,gc
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path.insert(0,str(ROOT/'phase5_mitigation'))
import screen as s
sys.path.insert(0,str(HERE/'vendor/src'))
from pift.config import Config,FieldSpec
from pift.serialize import render_segments
from src.representations import _random_order
OUT=HERE/'results';SPEC=s.SPEC;STEMS=s.STEMS
SCHEMA=Config(raw={'serialization':{'label_scheme':'key','separator':' | ','total_chars':10**9}},fields=[FieldSpec('title','title','protected')])
STOP=set('a an the of for with in on and or to is are by from at this that'.split())
def segments(p,attrs=None):
    result=[(key,key,p.get(key,'')) for key in ['title','class','category','description'] if p.get(key,'')]
    for a in p['attributes'] if attrs is None else attrs:
        if ':' in a:key,value=a.split(':',1);result.append((key,key,value))
        else:result.append(('attributes','attributes',a))
    return result
def protected(q,p):
    # User-approved WANDS adaptation: no inferred facets or lexical protection.
    return {'title'}
def labeled_build(p,condition,seeds):
    attrs=p['attributes']
    if condition=='C1_reverse_attribute_order':attrs=list(reversed(attrs))
    elif condition.startswith('C2_random_attribute_order_s'):attrs=_random_order(attrs,str(p['product_id']),seeds[int(condition.rsplit('s',1)[1])-1])
    return render_segments(segments(p,attrs),SCHEMA)
def config():
    c=json.loads((HERE/'config.json').read_text(encoding='utf8'));assert s.sha(HERE/'AUDIT_AND_PROTOCOL.md')==c['protocol_sha256'];return c
def configure_scope():
    s.HERE=HERE;s.OUT=OUT;s.setup=config;s.build=labeled_build
def freeze():
    if (HERE/'config.json').exists():return
    p,q,j,pi,qi=s.data();qv,pv=s.original_cache('wands','bge_base');old=json.loads((ROOT/'phase5_mitigation/config.json').read_text());oldrows=json.loads((ROOT/'phase5_mitigation/triples.json').read_text());rows=[];excluded=[]
    for x in oldrows:
        qid=x['query_id'];posit=x['positive'];neg=j[j.query_id.eq(qid)&j.label.eq('Irrelevant')].product_id.astype(str).tolist()
        ii=np.array([pi[n] for n in neg]);order=np.argsort(-(pv[ii]@qv[qi[qid]]),kind='stable');pool=[neg[i] for i in order[3:30] if pv[pi[neg[i]]]@pv[pi[posit]]<=.95]
        if len(pool)<3:excluded.append(x);continue
        ns=random.Random(int(s.hashkey(f'negative42:{qid}:{posit}')[:16],16)).sample(pool,3)
        query=str(q.iloc[qi[qid]]['query']);rows.append({'query_id':qid,'query':query,'positive':posit,'negatives':ns,'protect':sorted(protected(query,p[pi[posit]]))})
    s.dump(HERE/'triples.json',rows);s.dump(HERE/'excluded_training_rows.json',excluded)
    c={'frozen_utc':pd.Timestamp.now(tz='UTC').isoformat(),'train':old['train'],'validation':old['validation'],'test':old['test'],'seed':42,'epochs':5,'batch_size':128,'mini_batch_size':4,'lr':3e-5,'warmup_ratio':.1,'field_dropout':.15,'gpu_task_seconds_cap':28800,'model':SPEC,'training_rows':len(rows),'excluded_rows':len(excluded),'triples_sha256':s.sha(HERE/'triples.json'),'protocol_sha256':s.sha(HERE/'AUDIT_AND_PROTOCOL.md'),'author_commit':'64437335afd37b5e8f32d453eeb8e93af61c271d'}
    s.dump(HERE/'config.json',c);s.dump(HERE/'source_hashes.json',[{'file':str(f.relative_to(HERE)),'sha256':s.sha(f)} for f in (HERE/'vendor').rglob('*') if f.is_file()]);print('Frozen rows',len(rows),'excluded',len(excluded),flush=True)
def make_model():
    from sentence_transformers import SentenceTransformer,models
    tr=models.Transformer(SPEC['name'],max_seq_length=512,model_args={'revision':SPEC['revision'],'local_files_only':True},tokenizer_args={'revision':SPEC['revision'],'local_files_only':True})
    return SentenceTransformer(modules=[tr,models.Pooling(768,pooling_mode='cls'),models.Normalize()],device='cuda')
def train(name,augment):
    import torch
    from sentence_transformers.losses import CachedMultipleNegativesRankingLoss
    from transformers import get_linear_schedule_with_warmup
    ck=HERE/'checkpoints'/name/'weights.pt'
    if ck.exists():return
    c=config();assert s.sha(HERE/'triples.json')==c['triples_sha256'];s.guard();start=time.monotonic();torch.set_num_threads(4);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    p,q,j,pi,qi=s.data();rows=json.loads((HERE/'triples.json').read_text());model=make_model().train()
    known={(int(x.query_id),str(x.product_id)) for x in j.itertuples() if int(x.query_id) in c['train'] and x.label in ['Exact','Partial']}
    class LabelMaskedCachedMNRL(CachedMultipleNegativesRankingLoss):
        def calculate_loss(self,reps,with_backward=False):
            anchors=torch.cat(reps[0]);candidates=torch.cat([torch.cat(r) for r in reps[1:]]);n=len(anchors);total=0
            for b in range(0,n,self.mini_batch_size):
                e=b+self.mini_batch_size;scores=self.similarity_fct(anchors[b:e],candidates)*self.scale;scores=scores.masked_fill(self.invalid[b:e],-1e9)
                part=self.cross_entropy_loss(scores,torch.arange(b,min(e,n),device=scores.device))*len(scores)/n
                if with_backward:part.backward();part=part.detach()
                total=total+part
            return total
    lossfn=LabelMaskedCachedMNRL(model,mini_batch_size=4,show_progress_bar=False)
    opt=torch.optim.AdamW(model.parameters(),lr=3e-5,weight_decay=0.0);steps=math.ceil(len(rows)/128)*5;scheduler=get_linear_schedule_with_warmup(opt,math.ceil(.1*steps),steps)
    traces=[];logs=[];step=0
    for epoch in range(5):
        order=list(range(len(rows)));random.Random(42+epoch).shuffle(order)
        for off in range(0,len(rows),128):
            s.guard(start);batch=[rows[i] for i in order[off:off+128]];rng=random.Random(4200000+epoch*10000+off);ids=[[x['positive'] for x in batch]]+[[x['negatives'][k] for x in batch] for k in range(3)];texts=[[x['query'] for x in batch]]
            for col,group in enumerate(ids):
                ts=[]
                for row_i,pid in enumerate(group):
                    seg=segments(p[pi[pid]]);prot=set(batch[row_i]['protect']) if col==0 else set()
                    audit_rng=random.Random();audit_rng.setstate(rng.getstate());keep=SCHEMA.protected_labels|prot
                    retained=[i for i,v in enumerate(seg) if not augment or v[0] in keep or audit_rng.random()>.15]
                    # Direct author renderer: fresh RNG state per occurrence, dropout before permutation.
                    text=render_segments(seg,SCHEMA,rng=rng if augment else None,permute=augment,field_dropout=.15 if augment else 0,protect=prot)
                    ts.append(text);traces.append({'epoch':epoch,'step':step,'column':col,'product_id':pid,'query_id':batch[row_i]['query_id'],'sha256':s.hashkey(text),'protected':sorted(prot),'retained_segment_indices':retained,'rendered_segments':len(retained),'source_segments':len(seg)})
                texts.append(ts)
            allids=sum(ids,[]);mask=np.array([[(x['query_id'],pid) in known or pid==x['positive'] for pid in allids] for x in batch]);mask[np.arange(len(batch)),np.arange(len(batch))]=False;lossfn.invalid=torch.tensor(mask,device='cuda')
            features=[{k:v.cuda() for k,v in model.tokenize(t).items()} for t in texts]
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                loss=lossfn(features,None);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();scheduler.step()
            logs.append({'epoch':epoch,'step':step,'loss':loss.item(),'batch_rows':len(batch),'mean_valid_negatives':float((~mask).sum(1).mean()-1),'seconds':time.monotonic()-start});print(name,epoch,step,loss.item(),flush=True);step+=1
    ck.parent.mkdir(parents=True,exist_ok=True);torch.save(model[0].auto_model.cpu().state_dict(),ck);pd.DataFrame(logs).to_csv(ck.parent/'training.csv',index=False);pd.DataFrame(traces).to_json(ck.parent/'augmentation.jsonl',orient='records',lines=True)
    s.dump(ck.parent/'metadata.json',{'weights_sha256':s.sha(ck),'config_sha256':s.sha(HERE/'config.json'),'augment':augment,'loss':'label-masked CachedMultipleNegativesRankingLoss','seed':42,'sentence_transformers_version':__import__('sentence_transformers').__version__})
    s.cost(name+'_train',time.monotonic()-start);del model,opt,lossfn;gc.collect();torch.cuda.empty_cache()
def evaluate(name):
    # Reuse the established ranking, unit-normalization and target insertion pipeline.
    p,q,j,pi,qi=s.data();nrows=[];original_ranks=s.ranks
    def collect(scores):
        order,rr=original_ranks(scores);stem=STEMS[len(nrows)//len(q)]
        for qid in q.query_id:
            g=j[j.query_id.eq(qid)];idx=[pi[str(v)] for v in g.product_id];rk=rr[qi[int(qid)],idx];gain=g.label.map({'Exact':3.,'Partial':1.,'Irrelevant':0.}).to_numpy();sort=np.argsort(rk,kind='stable');ideal=np.sort(gain)[::-1][:10];den=(ideal/np.log2(np.arange(2,len(ideal)+2))).sum();top=rk<=10
            ndcg=float((gain[top]/np.log2(rk[top]+1)).sum()/den) if den else np.nan
            condensed=gain[sort][:10];cndcg=float((condensed/np.log2(np.arange(2,len(condensed)+2))).sum()/den) if den else np.nan
            nrows.append({'query_id':int(qid),'variant':stem,'nDCG@10_unjudged_zero':ndcg,'cNDCG@10':cndcg})
        return order,rr
    s.ranks=collect
    try:s.evaluate(name)
    finally:s.ranks=original_ranks
    if nrows:pd.DataFrame(nrows).to_csv(OUT/name/'ndcg_per_query.csv',index=False)
def run():
    configure_scope()
    # Do not compete with the legacy worker. Its durable boundary pause is logged.
    if not (HERE/'legacy_pause.json').exists():raise RuntimeError('Wait for legacy durable-boundary pause before starting GPU work.')
    for name,augment in [('Standard-FT',False),('Adapted-PI-FT',True)]:train(name,augment);evaluate(name)
    name='Labeled-ZeroShot';ck=HERE/'checkpoints'/name/'weights.pt'
    if not ck.exists():
        import torch
        model=make_model();ck.parent.mkdir(parents=True,exist_ok=True);torch.save(model[0].auto_model.cpu().state_dict(),ck);del model;torch.cuda.empty_cache()
    evaluate(name);s.dump(HERE/'run_complete.json',{'completed_utc':pd.Timestamp.now(tz='UTC').isoformat(),'scope':'source-aligned adapted WANDS experiment, not DevDataBench reproduction','cost_seconds':s.spent()})
if __name__=='__main__':
    action=sys.argv[1]
    if action=='freeze':freeze()
    else:
        configure_scope();start=time.monotonic();previous=s.spent()
        try:run()
        except BaseException:
            s.cost('interrupted_unaccounted_wall_time',max(0,time.monotonic()-start-(s.spent()-previous)));raise
