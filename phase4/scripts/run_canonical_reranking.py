from common import *
import torch
from transformers import AutoTokenizer,AutoModelForSequenceClassification
from threadpoolctl import threadpool_limits
threadpool_limits(4);torch.set_num_threads(4)
MODEL='cross-encoder/ms-marco-MiniLM-L-6-v2';REV='233902d25c440f23af6f7d6e94d2946bac0bee0a'

def run(ds,max_k=200):
    p,q,j=load(ds);folder=OUT/ds;choice=json.loads((P4/'config/selection.json').read_text())['method'];methods=['Original','hybrid',choice]
    orders={m:np.load(folder/f'bge_base_{m}_top1000.npy') for m in methods};qlookup=q.set_index('query_id')['query'].to_dict()
    with gzip.open(P2/f'data/mitigations/{ds}/M2C0.jsonl.gz','rt',encoding='utf8') as f:records=list(map(json.loads,f))
    assert [str(x['product_id']) for x in records]==[str(x['product_id']) for x in p], 'Canonical text must align with catalog indices'
    canonical=[x['text'] for x in records]
    cache=OUT/f'{ds}_canonical_reranker_scores.parquet'
    scores=pd.read_parquet(cache) if cache.exists() else pd.DataFrame(columns=['query_id','product_index','score'])
    present=set(zip(scores.query_id.astype(int),scores.product_index.astype(int)));pairs=[]
    for qi,qid in enumerate(q.query_id):
        candidates=sorted(set(int(x) for m in methods for x in orders[m][qi,:max_k]));pairs.extend((int(qid),i) for i in candidates if (int(qid),i) not in present)
    if pairs:
        tok=AutoTokenizer.from_pretrained(MODEL,revision=REV);net=AutoModelForSequenceClassification.from_pretrained(MODEL,revision=REV).to('cuda').half().eval();vals=[];times=[]
        for start in range(0,len(pairs),128):
            b=pairs[start:start+128];t=time.perf_counter();inp=tok([str(qlookup[qid]) for qid,i in b],[canonical[i] for qid,i in b],padding=True,truncation=True,max_length=256,return_tensors='pt');inp={k:v.to('cuda') for k,v in inp.items()}
            with torch.inference_mode():v=net(**inp).logits.float().flatten().cpu().numpy()
            times.append(time.perf_counter()-t);vals.extend(v)
            if start%16384==0:print(ds,'canonical reranking',start,len(pairs),flush=True)
        added=pd.DataFrame(pairs,columns=['query_id','product_index']);added['score']=vals;scores=pd.concat([scores,added],ignore_index=True);scores.to_parquet(cache,index=False)
        timing_path=OUT/f'{ds}_canonical_reranker_timing.json';previous=json.loads(timing_path.read_text()) if timing_path.exists() else {};total=sum(times)+previous.get('total_tokenization_and_inference_seconds',0)
        timing_path.write_text(json.dumps({'model':MODEL,'revision':REV,'joint_token_budget':256,'text':'historical M2C0, fixed for all candidates','unique_scored_pairs':len(scores),'batch_size':128,'max_candidate_K':max_k,'total_tokenization_and_inference_seconds':total,'seconds_per_pair_amortized':total/len(scores),'scope':'local batched GPU throughput; excludes model load, not online request latency'},indent=2));del net;torch.cuda.empty_cache()
    lookup={(int(r.query_id),int(r.product_index)):r.score for r in scores.itertuples()};ids=[str(x['product_id']) for x in p];j=j.copy();j['product_id']=j.product_id.astype(str);groups={int(qid):g for qid,g in j.groupby('query_id')};highest='Exact' if ds=='wands' else 'E';gains={'Exact':3,'Partial':1,'Irrelevant':0} if ds=='wands' else {'E':1,'S':.1,'C':.01,'I':0};rows=[];tops=[]
    for method in methods:
        for k in [50,100,200]+([500] if max_k>=500 else []):
            for qi,qid in enumerate(q.query_id):
                cand=orders[method][qi,:k];val=np.array([lookup[(int(qid),int(i))] for i in cand]);order=cand[np.lexsort((cand,-val))][:20];topids=[ids[i] for i in order];g=groups[int(qid)];h=set(g.loc[g.label.eq(highest),'product_id']);rel=dict(zip(g.product_id,g.label.map(gains)));v=np.array([rel[x] for x in topids if x in rel]);ideal=np.sort(g.label.map(gains).to_numpy())[::-1][:10];idcg=(ideal/np.log2(np.arange(2,len(ideal)+2))).sum();dcg=(v[:10]/np.log2(np.arange(2,min(10,len(v))+2))).sum()
                rows.append({'dataset':ds,'method':method,'candidate_K':k,'query_id':int(qid),'n_highest':len(h),'Recall@20':len(h&set(topids))/len(h) if h else np.nan,'returned_condensed_NDCG@10':dcg/idcg if idcg else np.nan,'unique_candidates':len(cand)})
                tops.extend((method,k,int(qid),pid,r+1) for r,pid in enumerate(topids))
    pq=pd.DataFrame(rows);pq.to_csv(OUT/f'{ds}_canonical_reranking_per_query.csv',index=False);pd.DataFrame(tops,columns=['method','candidate_K','query_id','product_id','rank']).to_parquet(OUT/f'{ds}_canonical_reranking_top20.parquet',index=False)
    eval_ids=q.loc[~dev_mask(q),'query_id'] if ds=='wands' else q.query_id;summary=[]
    for (method,k),g in pq[pq.query_id.isin(eval_ids)].groupby(['method','candidate_K']):
        for metric in ['Recall@20','returned_condensed_NDCG@10']:
            mu,lo,hi=boot(g[metric]);summary.append({'dataset':ds,'method':method,'candidate_K':k,'metric':metric,'mean':mu,'ci_low':lo,'ci_high':hi})
    pd.DataFrame(summary).to_csv(OUT/f'{ds}_canonical_reranking_summary.csv',index=False)

if __name__=='__main__':
    for ds in ['wands','esci']:run(ds)
    dev=pd.read_csv(OUT/'wands_dev/bge_base_Original_per_query.csv');gain=float((dev['Recall@500']-dev['Recall@200']).mean())
    # Conservative bound: every one of the three methods contributes 500 new pairs per query.
    bound=sum(3*500*len(load(ds)[1])*json.loads((OUT/f'{ds}_canonical_reranker_timing.json').read_text())['seconds_per_pair_amortized'] for ds in ['wands','esci'])
    decision={'development_original_gain_200_to_500':gain,'conservative_additional_seconds':bound,'local_extension_cap_seconds':1800,'triggered':gain>.02 and bound<1800,'basis':'frozen development recall threshold; measured K<=200 throughput; no evaluation effectiveness used'}
    (OUT/'reranking_extension_decision.json').write_text(json.dumps(decision,indent=2))
    if decision['triggered']:
        for ds in ['wands','esci']:run(ds,500)
