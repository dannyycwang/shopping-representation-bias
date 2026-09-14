import numpy as np
import pandas as pd
from .retrieval import inverse

def evaluate(order, scores, products, queries, labels):
    ranks = inverse(order); pids = products.product_id.to_numpy()
    lookup = {int(x):i for i,x in enumerate(pids)}
    groups = {int(q):g for q,g in labels.groupby('query_id')}
    rows, pair_rows, top_rows = [], [], []
    for qi, q in enumerate(queries.itertuples(index=False)):
        group = groups[q.query_id]
        ids = np.array([lookup[int(x)] for x in group.product_id])
        grades = group.grade.to_numpy()
        judged = dict(zip(ids, grades))
        ordered = order[qi]; judged_order = np.array([judged[int(i)] for i in ordered if int(i) in judged])
        ideal = np.sort(grades)[::-1]
        row = {'query_id':q.query_id, 'query':q.query, 'n_judged':len(ids), 'n_exact':int((grades==2).sum())}
        relevant = ids[grades>0]; exact = ids[grades==2]
        # Condensed NDCG excludes unknown judgments instead of declaring them irrelevant.
        for k in [10,20]:
            disc = 1/np.log2(np.arange(2, min(k,len(ideal))+2))
            idcg = ((2**ideal[:k]-1)*disc).sum()
            row[f'cNDCG@{k}'] = float(((2**judged_order[:k]-1)*disc).sum()/idcg) if idcg else np.nan
        for k in [10,20,50]:
            top = ordered[:k]
            row[f'Recall@{k}'] = float((ranks[qi,relevant]<=k).mean()) if len(relevant) else np.nan
            row[f'ExactHit@{k}'] = float((ranks[qi,exact]<=k).any()) if len(exact) else np.nan
            row[f'JudgedCoverage@{k}'] = sum(int(i) in judged for i in top)/k
        row['MRR'] = float(1/ranks[qi,relevant].min()) if len(relevant) else np.nan
        row['ExplicitIrrelevant@10'] = sum(judged.get(int(i),-1)==0 for i in ordered[:10])/10
        row['Unjudged@10'] = 1-row['JudgedCoverage@10']
        row['Top20BoundaryTied'] = float(scores[qi,ordered[19]]==scores[qi,ordered[20]])
        rows.append(row)
        for idx, grade in zip(ids, grades):
            if grade==2:
                pair_rows.append({'query_id':q.query_id,'product_id':int(pids[idx]),'rank':int(ranks[qi,idx]),
                                  'score':float(scores[qi,idx]),'zero_score':bool(scores[qi,idx]==0)})
        for r, idx in enumerate(ordered[:100],1):
            top_rows.append({'query_id':q.query_id,'product_id':int(pids[idx]),'rank':r,
                             'score':float(scores[qi,idx]),'grade':judged.get(int(idx),None)})
    return pd.DataFrame(rows), pd.DataFrame(pair_rows), pd.DataFrame(top_rows)

def paired_stats(a, b, seed, samples=10000):
    d = np.asarray(b)-np.asarray(a); d = d[np.isfinite(d)]
    if len(d)==0: return np.nan,np.nan,np.nan,np.nan
    rng = np.random.default_rng(seed)
    boots=[]; null=[]
    for start in range(0,samples,500):
        n=min(500,samples-start)
        boots.extend(d[rng.integers(len(d),size=(n,len(d)))].mean(1))
        null.extend((d*rng.choice([-1,1],size=(n,len(d)))).mean(1))
    lo,hi=np.quantile(boots,[.025,.975]); mean=d.mean()
    p=(1+np.sum(np.abs(null)>=abs(mean)-1e-15))/(samples+1)
    return float(mean),float(lo),float(hi),float(p)

def mean_ci(values, seed, samples=10000):
    values=np.asarray(values); values=values[np.isfinite(values)]
    rng=np.random.default_rng(seed); means=[]
    for start in range(0,samples,500):
        means.extend(values[rng.integers(len(values),size=(min(500,samples-start),len(values)))].mean(1))
    return np.quantile(means,[.025,.975])
