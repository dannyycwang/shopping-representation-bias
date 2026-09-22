"""Label-derived direct gains at original and condensed catalog ranks."""
import json
import numpy as np
import pandas as pd
from common import GAINS,OLD_GAINS,highest

def query_metrics(labels,absolute_ranks,product_ids,top_ids,dataset):
    labels=np.asarray(labels); rank=np.asarray(absolute_ranks,dtype=int); pids=np.asarray(product_ids,dtype=str)
    assert len(labels)==len(rank)==len(pids) and len(set(pids))==len(pids)
    assert len(set(rank))==len(rank) and (rank>=1).all()
    assert len(top_ids)>=20 and len(set(top_ids))==len(top_ids)
    for pid,pos in zip(pids,rank):
        if pos<=len(top_ids):assert str(top_ids[pos-1])==pid,('top ID/rank disagreement',pid,pos)
    assert set(labels)<=set(GAINS[dataset])
    gain=np.array([GAINS[dataset][x] for x in labels]); old=np.array([OLD_GAINS[dataset][x] for x in labels])
    h=labels==highest(dataset); old_h=old==max(OLD_GAINS[dataset].values()); new_h=gain==max(GAINS[dataset].values())
    assert np.array_equal(h,old_h) and np.array_equal(h,new_h)
    assert h.any()
    order=np.argsort(rank,kind='stable'); judged=set(pids)
    result={'n_judged':len(labels),'n_highest':int(h.sum()),'highest_support_ids':json.dumps(sorted(pids[h]),separators=(',',':'))}
    for k in [10,20]:
        for prefix,g in [('',gain),('old_',old)]:
            ideal=np.sort(g)[::-1][:k];den=float(np.sum(ideal/np.log2(np.arange(2,len(ideal)+2))))
            mask=rank<=k;dcg=float(np.sum(g[mask]/np.log2(rank[mask]+1)))
            condensed=g[order][:k];cdc=float(np.sum(condensed/np.log2(np.arange(2,len(condensed)+2))))
            assert den>0
            result[prefix+f'nDCG@{k}']=dcg/den;result[prefix+f'cNDCG@{k}']=cdc/den
        observed={str(v) for v in top_ids[:k] if str(v) in judged}
        expected=set(pids[rank<=k]);assert observed==expected,(dataset,k,observed^expected)
        result[f'JudgedCoverage@{k}']=len(observed)/k
    for k in [20,100]:
        included=set(pids[h & (rank<=k)])
        old_inc=set(pids[old_h & (rank<=k)]);new_inc=set(pids[new_h & (rank<=k)])
        assert included==old_inc==new_inc
        result[f'Recall@{k}']=len(included)/int(h.sum())
        result[f'included_ids_K{k}']=json.dumps(sorted(included),separators=(',',':'))
    return result

def evaluate_pairs(pairs,judgments,query_ids,top_by_query,dataset,meta):
    f=pairs[pairs.query_id.isin(query_ids)].copy();j=judgments[judgments.query_id.isin(query_ids)]
    f=f.set_index(['query_id','product_id']).sort_index();j=j.set_index(['query_id','product_id']).sort_index()
    assert f.index.equals(j.index),('incomplete judged support',meta,len(f),len(j))
    assert f.label.equals(j.label),('label disagreement',meta)
    rows=[]
    for qid,g in f.reset_index().groupby('query_id',sort=True):
        result=query_metrics(g.label,g['rank'],g.product_id,top_by_query[int(qid)],dataset)
        rows.append(dict(**meta,query_id=int(qid),**result))
    assert [r['query_id'] for r in rows]==query_ids
    return pd.DataFrame(rows)
