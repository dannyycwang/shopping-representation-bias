"""Paired query-cluster percentile bootstrap; one shared draw matrix, not iid swaps."""
from core import *

class QueryBootstrap:
    def __init__(self):
        partition=frame(HERE/'data/query_partition.csv')
        self.queries=sorted(partition.loc[partition.partition.eq('confirmation'),'query_id'])
        self.index={q:i for i,q in enumerate(self.queries)};n=len(self.queries)
        path=HERE/'results/bootstrap_indices.npy';path.parent.mkdir(parents=True,exist_ok=True)
        rng=np.random.default_rng(int(pick('query_bootstrap'),16));expected=rng.integers(0,n,(10000,n),dtype=np.int32)
        if path.exists():assert np.array_equal(np.load(path),expected)
        else:np.save(path,expected)
        w=np.zeros((10000,n),dtype=np.float64)
        np.add.at(w,(np.arange(10000)[:,None],expected),1);assert np.all(w.sum(1)==n);self.weights=w
        dump('results/bootstrap_manifest.json',dict(seed=SEED,purpose='query_bootstrap',generator='numpy.default_rng PCG64',
            draws=10000,query_ids=self.queries,indices_sha256=sha(path),interval='2.5/97.5 percentile',
            unit='query cluster; preserve all targets, swaps and model conditions; same draws for every contrast'))
    def aggregate(self,qids,values):
        x=np.zeros(len(self.queries),dtype=float)
        np.add.at(x,[self.index[int(q)] for q in qids],np.asarray(values,dtype=float));return x
    def estimate(self,qids,numerator,denominator):
        n=self.aggregate(qids,numerator);d=self.aggregate(qids,denominator)
        bn=self.weights@n;bd=self.weights@d;valid=bd>0
        draws=np.full(len(bn),np.nan);draws[valid]=bn[valid]/bd[valid]
        lo,hi=np.quantile(draws[valid],[.025,.975]) if valid.any() else (np.nan,np.nan)
        return dict(estimate=float(n.sum()/d.sum()) if d.sum() else np.nan,ci_low=float(lo),ci_high=float(hi),
            numerator_sum=float(n.sum()),denominator_pairs=int(d.sum()),valid_bootstrap_draws=int(valid.sum())),draws

def load_scores(stage='formal'):
    scores=pd.concat([frame(HERE/f'results/{stage}/{key}_scores.csv',dtype={'product_id':str}) for key in ['gte','harrier']],ignore_index=True)
    for col in ['in_top20','in_top20_A','all_competitor_scores_bitwise_unchanged']:
        assert scores[col].isin([True,False]).all()
    assert not scores.duplicated(['model','query_id','product_id','variant']).any()
    return scores

def concentration(rows,event_col,net_col,identity):
    g=rows.groupby('query_id').agg(events=(event_col,'sum'),net=(net_col,'sum'),pairs=('product_id','size')).reset_index()
    total=float(g.events.sum());net=float(g.net.sum());n=len(rows)
    top=g.sort_values(['events','query_id'],ascending=[False,True])
    loo=(net-g.net)/(n-g.pairs)
    summary=dict(**identity,event_total=total,event_queries=int(g.events.gt(0).sum()),
        max_query_event_share=float(top.events.iloc[0]/total) if total else np.nan,
        top5_query_event_share=float(top.events.head(5).sum()/total) if total else np.nan,
        leave_one_query_out_delta_min=float(loo.min()),leave_one_query_out_delta_max=float(loo.max()))
    return summary,[dict(**identity,**r) for r in g.to_dict('records')]
