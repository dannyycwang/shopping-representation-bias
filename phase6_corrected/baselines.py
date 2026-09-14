"""Reuse historical raw controls; supplement only missing graded ranks from frozen vectors."""
import numpy as np
import pandas as pd
from common_corrected import ROOT,HERE,STEMS,prior,measures

BASELINES=['Original','Canonical','Set-Mean','Set-Attention']


def baseline_frame(data,name,split):
    if name=='Set-Attention':
        vectors=np.load(ROOT/'phase5_mitigation/results/Set-Attention_s42/products.npy')
        f=data.judged_frame(vectors,split)
        # Exact historical ranks remain authoritative; preserve the frozen raw control.
        old=pd.read_parquet(ROOT/'phase5_mitigation/results/Set-Attention_s42/catalog_pairs.parquet')
        old.product_id=old.product_id.astype(str)
        old=old[old.query_id.isin(data.config[split])].set_index(['query_id','product_id'])
        exact=f.label.eq('Exact')
        ix=pd.MultiIndex.from_frame(f.loc[exact,['query_id','product_id']])
        differences=f.loc[exact,'C0'].to_numpy()-old.loc[ix,'C0'].to_numpy()
        f.attrs['historical_exact_reconciliation']={'changed_ranks':int(np.count_nonzero(differences)),
                                                   'max_rank_difference':int(np.abs(differences).max())}
        # Do not alter graded rank ordering to substitute an old arithmetic result.
        # Store original Exact columns separately for access-metric reuse in analysis.
        return f
    if name=='Original':
        f=None
        for s in STEMS:
            g=pd.read_parquet(ROOT/f'phase2/results/phase2_pair_ranks/wands_bge_base_native_{s}.parquet')
            g.product_id=g.product_id.astype(str)
            g=g[g.query_id.isin(data.config[split])]
            if f is None:f=g[['query_id','product_id','label']].copy()
            f=f.merge(g[['query_id','product_id','rank']].rename(columns={'rank':s}),on=['query_id','product_id'],validate='one_to_one')
        return f.reset_index(drop=True)
    stem='canonical_raw' if name=='Canonical' else 'set_mean'
    g=pd.read_parquet(ROOT/f'phase4/results/wands/bge_base_{stem}_pairs.parquet')
    g.product_id=g.product_id.astype(str)
    g=g[g.query_id.isin(data.config[split])]
    f=g[['query_id','product_id','label']].copy()
    for s in STEMS:f[s]=g['rank'].to_numpy()
    return f.reset_index(drop=True)


def historical_exact(name,split,cfg,target=False):
    folder='Set-Attention_s42' if name=='Set-Attention' else name
    f=pd.read_parquet(ROOT/'phase5_mitigation/results'/folder/('target_pairs.parquet' if target else 'catalog_pairs.parquet'))
    f.product_id=f.product_id.astype(str)
    return f[f.query_id.isin(cfg[split])].reset_index(drop=True)
