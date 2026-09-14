import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from common import attrs,texts,RULES,pair_metrics,ranks,norm

def record():return {'product_id':'p','title':'Fixed title','description':'Fixed description','attributes':['color: blue','type: chair','color: blue'],'section_order':['title','description','attributes'],'attribute_separator':'|'}

def test_input_order_independence_and_multiplicity():
    p=record();rev={**p,'attributes':list(reversed(p['attributes']))}
    for method in [*RULES,*[f'view{i}' for i in range(1,8)]]:
        assert attrs(p,method)==attrs(rev,method)
        assert attrs(p,method).count('color: blue')==2
        assert texts([p],method)==texts([rev],method)

def test_query_denominators_and_official_gains():
    q=pd.DataFrame({'query_id':[1,2,3]})
    pairs=pd.DataFrame({'query_id':[1,1,2,2,3],'product_id':['a','b','a','b','a'],'label':['E','S','S','C','I'],'rank':[2,1,2,1,1]})
    r=pair_metrics(pairs,q,'esci').set_index('query_id')
    assert len(r)==3 and np.isnan(r.loc[2,'Recall@100'])
    assert np.isfinite(r.loc[2,'cNDCG@10']) and np.isnan(r.loc[3,'cNDCG@10'])
    assert r.loc[1,'cNDCG@10']<1

def test_max_views_return_unique_products_and_centroid_norm():
    scores=np.array([[.7,.8,.6],[.9,.4,.8]])
    product_max=scores.max(axis=0)[None,:]
    order,rank=ranks(product_max)
    assert order.tolist()==[[0,1,2]] and len(set(order[0][:2]))==2
    v=norm(np.array([[1.,2.],[2.,1.]]));c=norm(v.mean(axis=0,keepdims=True))
    np.testing.assert_allclose(np.linalg.norm(c,axis=1),1)
