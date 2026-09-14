import itertools
import numpy as np
import pandas as pd
import pytest
import torch
from torch.nn import functional as F
from value_models import ValueAttention
from common_corrected import STEMS,graded_queries
from sampling import BalancedSampler


@pytest.mark.parametrize('method',['C1','C2','C3','C4'])
def test_uniform_initialization_original_values_and_learning(method):
    torch.manual_seed(42)
    model=ValueAttention(method,5,hidden=16,dim=24)
    a=F.normalize(torch.randn(2,4,24),dim=-1)
    mask=torch.tensor([[True,True,True,False],[False]*4])
    non=F.normalize(torch.randn(2,24),dim=1)
    ids=torch.tensor([[0,1,2,0],[0]*4])
    expected=F.normalize(.5*non+.5*F.normalize((a*mask[:,:,None]).sum(1),dim=1),dim=1)
    v,w,_=model.components(a,mask,non,ids)
    torch.testing.assert_close(v,expected,atol=1e-6,rtol=1e-6)
    torch.testing.assert_close(w[0,:3],torch.full((3,),1/3))
    opt=torch.optim.AdamW(model.parameters(),lr=.01)
    q=torch.randn(2,24)
    for _ in range(3):
        opt.zero_grad()
        (model(a,mask,non,ids)*q).sum().backward()
        opt.step()
    v,w,attr=model.components(a,mask,non,ids)
    assert torch.all(w>=0) and w[~mask].count_nonzero()==0
    torch.testing.assert_close(w.sum(1),torch.tensor([1.,0.]))
    torch.testing.assert_close(attr,F.normalize((w[:,:,None]*a).sum(1),dim=1))
    torch.testing.assert_close(v,F.normalize(.5*non+.5*attr,dim=1))
    assert model.project.weight.grad.abs().sum()>0
    assert not any(isinstance(m,torch.nn.Linear) and m.out_features==24 for m in model.modules())


@pytest.mark.parametrize('method',['C1','C2','C3','C4'])
def test_nonuniform_permutations_padding_duplicates_unknown(method):
    torch.manual_seed(43)
    model=ValueAttention(method,5,hidden=16,dim=24).eval()
    scorer=model.scorer[-1] if isinstance(model.scorer,torch.nn.Sequential) else model.scorer
    torch.nn.init.normal_(scorer.weight,std=.3)
    a=torch.randn(1,4,24);a[:,2]=a[:,0]
    mask=torch.tensor([[True,True,True,False]])
    ids=torch.tensor([[0,2,0,0]])
    non=F.normalize(torch.randn(1,24),dim=1)
    with torch.no_grad():
        expected=model(a,mask,non,ids)
        for permutation in itertools.permutations(range(4)):
            ix=list(permutation)
            torch.testing.assert_close(expected,model(a[:,ix],mask[:,ix],non,ids[:,ix]),atol=1e-6,rtol=1e-6)
        mask[:,2]=False
        assert not torch.allclose(expected,model(a,mask,non,ids),atol=1e-6)


def test_graded_query_eligibility_and_condensation():
    f=pd.DataFrame({'query_id':[1,1,1,2,2,3],'product_id':['a','b','c','d','e','f'],
                    'label':['Exact','Partial','Irrelevant','Partial','Irrelevant','Irrelevant']})
    for s in STEMS:f[s]=[100,1,2,500,20,1]
    result=graded_queries(f)
    expected=(1+3/np.log2(4))/(3+1/np.log2(3))
    assert np.isclose(result.loc[1,'cNDCG@10'],expected)
    assert np.isclose(result.loc[2,'cNDCG@10'],1/np.log2(3))
    assert np.isnan(result.loc[3,'cNDCG@10'])
    from common import pair_metrics
    existing=pair_metrics(f.assign(rank=f.C0),pd.DataFrame({'query_id':[1,2,3]}),'wands').set_index('query_id')
    np.testing.assert_allclose(result['cNDCG@10'],existing['cNDCG@10'],equal_nan=True)


def test_equal_query_sampling_and_grade_weights():
    groups={1:{'Exact':['a','b'],'Partial':['c'],'Irrelevant':['d']},2:{'Exact':[],'Partial':['e'],'Irrelevant':['f']},3:{'Exact':['g'],'Partial':[],'Irrelevant':[]}}
    sampler=BalancedSampler(groups,42,64)
    rows=pd.DataFrame(sampler.epoch(1)+sampler.epoch(2))
    assert rows.groupby(['epoch','query_id']).size().eq(64).all()
    assert set(rows.query_id)=={1,2}
    assert set(rows[rows.query_id==1].weight)=={.5,1.}
    assert rows.equals(pd.DataFrame(BalancedSampler(groups,42,64).epoch(1)+BalancedSampler(groups,42,64).epoch(2)))
