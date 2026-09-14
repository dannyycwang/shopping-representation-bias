import unittest
import numpy as np
import pandas as pd
import torch
from screen import rank_target, ranks, metric_frame, STEMS, set_class, setup, data

class Integrity(unittest.TestCase):
    def test_target_removal_and_ties(self):
        b=np.array([.8,.6,.6,.2],dtype='f4')
        for i in range(4):
            for value in [.9,.8,.6,.1]:
                v=np.float32(value);changed=b.copy();changed[i]=v
                self.assertEqual(rank_target(b,np.array([i]),np.array([v]))[0],ranks(changed[None])[1][0,i])
    def test_metrics_population(self):
        f=pd.DataFrame({'query_id':[1,1,2],'product_id':['a','b','c']})
        for s in STEMS:f[s]=[1,200,200]
        f['C1']=[200,200,200]
        d,q=metric_frame(f)
        self.assertAlmostEqual(q['VI@20'].mean(),.25)
        self.assertAlmostEqual(q['Never@100'].mean(),.75)
        self.assertAlmostEqual(q['Robust@100'].mean(),0)
        self.assertAlmostEqual(q['Recall@100'].mean(),6/28)
    def test_attention_invariance_empty_and_multiplicity(self):
        torch.manual_seed(42);m=set_class()().eval();a=torch.randn(3,5,768);mask=torch.ones(3,5,dtype=torch.bool);mask[0]=False;non=torch.randn(3,768)
        with torch.no_grad():
            v=m(a,mask,non);w=m(a[:,[3,1,4,0,2]],mask[:,[3,1,4,0,2]],non)
        self.assertLess((v-w).norm(dim=1).max().item(),1e-6)
        self.assertTrue(torch.isfinite(v).all())
    def test_splits_and_negative_labels(self):
        import json
        from screen import HERE
        c=setup();p,q,j,pi,qi=data();labels={(int(r.query_id),str(r.product_id)):r.label for r in j.itertuples()}
        for x in json.loads((HERE/'triples.json').read_text()):
            self.assertIn(x['query_id'],c['train']);self.assertNotIn(x['query_id'],c['test']);self.assertNotEqual(x['positive'],x['negative'])
            self.assertEqual(labels[x['query_id'],x['positive']],'Exact');self.assertEqual(labels[x['query_id'],x['negative']],'Irrelevant')
    def test_exact_serialization_reuse(self):
        import gzip,json
        from collections import Counter
        from screen import ROOT,CFG,build
        p,*_=data(); chosen=set(np.linspace(0,len(p)-1,100,dtype=int))
        for stem,condition in zip(STEMS,CFG['primary_variant_family']):
            with gzip.open(ROOT/f'phase2/data/representations/wands/{stem}.jsonl.gz','rt',encoding='utf8') as f:
                for i,line in enumerate(f):
                    if i not in chosen:continue
                    x=json.loads(line);text=build(p[i],condition,CFG['attribute_permutation_seeds'])
                    self.assertEqual(text,x['text'])
                    self.assertEqual(Counter(text),Counter(build(p[i],CFG['primary_variant_family'][0],CFG['attribute_permutation_seeds'])))
if __name__=='__main__':unittest.main()
