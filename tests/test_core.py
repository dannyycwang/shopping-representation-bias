import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import unittest
from collections import Counter
import numpy as np
import pandas as pd
from src.representations import build,audit,tokens,FIELDS
from src.retrieval import bm25,rank,inverse,hybrid
from src.evaluation import evaluate,paired_stats

class CoreTests(unittest.TestCase):
    def test_duplicate_conflicting_and_empty_keys_survive(self):
        row=dict(zip(FIELDS,['Oak bed','Beds','Furniture / Beds','Not suitable outdoors.','width:10|width:12| : no|material:oak']))
        for condition in ['R0','R2','R3','R4','R5','R8','C_repeat']:
            text=build(row,condition)
            self.assertTrue(audit(row,text)['pass'])
            self.assertEqual(audit(row,text)['conflicting_keys'],1)
            self.assertIn('Not suitable outdoors.',text)
        self.assertEqual(Counter(tokens(build(row,'R0'))),Counter(tokens(build(row,'R4'))))
        self.assertFalse(audit(row,build(row,'R0')+' capacity:999999')['pass'])

    def test_bm25_order_control(self):
        texts=['oak bed width:10|wood:yes','steel chair','wood dining table']
        changed=[x.replace('|','\n- ') for x in texts]
        np.testing.assert_array_equal(bm25(texts,['oak wood']),bm25(changed,['oak wood']))
        self.assertEqual(rank(bm25(texts,['oak']))[0,0],0)

    def test_ties_and_inverse(self):
        scores=np.array([[1.,2.,2.,0.]])
        np.testing.assert_array_equal(rank(scores),[[1,2,0,3]])
        np.testing.assert_array_equal(inverse(rank(scores)),[[3,1,2,4]])

    def test_unknown_is_not_negative(self):
        p=pd.DataFrame({'product_id':[0,1,2]}); q=pd.DataFrame({'query_id':[1],'query':['bed']})
        labels=pd.DataFrame({'query_id':[1,1],'product_id':[1,2],'grade':[2,0]})
        # evaluate expects top 20 so extend synthetic catalog with unjudged entries.
        p=pd.DataFrame({'product_id':list(range(30))})
        scores=np.array([np.arange(30,0,-1)],dtype=float)
        out,pairs,top=evaluate(rank(scores),scores,p,q,labels)
        self.assertEqual(out.iloc[0]['cNDCG@10'],1.)
        self.assertEqual(out.iloc[0]['MRR'],.5)
        self.assertEqual(out.iloc[0]['JudgedCoverage@10'],.2)
        self.assertEqual(out.iloc[0]['ExplicitIrrelevant@10'],.1)
        self.assertTrue(pd.isna(top.iloc[0]['grade']))

    def test_zero_effect(self):
        d,lo,hi,p=paired_stats([.1,.2],[.1,.2],42,1000)
        self.assertEqual((d,lo,hi,p),(0.,0.,0.,1.))

    def test_full_rrf(self):
        a=np.array([[0,1,2]]); b=np.array([[2,1,0]])
        s=hybrid(a,b,60)
        self.assertAlmostEqual(float(s[0,0]),1/61+1/63,places=7)

if __name__=='__main__': unittest.main()
