"""Hand-computable tests run before large reevaluation."""
import unittest
import numpy as np
from common import ranks
from metrics import query_metrics,evaluate_pairs
import pandas as pd

def calc(labels,positions,pids=None):
    pids=pids or [str(i) for i in range(len(labels))]
    top=['u'+str(i) for i in range(120)]
    for pid,pos in zip(pids,positions):top[pos-1]=pid
    return query_metrics(labels,positions,pids,top,'esci')

class MetricTests(unittest.TestCase):
    def test_e_s_c_direct_gains(self):
        r=calc(['C','E','S'],[1,2,3]);den=1+.1/np.log2(3)+.01/2
        expected=(.01+1/np.log2(3)+.1/2)/den
        self.assertAlmostEqual(r['nDCG@10'],expected,places=14)
        self.assertAlmostEqual(r['cNDCG@10'],expected,places=14)
        self.assertNotAlmostEqual(r['old_nDCG@10'],expected,places=5)
    def test_unjudged_first_changes_discount(self):
        r=calc(['E','S','C'],[2,3,4]);den=1+.1/np.log2(3)+.01/2
        self.assertAlmostEqual(r['nDCG@10'],(1/np.log2(3)+.1/2+.01/np.log2(5))/den)
        self.assertEqual(r['cNDCG@10'],1.)
        self.assertEqual(r['JudgedCoverage@10'],.3)
    def test_same_gain_sequences_different_membership(self):
        a=calc(['E','E','I'],[1,21,22],['a','b','c'])
        b=calc(['E','E','I'],[21,1,22],['a','b','c'])
        self.assertEqual(a['nDCG@20'],b['nDCG@20'])
        self.assertNotEqual(a['included_ids_K20'],b['included_ids_K20'])
    def test_catalog_tie_break(self):
        order,rank=ranks(np.array([[.2,.5,.5,.2]],dtype='float32'))
        np.testing.assert_array_equal(order,[[1,2,0,3]])
        np.testing.assert_array_equal(rank,[[3,1,2,4]])
    def test_original_cutoff_and_condensed(self):
        r=calc(['E','S','C','I'],[11,21,22,1])
        self.assertEqual(r['nDCG@10'],0.)
        self.assertGreater(r['nDCG@20'],0.)
        self.assertGreater(r['cNDCG@10'],0.)
        self.assertEqual(r['JudgedCoverage@10'],.1)
    def test_recall_gain_swap_invariant(self):
        r=calc(['E','E','S','C'],[20,101,1,2])
        self.assertEqual(r['Recall@20'],.5);self.assertEqual(r['Recall@100'],.5)
    def test_wands_direct_not_exponential(self):
        top=[str(i) for i in range(120)]
        r=query_metrics(['Partial','Exact','Irrelevant'],[1,2,3],['0','1','2'],top,'wands')
        self.assertAlmostEqual(r['nDCG@10'],(1+3/np.log2(3))/(3+1/np.log2(3)))
    def test_incomplete_judgment_rejected(self):
        j=pd.DataFrame({'query_id':[1,1],'product_id':['a','b'],'label':['E','S']})
        with self.assertRaises(AssertionError):
            evaluate_pairs(j.iloc[:1].assign(rank=1),j,[1],{1:['a']+['u'+str(i) for i in range(20)]},'esci',{})

if __name__=='__main__':unittest.main()
