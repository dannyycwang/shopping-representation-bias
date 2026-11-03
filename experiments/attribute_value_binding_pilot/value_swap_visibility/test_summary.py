"""Protect Top-20 denominators and signs with boundary and empty-risk-set cases."""
import unittest
from analyze import summarize
from common import pd

class SummaryTests(unittest.TestCase):
    def fixture(self,a,c):
        f=pd.DataFrame(dict(query_id=range(len(a)),product_id=range(len(a)),rank_A=a,rank_C=c))
        f['rank_delta']=f.rank_C-f.rank_A
        f['score_delta']=-f.rank_delta/100
        f['in_top20_A']=f.rank_A.le(20)
        f['top20_entry']=f.rank_A.gt(20)&f.rank_C.le(20)
        f['top20_exit']=f.rank_A.le(20)&f.rank_C.gt(20)
        f['direction']=f.rank_delta.map(lambda d:'improved' if d<0 else 'declined' if d>0 else 'unchanged')
        return f
    def test_boundary_and_risk_set_denominators(self):
        s=summarize(self.fixture([1,20,21,30,100,3],[2,21,20,2,100,1]),'test')
        self.assertEqual((s['improved_n'],s['declined_n'],s['unchanged_n']),(3,2,1))
        self.assertEqual((s['entry_n'],s['entry_den']),(2,3))
        self.assertEqual((s['exit_n'],s['exit_den']),(1,3))
        self.assertEqual((s['crossing_n'],s['crossing_den']),(3,6))
    def test_zero_denominator_is_na(self):
        outside=summarize(self.fixture([21,100],[22,99]),'test')
        self.assertIsNone(outside['exit_rate']);self.assertEqual(outside['exit_den'],0)
        inside=summarize(self.fixture([1,2],[3,2]),'test')
        self.assertIsNone(inside['entry_rate']);self.assertEqual(inside['entry_den'],0)

if __name__=='__main__':unittest.main()
