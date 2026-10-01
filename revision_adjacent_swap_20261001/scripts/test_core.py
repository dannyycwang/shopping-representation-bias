"""Scientific correctness tests; synthetic fixtures are never experiment results."""
import unittest
from common import *

class CoreTests(unittest.TestCase):
    def test_replacement_with_ties_and_self_exclusion(self):
        rng=np.random.default_rng(882)
        for n in [3,21,103]:
            for trial in range(10):
                base=rng.integers(-3,4,size=n).astype('float32')/7
                for target in range(n):
                    scores=np.r_[base[target],np.float32(-2),np.float32(2),rng.integers(-3,4,size=5).astype('float32')/7].astype('float32')
                    ranks=replacement(base,np.full(len(scores),target),scores)
                    for s,r in zip(scores,ranks):
                        actual=base.copy();actual[target]=s
                        self.assertEqual(int(r),list(np.argsort(-actual,kind='stable')).index(target)+1)
                        for k in [1,min(n-1,20),min(n-1,100)]:
                            j,threshold=boundary(base,target,k)
                            self.assertEqual(r<=k,s>threshold or s==threshold and target<j)
    def test_multiset_duplicates_adjacent_mapping(self):
        p=dict(product_id='a',title='Fixed 12',description='desc',attributes=['x: 1','x: 1','y: 2','z: 3'],
            attribute_separator=' | ',section_order=['title','description','attributes'])
        vs,noops=variants(p);self.assertEqual(noops,1);self.assertEqual(len(vs),2)
        self.assertEqual([v['positions'] for v in vs],[[2],[3]])
        for v in vs:self.assertTrue(audit(p,v)['pass'])
        bad=dict(vs[0]);bad['attributes']=['x: 1','y: 2','z: 3','x: 1']
        self.assertFalse(audit(p,bad)['adjacent_transposition'])
    def test_hash_and_no_swaps(self):
        self.assertEqual(selection_hash('query','wands',2),digest('["20261001","query","wands","2"]'))
        p=dict(product_id='a',title='Fixed',description='',attributes=['a','a'],attribute_separator='|',section_order=['title','attributes'])
        self.assertEqual(variants(p),([],1))

if __name__=='__main__':unittest.main()
