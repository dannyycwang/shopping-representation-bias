import unittest,random,json
from collections import Counter
import run as r
class Integrity(unittest.TestCase):
    def test_author_renderer_fresh_and_protected(self):
        seg=[('title','title','T')]+[(f'field{i}',f'field{i}',str(i)) for i in range(10)]
        rng=random.Random(42);out=[r.render_segments(seg,r.SCHEMA,rng,True,.15,{'field0'}) for _ in range(100)]
        self.assertGreater(len(set(out)),90)
        self.assertTrue(all('title: T' in x and 'field0: 0' in x for x in out))
        self.assertTrue(any('field1: 1' not in x for x in out))
        self.assertEqual(len(seg),11)
    def test_duplicate_atoms_and_native_keys(self):
        p={'title':'title','attributes':['color:blue','color:blue','free value']}
        seg=r.segments(p);self.assertEqual(seg.count(('color','color','blue')),2)
        self.assertIn(('attributes','attributes','free value'),seg)
    def test_seven_fact_inventory(self):
        p,*_=r.s.data()
        for x in p[::430]:
            base=r.labeled_build(x,r.s.CFG['primary_variant_family'][0],r.s.CFG['attribute_permutation_seeds'])
            for c in r.s.CFG['primary_variant_family']:
                self.assertEqual(Counter(base),Counter(r.labeled_build(x,c,r.s.CFG['attribute_permutation_seeds'])))
    def test_training_support_and_negatives(self):
        c=r.config();_,_,j,_,_=r.s.data();labels={(int(x.query_id),str(x.product_id)):x.label for x in j.itertuples()}
        for x in json.loads((r.HERE/'triples.json').read_text()):
            self.assertIn(x['query_id'],c['train']);self.assertNotIn(x['query_id'],c['test']);self.assertEqual(labels[x['query_id'],x['positive']],'Exact')
            self.assertEqual(len(set(x['negatives'])),3)
            for pid in x['negatives']:self.assertEqual(labels[x['query_id'],pid],'Irrelevant')
    def test_budget_and_freeze(self):
        c=r.config();self.assertEqual(c['batch_size'],128);self.assertEqual(c['field_dropout'],.15);self.assertEqual(c['epochs'],5)
        self.assertEqual(r.s.sha(r.HERE/'triples.json'),c['triples_sha256'])
    def test_title_only_amendment(self):
        rows=json.loads((r.HERE/'triples.json').read_text())
        old=json.loads((r.HERE/'archive/before_title_only/triples.json').read_text())
        self.assertEqual(len(rows),len(old))
        for a,b in zip(rows,old):
            self.assertEqual(a['protect'],['title'])
            self.assertEqual({k:v for k,v in a.items() if k!='protect'},{k:v for k,v in b.items() if k!='protect'})
        self.assertEqual(r.protected('blue cotton chair',{}),{'title'})
if __name__=='__main__':unittest.main()
