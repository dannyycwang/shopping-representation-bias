from core import *
from analyze import source_choices
from bootstrap_stats import QueryBootstrap
import unittest

class Tests(unittest.TestCase):
    def product(self,attrs):return dict(product_id='x',title='red 19 inch remains',description='red remains here',
        **{'class':'item'},category='category',attributes=attrs,attribute_separator='|',section_order=['title','class','category','description','attributes'])
    def test_five_distinct_sets_and_untouched_blocks(self):
        p=self.product(['a:red','dup:x','dup:y','b:19 inch','c:yes','d:metal']);s,_=choose_five(p)
        self.assertEqual(len(s),5);self.assertEqual(len({(r['entry_index1'],r['entry_index2']) for r in s}),5)
        self.assertTrue(all(r['entry_index1'] not in [1,2] and r['entry_index2'] not in [1,2] for r in s))
        v=variants(p,s);self.assertEqual(len(v),11)
        for r in v:self.assertTrue(r['text'].startswith('red 19 inch remains\nitem\ncategory\nred remains here\n'))
        self.assertEqual(s,choose_five(p)[0])
    def test_fewer_than_five_and_missing(self):
        s,a=choose_five(self.product(['a:x','b:y','c:N/A','d:unknown']))
        self.assertEqual(s,[]);self.assertEqual(a['valid_pairs'],1)
    def test_source_only_ties_and_nested_budgets(self):
        f=pd.DataFrame([dict(query_id=1,product_id='p',variant=v,condition=v[0],j=j,score=s)
            for v,j,s in [('A',0,.5),('C1',1,.5),('C2',2,.7),('C3',3,.7),('C4',4,.6),('C5',5,.8),('O1',1,.99)]])
        self.assertEqual(source_choices(f,'C',1).iloc[0].variant,'A')
        self.assertEqual(source_choices(f,'C',3).iloc[0].variant,'C2')
        self.assertEqual(source_choices(f,'C',5).iloc[0].variant,'C5')
        self.assertEqual(source_choices(f,'O',1).iloc[0].variant,'O1')
    def test_cluster_ratio_keeps_all_targets(self):
        b=QueryBootstrap.__new__(QueryBootstrap);b.queries=[1,2];b.index={1:0,2:1}
        b.weights=np.array([[2,0],[1,1],[0,2]],dtype=float)
        # Query 1 has two pairs, query 2 one; result is pair weighted, not query-mean.
        stats,draws=b.estimate([1,1,2],[1,0,0],[1,1,1])
        self.assertEqual(stats['estimate'],1/3);np.testing.assert_array_equal(draws,[.5,1/3,0])
    def test_rank_target_ties(self):
        base=np.array([.7,.7,.6,.5],dtype='float32')
        for ix in range(4):
            for value in [.4,.5,.6,.7,.8]:
                scores=base.copy();scores[ix]=value
                expected=np.flatnonzero(np.argsort(-scores,kind='stable')==ix)[0]+1
                self.assertEqual(rank_target(base,np.array([ix]),np.array([value],dtype='float32'))[0],expected)

def audit_design():
    freeze_check();t=Timer();products,queries,j=load();pairs=frame(HERE/'data/selected_pairs.csv',dtype={'product_id':str})
    maps=frame(HERE/'data/selected_swaps.csv',dtype={'product_id':str});full=pd.read_parquet(HERE/'data/full_inputs.parquet').set_index(['product_id','variant'])
    dev=frame(HERE/'data/development_pairs.csv',dtype={'product_id':str});assert set(pairs.query_id).isdisjoint(dev.query_id)
    assert pairs.groupby('query_id').size().max()<=5 and not pairs.duplicated(['query_id','product_id']).any()
    exact=set(zip(j.loc[j.label.eq('Exact'),'query_id'],j.loc[j.label.eq('Exact'),'product_id']))
    assert set(zip(pairs.query_id,pairs.product_id))<=exact
    audit=[]
    for pid,g in maps.groupby('product_id',sort=False):
        p=products[int(g.iloc[0].catalog_index)];chosen,_=choose_five(p)
        for expected,record in zip(chosen,g.to_dict('records')):assert all(record[k]==v for k,v in expected.items())
        for r in variants(p,chosen):
            actual=full.loc[(pid,r['variant'])];assert actual.text==r['text'] and actual.text_sha256==digest(r['text'])
            audit.append(dict(product_id=pid,variant=r['variant'],lossless_intervention=True,unselected_content_unchanged=True))
    csv('qa/intervention_audit.csv',audit)
    for model in ['gte','harrier']:
        a=frame(HERE/f'data/token_audit_{model}.csv',dtype={'product_id':str});assert a[a.product_id.isin(set(pairs.product_id))].fits.all()
    timed('qa/design_validation.json',t,passed=True,pairs=len(pairs),products=maps.product_id.nunique(),versions_checked=len(audit))

if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Tests);result=unittest.TextTestRunner().run(suite)
    assert result.wasSuccessful();audit_design();print('DESIGN VERIFIED',flush=True)
