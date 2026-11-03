from common import *
import unittest

class Tests(unittest.TestCase):
    def product(self,attrs):return dict(product_id=5,title='purple 19 inch',description='purple remains in description',
        **{'class':'item'},category='category',attributes=attrs,attribute_separator='|',section_order=['title','class','category','description','attributes'])
    def test_swap_only_selected_entries(self):
        p=self.product(['size:19 inch','dup:x','dup:y','color: purple '])
        s,_=select_pair(p);self.assertEqual([s['entry_index1'],s['entry_index2']],[0,3])
        a,c=swap(p,s)
        self.assertEqual(c,'purple 19 inch\nitem\ncategory\npurple remains in description\nsize: purple |dup:x|dup:y|color:19 inch')
    def test_duplicates_not_whole_product_exclusion(self):
        p=self.product(['a:one',' A :two','b:yes','c:no'])
        s,_=select_pair(p);self.assertEqual((s['entry_index1'],s['entry_index2']),(2,3))
    def test_missing_and_no_type_whitelist(self):
        p=self.product(['a:N/A','b:unknown','c:yes','d:0'])
        s,_=select_pair(p);self.assertEqual({s['value1_A'],s['value2_A']},{'yes','0'})
        self.assertIsNone(select_pair(self.product(['a: red','b:RED ']))[0])
    def test_first_colon_and_seed(self):
        p=self.product(['anything:a:b','another:some arbitrary text'])
        s,_=select_pair(p);self.assertEqual(s,select_pair(p)[0]);self.assertIn('a:b',[s['value1_A'],s['value2_A']])
        swap(p,s)
    def test_target_formula_with_ties(self):
        rng=np.random.default_rng(SEED)
        for _ in range(25):
            base=rng.integers(-3,4,31).astype('float32')
            for ix in [0,12,30]:
                for value in range(-4,5):
                    new=base.copy();new[ix]=value
                    expected=np.flatnonzero(np.argsort(-new,kind='stable')==ix)[0]+1
                    actual=rank_target(base,np.array([ix]),np.array([value],dtype='float32'))[0]
                    self.assertEqual(expected,actual)

if __name__=='__main__':unittest.main()
