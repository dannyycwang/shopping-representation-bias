import json, unittest
from pathlib import Path
from core import classify, bounds, joint_status, product_type, published_rank
P=json.loads((Path(__file__).parent/'protocol.json').read_text(encoding='utf8'))
def p(attrs,description='',title='item',cls=''):return dict(attributes=attrs,description=description,title=title,**{'class':cls})
def c(kind='color',value='white',scope='whole',modifier='pillow'):return dict(constraint_type=kind,normalized_value=value,scope=scope,modifier=modifier)
class SemanticTests(unittest.TestCase):
 def test_unknown_outside_rank_not_fabricated(self):
  self.assertEqual(published_rank({'a':6},'b'),'>20')
  self.assertEqual(published_rank({'a':6,'b':69},'b'),69)
 def test_taxonomy_does_not_invent_contradiction(self):
  self.assertEqual(product_type(p([],title='traditional rocking chair',cls='Patio Rockers & Gliders'),'chair')['status'],'MATCH')
  self.assertEqual(product_type(p([],title='porch swing',cls='Porch Swings|Hammocks'),'chair')['status'],'UNKNOWN')
 def test_missing_not_contradiction(self):self.assertEqual(classify(p([]),c(),P)['status'],'UNKNOWN')
 def test_description_variant_and_structural_scope(self):
  self.assertEqual(classify(p(['material: wool'],'Many of the designs in this collection are accented with viscose.'),c('material','wool','whole','rug'),P)['status'],'AMBIGUOUS')
  self.assertEqual(classify(p(['framematerial: solid wood'],'A strong aluminum rocking base provides stability.'),c('material','wood','frame','chair'),P)['status'],'AMBIGUOUS')
 def test_mixed_and_variant_uncertainty(self):
  for attrs in [['color: white/blue'],['color: white','color: purple']]:self.assertEqual(classify(p(attrs),c(),P)['status'],'AMBIGUOUS')
 def test_shared_unknown_cancels(self):
  s={'a':'MATCH','b':'CONTRADICTION','u':'UNKNOWN'};x=bounds(['a','u'],['b','u'],s)
  self.assertEqual(x['loose_bounds'],[-2,0]);self.assertEqual(x['tight_bounds'],[-1,-1])
 def test_full_set_and_exact_subset_differ(self):
  s={'a':'MATCH','b':'CONTRADICTION','x':'CONTRADICTION','y':'MATCH'}
  self.assertEqual(bounds(['a','x'],['b','y'],s)['confirmed_delta'],0)
  self.assertEqual(bounds(['a'],['b'],s)['confirmed_delta'],-1)
 def test_frame_not_whole_chair(self):
  x=p(['framematerial: plastic/resin'],'textured wood grain','weather wood chair','Adirondack Chairs')
  self.assertEqual(classify(x,c('material','wood','frame','chair'),P)['status'],'CONTRADICTION')
  self.assertEqual(classify(x,c('material','wood','whole','chair'),P)['status'],'UNKNOWN')
 def test_text_conflict_not_ignored(self):self.assertEqual(classify(p(['color: white'],'a blue pillow'),c(),P)['status'],'AMBIGUOUS')
 def test_cover_is_not_complete_pillow(self):self.assertEqual(product_type(p(['producttype: pillow cover'],title='white pillow cover',cls='Accent Pillows'),'pillow')['status'],'CONTRADICTION')
 def test_joint_does_not_ignore_unassessed(self):
  self.assertEqual(joint_status(['MATCH'],'MATCH',[],unassessed=True),'UNKNOWN')
  self.assertEqual(joint_status(['MATCH'],'MATCH',[],scope_narrowed=True),'AMBIGUOUS')
 def test_disappearance_requires_resolved_after(self):
  self.assertFalse(bounds(['a'],['u'],{'a':'MATCH','u':'UNKNOWN'})['disappearance'])
 def test_wood_engineered_distinction(self):self.assertEqual(classify(p(['framematerial: manufactured wood']),c('material','wood','frame','chair'),P)['status'],'AMBIGUOUS')
 def test_nominal_size_no_numeric_conversion(self):
  self.assertEqual(classify(p(['mattresssize: full / double']),c('nominal_size','full','bedding','bed'),P)['status'],'MATCH')
  self.assertEqual(classify(p(['mattresssize: california king']),c('nominal_size','king','bedding','bed'),P)['status'],'CONTRADICTION')
if __name__=='__main__':unittest.main()
