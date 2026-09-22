from common import *
import argparse
from collections import Counter

def main():
 verify();plan=pd.read_parquet(HERE/'data/coverage_plan.parquet');old=pd.read_csv(OLD/'data/all_seven_token_lengths.csv',dtype={'product_id':str});checks=[]
 frozen=read(HERE/'data/COVERAGE_PLAN_FROZEN.json');assert sha(HERE/'data/coverage_plan.parquet')==frozen['plan_sha256']
 for ds in ['wands','esci']:
  pp=plan[plan.dataset==ds];p,_=load(ds);lookup={str(x['product_id']):x for x in p}
  assert not pp.duplicated(['product_id','schedule']).any()
  for model in ['minilm','bge_base']:
   prior=old[(old.dataset==ds)&(old.model==model)].set_index('product_id')
   for s in STEMS:
    g=pp[pp.schedule==s];assert np.array_equal(g['tokens_'+model],prior.loc[g.product_id,'tokens_'+s])
   checks.append(dict(dataset=ds,model=model,old_seven_token_counts_identical=True,targets=pp.product_id.nunique(),schedule_records=len(pp)))
  if ds=='wands':assert pp.groupby('product_id').size().eq(32).all()
  else:
   ex=pp[pp.schedule.str.startswith('E')];assert ex.attribute_count.max()<=3
   assert ex.groupby('product_id').size().le(6).all()
   assert ex.groupby('product_id').text_sha256.nunique().equals(ex.groupby('product_id').size())
  # Recreate every schedule and audit whole-entry multiset; text/encoding outcomes do not influence these checks.
  from coverage_prepare import text_for
  for r in pp.itertuples():
   product=lookup[r.product_id];assert textsha(text_for(product,r.schedule))==r.text_sha256
   if not r.schedule.startswith('E'):assert Counter(ordered_attrs(product,r.schedule))==Counter(product['attributes'])
 dump(HERE/'qa/coverage_plan_validation.json',dict(checks=checks,all_original_token_lengths_match=True,whole_entry_multisets_preserved=True,all_text_hashes_match=True,esci_at_most_six_distinct_orders=True))
 print('COVERAGE PLAN VALIDATED',flush=True)

if __name__=='__main__':main()
