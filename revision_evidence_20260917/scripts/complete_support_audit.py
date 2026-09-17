from common import *

fit=pd.read_csv(DATA/'all_seven_token_lengths.csv',dtype={'product_id':str})
rows=[]
for ds in ['wands','esci']:
    old=pd.read_csv(ROOT/f'phase4/results/{ds}_product_features.csv',dtype={'product_id':str}).set_index('product_id').sort_index()
    bge=fit[(fit.dataset==ds)&(fit.model=='bge_base')].set_index('product_id').sort_index()
    mini=fit[(fit.dataset==ds)&(fit.model=='minilm')].set_index('product_id').sort_index()
    assert old.index.equals(bge.index)
    assert np.array_equal(old.max_tokens_seven,bge.max_tokens_seven)
    assert np.array_equal(old.fully_fits,bge.fully_fits)
    rows.append(dict(dataset=ds,historical_BGE_counts_exact_match=True,
       products_falsely_marked_fitting_if_BGE_mask_used_for_MiniLM=int((bge.fully_fits&~mini.fully_fits).sum())))
pd.DataFrame(rows).to_csv(DATA/'historical_fitting_mask_audit.csv',index=False)
integrity=pd.read_csv(DATA/'serialization_integrity.csv')
counts=integrity.groupby(['dataset','distinct_serializations_seven']).size().reset_index(name='products')
counts.to_csv(DATA/'serialization_diversity_summary.csv',index=False)
assert integrity.raw_seven_exact.all() and integrity.all_three_canonical_rules_invariant.all()

# An explicitly post-hoc illustrative case. It does not replace population evidence or the introduction.
t=pairread(ROOT/'phase3/results/target_only_permutations/wands_minilm_pairs.parquet')
f=fit[(fit.dataset=='wands')&(fit.model=='minilm')]
t=t.merge(f,on='product_id',validate='many_to_one')
t=t[t.fully_fits&t.query_id.isin(SPLITS['wands_heldout'])]
t=t[(t[STEMS].min(axis=1)<=20)&(t[STEMS].max(axis=1)>20)].copy()
t['range_seven']=t[STEMS].max(axis=1)-t[STEMS].min(axis=1)
t=t.sort_values(['range_seven','query_id','product_id'],ascending=[False,True,True])
if len(t):
    r=t.iloc[0];p=next(x for x in products('wands') if str(x['product_id'])==r.product_id)
    q=pd.read_csv(ROOT/'phase2/data/processed/wands_queries.csv').set_index('query_id').loc[int(r.query_id),'query']
    dump(DATA/'posthoc_fully_fitting_illustrative_case.json',dict(dataset='wands',model='minilm',query_id=int(r.query_id),product_id=r.product_id,
      query=q,product_title=p['title'],product_record=p,highest_label='Exact',cap=256,
      ranks={s:int(r[s]) for s in STEMS},tokens_with_special={s:int(r['tokens_'+s]) for s in STEMS},
      fully_fits_all_orders=True,competitors='full raw C0 catalog; target old entry removed',
      selection='Post hoc: largest seven-order target rank range among held-out WANDS/MiniLM all-order-fitting highest-label pairs crossing K=20; ties query_id then product_id. Illustrative only, not independent population evidence.',
      manuscript_action='No replacement of the Introduction example or its historical artifacts.'))
print('SUPPORT AUDIT COMPLETE',flush=True)
