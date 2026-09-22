from common import *
import argparse
from collections import Counter

def boundary_context():
 prov=provenance();diff=pd.read_csv(HERE/'qa/boundary_rank_discrepancies.csv',dtype={'product_id':str});out=[]
 for r in diff.itertuples():
  p,q=load(r.dataset);pi={str(v['product_id']):i for i,v in enumerate(p)};qi={int(v):i for i,v in enumerate(q.query_id)};pr=prov[(prov.dataset==r.dataset)&(prov.model==r.model)].set_index('schedule')
  qv=np.load(ROOT/pr.loc['C0','query_embedding']);pv=np.load(ROOT/pr.loc['C0','product_embedding'],mmap_mode='r');base=qv@pv.T;bb=base[qi[r.query_id]];idx=pi[r.product_id];order=np.argsort(-bb,kind='stable');other=order[order!=idx]
  for pos in range(max(0,r.saved_rank-4),min(len(other),r.reconstructed_rank+3)):
   j=other[pos];out.append(dict(dataset=r.dataset,model=r.model,query_id=r.query_id,product_id=r.product_id,schedule=r.schedule,saved_rank=r.saved_rank,reconstructed_rank=r.reconstructed_rank,competitor_rank_excluding_target=pos+1,competitor_id=str(p[j]['product_id']),competitor_index=int(j),competitor_score=float(bb[j]),target_score=r.score,competitor_minus_target=float(bb[j])-r.score,fp32_ulps=abs(int(np.float32(bb[j]).view('int32'))-int(np.float32(r.score).view('int32')))))
 pd.DataFrame(out).to_csv(HERE/'qa/boundary_discrepancy_neighbors.csv',index=False)
 prior=pd.read_csv(OLD/'data/target_rank_precision_differences.csv',dtype={'product_id':str})
 prior.to_csv(HERE/'qa/inherited_precision_differences.csv',index=False)

def xai_quality():
 qa=pd.read_csv(HERE/'qa/xai_forward_completeness.csv',dtype={'product_id':str});entry=pd.read_parquet(HERE/'data/xai_entry_attributions.parquet');token=pd.read_parquet(HERE/'data/xai_token_attributions.parquet');panel=pd.read_csv(HERE/'XAI_CASE_MANIFEST.csv',dtype={'product_id':str})
 assert len(panel)==25 and len(qa)==100
 assert qa.groupby(['model','query_id','product_id']).size().eq(4).all()
 rows=[]
 for key,g in entry.groupby(['model','query_id','product_id','variant']):
  z=g.pivot_table(index=['entry_id','entry_text'],columns='baseline',values='attribution',aggfunc='sum',fill_value=0)
  both=z[['zero','pad']];active=(both.abs().sum(1)>1e-12)
  corr=both.loc[active].corr().iloc[0,1] if active.sum()>1 and both.loc[active].std().gt(0).all() else np.nan
  rows.append(dict(model=key[0],query_id=key[1],product_id=key[2],variant=key[3],entries=int(len(z)),active_entries=int(active.sum()),zero_pad_entry_pearson=corr,sign_agreement=float((np.sign(z.loc[active,'zero'])==np.sign(z.loc[active,'pad'])).mean()) if active.any() else np.nan,absolute_entry_difference=float(abs(z['zero']-z['pad']).sum())))
 pd.DataFrame(rows).to_csv(HERE/'qa/xai_baseline_per_variant.csv',index=False)
 changes=[]
 for key,g in entry.groupby(['model','query_id','product_id']):
  z=g.pivot_table(index=['entry_id','entry_text'],columns=['baseline','variant'],values='attribution',aggfunc='sum',fill_value=0)
  dz=z[('zero','maximum')]-z[('zero','minimum')];dp=z[('pad','maximum')]-z[('pad','minimum')];active=dz.abs()+dp.abs()>1e-12
  e=pd.DataFrame(dict(entry_id=z.index.get_level_values('entry_id'),entry_text=z.index.get_level_values('entry_text'),zero_delta=dz.to_numpy(),pad_delta=dp.to_numpy())).assign(model=key[0],query_id=key[1],product_id=key[2]);changes.append(e)
 pd.concat(changes).to_parquet(HERE/'data/xai_aligned_entry_changes.parquet',index=False)
 crows=[]
 for key,g in pd.concat(changes).groupby(['model','query_id','product_id']):
  active=g.zero_delta.abs()+g.pad_delta.abs()>1e-12;gg=g[active];corr=gg[['zero_delta','pad_delta']].corr().iloc[0,1] if len(gg)>1 else np.nan
  q=qa[(qa.model==key[0])&(qa.query_id==key[1])&(qa.product_id==key[2])]
  crows.append(dict(model=key[0],query_id=key[1],product_id=key[2],display=q.display.iloc[0],zero_pad_change_pearson=corr,change_sign_agreement=float((np.sign(gg.zero_delta)==np.sign(gg.pad_delta)).mean()),zero_change_l1=float(g.zero_delta.abs().sum()),pad_change_l1=float(g.pad_delta.abs().sum()),both_variant_boundary_guard=bool(q.variant_boundary_guard.all()),all_completeness_passed=bool(q.completeness_passed.fillna(False).all())))
 pd.DataFrame(crows).to_csv(HERE/'qa/xai_baseline_per_pair.csv',index=False)
 # Completeness and accounting are independent checks.
 keys=['model','query_id','product_id','variant','baseline']
 sums=token.groupby(keys).attribution.sum();esums=entry.groupby(keys).attribution.sum();assert np.allclose(sums,esums,atol=1e-10,rtol=0)
 completed=qa[qa.status.isin(['passed','completeness_failed'])].set_index(keys).sort_index()
 assert np.allclose(sums.sort_index(),completed.attribution_sum,atol=1e-10,rtol=0)
 recomputed=(completed.completeness_residual.abs()<=np.maximum(1e-4,.01*completed.score_difference.abs()))
 assert np.array_equal(recomputed,completed.completeness_passed)
 fails=qa[qa.status.ne('passed')].replace({np.nan:None}).to_dict('records');dump(HERE/'qa/xai_failures.json',fails)
 dump(HERE/'qa/xai_accounting.json',dict(pairs=25,variants=50,baseline_runs=len(qa),completed_runs=len(completed),completeness_passes=int(qa.completeness_passed.fillna(False).sum()),failures=len(fails),all_cases_retained=True,token_rows=len(token),entry_rows=len(entry),token_entry_sum_match=True,max_forward_ids_embeds_error=float(qa.input_ids_vs_embeds_error.max()),max_diagnostic_native_error=float(qa.diagnostic_native_error.max()),changed_diagnostic_inclusions=int(qa.drop_duplicates(keys[:-1]).eval('diagnostic_inclusion != native_inclusion').sum()),steps_counts={str(k):int(v) for k,v in qa.steps.value_counts().items()}))

def compact_table():
 f=pd.read_csv(HERE/'tables/control_transitions.csv');f=f[(f.K==20)&f.aggregation.eq('query_macro')];rows=[]
 for key,g in f.groupby(['dataset','model','rule'],sort=False):
  row=dict(dataset=key[0],model=key[1],rule=key[2])
  for fam in ['dense','hybrid']:
   z=g[g.family==fam].set_index(['reference_state','control_state'])
   for label,cell in [('always_in_lost',('persistent_inclusion','omitted')),('always_out_gained',('persistent_omission','included')),('crossing_included',('crossing','included')),('crossing_omitted',('crossing','omitted'))]:
    row[fam+'_'+label]=float(z.loc[cell,'cell_fraction'])
  rows.append(row)
 df=pd.DataFrame(rows);df.to_csv(HERE/'tables/main_control_transitions.csv',index=False)
 lines=['| Dataset / encoder / rule | Dense lost | gained | crossing in | out | Hybrid lost | gained | crossing in | out |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
 short={'lexical_ascending':'asc','lexical_descending':'desc','field_priority_type':'field'}
 for r in rows:lines.append('| '+r['dataset']+' / '+r['model']+' / '+short[r['rule']]+' | '+' | '.join(f'{r[c]*100:.2f}' for c in df.columns[3:])+' |')
 (HERE/'tables/main_control_transitions.md').write_text('\n'.join(lines)+'\n\nAll cells are percentage points of highest-label support, query macro, K=20. Lost = reference always-in but fixed-control omitted; gained = reference always-out but fixed-control included. Crossing columns partition reference crossing mass. Dense and hybrid use their respective raw catalog-wide seven-schedule references. Full six-cell tables, denominators, conditional rates and uncorrected query-cluster intervals are in control_transitions.csv.\n',encoding='utf8')
 tex=['\\begin{tabular}{lllrrrrrrrr}','\\toprule','Dataset & Encoder & Rule & \\multicolumn{4}{c}{Dense} & \\multicolumn{4}{c}{Hybrid} \\\\', ' & & & Lost & Gained & Cross in & Cross out & Lost & Gained & Cross in & Cross out \\\\','\\midrule']
 for r in rows:tex.append(' & '.join([r['dataset'].upper(),r['model'].replace('_','\\_'),short[r['rule']]]+[f'{r[c]*100:.2f}' for c in df.columns[3:]])+' \\\\')
 tex+=['\\bottomrule','\\end{tabular}'];(HERE/'tables/main_control_transitions.tex').write_text('\n'.join(tex)+'\n',encoding='utf8')

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--xai',action='store_true');a=ap.parse_args();boundary_context();compact_table()
 if a.xai:xai_quality()
 print('SUPPLEMENT QA COMPLETE',flush=True)
