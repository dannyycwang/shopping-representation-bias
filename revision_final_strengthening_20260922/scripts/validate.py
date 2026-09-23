"""Completion checks: required scientific invariants, traceability and publication scope."""
from common import *
import argparse
def initial():
 verify();manifest=read(HERE/'INPUT_MANIFEST.json');assert sha(HERE/'INPUT_MANIFEST.json')==read(HERE/'PROTOCOL.json')['input_manifest_sha256']
 for r in manifest['files']:
  path=Path(r['path']);path=path if path.is_absolute() else ROOT/path
  assert path.exists() and sha(path)==r['sha256'],str(path)
 for name,digest in manifest['preserved_working_files'].items():assert sha(ROOT/name)==digest,name
 return dict(frozen_protocol=True,input_hashes_checked=len(manifest['files']),preexisting_files_preserved=len(manifest['preserved_working_files']))
def final():
 result=initial();ba=pd.read_csv(HERE/'qa/boundary_reconstruction.csv')
 assert len(ba)==6 and ba.pairs.tolist()==[21299]*3+[4434]*3 and ba.boundary_mismatches.sum()==0 and ba.rank_mismatches.sum()==1
 prev=pd.read_csv(HERE/'qa/inherited_precision_differences.csv',dtype={'product_id':str});now=pd.read_csv(HERE/'qa/boundary_rank_discrepancies.csv',dtype={'product_id':str})
 assert len(prev)==len(now)==1
 for c in ['dataset','model','schedule','query_id','product_id','reconstructed_rank']:assert prev[c].iloc[0]==now[c].iloc[0]
 assert prev.inherited_rank.iloc[0]==now.saved_rank.iloc[0]
 result['boundary']=dict(records=int(ba.records.sum()),supports_correct=True,no_cutoff_contradiction=True,sole_rank_difference_matches_inherited=True)
 panel=pd.read_csv(HERE/'XAI_CASE_MANIFEST.csv',dtype={'product_id':str});variants=pd.read_csv(HERE/'data/xai_variant_manifest.csv',dtype={'product_id':str});x=read(HERE/'qa/xai_accounting.json');assert len(panel)==25 and len(variants)==50 and x['baseline_runs']==100 and x['token_entry_sum_match']
 assert sha(HERE/'XAI_CASE_MANIFEST.csv')==read(HERE/'PROTOCOL.json')['B']['panel_sha256']
 assert variants.fully_fitting.all() and variants.groupby(['model','query_id','product_id']).size().eq(2).all()
 result['xai']=x
 c=pd.read_csv(HERE/'tables/control_transitions.csv');ident=pd.read_parquet(HERE/'data/control_transition_identities.parquet');pq=pd.read_parquet(HERE/'data/control_transitions_per_query.parquet')
 group=['dataset','model','family','rule','K'];assert len(c.groupby(group))==48 and len(c)==576
 for key,g in c.groupby(group+['aggregation']):assert len(g)==6 and abs(g.cell_fraction.sum()-1)<1e-12
 assert not ident.duplicated(group+['query_id','product_id']).any()
 ig=ident.groupby(group+['query_id','reference_state','control_state']).size();expected=pq.set_index(group+['query_id','reference_state','control_state']).cell_pairs
 assert np.array_equal(ig.reindex(expected.index,fill_value=0),expected)
 assert pq.groupby(group+['query_id']).cell_fraction.sum().sub(1).abs().max()<1e-12
 # Recompute included-column Recall directly from inherited fixed-control records.
 inherited=pd.read_parquet(GRADED/'data/per_query.parquet')
 for key,g in pq.groupby(group):
  ds,model,family,rule,k=key;method=('canonical_hybrid_' if family=='hybrid' else '')+rule
  old=inherited[(inherited.dataset==ds)&inherited.model.eq(model)&inherited.method.eq(method)].set_index('query_id')
  col=g[g.control_state.eq('included')].groupby('query_id').cell_fraction.sum()
  actual=old.loc[col.index,f'included_ids_K{k}'].map(lambda s:len(json.loads(s)))/old.loc[col.index,'highest_support_ids'].map(lambda s:len(json.loads(s)))
  assert np.allclose(col,actual,rtol=0,atol=1e-12)
  table=c.copy()
  for name,value in zip(group,key):table=table[table[name].eq(value)]
  for agg in ['query_macro','pair_micro']:
   expected_mass=g.groupby(['reference_state','control_state']).cell_fraction.mean() if agg=='query_macro' else g.groupby(['reference_state','control_state']).cell_pairs.sum()/g.groupby('query_id').Hq.first().sum()
   actual_mass=table[table.aggregation.eq(agg)].set_index(['reference_state','control_state']).cell_fraction
   assert np.allclose(actual_mass.sort_index(),expected_mass.sort_index(),rtol=0,atol=1e-12)
 result['transitions']=dict(configurations=48,summary_rows=len(c),identity_records=len(ident),all_cells_partition=True,fixed_recall_verified=True)
 plan=read(HERE/'qa/coverage_plan_validation.json');assert plan['all_original_token_lengths_match'] and plan['all_text_hashes_match']
 frozen=read(HERE/'data/COVERAGE_PLAN_FROZEN.json')
 for name,key in [('coverage_plan','plan_sha256'),('coverage_input_aliases','alias_sha256'),('coverage_common_support','support_sha256')]:assert sha(HERE/f'data/{name}.parquet')==frozen[key]
 cov=pd.read_parquet(HERE/'data/coverage_pair_states.parquet');supports=pd.read_parquet(HERE/'data/coverage_common_support.parquet');summary=pd.read_csv(HERE/'tables/coverage_summary.csv');validation=read(HERE/'qa/coverage_validation.json')
 assert len(validation)==4 and all(r['boundary_discrepancies']==0 and r['monotonicity_passed'] for r in validation)
 for key,g in cov.groupby(['dataset','model','support','K']):
  ds,model,scope,k=key;families=['7','16','32'] if ds=='wands' else ['7','exhaustive_target'];last=None
  h=supports[(supports.dataset==ds)&supports.model.eq(model)];h=h if scope=='full' else h[h.fully_fitting]
  ix=pd.MultiIndex.from_frame(h[['query_id','product_id']]).sort_values()
  for fam in families:
   curr=g[g.family.eq(fam)].set_index(['query_id','product_id']).sort_index();assert curr.index.equals(ix)
   assert np.allclose(curr[['persistent_inclusion','VI','persistent_omission']].sum(1),1)
   if last is not None:
    assert curr.VI.ge(last.VI).all() and curr.persistent_inclusion.le(last.persistent_inclusion).all() and curr.persistent_omission.le(last.persistent_omission).all()
   last=curr
   for metric in ['persistent_inclusion','VI','persistent_omission','tested_order_inclusion_frequency']:
    for agg in ['query_macro','pair_micro']:
     expected_mean=curr[metric].groupby(level='query_id').mean().mean() if agg=='query_macro' else curr[metric].mean()
     row=summary[(summary.dataset==ds)&summary.model.eq(model)&summary.support.eq(scope)&summary.K.eq(k)&summary.family.eq(fam)&summary.metric.eq(metric)&summary.aggregation.eq(agg)]
     assert len(row)==1 and abs(float(row['mean'].iloc[0])-expected_mean)<1e-12
   if fam=='7':
    orig=pd.read_parquet(HERE/f'data/{ds}_{model}_boundary_records.parquet');orig=orig[orig.K.eq(k)].set_index(['query_id','product_id'])
    originals=orig.loc[ix].groupby(level=[0,1]).inclusion
    expected_vi=(originals.max()&~originals.min()).astype(float)
    assert np.array_equal(expected_vi.sort_index(),curr.VI)
  rows=summary[(summary.dataset==ds)&summary.model.eq(model)&summary.support.eq(scope)&summary.K.eq(k)]
  assert rows.queries.nunique()==rows.pairs.nunique()==1
 result['coverage']=dict(datasets_models=4,fixed_supports=True,original_seven_states_match_A=True,monotonicity_per_pair=True,all_planned_inputs_encoded=True,statistics_rows=len(summary))
 resume=(HERE/'qa/coverage_resume_minilm.log').read_text(encoding='utf8');assert 'ENCODING COMPLETE minilm' in resume and '\nENCODED ' not in resume
 result['cache_resume']=dict(tested=True,reused_blocks=sum(line.startswith('CACHE ') for line in resume.splitlines()),new_blocks_encoded=0)
 required=['README.md','BOUNDARY_AUDIT.md','XAI_AUDIT.md','CONTROL_TRANSITIONS.md','PERMUTATION_COVERAGE.md','CLAIM_VERDICTS.md','CHAPTER6_EVIDENCE_MAP.md','REPRODUCE.md']
 for n in required:assert (HERE/n).stat().st_size>100
 for name in ['main_query_conditioned_cases','supplement_french_molding','supplement_permutation_coverage']:
  for ext in ['pdf','svg','png']:assert (HERE/f'figures/{name}.{ext}').stat().st_size>1000
 assert read(HERE/'qa/visual_qa.json')['inspected']
 result['deliverables']=dict(required_reports=len(required),figures=3,formats=['pdf','svg','png'],main_figure_budget=1,main_table_budget=1,visual_inspection=True)
 for r in read(HERE/'SUPPLEMENTAL_SOURCE_MANIFEST.json'):assert sha(ROOT/r['path'])==r['sha256']
 # Payloads are included in cache inventory, not in Git's artifact manifest.
 caches=[];outputs=[]
 for path in sorted(HERE.rglob('*')):
  if not path.is_file() or '__pycache__' in path.parts:continue
  rel=path.relative_to(HERE).as_posix()
  if rel.endswith(('.npz','.npy')) and rel.startswith('cache/'):
   caches.append(dict(path=rel,bytes=path.stat().st_size,sha256=sha(path),local_available=True,git_distributed=False));continue
  if rel in ['OUTPUT_MANIFEST.json','CACHE_MANIFEST.json','qa/final_validation.json','qa/final_validation.log']:continue
  assert path.stat().st_size<100*1024**2,rel
  outputs.append(dict(path=rel,bytes=path.stat().st_size,sha256=sha(path),git_distributed=True))
 dump(HERE/'CACHE_MANIFEST.json',dict(payloads=caches,restore='Local ignored payloads remain available; rebuild new vectors using coverage_encode.py, restore exact original input payload hashes separately.',total_bytes=sum(r['bytes'] for r in caches)))
 dump(HERE/'OUTPUT_MANIFEST.json',dict(files=outputs,exclusions=['manifests themselves','qa/final_validation.json and .log','ignored generated vector/score payloads','__pycache__'],total_bytes=sum(r['bytes'] for r in outputs)))
 result['output_files']=len(outputs);result['ignored_cache_files']=len(caches);result['completion']='A-D executed; all selected cases accounted including failures; no resource blocks';dump(HERE/'qa/final_validation.json',result)
 print('FINAL VALIDATION PASS',json.dumps(result),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--initial',action='store_true');a=ap.parse_args()
 if a.initial:dump(HERE/'qa/input_recheck.json',initial());print('INPUT HASHES PASS',flush=True)
 else:final()
