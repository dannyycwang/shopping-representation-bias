from common import *
from datetime import datetime,timezone
from huggingface_hub import snapshot_download
import importlib.metadata

def main():
 assert not (HERE/'PROTOCOL.json').exists(),'Do not overwrite frozen protocol.'
 for name in ['data','qa','tables','figures','cache','inputs']:(HERE/name).mkdir(exist_ok=True)
 sources=set();prior={e['path']:e.get('sha256') for e in read(GRADED/'data/input_manifest.json')['files']}
 def add(p):sources.add(ROOT/p if not Path(p).is_absolute() else Path(p))
 paths=['revision_graded_20260922/README.md','revision_graded_20260922/GRADED_METRIC_AUDIT.md','revision_graded_20260922/CLAIM_VERDICTS.md','revision_graded_20260922/PROTOCOL.json','revision_graded_20260922/tables/effectiveness_wide.csv','revision_graded_20260922/tables/paired_contrasts.csv','revision_graded_20260922/data/per_query.parquet','revision_graded_20260922/data/rq2_membership_ndcg_per_query.parquet','revision_graded_20260922/data/persistent_states_per_query.parquet','revision_graded_20260922/data/input_manifest.json','revision_completion_20260921/CLAIM_VERDICTS.md','revision_evidence_20260917/data/encoder_profiles_and_rank_sources.csv','revision_evidence_20260917/scripts/verify_target_rank_sources.py','revision_evidence_20260917/figure_revision/data/figure1_case_audit.json','revision_evidence_20260917/data/posthoc_fully_fitting_illustrative_case.json','revision_evidence_20260917/data/all_seven_token_lengths.csv','phase3/results/target_only_permutations/provenance.json','phase3/scripts/analyze_target_only_permutations.py','phase2/src/encoding.py','phase2/src/representations.py','phase2/config/phase2.json']
 for p in paths:add(p)
 for r in provenance().itertuples():
  for p in [r.query_embedding,r.product_embedding,r.rank_source]:
   add(p)
   if p.endswith('.npy'):add(Path(p).with_suffix('.json'))
 for ds in ['wands','esci']:
  for model in MODELS:add(f'phase3/results/target_only_permutations/{ds}_{model}_pairs.parquet')
  for ext in ['products.jsonl.gz','queries.csv','judgments.csv']:add(f'phase2/data/processed/{ds}_{ext}')
  add(f'revision_graded_20260922/data/{ds}_highest_support.csv')
  h=pd.read_csv(GRADED/f'data/{ds}_highest_support.csv',dtype={'product_id':str});h.to_csv(HERE/f'data/{ds}_primary_support.csv',index=False)
 models={}
 for model in ['minilm','bge_base']:
  m=model_spec(model);local=Path(snapshot_download(m['name'],revision=m['revision'],local_files_only=True));models[model]=str(local)
  for name in ['config.json','model.safetensors','pytorch_model.bin','tokenizer.json','tokenizer_config.json','special_tokens_map.json','vocab.txt']:
   if (local/name).exists():add(local/name)
 tokens=pd.read_csv(OLD/'data/all_seven_token_lengths.csv',dtype={'product_id':str});products,_=load('wands');lookup={str(x['product_id']):x for x in products}
 candidates=[];selected=[]
 for model in ['minilm','bge_base']:
  t=target_frame('wands',model);t=t.merge(tokens[(tokens.dataset=='wands')&(tokens.model==model)][['product_id','max_tokens_seven','fully_fits']],on='product_id',validate='many_to_one')
  rr=t[STEMS].to_numpy();t['state']=np.where((rr<=20).all(1),'persistent_inclusion',np.where((rr>20).all(1),'persistent_omission','crossing'));t['rank_span']=rr.max(1)-rr.min(1);t=t[t.fully_fits].copy();t['model']=model;t['attribute_count']=t.product_id.map(lambda x:len(lookup[x]['attributes']));t['selection_hash']=[textsha(f'2026092201|{model}|{s}|{q}|{p}') for s,q,p in zip(t.state,t.query_id,t.product_id)]
  t=t.sort_values(['state','selection_hash','query_id','product_id']);t['selected']=False
  for state,g in t.groupby('state'):
   chosen=[];qs=set()
   for i,row in g.iterrows():
    if row.query_id not in qs:chosen.append(i);qs.add(row.query_id)
    if len(chosen)==4:break
   if len(chosen)<4:chosen+=list(g.index.difference(chosen))[:4-len(chosen)]
   t.loc[chosen,'selected']=True
  candidates.append(t[['model','query_id','product_id','state','max_tokens_seven','attribute_count','rank_span','selection_hash','selected']]);selected.append(t[t.selected][candidates[-1].columns])
 panel=pd.concat(selected,ignore_index=True);panel['selection']='stratified_hash_panel';panel['display']='supplement'
 assert len(panel)==24
 cross=panel[(panel.model=='bge_base')&(panel.state=='crossing')];median=cross.rank_span.median();idx=(cross.assign(distance=(cross.rank_span-median).abs()).sort_values(['distance','selection_hash']).index[0]);panel.loc[idx,'display']='main_crossing';chosen=panel.loc[idx]
 stable=panel[(panel.model=='bge_base')&(panel.state=='persistent_inclusion')].copy();stable['distance']=(stable.max_tokens_seven-chosen.max_tokens_seven).abs()/512+(stable.attribute_count-chosen.attribute_count).abs()/max(1,chosen.attribute_count);idx=stable.sort_values(['distance','selection_hash']).index[0];panel.loc[idx,'display']='main_stable_control'
 if not ((panel.model=='minilm')&(panel.query_id==359)&(panel.product_id=='12575')).any():
  old=next(f[(f.query_id==359)&(f.product_id=='12575')].iloc[0] for f in candidates if (f.model=='minilm').all())
  panel=pd.concat([panel,pd.DataFrame([{**old.to_dict(),'selected':True,'selection':'post_hoc_historical','display':'supplement_historical'}])],ignore_index=True)
 panel.to_csv(HERE/'XAI_CASE_MANIFEST.csv',index=False);pd.concat(candidates).to_csv(HERE/'data/xai_all_candidates.csv',index=False)
 coverage_ids=sorted(support('wands'),key=lambda q:(textsha('coverage-20260922|'+str(q)),q))[:128]
 h=pd.read_csv(HERE/'data/wands_primary_support.csv',dtype={'product_id':str});h=h[h.query_id.isin(coverage_ids)];h.to_csv(HERE/'data/wands_coverage_support.csv',index=False)
 preserved={p:sha(ROOT/p) for p in git('diff','--name-only').splitlines()}
 # Existing untracked author files are also preserved, excluding this new package.
 for p in git('ls-files','--others','--exclude-standard').splitlines():
  if not p.startswith(HERE.name+'/') and (ROOT/p).is_file():preserved[p]=sha(ROOT/p)
 entries=[];tracked=set(git('ls-files').splitlines())
 for i,p in enumerate(sorted(sources)):
  rel=p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else str(p);e=dict(path=rel,local_exists=p.is_file(),git_tracked=rel in tracked)
  if p.is_file():
   e.update(bytes=p.stat().st_size,sha256=sha(p))
   if prior.get(rel):assert e['sha256']==prior[rel],rel
  entries.append(e)
  if i%60==0:print('HASH',i,len(sources),flush=True)
 dump(HERE/'INPUT_MANIFEST.json',dict(files=entries,preserved_working_files=preserved,head=git('rev-parse','HEAD'),branch=git('branch','--show-current'),reviewed_baseline='29a699794746b4ecc916f69e75e7d05f737677f7',merge_base=git('merge-base','HEAD','29a699794746b4ecc916f69e75e7d05f737677f7'),tree_diff=git('diff','--stat','HEAD','29a699794746b4ecc916f69e75e7d05f737677f7')))
 protocol=dict(frozen_utc=datetime.now(timezone.utc).isoformat(),status='follow-up analysis plan after observed historical findings; not whole-study preregistration',input_manifest_sha256=sha(HERE/'INPUT_MANIFEST.json'),primary_populations=read(GRADED/'PROTOCOL.json')['populations'],model_specs=CFG['models'],local_models=models,query_prefix='',precision='native CUDA FP16 model / FP32 pooling and normalized embeddings; FP32 gradient diagnostic separately audited',ties='descending saved FP32 score then original catalog index; old target removed',
  A=dict(models=MODELS,datasets=['wands','esci'],schedules=STEMS,K=[20,100],scores='C0 complete matrix from exact audited query/product vectors; C0 target uses same matrix value; alternative target uses saved phase2 pair score',ranks='authoritative phase3 target_only_permutations; reconstruction differences retained without tolerance',fitting='original all-seven token support; separate from coverage fitting',pair_distribution_weights='query_macro: weight 1/(nonempty query count * Hq support size), then condition and renormalize on state; pair_micro uniform pairs'),
  B=dict(seed=2026092201,selection_hash='sha256(seed|model|state|query_id|product_id)',quotas='4 crossing/4 persistent inclusion/4 persistent omission per WANDS MiniLM/BGE, all-seven fitting; distinct queries within stratum first',panel_sha256=sha(HERE/'XAI_CASE_MANIFEST.csv'),variants='min/max native target scores over original seven, original schedule order resolves ties',display='BGE crossing nearest selected-stratum median rank span, hash ties; BGE stable-inclusion control minimum |token delta|/512 + |attribute-count delta|/max(1,crossing attribute count), hash ties; historical french molding supplementary',scalar='dot fixed cached normalized query vector with normalized product embedding',adapter='same pinned tokenizer/weights/native pooling, model.eval(), gradient-enabled FP32 diagnostic; input_ids and inputs_embeds agreement checked',baselines=['zero word embedding on nonspecial nonpadding tokens','PAD word embedding on same nonspecial nonpadding tokens'],preserved='special/padding embeddings, attention mask, position and token-type IDs',integration='Gauss-Legendre quadrature along straight line, 64 then128 then256 nodes; stop when completeness passes',completeness='absolute residual <= max(1e-4,.01*abs(input score-baseline score)) separately per input/baseline',boundary_guard='both native inclusion decisions preserved; each absolute native margin >5*absolute diagnostic/native discrepancy',failure_policy='retain all failures and numerical ambiguity; no case replacement',span_assignment='token offset wholly within original occurrence span -> that entry; overlap -> boundary; fixed blocks, separator, special and padding retained separately',display_entries='first 8 attribute occurrence IDs in original source order, plus other content sum; no selection by attribution change'),
  C=dict(families=['raw dense seven -> pure canonical dense','raw hybrid seven -> canonical BM25'],datasets=['wands','esci'],models=['minilm','bge_base'],rules=RULES,K=[20,100],identities='highest_support_ids and included_ids_K from saved per_query',conditional='ratio of query-macro cell masses primary; pair-micro ratio secondary; zero denominator NA'),
  D=dict(wands_query_ids=coverage_ids,wands_selection='first128 sha256(coverage-20260922|decimal query id), numeric tie',wands_pairs=len(h),wands_unique_products=int(h.product_id.nunique()),families=[7,16,32],new_seeds=list(range(2026092201,2026092226)),datasets=['wands','esci'],models=['minilm','bge_base'],esci='all499 E target supports, every distinct whole-entry permutation (max3 entries/max6 orders); original duplicate observations retained',intervention='target-only, full rawC0 competitor catalog fixed; no GTE/no catalog re-encoding',common_fit='all32 schedules WANDS; all distinct permutations ESCI; same nonempty query support at each family size',cache='text SHA256 plus full encoder profile; encode only missing distinct target strings; original7 saved outcomes remain authoritative',monotonicity='VI nondecreasing; persistent inclusion/omission nonincreasing on fixed support; structural check, not discovery'),
  bootstrap=dict(draws=DRAWS,seed=SEED,unit='paired query cluster, all methods/schedules together',interval='95% percentile uncorrected'),compute_estimate=dict(coverage_upper_bound_new_wands_texts_per_model=int(h.product_id.nunique()*25),esci_upper_bound_texts_per_product=6,xai_pairs=len(panel),xai_input_variants=2*len(panel),xai_max_integration_evaluations=len(panel)*2*2*(64+128+256),runtime='estimated MiniLM tens of minutes; BGE roughly1–3 GPU hours for worst-case160k targets, before actual tokenizer workload and cache dedup audit; XAI few to tens of minutes; exact preparation estimates saved before scoring',new_training=False),scope='four bounded workstreams; no method/seed/case selection after outcomes; no manuscript edits; dedicated branch pushed but not merged')
 dump(HERE/'PROTOCOL.json',protocol);(HERE/'PROTOCOL.sha256').write_text(sha(HERE/'PROTOCOL.json')+'\n')
 dump(HERE/'qa/runtime.json',dict(python=sys.version,packages={n:importlib.metadata.version(n) for n in ['torch','transformers','numpy','pandas','scipy','pyarrow']},threads=4))
 brief=Path(r'C:/Users/ycw/Downloads/FINAL_CODEX_STRENGTHENING_20260922.md');(HERE/'inputs/EXECUTION_BRIEF.md').write_bytes(brief.read_bytes())
 print('FROZEN',sha(HERE/'PROTOCOL.json'),'XAI',len(panel),'coverage pairs',len(h),flush=True)

if __name__=='__main__':main()
