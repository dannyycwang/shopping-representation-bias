"""Freeze design and hash inputs before evaluating any new scores."""
from common import *
from datetime import datetime,timezone
import platform

def main():
    for name in ['data','tables','qa','inputs']: (HERE/name).mkdir(parents=True,exist_ok=True)
    assert not (HERE/'PROTOCOL.json').exists(), 'Protocol already frozen; do not overwrite.'
    populations={}; sources=set(); conditions=[]
    def add(path):
        p=Path(path); p=p if p.is_absolute() else ROOT/p; sources.add(p)
        return p.relative_to(ROOT).as_posix()
    manifest=read(COMPLETE/'data/source_manifest.json')
    add(COMPLETE/'data/source_manifest.json')
    for ds in ['wands','esci']:
        p,q,j=load(ds); pop=next(x for x in manifest['populations'] if x['dataset']==ds)
        ids=pop['eligible_ids']; jj=j[j.query_id.isin(ids)]
        h=jj[jj.label.eq(highest(ds))]
        assert (len(ids),len(h),len(p))==((308,21299,42994) if ds=='wands' else (499,4434,10076))
        assert sorted(h.query_id.unique())==ids
        assert not jj.duplicated(['query_id','product_id']).any()
        for ext in ['products.jsonl.gz','queries.csv','judgments.csv']:add(f'phase2/data/processed/{ds}_{ext}')
        for kind in ['catalog','queries']:
            src=COMPLETE/f'data/{ds}_{kind}_ordering.csv';add(src)
            (HERE/f'data/{ds}_{kind}_ordering.csv').write_bytes(src.read_bytes())
        catalog=pd.read_csv(HERE/f'data/{ds}_catalog_ordering.csv',dtype={'catalog_id':str})
        assert catalog.catalog_id.tolist()==[str(x['product_id']) for x in p]
        assert pd.read_csv(HERE/f'data/{ds}_queries_ordering.csv').queries_id.tolist()==q.query_id.tolist()
        h[['query_id','product_id','label']].sort_values(['query_id','product_id']).to_csv(HERE/f'data/{ds}_highest_support.csv',index=False)
        jj[['query_id','product_id','label']].sort_values(['query_id','product_id']).to_parquet(HERE/f'data/{ds}_judged_support.parquet',index=False)
        populations[ds]=dict(eligible_ids=ids,eligible_queries=len(ids),highest_label_pairs=len(h),judged_pairs=len(jj),catalog_size=len(p),
                            highest_support_sha256=sha(HERE/f'data/{ds}_highest_support.csv'),judged_support_sha256=sha(HERE/f'data/{ds}_judged_support.parquet'))
        for model in MODELS:
            for s in STEMS:
                stem=f'{ds}_{model}_native_{s}'
                conditions.append(dict(dataset=ds,model=model,method='raw',schedule=s,
                    pairs=add(f'phase2/results/phase2_pair_ranks/{stem}.parquet'),top=add(f'phase2/results/phase2_top50/{stem}.parquet')))
                add(f'phase2/results/phase2_per_query/{stem}.csv')
            for rule in RULES:
                sp=OLD/f'experiments/{ds}_{model}_{rule}_source.json';add(sp);s=read(sp)
                rp=ROOT/s['rank_source']['relative_path'];assert sha(rp)==s['rank_source']['sha256']
                ep=ROOT/s['product_embedding_metadata']['relative_path'];add(ep);add(ep.with_suffix('.npy'))
                conditions.append(dict(dataset=ds,model=model,method=rule,schedule='fixed',pairs=add(rp),embedding=ep.with_suffix('.npy').relative_to(ROOT).as_posix()))
                for k in ['query_embedding','query_embedding_metadata']:add(s[k]['relative_path'])
            if model=='gte_modernbert':continue
            for method in ['set_mean','centroid_2','max_2']+(['centroid_4','max_4','centroid_7','max_7'] if model=='bge_base' else []):
                base=f'phase4/results/{ds}/{model}_{method}'
                conditions.append(dict(dataset=ds,model=model,method=method,schedule='fixed',pairs=add(base+'_pairs.parquet'),top=add(base+'_top1000.npy')))
                add(base+'_scores.npy')
            for method in ['Original','hybrid']:
                for suffix in ['_pairs.parquet','_top1000.npy','_scores.npy']:add(f'phase4/results/{ds}/{model}_{method}'+suffix)
            add(COMPLETE/f'data/{ds}_{model}_hybrid_seven_pairs.parquet')
            add(COMPLETE/f'data/{ds}_{model}_reconstructed_raw_seven_pairs.parquet')
            for s in STEMS:add(COMPLETE/f'data/{ds}_{model}_hybrid_{s}_top1000.npz')
        base=f'phase4/results/{ds}/BM25'
        conditions.append(dict(dataset=ds,model='lexical',method='BM25',schedule='fixed',pairs=add(base+'_pairs.parquet'),top=add(base+'_top1000.npy')))
        for suf in ['_scores.npy','_cost.json']:add(base+suf)
    provenance=OLD/'data/encoder_profiles_and_rank_sources.csv';add(provenance)
    for row in pd.read_csv(provenance).itertuples():
        for path in [row.product_embedding,row.query_embedding]:add(path);add(Path(path).with_suffix('.json'))
    for p in ['phase2/config/phase2.json','phase2/scripts/prepare_data.py','phase2/src/evaluation.py','phase2/src/representations.py','phase4/scripts/common.py','phase4/scripts/run_methods.py','phase4/config/splits.json','phase4/config/selection.json',
              'revision_completion_20260921/data/strategy_costs.csv','revision_completion_20260921/data/hybrid_reconstruction_audit.csv','revision_completion_20260921/data/structural_invariance_checks.csv','revision_completion_20260921/tables/strategy_joint_complete.csv']: 
        if (ROOT/p).exists():add(p)
    status=git('status','--porcelain=v1','-uall')
    (HERE/'inputs/initial_git_status.txt').write_text(status+'\n',encoding='utf8')
    preexisting={}
    for line in status.splitlines():
        rel=line[3:].strip('"')
        if rel.startswith(HERE.name+'/'):continue
        if (ROOT/rel).is_file():preexisting[rel]=sha(ROOT/rel)
    # Historical manifest supplies expected content hashes where already available.
    prior={x['path']:x.get('sha256') for x in manifest['files']}
    tracked=set(git('ls-files').splitlines()); entries=[]
    for i,p in enumerate(sorted(sources)):
        rel=p.relative_to(ROOT).as_posix();exists=p.is_file()
        e=dict(path=rel,local_exists=exists,git_tracked=rel in tracked,metadata_tracked=p.with_suffix('.json').relative_to(ROOT).as_posix() in tracked)
        if exists:
            e.update(bytes=p.stat().st_size,sha256=sha(p))
            if prior.get(rel): assert e['sha256']==prior[rel],('historical hash mismatch',rel)
            if p.suffix=='.npy':
                a=np.load(p,mmap_mode='r');e.update(shape=list(a.shape),dtype=str(a.dtype))
        entries.append(e)
        if i%80==0:print('HASH',i,len(sources),flush=True)
    dump(HERE/'data/input_manifest.json',dict(files=entries,preexisting_files=preexisting,git_head=git('rev-parse','HEAD'),branch=git('branch','--show-current'),reference='32b24160afcaa229a8b7057c04ecf638c7fae488',merge_base=git('merge-base','HEAD','32b24160afcaa229a8b7057c04ecf638c7fae488'),tree_diff=git('diff','--stat','HEAD','32b24160afcaa229a8b7057c04ecf638c7fae488')))
    dump(HERE/'data/conditions.json',conditions)
    protocol=dict(frozen_utc=datetime.now(timezone.utc).isoformat(),populations=populations,schedules=STEMS,models=MODELS,gains=GAINS,
       old_gains=OLD_GAINS,direct_gains=True,graded_cutoffs=[10,20],recall_cutoffs=[20,100],all_judgments_retained=True,
       full_ndcg='Original full catalog absolute ranks. Unjudged gain zero for computation only; same-query all-judged descending-gain IDCG.',
       condensed_cndcg='Remove unjudged before discounting. Never label as ordinary nDCG.',coverage='Fraction of original top K IDs with any judgment, including zero-gain labels.',
       bootstrap=dict(seed=SEED,draws=DRAWS,unit='query cluster; identical resamples for all methods and schedules within a dataset',interval='95% percentile, uncorrected'),
       raw='All seven schedules x all three frozen encoders x both datasets; historical all-judged ranks authoritative.',controls=['BM25',*RULES,'set_mean','centroid_2','max_2','raw_hybrid_seven'],
       secondary='GTE canonical and BGE centroid/multi-vector M=4/7; all budgets visible; no retrospective selection.',
       hybrid=dict(models=['minilm','bge_base'],rules=RULES,rrf_constant=60,equal_weights=True,arithmetic='float32',branch_ranks='full catalog',ties='ascending fixed catalog index',query_embeddings='common historical query matrix',comparison='Each raw hybrid schedule paired with corresponding raw; mean query contrasts over seven schedules.'),
       numerical_policy='Audit all judged reconstructed raw ranks and original top lists against saved ranks. Keep saved raw ranks authoritative. Hybrid must reproduce saved highest ranks and top1000 exactly; otherwise stop that configuration. Canonical fusion uses saved full scores where available, otherwise frozen embeddings; audit every judged rank before fusion.',
       rq2=dict(contrasts='Each C1/C2s1..C2s5 minus C0, every dataset/encoder; raw and hybrid separately',membership='Highest-label Top20 gains/losses',metrics=['nDCG@20','nDCG@10'],thresholds=[.001,.005,.01],threshold_interpretation='descriptive sensitivity, not equivalence tests',numerical_equality_atol=ZERO_TOL,rtol=0,distributions='per-query signed/absolute deltas plus complete changed-query ECDF',denominators='all eligible queries and changed queries retained',ndcg10_note='Separate diagnostic at K10 matched to K20 membership; cutoff differs, no identical-event interpretation'),
       p1='12 canonical-lexical hybrids, no tuning. Compare to own dense branch, BM25, and mean of seven corresponding raw-hybrid references; report structural fixed-index VI=0, persistent states, membership, costs, paired intervals.',
       p2='Optional permutation expansion deferred: no new encodings authorized by this frozen P0/P1 execution budget. Existing seven schedules retained.',
       scope='No training, model download, paid API, historical-result overwrite, or manuscript integration.',
       input_manifest_sha256=sha(HERE/'data/input_manifest.json'),conditions_sha256=sha(HERE/'data/conditions.json'))
    dump(HERE/'GAIN_MAPPING.json',dict(old=OLD_GAINS,new=GAINS,source='https://arxiv.org/pdf/2206.06588 section 3.1',source_checked_utc=datetime.now(timezone.utc).isoformat(),scope='evaluation from labels; frozen ranks, labels, Hq, query embeddings and retrievers unchanged'))
    dump(HERE/'PROTOCOL.json',protocol)
    (HERE/'PROTOCOL.sha256').write_text(sha(HERE/'PROTOCOL.json')+'\n')
    dump(HERE/'qa/runtime.json',dict(python=sys.version,numpy=np.__version__,pandas=pd.__version__,platform=platform.platform(),blas_threads=4,encoding_performed=False))
    missing=[x['path'] for x in entries if not x['local_exists']]
    dump(HERE/'qa/payload_availability.json',dict(inputs=len(entries),missing=missing,locally_available_ignored_payloads=sum(e['local_exists'] and not e['git_tracked'] for e in entries)))
    print('FROZEN',sha(HERE/'PROTOCOL.json'),'conditions',len(conditions),'missing',missing,flush=True)

if __name__=='__main__':main()
