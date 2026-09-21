"""Freeze scope and inventory local payloads separately from Git metadata."""
from common import *
import subprocess, zipfile, platform, datetime, argparse

def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT).decode('utf8').strip()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--chapter-package',type=Path,default=Path(r'C:/Users/ycw/Downloads/chapter56_revision_package.zip'))
    parser.add_argument('--reference-pdf',type=Path,default=Path(r'C:/Users/ycw/Downloads/Same_Product__Different_Visibility__Representation_Robustness_in_Neural_E_Commerce_Retrieval (6).pdf'))
    args=parser.parse_args();package=args.chapter_package
    with zipfile.ZipFile(package) as z:
        for info in z.infolist():
            target=(HERE/'inputs'/info.filename).resolve()
            assert target.is_relative_to((HERE/'inputs').resolve())
            if not info.is_dir():
                target.parent.mkdir(parents=True,exist_ok=True)
                if target.exists():assert target.read_bytes()==z.read(info)
                else:target.write_bytes(z.read(info))
    tracked=set(git('ls-files').splitlines()); sources=set()
    def add(path):sources.add(Path(path).resolve())
    add(package)
    add(Path(r'C:/Users/ycw/.codex/attachments/234d831c-e6dc-4e87-a08b-efeb80407f1e/pasted-text.txt'))
    pdf=args.reference_pdf;add(pdf)
    for path in ['PROSPECTIVE_PROTOCOL.json','EVIDENCE_STATUS.md','CANONICAL_CONTROLS.md','data/evaluation_populations.json','data/membership_changes_sources.json','data/inclusion_states_sources.json','data/canonical_sources.json','data/all_seven_token_lengths.csv','data/encoder_profiles_and_rank_sources.csv','data/canonical_serialization_audit.csv']:
        add(OLD/path)
    for path in ['config/EXPERIMENT_PROTOCOL.frozen.md','config/splits.json','config/selection.json','scripts/common.py','scripts/run_methods.py','scripts/audit_shared_query_cache.py','results/shared_query_cache_audit.csv','results/encoding_costs.csv','results/unique_view_summary.csv']:
        add(ROOT/'phase4'/path)
    for path in ['config/phase2.json','src/representations.py','src/encoding.py','src/evaluation.py','scripts/prepare_data.py']:add(ROOT/'phase2'/path)
    provenance=pd.read_csv(OLD/'data/encoder_profiles_and_rank_sources.csv')
    for r in provenance.itertuples(index=False):
        for p in [r.rank_source,r.product_embedding,r.query_embedding]:
            add(ROOT/p)
            if p.endswith('.npy'):add((ROOT/p).with_suffix('.json'))
    populations=[]; ordering=[]
    for ds in ['wands','esci']:
        products,queries,judgments=p4.load(ds)
        pids=[str(p['product_id']) for p in products];qids=queries.query_id.astype(int).tolist()
        assert len(pids)==len(set(pids));assert len(qids)==len(set(qids))
        for kind,ids in [('catalog',pids),('queries',qids)]:
            pd.DataFrame({'index':range(len(ids)),kind+'_id':ids}).to_csv(DATA/f'{ds}_{kind}_ordering.csv',index=False)
            ordering.append(dict(dataset=ds,axis=kind,size=len(ids),ordered_ids_sha256=digest_ids(ids)))
        j=judgments.copy();j.product_id=j.product_id.astype(str);h=relevant(j,ds)
        assert h.index.is_unique and set(h.index.get_level_values('product_id')).issubset(pids)
        populations.append(dict(dataset=ds,requested_queries=len(primary(ds)),eligible_queries=h.index.get_level_values(0).nunique(),highest_label_pairs=len(h),catalog_size=len(pids),requested_ids=primary(ds),eligible_ids=sorted(h.index.get_level_values(0).unique().tolist())))
        for ext in ['products.jsonl.gz','queries.csv','judgments.csv']:add(ROOT/f'phase2/data/processed/{ds}_{ext}')
        for s in STEMS:add(ROOT/f'phase2/data/representations/{ds}/{s}.jsonl.gz')
        for model in MODELS:
            add(ROOT/f'phase3/results/target_only_permutations/{ds}_{model}_pairs.parquet')
            for rule in RULES:
                rank,src=canonical_source(ds,model,rule); add(rank);add(OLD/f'experiments/{ds}_{model}_{rule}_source.json')
                for value in src.values():
                    if isinstance(value,dict) and 'relative_path' in value:add(ROOT/value['relative_path'])
        for pattern in ['*_pairs.parquet','*_per_query.csv','*_scores.npy','*_top1000.npy','*cost*']:
            for p in (ROOT/f'phase4/results/{ds}').glob(pattern):add(p)
        for pattern in [f'{ds}*transitions.csv',f'{ds}*timing.csv',f'{ds}_unique_views*']:
            for p in (ROOT/'phase4/results').glob(pattern):add(p)
        for p in (ROOT/'phase4/results/embeddings').glob(f'{ds}_*'):
            if p.suffix in ['.json','.npy']:add(p)
        for p in (ROOT/'phase3/results/phase3_invariant_method/embeddings').glob(f'{ds}_*'):add(p)
    for folder in ['data','scripts','assets/fonts','figures']:
        for p in (OLD/'figure_revision'/folder).rglob('*'):
            if p.is_file() and p.suffix not in ['.pyc','.png']:add(p)
    for p in (HERE/'inputs').rglob('*'):
        if p.is_file():add(p)
    for p in [OLD/'data/membership_changes_per_query.csv',OLD/'data/membership_changes_summary.csv',OLD/'data/rq2_cancellation_companion.csv',OLD/'data/inclusion_states_summary.csv']:add(p)
    # Preserve every pre-existing tracked local edit, including manuscript binaries.
    modified=git('diff','--name-only').splitlines()
    for p in modified:add(ROOT/p)
    entries=[]
    for i,p in enumerate(sorted(sources)):
        rel=p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else str(p)
        metarel=p.with_suffix('.json').relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else ''
        exists=p.is_file(); entry=dict(path=rel,local_exists=exists,git_tracked=rel in tracked,git_distribution='payload_tracked' if rel in tracked else 'metadata_only' if metarel in tracked else 'not_tracked',
            availability='local_file' if exists else 'metadata_only' if p.with_suffix('.json').is_file() else 'missing')
        if exists:
            entry.update(bytes=p.stat().st_size,sha256=sha(p))
            if p.suffix=='.npy':
                a=np.load(p,mmap_mode='r');entry.update(shape=list(a.shape),dtype=str(a.dtype),array_bytes=a.nbytes)
                if p.with_suffix('.json').is_file():
                    m=read(p.with_suffix('.json'));entry['embedding_metadata']={k:m[k] for k in ['model','profile','precision','implementation','source_sha256','seconds'] if k in m}
        entries.append(entry)
        if i%75==0:print('HASH',i,len(sources),flush=True)
    dump(DATA/'source_manifest.json',dict(git_head=git('rev-parse','HEAD'),reference_head='b73e52ae4a787e33c5773673694464b8f2743e56',tracked_tree_diff=git('diff','--stat','HEAD','b73e52ae4a787e33c5773673694464b8f2743e56'),files=entries,ordering=ordering,populations=populations,preexisting_modified=modified))
    protocol=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),reference_head='b73e52ae4a787e33c5773673694464b8f2743e56',actual_head=git('rev-parse','HEAD'),models=CFG['models'],primary_populations=populations,
      schedules=STEMS,raw_seeds=list(range(20260911,20260916)),view_seeds=list(range(20260971,20260978)),cutoffs=[20,100],diagnostic_cutoffs=[20,50,100,200,500,1000],bootstrap=dict(draws=DRAWS,seed=SEED,unit='paired query cluster',interval='95% percentile; uncorrected; no equivalence claim'),
      hybrid=dict(cells='WANDS/ESCI x MiniLM/BGE',bm25='phase4 common.py; full source-order text; lowercase [a-z0-9]+ counts; sorted unique query tokens; k1=1.2,b=.75',fusion='FP32 1/(60+bm25 rank)+1/(60+dense rank); complete catalog; stable original catalog-index ties',dense_source='authoritative saved full scores when available, else exact dot products from audited historical embeddings',numerical_policy='audit all saved highest-label ranks and K membership; save reconstructed raw and hybrid on common scores; do not overwrite history'),
      strategies=['raw_seven','raw_C0',*RULES,'set_mean','centroid_2','max_2','BM25','hybrid_seven'],budget_sweep='BGE centroid/max M=4,7 only; no winner selection',canonical='reuse all 18 existing conditions; M2 and rule_appearance excluded from pure rules',set_mean='0.5 nonattribute vector + 0.5 normalized attribute mean; final L2 norm; Phase IV common historical queries',
      support='all Hq including absent-in-both; within-query first then query macro; pair micro separate; WANDS dev excluded; ESCI existing sample, not untouched',invariance='single fixed index; structural VI=0 for canonical, set mean, canonical-seeded views, BM25',target_only='separate counterfactual indexes, never one shared-index Recall',fitting='encoder-specific all-seven token counts including special tokens; target filtering only; competitors unchanged',cost='reuse measured cold construction metadata; cache-hit elapsed not cold cost; exact local timings not ANN',scope='P0-P3; no P4, training, paid APIs or Chapters 1-4 edits')
    if not (HERE/'PROTOCOL.json').exists():dump(HERE/'PROTOCOL.json',protocol)
    (HERE/'PROTOCOL.sha256').write_text(sha(HERE/'PROTOCOL.json')+'\n')
    dump(HERE/'qa/runtime.json',dict(python=sys.version,platform=platform.platform(),numpy=np.__version__,pandas=pd.__version__,threads=4))
    missing=[e['path'] for e in entries if not e['local_exists']]
    text=f'''# Source audit\n\nActual HEAD: `{git('rev-parse','HEAD')}`. Requested reference: `b73e52ae4a787e33c5773673694464b8f2743e56`. The reference was fetched read-only; its tracked tree is identical to HEAD (two merge commits differ). No reset, checkout, or historical writer was run. No applicable AGENTS.md was found in the repository or workspace ancestors. Pre-existing local edits are hashed in the manifest and preserved.\n\n`data/source_manifest.json` distinguishes local payload existence, Git tracking, and metadata-only/missing states. In particular, locally available ignored NPY caches are not claimed to be distributed by Git. Array shapes, dtypes, exact hashes, model revisions, precision and implementation metadata are recorded. Ordered query/catalog IDs are exported separately. Raw caches come from the historical provenance audit; no re-encoding or canonical rerun is needed.\n\nPrimary supports: WANDS 384 requested / 308 eligible / 21,299 Exact pairs / 42,994 products; ESCI 500 requested / 499 eligible / 4,434 E pairs / 10,076 products. The 96 WANDS development queries are excluded. ESCI is the existing evaluation sample. Hq includes absent-in-both products. Query macro and pair micro remain separate.\n\nAll 18 canonical conditions are reused through `canonical_sources.json` and their source records. Pure ascending is `canonical_raw`; M2 framing and historical `rule_appearance` are excluded. WANDS pipe entries and ESCI brand/color/whole bullet-field entries remain intact. Duplicates and repeated views remain.\n\n`shared_query_cache_audit.csv` documents nonzero differences between historical Phase III Set-Mean queries and the common Original query matrix. This package uses Phase IV Set-Mean ranks/scores, which use the common queries; Phase III Set-Mean ranks are not substituted.\n\nThe supplied chapter package is preserved byte-for-byte under `inputs/chapter56_revision/`; updated copies will be separate. Its embedded execution prompt is supporting material, not an override of the user's initial scope. PDF version (6) is available. Complete LaTeX matching that PDF has not been supplied, so full-manuscript page count/float placement remains author integration work.\n\nMissing analysis inputs: {missing or 'none among the inventoried required artifacts'}. Numerical and score reconstruction compatibility are checked separately in `data/hybrid_reconstruction_audit.csv`; presence alone is not treated as compatibility.\n'''
    (HERE/'SOURCE_AUDIT.md').write_text(text,encoding='utf8')
    print('AUDIT COMPLETE',len(entries),'files',flush=True)

if __name__=='__main__':main()
