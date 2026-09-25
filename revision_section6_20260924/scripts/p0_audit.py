"""Read-only P0 checks and frozen provenance before any new inference."""
from common import *
from collections import Counter
import importlib.metadata, platform, shutil

def main():
    checks=[];manifest=[]
    def check(name,ok,detail=''):
        checks.append(dict(check=name,status='PASS' if bool(ok) else 'NEEDS_CORRECTION',detail=detail))
    snapshot=HERE/'qa/starting_state.json'
    if not snapshot.exists():
        changed=git('diff','--name-only').splitlines()
        dump(snapshot,dict(time_utc=now(),revision=git('rev-parse','HEAD'),branch=git('branch','--show-current'),status=git('status','--short'),preexisting_files={p:sha(ROOT/p) for p in changed if (ROOT/p).is_file()}))
    split=read(ROOT/'phase4/config/splits.json')
    for ds in ['wands','esci']:
        p,q,j=load(ds);ids=support(ds);label='Exact' if ds=='wands' else 'E'
        h=j[j.label.eq(label)&j.query_id.isin(ids)]
        check(ds+' primary support',len(ids)==(308 if ds=='wands' else 499) and len(h)==(21299 if ds=='wands' else 4434))
        check(ds+' catalog ordering',list(map(str,[x['product_id'] for x in p]))==pd.read_csv(GRADED/f'data/{ds}_catalog_ordering.csv',dtype=str)['catalog_id'].tolist())
        if ds=='wands':
            order=sorted(map(int,q.query_id),key=lambda x:digest(f'phase3-dev:{x}'))
            check('WANDS deterministic 96/384 split',order[:96]==split['wands_dev'] and set(order[96:])==set(split['wands_heldout']))
            check('WANDS 308 = heldout with Exact',sorted(j[j.query_id.isin(split['wands_heldout'])&j.label.eq(label)].query_id.unique())==ids)
        else:
            raw=ROOT/'phase2/data/esci_repo/shopping_queries_dataset/shopping_queries_dataset_examples.parquet'
            if raw.exists():
                ex=pd.read_parquet(raw,filters=[('product_locale','==','us'),('small_version','==',1),('split','==','test')],columns=['query_id','product_id'])
                sample=sorted(ex.query_id.unique(),key=lambda x:digest(f'20260904:{int(x)}'))[:500]
                check('ESCI hash sample of 500',set(sample)==set(q.query_id))
                check('ESCI pooled 10076 catalog',set(ex[ex.query_id.isin(sample)].product_id)==set(str(x['product_id']) for x in p) and len(p)==10076)
        csv(pd.DataFrame({'query_id':sorted(q.query_id),'eligible':[x in ids for x in sorted(q.query_id)],'development':[x in split['wands_dev'] if ds=='wands' else False for x in sorted(q.query_id)]}),HERE/f'data/{ds}_sampling_audit.csv')
        # Every whole-entry occurrence and all fixed fields survive all seven serializations.
        for s,condition in zip(STEMS,CFG['primary_variant_family']):
            for product in p:
                ordered=attrs(product,s)
                assert Counter(ordered)==Counter(product['attributes'])
                assert text_for(product,s)==REP.build(product,condition,CFG['attribute_permutation_seeds'])
            check(f'{ds} {s}: whole-entry multiplicities and exact original serializer',True,len(p))
    fits=pd.read_csv(OLD/'data/all_seven_token_lengths.csv',dtype={'product_id':str})
    check('fully fitting includes complete native token counts',np.array_equal(fits.fully_fits,fits[[f'tokens_{s}' for s in STEMS]].max(axis=1)<=fits.tokenizer_cap),'Special tokens included, empty prefixes; tokenizer hashes and prior token audit retained.')
    for ds in ['wands','esci']:
        f=fits[(fits.dataset==ds)&fits.model.eq('gte_modernbert')]
        check(ds+' GTE full equals fully fitting',f.fully_fits.all())
    # Test the historical replacement rank against explicit single-target replacement, with ties.
    import ast
    path=ROOT/'phase3/scripts/analyze_target_only_permutations.py';tree=ast.parse(path.read_text(encoding='utf8'))
    fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='rank_target')
    env={'np':np};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),env)
    # Existing focused validation already exercises the exact historical function signature.
    test=read(OLD/'data/target_rank_algorithm_tests.json')
    check('historical target-only focused tests available',test['all_passed'] and sha(path)==test['saved_function_source']['sha256'],'11,600 tied replacements; one deep historical reconstruction discrepancy retained.')
    rng=np.random.default_rng(20260925)
    for n in [2,5,20,100]:
        for iteration in range(100):
            b=rng.integers(-3,4,n).astype('f4');inds=np.arange(n);v=rng.integers(-3,4,n).astype('f4')
            actual=env['rank_target'](b,inds,v)
            expected=[]
            for i,val in enumerate(v):
                replaced=b.copy();replaced[i]=val;expected.append(int(np.where(np.argsort(-replaced,kind='stable')==i)[0][0])+1)
            assert np.array_equal(actual,expected)
    check('fresh single-target replacement and deterministic tied ranks',True,'12,700 explicit replacements; old copy removed')
    for p in [OLD/'data/validation_report.json',GRADED/'qa/validation.json',COMPLETE/'qa/validation.json',FINAL/'qa/final_validation.json',FINAL/'qa/coverage_plan_validation.json']:
        if p.exists():manifest.append(dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p),role='reused validation; no historical writer rerun'))
    for folder in ['revision_evidence_20260917','revision_completion_20260921','revision_graded_20260922','revision_final_strengthening_20260922']:
        for p in (ROOT/folder).glob('**/*'):
            if p.is_file() and p.suffix in ['.csv','.parquet','.json','.py','.tex','.md'] and not any(x in p.parts for x in ['cache','xai','inputs','__pycache__']):
                manifest.append(dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p),role='saved evidence/source inventory'))
    pdf=Path.home()/'Downloads/Same_Product__Different_Visibility__Representation_Robustness_in_Neural_E_Commerce_Retrieval (11).pdf'
    manifest.append(dict(path=str(pdf),bytes=pdf.stat().st_size,sha256=sha(pdf),role='reviewed manuscript v11'))
    csv(pd.DataFrame(manifest).drop_duplicates('path'),HERE/'data/source_inventory.csv')
    dump(HERE/'qa/p0_checks.json',checks)
    versions={}
    for name in ['numpy','pandas','pyarrow','torch','transformers','tokenizers','matplotlib','scipy','safetensors','threadpoolctl']:
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]='unavailable'
    dump(HERE/'qa/environment.json',dict(time_utc=now(),python=sys.version,executable=sys.executable,platform=platform.platform(),packages=versions,disk_free=shutil.disk_usage(ROOT).free))
    categories=[
      ('Primary splits, serializers, ranks, fitting support','already complete','Reuse exact saved IDs, token audits and rank payloads.'),
      ('Graded metrics and canonical hybrids','already complete','49,227 corrected per-query records; direct gains, source references retained.'),
      ('Claim ledger and complete joined companions','recompute from saved outputs','Cross-check full precision and join existing records; no inference.'),
      ('RQ2 per-contrast counts and factual replacement case','recompute from saved outputs','Use raw saved ranks and membership sets.'),
      ('Compact figures, main table, paragraph handoff','presentation only','v11 found locally; current full matching TeX unavailable.'),
      ('Harmonized expansion supplement','new forward pass required','Historical 7 plus later forwards use mixed Transformers versions and batch sizes. Freeze existing cohort and all orders; re-encode queries and C0 catalog too.'),
      ('Complete TeX corresponding exactly to PDF v11','unavailable','Provide targeted replacement fragments and paragraph map; preserve older paper sources.'),
      ('IG/attention, training and method sweeps','already complete','Historical outputs retained; outside requested revision scope.')]
    csv(pd.DataFrame(categories,columns=['task','classification','decision']),HERE/'data/task_classification.csv')
    text='# P0 audit checkpoint\n\nRevision: '+git('rev-parse','HEAD')+'\n\n'
    text+='\n'.join(f'- {r[1]}: {r[0]}. {r[2]}' for r in categories)
    text+='\n\nAll historical inputs remain read-only. This is a follow-up after observed outcomes, not new preregistration. Detailed checks: qa/p0_checks.json. Harmonization is separately versioned and cannot replace primary historical seven-schedule estimates.\n'
    (HERE/'P0_CHECKPOINT.md').write_text(text,encoding='utf8')
    print(json.dumps(checks,ensure_ascii=True));assert all(x['status']=='PASS' for x in checks)

if __name__=='__main__':main()
