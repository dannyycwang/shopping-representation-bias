"""Independent accounting tests, historical reconciliation and artifact integrity."""
from common import *
from analyze_membership import per_query
from pypdf import PdfReader
import subprocess, importlib.metadata as md

def main():
    checks=[]
    toy=pd.DataFrame({'query_id':[1,1,1,1,2],'product_id':['a','b','c','d','e'],
                      'C0':[1,21,1,21,1],'ALT':[2,1,22,30,21]})
    q=per_query(toy,'C0','ALT',20)
    assert list(q.loc[1,['retained','newly_included','newly_omitted','absent_in_both']])==[1,1,1,1]
    assert q.loc[1,'net_delta']==0 and q.loc[1,'membership_change_fraction']==.5
    assert q.loc[1,'exact_zero_net_with_changes'] and not q.loc[2,'exact_zero_net_with_changes']
    assert q.net_delta.mean()==-.5
    assert ((q.newly_included-q.newly_omitted).sum()/q.relevant_count.sum())==-.2
    checks.append(dict(check='hand-calculated four states, within-query cancellation, macro != micro',passed=True))
    p=pd.read_csv(DATA/'membership_changes_per_query.csv')
    keys=['dataset','model','population','K','reference','alternative','query_id']
    assert not p.duplicated(keys).any()
    counts=p[['retained','newly_included','newly_omitted','absent_in_both']].to_numpy(dtype=np.int64)
    assert np.array_equal(counts.sum(1),p.relevant_count)
    assert np.array_equal(p.exact_zero_net_with_changes,(p.newly_included==p.newly_omitted)&((p.newly_included+p.newly_omitted)>0))
    assert np.allclose(p.net_delta,p.alternative_recall-p.reference_recall,atol=1e-14,rtol=0)
    assert np.allclose(p.membership_change_fraction,p.gain_fraction+p.loss_fraction,atol=1e-14,rtol=0)
    assert (p.groupby(['dataset','model','population','K','query_id']).size()==21).all()
    assert (p.groupby(['dataset','model','population','K','query_id']).relevant_count.nunique()==1).all()
    checks.append(dict(check='all empirical query rows: accounting, exact cancellation, all 21 comparisons, fixed Hq',passed=True,rows=len(p)))
    summary=pd.read_csv(DATA/'membership_changes_summary.csv');assert len(summary)==1008
    primary=summary[(summary.population=='primary_existing_evaluation')&(summary.reference=='C0')]
    assert len(primary)==144
    for group,g in summary.groupby(['dataset','model','population','K','reference','alternative']):
        ds,model,pop,k,ref,alt=group
        z=p[(p.dataset==ds)&(p.model==model)&(p.population==pop)&(p.K==k)&(p.reference==ref)&(p.alternative==alt)]
        for r in g.itertuples():
            expected=z.net_delta.mean() if r.aggregation=='query_macro' else (z.newly_included-z.newly_omitted).sum()/z.relevant_count.sum()
            assert abs(expected-r.net_delta)<1e-14
            assert r.queries_with_exact_zero_net_and_changes==z.exact_zero_net_with_changes.sum()
            assert r.queries_with_any_change==z.any_membership_change.sum()
    checks.append(dict(check='summary independently reconciled to every per-query contrast',passed=True,rows=len(summary)))
    sourcefile=json.loads((DATA/'membership_changes_sources.json').read_text(encoding='utf8'))
    for x in sourcefile['source_files']:assert sha(x['path'])==x['sha256']
    protocol=json.loads((HERE/'PROSPECTIVE_PROTOCOL.json').read_text(encoding='utf8'))
    assert sha(protocol['framing_pdf'])==protocol['framing_pdf_sha256']
    for rel,digest in protocol['source_hashes'].items():assert sha(ROOT/rel)==digest
    checks.append(dict(check='inherited raw inputs, manuscript PDF and frozen source definitions preserved',passed=True))
    recon=[]
    statepath=DATA/'inclusion_states_summary.csv'
    if statepath.exists():
        st=pd.read_csv(statepath)
        assert np.allclose(st.Always+st.VI+st.Never,1,atol=1e-14,rtol=0)
        for intervention,path in [('catalog_wide',ROOT/'phase2/results/phase2_tables/table1_representation_sensitivity_native.csv'),('target_only',ROOT/'phase3/results/target_only_permutations/summary.csv')]:
            old=pd.read_csv(path)
            for r in old.itertuples():
                z=st[(st.dataset==r.dataset)&(st.model==r.retriever)&(st.population=='historical_full')&(st.target_support=='full')&(st.intervention==intervention)&(st.K==20)]
                for agg,col in [('query_macro','VI@20_query_macro'),('pair_micro','VI@20_micro')]:
                    inherited=float(old.loc[(old.dataset==r.dataset)&(old.retriever==r.retriever),col].iloc[0]);derived=float(z.loc[z.aggregation==agg,'VI'].iloc[0])
                    assert abs(derived-inherited)<1e-12
                    recon.append(dict(dataset=r.dataset,model=r.retriever,intervention=intervention,aggregation=agg,metric='VI@20',inherited=inherited,derived=derived))
        phase6=pd.read_csv(ROOT/'PHASE6_RESULTS.csv');old=phase6[(phase6.method=='Original')&(phase6.split=='test')&(phase6['mode']=='catalog')].iloc[0]
        z=st[(st.dataset=='wands')&(st.model=='bge_base')&(st.population=='primary_existing_evaluation')&(st.target_support=='full')&(st.intervention=='catalog_wide')&(st.aggregation=='query_macro')]
        for k in [20,100]:
            r=z[z.K==k].iloc[0]
            assert abs(r.mean_over_seven_schedules_Recall*100-old[f'Recall@{k}'])<1e-9
            recon.append(dict(dataset='wands',model='bge_base',intervention='catalog_wide',aggregation='query_macro',metric=f'PhaseVI Original Recall@{k} = mean-over-seven, not C0',inherited=old[f'Recall@{k}']/100,derived=r.mean_over_seven_schedules_Recall,C0_Recall=r.C0_Recall))
        r=z[z.K==100].iloc[0]
        for legacy,current in [('Robust@100','Always'),('VI@100','VI'),('Never@100','Never')]:
            assert abs(r[current]*100-old[legacy])<1e-9
            recon.append(dict(dataset='wands',model='bge_base',intervention='catalog_wide',aggregation='query_macro',metric=f'PhaseVI {legacy} = {current}',inherited=old[legacy]/100,derived=r[current]))
        old_target=phase6[(phase6.method=='Original')&(phase6.split=='test')&(phase6['mode']=='target')].iloc[0]
        for k in [20,100]:
            target=st[(st.dataset=='wands')&(st.model=='bge_base')&(st.population=='primary_existing_evaluation')&(st.target_support=='full')&(st.intervention=='target_only')&(st.aggregation=='query_macro')&(st.K==k)].iloc[0]
            assert abs(target.mean_schedule_inclusion*100-old_target[f'Inclusion@{k}'])<1e-9
            recon.append(dict(dataset='wands',model='bge_base',intervention='target_only',aggregation='query_macro',metric=f'PhaseVI Original Inclusion@{k}, separate target indexes',inherited=old_target[f'Inclusion@{k}']/100,derived=target.mean_schedule_inclusion))
        checks.append(dict(check='all three states sum to one; historical VI and PhaseVI mean-schedule metric reconciled',passed=True))
    pd.DataFrame(recon).to_csv(DATA/'baseline_metric_reconciliation.csv',index=False)
    if (DATA/'canonical_control_availability.csv').exists():
        a=pd.read_csv(DATA/'canonical_control_availability.csv')
        if a.status.eq('available').all():
            states=pd.read_csv(DATA/'canonical_inclusion_states.csv')
            assert np.all(states.VI==0) and np.allclose(states.Always+states.Never,1)
            assert set(states.alternative)==set(RULES)
            addendum=json.loads((HERE/'BGE_EXECUTION_ADDENDUM.json').read_text(encoding='utf8'))
            assert sha(HERE/'scripts/run_canonical_controls.py')==addendum['runner_sha256']
            for condition in addendum['scope']:
                record=json.loads((HERE/'experiments'/f'{condition}_source.json').read_text(encoding='utf8'))
                assert record['runtime']['effective_max_batch_size']==16
                assert record['runtime']['batching_addendum']['sha256']==sha(HERE/'BGE_EXECUTION_ADDENDUM.json')
            checks.append(dict(check='all 18 frozen canonical conditions available; three rules retain separate outcomes',passed=True))
    pdfs=[]
    for name,expected in [('figure2_intervention_and_states.pdf',1),('rq2_membership_changes.pdf',2),('rq1_target_fully_fitting.pdf',1)]:
        path=FIG/name
        if not path.exists():continue
        doc=PdfReader(path);assert len(doc.pages)==expected
        for page in doc.pages:
            assert abs(float(page.mediabox.width)-7.1*72)<.1
            assert len(page.images)==0, 'Final scientific figure must be vector'
        pdfs.append(dict(file=name,pages=len(doc.pages),width_inches=7.1,raster_images=0,sha256=sha(path)))
    checks.append(dict(check='final PDF dimensions/page counts and zero raster image objects',passed=True,figures=len(pdfs)))
    dump(DATA/'validation_report.json',dict(checks=checks,pdfs=pdfs,notes='Visual proof inspection is recorded separately in qa/VISUAL_QA.md.'))
    versions={p:md.version(p) for p in ['numpy','pandas','pyarrow','matplotlib','fonttools','pypdf','threadpoolctl']}
    dump(HERE/'analysis_runtime.json',dict(python=sys.version,packages=versions))
    final_status=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True)
    (HERE/'final_git_status.txt').write_text(final_status,encoding='utf8')
    manifest={str(p.relative_to(HERE)).replace('\\','/'):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(HERE.rglob('*'))
              if p.is_file() and '__pycache__' not in p.parts and not ('embeddings' in p.parts and p.suffix=='.npy')
              and ('qa' not in p.parts or p.suffix=='.md') and p.name not in ['DELIVERY_MANIFEST.json','evidence_package.zip','FINALIZATION_COMPLETE.json'] and not p.name.endswith('.log')}
    dump(HERE/'DELIVERY_MANIFEST.json',dict(files=manifest,embedding_metadata='experiments/*_source.json links exact embedding provenance; JSON metadata is included and large .npy caches are excluded from this portable manifest.'))
    print('VALIDATED',len(checks),'checks;',len(manifest),'delivery files',flush=True)

if __name__=='__main__':main()
