"""Evidence checks, source preservation, and protocol/denominator consistency."""
from common import *
from pypdf import PdfReader
import subprocess

def main():
    manifest=read(DATA/'source_manifest.json');verified=0
    for e in manifest['files']:
        if not e['local_exists']:continue
        p=ROOT/e['path'];assert p.exists() and sha(p)==e['sha256'],('changed source',p)
        verified+=1
    for e in read(DATA/'additional_cost_sources.json')['files']:
        assert sha(ROOT/e['path'])==e['sha256']
    def walk(x):
        nonlocal verified
        if isinstance(x,dict):
            if 'relative_path' in x and 'sha256' in x:
                p=ROOT/x['relative_path'];assert p.exists() and sha(p)==x['sha256'],('historical manifest mismatch',p);verified+=1
            for v in x.values():walk(v)
        elif isinstance(x,list):
            for v in x:walk(v)
    for name in ['membership_changes_sources','inclusion_states_sources','canonical_sources']:walk(read(OLD/f'data/{name}.json'))
    inp=HERE/'inputs/chapter56_revision';package=read(inp/'PACKAGE_MANIFEST.json')
    for e in package['files']:assert sha(inp/e['path'])==e['sha256']
    assert sha(HERE/'PROTOCOL.json')==(HERE/'PROTOCOL.sha256').read_text().strip()
    recon=pd.read_csv(DATA/'hybrid_reconstruction_audit.csv');assert len(recon)==28
    assert recon.saved_rank_mismatches.eq(0).all()
    assert recon[['raw_membership_mismatches_K20','raw_membership_mismatches_K100']].eq(0).all().all()
    c0=recon[recon.schedule=='C0'];assert c0.C0_hybrid_rank_exact.all() and c0.C0_hybrid_fullscore_exact.all()
    s=pd.read_csv(DATA/'strategy_summary.csv');states=s[~s.method.eq('raw_C0')]
    assert np.allclose(states[['persistent_inclusion','VI','persistent_omission']].sum(axis=1),1)
    invariant=states[states.invariance_basis.str.startswith('structural')];assert invariant.VI.eq(0).all()
    assert np.allclose(invariant.source_order_recall,invariant.seven_schedule_mean_recall)
    mq=pd.read_csv(DATA/'membership_per_query.csv')
    assert (mq[['retained','newly_included','newly_omitted','absent_in_both']].sum(axis=1)==mq.n_highest).all()
    assert np.allclose((mq.newly_included-mq.newly_omitted)/mq.n_highest,mq.delta_recall)
    assert np.array_equal(mq.exact_cancellation,(mq.newly_included==mq.newly_omitted)&(mq.newly_included>0))
    for ds,nq,npair in [('wands',308,21299),('esci',499,4434)]:
        x=s[s.dataset==ds];assert x.eligible_queries.eq(nq).all() and x.highest_label_pairs.eq(npair).all()
    cutoff=pd.read_csv(DATA/'cutoff_summary.csv');den=pd.read_csv(DATA/'cutoff_denominators.csv')
    for key,g in den.groupby(['dataset','model','intervention','target_support']):
        assert g.support_ids_sha256.nunique()==1 and g.eligible_queries.nunique()==1 and g.highest_label_pairs.nunique()==1
    for key,g in cutoff.groupby(['dataset','model','intervention','target_support','aggregation']):
        g=g.sort_values('K');assert (g.persistent_inclusion.diff().dropna()>=-1e-14).all();assert (g.persistent_omission.diff().dropna()<=1e-14).all()
        assert np.allclose(g[['persistent_inclusion','VI','persistent_omission']].sum(axis=1),1)
    # Reconcile new cutoff calculations with the existing mandatory K20/K100 states.
    old=pd.read_csv(OLD/'data/inclusion_states_summary.csv');old=old[old.population=='primary_existing_evaluation']
    matched=cutoff[cutoff.K.isin([20,100])].merge(old,on=['dataset','model','intervention','target_support','aggregation','K'],suffixes=('','_old'),validate='one_to_one')
    assert len(matched)==96
    for a,b in [('persistent_inclusion','Always'),('VI','VI_old'),('persistent_omission','Never')]:assert np.allclose(matched[a],matched[b],rtol=0,atol=1e-14)
    cs=pd.read_csv(OLD/'data/canonical_primary_comparison_table.csv')
    assert len(s[s.method.isin(RULES)&s.aggregation.eq('query_macro')])==36
    # Quantitative prose crosschecks including source/mean naming and the single positive-positive cell.
    hybrid=pd.read_csv(DATA/'matched_hybrid_raw_contrasts.csv');hybrid=hybrid[hybrid.aggregation=='query_macro']
    both=hybrid[(hybrid.seven_schedule_mean_recall>0)&(hybrid.VI>0)]
    assert len(both)==1 and both.iloc[0].dataset=='esci' and both.iloc[0].model=='bge_base' and both.iloc[0].K==100
    assert not ((hybrid.seven_schedule_mean_recall_ci_low>0)&(hybrid.VI_ci_low>0)).any()
    # All pre-existing tracked user edits are preserved; only the new directory is authored.
    modified=subprocess.check_output(['git','diff','--name-only'],cwd=ROOT).decode().splitlines()
    assert set(modified)==set(manifest['preexisting_modified'])
    figure_checks=[]
    for p in sorted((HERE/'figures').glob('*.pdf')):
        pdf=PdfReader(p);assert len(pdf.pages)==1
        page=pdf.pages[0];assert abs(float(page.mediabox.width)/72-7.1)<.01
        assert len(list(page.images))==0,(p,'raster figure')
        figure_checks.append(dict(path=p.relative_to(HERE).as_posix(),pages=1,width_inches=float(page.mediabox.width)/72,embedded_raster_images=0))
    assert len(figure_checks)==7
    latex=read(HERE/'qa/latex_check.json');assert not latex['acm_problems'] and not latex['artifact_table_problems'],latex
    for r in read(HERE/'qa/figure_geometry.json'):assert r['minimum_font_pt']>=8 and not r['text_outside_page']
    dump(HERE/'qa/validation.json',dict(status='PASS',verified_source_hash_records=verified,hybrid_conditions=28,canonical_conditions=18,primary_joint_cells=4,preserved_user_edits=modified,highest_rank_reconstruction_mismatches=0,old_states_reconciled_rows=len(matched),figures=figure_checks,checks=['all Hq including absent-in-both','states sum one','integer membership partition and gain-loss identity','exact positive cancellation','C0 hybrid full-score/rank equality','fixed cutoff support','monotone persistent inclusion/omission','shared historical query/cache compatibility','all structural VI zero','7 vector figures at 7.1 inches','ACM/table compile no overflow or unresolved references'],full_manuscript_layout='not verified; matching complete LaTeX not supplied'))
    print('VALIDATION PASS',verified,'source hash records',flush=True)

if __name__=='__main__':main()
