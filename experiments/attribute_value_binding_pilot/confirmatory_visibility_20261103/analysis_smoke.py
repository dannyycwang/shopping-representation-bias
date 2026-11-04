"""Exercise the analysis on completed smoke rows, isolated from formal summaries."""
from core import *
import analyze
from bootstrap_stats import QueryBootstrap,load_scores

def main():
    t=Timer();raw=load_scores('smoke');original_csv=analyze.csv
    analyze.csv=lambda name,rows:original_csv('qa/smoke_statistics/'+Path(name).name,rows)
    boot=QueryBootstrap();a=analyze.rq1(raw,boot);b=analyze.rq2(raw,boot)
    assert a[a.stratum.eq('all')].pairs.eq(5).all() and b.pairs.eq(5).all()
    assert b[(b.evaluation=='source')&(b.metric=='exit')].raw_events.eq(0).all()
    assert b[(b.evaluation=='source')&(b.metric=='delta_inclusion')].estimate.ge(0).all()
    choices=frame(HERE/'qa/smoke_statistics/rq2_choices.csv');assert len(choices)==2*2*3*5
    reference=[]
    for model in ['gte','harrier']:
        old=frame(HERE/f'results/smoke_original/{model}_scores.csv',dtype={'product_id':str})
        new=frame(HERE/f'results/smoke/{model}_scores.csv',dtype={'product_id':str})
        if model=='gte':pd.testing.assert_frame_equal(old,new,check_exact=True)
        else:
            # Re-score original-companion vectors using the pinned full-query
            # scoring profile; historical row-score smoke is preserved, not mixed.
            from run import cached_forward
            plan=read(HERE/'cache_plan_harrier.json');cat=np.load(ROOT/plan['files']['product']['path'],mmap_mode='r')[:plan['source_catalog_slice'][1]]
            q=np.load(ROOT/plan['files']['query']['path']);work=np.array(cat,copy=True,order='C');base=q@work.T
            full=pd.read_parquet(HERE/'data/full_inputs.parquet').set_index(['product_id','variant'])
            def missing(*args):raise AssertionError('Original-context smoke vectors missing; execute packing validation and smoke first')
            for r in new.itertuples():
                loc=plan['product_map'][r.product_id];qi=plan['query_map'][str(r.query_id)]['baseline_index'];ix=loc['baseline_index']
                if r.variant=='A':scores=base[qi]
                else:
                    texts=list(plan['catalog_batches'][loc['batch']]['texts']);texts[loc['offset']]=full.loc[(r.product_id,r.variant)].text
                    arr,reused=cached_forward('harrier',missing,texts,'document',dict(batch=loc['batch'],variant=r.variant,product_id=r.product_id),'cpu_smoke_reference');assert reused
                    try:work[ix]=arr[loc['offset']];scores=(q@work.T)[qi]
                    finally:work[ix]=cat[ix]
                expected=float(scores[ix]);rank=int(np.flatnonzero(np.argsort(-scores,kind='stable')==ix)[0]+1)
                assert r.score==expected and r.rank==rank
                reference.append(dict(query_id=r.query_id,product_id=r.product_id,variant=r.variant,score=expected,rank=rank,original_companion_reference=True))
    csv('qa/smoke_statistics/harrier_full_matrix_original_context_reference.csv',reference)
    timed('qa/analysis_smoke.json',t,passed=True,pairs=5,raw_rows=len(raw),selection_rows=len(choices),
        original_companion_and_packed_embeddings_and_profile_correct_scores_bitwise_equal=True,
        harrier_reference_full_sort_checks=len(reference),historical_row_score_harrier_smoke_not_used_as_formal_profile=True,not_formal_results=True)
    print('ANALYSIS SMOKE PASSED',flush=True)

if __name__=='__main__':main()
