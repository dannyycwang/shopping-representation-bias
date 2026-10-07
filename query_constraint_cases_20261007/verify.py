"""Independent arithmetic/source verification: deliberately does not import core.

Recomputes rank-based metrics, all sets, four-state counts and productwise bounds.
This checks arithmetic/provenance, not an independent human semantic annotation.
"""
import math
from collections import Counter
from common import *
STATES=('MATCH','CONTRADICTION','UNKNOWN','AMBIGUOUS')

def independent_bounds(a,b,state):
    a,b=set(a),set(b)
    def tally(s):return {v:sum(state[p]==v for p in s) for v in STATES}
    x,y=tally(a),tally(b);lost,gained=tally(a-b),tally(b-a)
    # Each product has one static truth value in [0,1] if unresolved.
    # Common products get coefficient zero, which enforces coupled uncertainty.
    low=high=0
    for p in a|b:
        coefficient=int(p in b)-int(p in a)
        interval=(1,1) if state[p]=='MATCH' else (0,0) if state[p]=='CONTRADICTION' else (0,1)
        contribution=[coefficient*v for v in interval]
        low+=min(contribution);high+=max(contribution)
    u=lambda s:s['UNKNOWN']+s['AMBIGUOUS']
    return dict(before=x,after=y,lost=lost,gained=gained,confirmed_delta=y['MATCH']-x['MATCH'],
        loose_bounds=[y['MATCH']-(x['MATCH']+u(x)),y['MATCH']+u(y)-x['MATCH']],tight_bounds=[low,high],
        proven_decline=high<0,proven_increase=low>0,disappearance=x['MATCH']>0 and y['MATCH']+u(y)==0)

def main():
    protocol=read('query_constraint_cases_20261007/protocol.json')
    products,queries,j=load();labels={int(q):dict(zip(g.product_id,g.label)) for q,g in j.groupby('query_id')}
    raw=pd.read_csv(src('data/raw/label.csv'),sep='\t',dtype={'product_id':str})
    key=['query_id','product_id'];counts=raw.groupby(key).label.nunique();bad=set(counts[counts>1].index)
    clean=raw[[tuple(x) not in bad for x in raw[key].itertuples(index=False,name=None)]]
    duplicate_rows=int(clean.duplicated(key).sum());clean=clean.drop_duplicates(key)
    pd.testing.assert_frame_equal(clean[key+['label']].sort_values(key).reset_index(drop=True),j[key+['label']].sort_values(key).reset_index(drop=True))
    rawq=pd.read_csv(src('data/raw/query.csv'),sep='\t')
    pd.testing.assert_frame_equal(rawq[['query_id','query']].sort_values('query_id').reset_index(drop=True),queries[['query_id','query']].sort_values('query_id').reset_index(drop=True))
    heldout=read('phase4/config/splits.json')['wands_heldout']
    eligible=sorted(q for q in heldout if any(l=='Exact' for l in labels[q].values()))
    assert eligible==protocol['eligible_query_ids']
    tops={(r['model'],r['schedule'],r['query_id']):r['top20'] for r in lines('data/validated_top20.jsonl')}
    membership=lines('data/membership.jsonl');summaries=lines('data/query_comparisons.jsonl')
    ev=lines('candidate_evidence.jsonl');evidence={(r['query_id'],r['product_id'],r['constraint']['constraint_id']):r for r in ev}
    qdesign={r['query_id']:r for r in protocol['query_design']}
    gain={'Exact':3,'Partial':1,'Irrelevant':0}
    def met(top,qid):
        h={p for p,l in labels[qid].items() if l=='Exact'};gs=[gain.get(labels[qid].get(p),0) for p in top]
        ideal=sorted([gain[l] for l in labels[qid].values()],reverse=True)[:20]
        dcg=math.fsum(v/math.log2(i+2) for i,v in enumerate(gs));idcg=math.fsum(v/math.log2(i+2) for i,v in enumerate(ideal))
        return dict(relevant_count=len(set(top)&h),recall=len(set(top)&h)/len(h),dcg=dcg,ndcg=dcg/idcg,qrel_counts=dict(Counter(labels[qid].get(p,'Unjudged') for p in top)))
    for r in membership:
        qid=r['query_id'];a=tops[r['model'],r['before_schedule'],qid];b=tops[r['model'],r['after_schedule'],qid]
        h={p for p,l in labels[qid].items() if l=='Exact'}
        assert r['highest_n']==len(h)
        assert r['all_lost']==sorted(set(a)-set(b)) and r['all_gained']==sorted(set(b)-set(a))
        assert r['relevant_lost']==sorted((set(a)-set(b))&h) and r['relevant_gained']==sorted((set(b)-set(a))&h)
        for side,top in [('before',a),('after',b)]:
            m=met(top,qid)
            assert m['relevant_count']==r['relevant_'+side]
            for k in ['recall','dcg','ndcg']:assert abs(m[k]-r[k+'_'+side])<1e-13,(r['comparison'],qid,k)
            for label in ['Exact','Partial','Irrelevant','Unjudged']:assert m['qrel_counts'].get(label,0)==r['qrel_counts_'+side][label]
        assert r['recall_equal']==(r['relevant_before']==r['relevant_after'])
        assert r['exact_cancellation']==(len(r['relevant_lost'])==len(r['relevant_gained'])>0)
    for r in summaries:
        if not r['scanned']:continue
        qid=r['query_id'];a=set(tops[r['model'],r['before_schedule'],qid]);b=set(tops[r['model'],r['after_schedule'],qid]);h={p for p,l in labels[qid].items() if l=='Exact'}
        states={p:evidence[qid,p,r['constraint_id']]['status'] for p in a|b}
        assert independent_bounds(a,b,states)==r['full']
        assert independent_bounds(a&h,b&h,states)==r['exact']
        joints={}
        for p in a|b:
            rows=[evidence[qid,p,c] for c in qdesign[qid]['constraints']]
            ss=[x['status'] for x in rows]+[rows[0]['product_type_assessment']['status']]+[x['status'] for x in rows[0]['use_assessments'].values()]
            joints[p]='CONTRADICTION' if 'CONTRADICTION' in ss else 'AMBIGUOUS' if 'AMBIGUOUS' in ss or any(x['constraint']['scope_narrowed'] for x in rows) else 'UNKNOWN' if 'UNKNOWN' in ss or qdesign[qid]['other_explicit_clauses'] else 'MATCH'
        assert independent_bounds(a,b,joints)==r['joint']
        assert r['candidate']==(r['recall_equal'] and any(states[p]=='MATCH' for p in a-b) and any(states[p]=='CONTRADICTION' for p in b-a))
    idx=read('query_constraint_cases_20261007/case_index.json');casechecks=[]
    pmap={str(p['product_id']):p for p in products}
    for ci in idx:
        c=read('query_constraint_cases_20261007/top_cases/'+ci['case_name']+'/audit.json');s=c['summary'];qid=s['query_id']
        assert c['top_before']==tops[s['model'],'C0',qid] and c['top_after']==tops[s['model'],'C1',qid]
        for p in c['source_products']:assert p==pmap[str(p['product_id'])]
        assert len(c['products'])==len(set(c['top_before'])|set(c['top_after']))
        for schedule,x in c['matched_scoring'].items():
            m=met(x['top20'],qid)
            for k in ['recall','dcg','ndcg']:assert abs(m[k]-x['metrics'][k])<1e-13
        token=pd.read_csv(HERE/'top_cases'/ci['case_name']/'token_audit.csv')
        assert len(token)==len(c['products'])*2
        assert (token.tokens_including_special==token.tokens_without_special+token.special_tokens).all()
        assert (token.fully_fitting==(token.tokens_including_special<=token.max_length)).all()
        casechecks.append(dict(query_id=qid,union_complete=True,all_saved_source_records_match=True,all_metrics_verified=True,
            top20_cache_reproduction=c['reproduced'],grade=c['final_grade']))
    preservation=read('query_constraint_cases_20261007/qa/preexisting_state.json')
    changed=[r['path'] for r in preservation['files'] if not (ROOT/r['path']).exists() or sha(ROOT/r['path'])!=r['sha256']]
    assert not changed,('Preexisting files changed',changed)
    manifest=read('query_constraint_cases_20261007/input_manifest.json')
    for f in manifest['inputs']:assert sha(ROOT/f['path'])==f['sha256'],f['path']
    for f in read('query_constraint_cases_20261007/qa/cache_tokenizer_manifest.json'):
        path=Path(f['path']);path=path if path.is_absolute() else ROOT/path
        assert sha(path)==f['sha256'],f['path']
    missing=[]
    for comp in protocol['comparisons']:
        rr=[r for r in summaries if r['comparison']==comp['comparison'] and r['scanned']]
        for pop in ['full','exact','joint']:
            for side in ['before','after']:
                tally={state:sum(r[pop][side][state] for r in rr) for state in STATES}
                missing.append(dict(comparison=comp['comparison'],population=pop,side=side,clause_rows=len(rr),**tally,total_product_clause_instances=sum(tally.values())))
    pd.DataFrame(missing).to_csv(HERE/'missingness_summary.csv',index=False)
    dump('qa/independent_verification.json',dict(passed=True,catalog_size=len(products),all_queries=len(queries),heldout_queries=len(heldout),
        no_exact_heldout=len(heldout)-len(eligible),eligible_queries=len(eligible),highest_pairs=sum(l=='Exact' for q in eligible for l in labels[q].values()),
        raw_qrel_conflicting_pairs_removed=len(bad),raw_duplicate_rows_removed=duplicate_rows,clean_judged_pairs=len(j),
        query_comparisons=len(membership),all_summary_rows=len(summaries),assessed_clause_comparisons=sum(r['scanned'] for r in summaries),
        independent_metric_max_tolerance=1e-13,tolerance_purpose='Arithmetic verification only; equality labels use identical gain sequence or exact computed equality.',
        evidence_records=len(ev),cases=casechecks,protected_files=len(preservation['files']),protected_file_changes=changed,
        input_and_cache_hashes_verified=True,semantic_scope='Independent arithmetic; not an independent human adjudicator. Source-description caveats remain visible.'))
    print('Independent verification passed:',len(membership),'query contrasts;',sum(r['scanned'] for r in summaries),'clause contrasts;',len(idx),'cases;',len(preservation['files']),'protected files unchanged.')

if __name__=='__main__':main()
