"""Execute the frozen 12-comparison search; retain all directions and uncertainty."""
import collections, math
import numpy as np
from common import *
from core import *

def main():
    protocol=read('query_constraint_cases_20261007/protocol.json')
    assert sha(HERE/'protocol.json')==(HERE/'protocol.sha256').read_text().strip()
    assert sha(HERE/'query_constraints.csv')==protocol['query_constraint_sha256']
    assert sha(HERE/'query_design.py')==protocol['query_design_source_sha256']
    products,queries,j=load();pmap={str(p['product_id']):p for p in products};catalog=set(pmap)
    ids=protocol['eligible_query_ids'];qmap={q['query_id']:q for q in protocol['query_design']}
    labels={int(qid):dict(zip(g.product_id,g.label)) for qid,g in j.groupby('query_id')}
    highest={qid:{p for p,l in labels[qid].items() if l=='Exact'} for qid in ids}
    qrs=pd.read_csv(HERE/'query_constraints.csv',keep_default_na=False)
    constraints={qid:[] for qid in ids}
    for row in qrs[qrs.included.eq(True)].to_dict('records'):
        for col in ['query_ambiguous','scope_narrowed','known_seed']:row[col]=bool(row[col])
        row['other_explicit_clauses']=json.loads(row['other_explicit_clauses']);constraints[row['query_id']].append(row)
    runs=read('query_constraint_cases_20261007/data/runs.json')
    tops={};ranklookup={};scorelookup={};allp=collections.defaultdict(set)
    saved_metrics=pd.read_parquet(src('revision_graded_20260922/data/wands_per_query.parquet'))
    primary_prev=[json.loads(s) for s in src('option_coverage_20261007/data/validated_top20.jsonl').read_text(encoding='utf8').splitlines()]
    previous={(r['run'],r['query_id']):r for r in primary_prev}
    top_records=[];checks=[]
    for run in runs:
        if not run['available_rankings']:continue
        model,schedule=run['model'],run['schedule'];key=(model,schedule)
        f=pd.read_parquet(src(run['top_path']));f.product_id=f.product_id.astype(str)
        assert not f.duplicated(['query_id','product_id']).any()
        f=f[f.query_id.isin(ids)]
        assert set(f.query_id)==set(ids)
        tops[key]={int(q):g.sort_values('rank').head(20).product_id.tolist() for q,g in f.groupby('query_id')}
        ranklookup[key]={(int(r.query_id),r.product_id):int(r.rank) for r in f.itertuples()}
        scorelookup[key]={(int(r.query_id),r.product_id):float(r.score) for r in f.itertuples()}
        pairs=pd.read_parquet(src(run['pairs_path']));pairs.product_id=pairs.product_id.astype(str)
        a=pairs[['query_id','product_id','label']].sort_values(['query_id','product_id']).reset_index(drop=True)
        b=j[['query_id','product_id','label']].sort_values(['query_id','product_id']).reset_index(drop=True)
        pd.testing.assert_frame_equal(a,b)
        actual=pairs[pairs.query_id.isin(ids)&pairs['rank'].le(20)]
        saved=saved_metrics[saved_metrics.model.eq(model)&saved_metrics.method.eq('raw')&saved_metrics.schedule.eq(schedule)].set_index('query_id')
        groups=actual.groupby('query_id')
        for qid in ids:
            top=tops[key][qid]
            assert len(top)==len(set(top))==20 and set(top)<=catalog
            pairg=groups.get_group(qid) if qid in groups.groups else actual.iloc[:0]
            expected=dict(zip(pairg.product_id,pairg['rank']))
            observed={p:i+1 for i,p in enumerate(top) if p in labels[qid]}
            assert observed==expected
            metrics=metric(top,labels[qid],highest[qid])
            assert abs(metrics['recall']-saved.loc[qid,'Recall@20'])<1e-14
            assert abs(metrics['ndcg']-saved.loc[qid,'nDCG@20'])<1e-14
            if model=='bge_base':assert top==previous[schedule,qid]['top20']
            top_records.append(dict(model=model,schedule=schedule,query_id=qid,top20=top,**metrics))
            if constraints[qid]:allp[qid].update(top)
        for r in pairs[pairs.query_id.isin([q for q in ids if constraints[q]])].itertuples():
            ranklookup[key][int(r.query_id),r.product_id]=int(r.rank)
            scorelookup[key][int(r.query_id),r.product_id]=float(r.score)
        checks.append(dict(model=model,schedule=schedule,queries=len(ids),complete_unique_lists=True,qrels_agree=True,judged_ranks_agree=True,metrics_agree=True))
    save('data/validated_top20.jsonl',top_records)
    stats={(r['model'],r['schedule'],r['query_id']):r for r in top_records}
    evidence=[];ev={};types={};uses={};joints={}
    source_products=set()
    for qid,union in sorted(allp.items()):
        q=qmap[qid]
        for pid in sorted(union):
            p=pmap[pid];source_products.add(pid)
            t=product_type(p,q['product_head']);types[qid,pid]=t
            us={u:use_status(p,u) for u in q['use_clauses']};uses[qid,pid]=us
            statuslist=[]
            ranks={m+'_'+s:published_rank(ranklookup[m,s],(qid,pid)) for m,s in tops}
            schedule=[m+'_'+s for (m,s),top in tops.items() if pid in top[qid]]
            for c in constraints[qid]:
                result=classify(p,c,protocol);ev[qid,pid,c['constraint_id']]=result;statuslist.append(result['status'])
                evidence.append(dict(query_id=qid,query=q['query'],product_id=pid,schedule=schedule,constraint=c,
                    **result,qrel=labels[qid].get(pid,'Unjudged'),ranks=ranks,title=p['title'],product_class=p['class'],
                    original_description=p['description'],product_type_assessment=t,use_assessments=us,
                    source={'catalog':'phase2/data/processed/wands_products.jsonl.gz','original':'data/raw/product.csv','product_id':pid,'raw_attribute_index_base':0},
                    full_attribute_record='catalog_evidence.jsonl#product_id='+pid))
            joints[qid,pid]=joint_status(statuslist,t['status'],[u['status'] for u in us.values()],any(c['scope_narrowed'] for c in constraints[qid]),bool(q['other_explicit_clauses']))
    save('candidate_evidence.jsonl',evidence)
    save('catalog_evidence.jsonl',[pmap[p] for p in sorted(source_products)])
    case_records=[];summary=[];membership_rows=[]
    for comp in protocol['comparisons']:
        if not comp['available']:continue
        model,s0,s1=comp['model'],comp['before'],comp['after']
        for qid in ids:
            t0,t1=tops[model,s0][qid],tops[model,s1][qid];a,b=set(t0),set(t1);h=highest[qid]
            r0,r1=a&h,b&h;l,g=a-b,b-a
            ms0,ms1=stats[model,s0,qid],stats[model,s1,qid]
            delta=ms1['ndcg']-ms0['ndcg']
            equality='exact_gain_sequence' if ms0['gain_sequence']==ms1['gain_sequence'] else 'numerically_equal_only' if delta==0 else 'near_equal_numeric_only' if abs(delta)<=1e-12 else 'increased' if delta>0 else 'decreased'
            common=dict(comparison=comp['comparison'],primary=comp['primary'],model=model,before_schedule=s0,after_schedule=s1,query_id=qid,
                query=qmap[qid]['query'],highest_n=len(h),relevant_before=len(r0),relevant_after=len(r1),recall_before=ms0['recall'],recall_after=ms1['recall'],
                recall_equal=len(r0)==len(r1),exact_cancellation=len(r0-r1)==len(r1-r0)>0,relevant_lost=sorted(r0-r1),relevant_gained=sorted(r1-r0),
                all_lost=sorted(l),all_gained=sorted(g),n_all_changes=len(l)+len(g),dcg_before=ms0['dcg'],dcg_after=ms1['dcg'],
                ndcg_before=ms0['ndcg'],ndcg_after=ms1['ndcg'],delta_ndcg=delta,ndcg_equality=equality,
                qrel_counts_before=ms0['qrel_counts'],qrel_counts_after=ms1['qrel_counts'])
            membership_rows.append(common)
            if not constraints[qid]:
                summary.append(dict(**common,constraint_id=f'{qid}:none',constraint_type='none',constraint='',scope='',scanned=False,
                    exclusion_reason=qmap[qid]['exclusion_reason'],candidate=False,grade='not_applicable',direction='not_assessed'))
                continue
            union=a|b
            for c in constraints[qid]:
                cid=c['constraint_id'];states={p:ev[qid,p,cid]['status'] for p in union}
                full=bounds(a,b,states);rel=bounds(r0,r1,states)
                joint=bounds(a,b,{p:joints[qid,p] for p in union})
                candidate=common['recall_equal'] and full['lost']['MATCH']>0 and full['gained']['CONTRADICTION']>0
                type_clear=all(types[qid,p]['status']=='MATCH' for p in l|g)
                clean=not c['query_ambiguous'] and type_clear
                grade=('A' if full['proven_decline'] else 'B') if candidate and clean else 'C' if candidate else 'not_candidate'
                direction='proved_decrease' if full['proven_decline'] else 'proved_increase' if full['proven_increase'] else 'confirmed_decrease_uncertain' if full['confirmed_delta']<0 else 'confirmed_increase_uncertain' if full['confirmed_delta']>0 else 'confirmed_unchanged'
                reasons=[]
                if c['query_ambiguous']:reasons.append('query modifier/name/scope ambiguous')
                if not type_clear:reasons.append('some entering/leaving product types are contradictory or ambiguous; not a pure attribute substitution')
                if not full['proven_decline']:reasons.append('full Top20 net decline not established')
                if not candidate:reasons.append('equal Recall + MATCH-out + CONTRADICTION-in criterion not met')
                row=dict(**common,constraint_id=cid,constraint_type=c['constraint_type'],constraint=c['normalized_value'],scope=c['scope'],scanned=True,
                    exclusion_reason='',candidate=candidate,grade=grade,grading_stage='automated conservative screen; selected-case source audit still required',
                    direction=direction,query_ambiguous=c['query_ambiguous'],type_clear=type_clear,scope_narrowed=c['scope_narrowed'],
                    reasons=reasons,full=full,exact=rel,joint=joint)
                summary.append(row)
                if candidate:
                    records=[]
                    for pid in sorted(union):
                        records.append(dict(product_id=pid,title=pmap[pid]['title'],qrel=labels[qid].get(pid,'Unjudged'),
                            before_rank=published_rank(ranklookup[model,s0],(qid,pid)),after_rank=published_rank(ranklookup[model,s1],(qid,pid)),
                            before_score=scorelookup[model,s0].get((qid,pid)),after_score=scorelookup[model,s1].get((qid,pid)),
                            membership='retained' if pid in a&b else 'lost' if pid in l else 'gained',
                            constraint_assessment=ev[qid,pid,cid],type_assessment=types[qid,pid],use_assessments=uses[qid,pid],
                            joint_status=joints[qid,pid]))
                    case_records.append(dict(summary=row,top_before=t0,top_after=t1,products=records))
    save('data/query_comparisons.jsonl',summary);save('data/membership.jsonl',membership_rows);save('data/all_candidates.jsonl',case_records)
    flat=[]
    for r in summary:
        rr={k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in r.items() if k not in ['full','exact','joint']}
        for pop in ['full','exact','joint']:
            if pop in r:
                for side in ['before','after','lost','gained']:
                    rr.update({f'{pop}_{side}_{s}':n for s,n in r[pop][side].items()})
                for k in ['confirmed_delta','proven_decline','proven_increase','disappearance']:rr[f'{pop}_{k}']=r[pop][k]
                for b in ['loose_bounds','tight_bounds']:
                    rr[pop+'_'+b+'_low'],rr[pop+'_'+b+'_high']=r[pop][b]
        flat.append(rr)
    pd.DataFrame(flat).to_csv(HERE/'query_comparison_summary.csv',index=False)
    cand=[r for r in summary if r['candidate']]
    cand.sort(key=lambda r:(r['grade']=='C', {'A':0,'B':1,'C':2}[r['grade']],not r['primary'],r['ndcg_equality']!='exact_gain_sequence',r['n_all_changes'],r['query_id'],r['model'],r['after_schedule'],r['constraint_id']))
    candidate_sort=[]
    for i,r in enumerate(cand):candidate_sort.append(dict(selection_rank=i+1,comparison=r['comparison'],query_id=r['query_id'],query=r['query'],constraint_id=r['constraint_id'],constraint=r['constraint'],scope=r['scope'],grade=r['grade'],primary=r['primary'],ndcg_equality=r['ndcg_equality'],transitions=r['n_all_changes'],full_match_before=r['full']['before']['MATCH'],full_match_after=r['full']['after']['MATCH'],tight_bounds=r['full']['tight_bounds'],reasons='; '.join(r['reasons']),cache_reproduction='pending selected cases only'))
    pd.DataFrame(candidate_sort).to_csv(HERE/'candidate_ranking.csv',index=False)
    cohorts=[]
    for comp in protocol['comparisons']:
        mr=[r for r in membership_rows if r['comparison']==comp['comparison']]
        rr=[r for r in summary if r['comparison']==comp['comparison'] and r['scanned']]
        cohorts.append(dict(comparison=comp['comparison'],primary=comp['primary'],evaluation_queries=len(mr),attribute_queries=len({r['query_id'] for r in rr}),clause_rows=len(rr),
            cancellation_queries=sum(r['exact_cancellation'] for r in mr),equal_recall_queries=sum(r['recall_equal'] for r in mr),
            candidates=sum(r['candidate'] for r in rr),candidate_queries=len({r['query_id'] for r in rr if r['candidate']}),
            grade_A=sum(r['grade']=='A' for r in rr),grade_B=sum(r['grade']=='B' for r in rr),grade_C=sum(r['grade']=='C' for r in rr),
            **{d:sum(r['direction']==d for r in rr) for d in ['proved_decrease','proved_increase','confirmed_decrease_uncertain','confirmed_increase_uncertain','confirmed_unchanged']}))
    pd.DataFrame(cohorts).to_csv(HERE/'comparison_summary.csv',index=False)
    dump('qa/scan_checks.json',dict(protocol_sha256=sha(HERE/'protocol.json'),run_checks=checks,eligible_queries=len(ids),
         primary_cancellation=sum(r['exact_cancellation'] for r in membership_rows if r['primary']),candidate_records=len(case_records),
         query_comparison_count=len(membership_rows),clause_summary_rows=len(summary),scanned_clause_rows=sum(r['scanned'] for r in summary),
         evidence_records=len(evidence),unique_products=len(source_products),unknown_top20_ids=0,saved_rank_mismatches=0,saved_metric_mismatches=0))
    print(pd.DataFrame(cohorts).to_string(index=False));print('CANDIDATES',len(cand))
    print(pd.DataFrame(candidate_sort).head(25).to_string(index=False))

if __name__=='__main__':main()
