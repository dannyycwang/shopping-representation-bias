"""No model inference. Reuse all-judged ranks; audit full-rank fusion."""
from common import *
from metrics import evaluate_pairs
from collections import Counter
import importlib.util,time,unittest

def main():
    verify_protocol()
    suite=unittest.defaultTestLoader.discover(str(HERE/'scripts'),pattern='test_metrics.py')
    result=unittest.TextTestRunner(verbosity=1).run(suite)
    assert result.wasSuccessful()
    dump(HERE/'qa/evaluator_tests.json',dict(tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),passed=True,before_reevaluation=True))
    provenance=pd.read_csv(OLD/'data/encoder_profiles_and_rank_sources.csv')
    conditions=read(HERE/'data/conditions.json')
    spec=importlib.util.spec_from_file_location('frozen_representations',ROOT/'phase2/src/representations.py')
    reps=importlib.util.module_from_spec(spec);spec.loader.exec_module(reps)
    output=[]; audits=[];differences=[];labelaudit=[];structural=[];costs=[];generated=[]
    started=time.perf_counter()
    for ds in ['wands','esci']:
        p,q,j=load(ds);ids=support(ds);pids=np.array([str(x['product_id']) for x in p]);pi={v:i for i,v in enumerate(pids)};qi={int(v):i for i,v in enumerate(q.query_id)}
        a=np.array([qi[int(v)] for v in j.query_id]);b=np.array([pi[str(v)] for v in j.product_id])
        take=np.array([qi[x] for x in ids]);hmask=j.label.eq(highest(ds)).to_numpy()&j.query_id.isin(ids).to_numpy()
        expected=sortedpairs(j)
        for label,g in j.groupby('label'):
            assert (g.grade==OLD_GAINS[ds][label]).all()
            labelaudit.append(dict(dataset=ds,label=label,rows_all_queries=len(g),rows_eligible=int(g.query_id.isin(ids).sum()),stored_grade=float(g.grade.iloc[0]),old_gain=OLD_GAINS[ds][label],corrected_gain=GAINS[ds][label],all_stored_values_match_old=True))
        def tops(order):return {int(qid):pids[order[qi[int(qid)],:min(1000,order.shape[1])]].tolist() for qid in ids}
        def savedtop(path):
            if path.suffix=='.npy':return tops(np.load(path))
            t=pairread(path);return {int(k):g.sort_values('rank').product_id.tolist() for k,g in t[t.query_id.isin(ids)].groupby('query_id')}
        def makepairs(rank,score):
            f=j[['query_id','product_id','label']].copy();f['rank']=rank[a,b];f['score']=score[a,b];return f
        def check(reference,actual,meta,strict=False):
            ref=sortedpairs(reference);act=sortedpairs(actual)
            assert ref.index.equals(act.index) and ref.label.equals(act.label)
            mask=ref.index.get_level_values('query_id').isin(ids);rr=ref['rank'].to_numpy();ar=act['rank'].to_numpy();m=(rr!=ar)&mask
            hm=ref.label.eq(highest(ds)).to_numpy()&mask
            row=dict(**meta,all_judged_rank_mismatches=int(m.sum()),highest_rank_mismatches=int(np.sum((rr!=ar)&hm)),max_rank_difference=int(np.max(np.abs(rr[mask]-ar[mask]))))
            for k in [10,20,100]:
                row[f'all_judged_membership_mismatches_K{k}']=int(np.sum(((rr<=k)!=(ar<=k))&mask))
                row[f'highest_membership_mismatches_K{k}']=int(np.sum(((rr<=k)!=(ar<=k))&hm))
            for i in np.flatnonzero(m):differences.append(dict(**meta,query_id=int(ref.index[i][0]),product_id=ref.index[i][1],label=ref.label.iloc[i],saved_rank=int(rr[i]),computed_rank=int(ar[i])))
            audits.append(row)
            if strict:assert not m.any(),row
            return row
        def emit(frame,top,model,method,schedule,source):
            out=evaluate_pairs(frame,j,ids,top,ds,dict(dataset=ds,model=model,method=method,schedule=schedule,rank_source=source,catalog_size=len(p)))
            output.append(out)
            print('EVALUATED',ds,model,method,schedule,'queries',len(out),flush=True)
        # All saved conditions except canonical (which also needs catalog ranking for P1).
        for c in conditions:
            if c['dataset']!=ds or c['method'] in RULES:continue
            frame=pairread(ROOT/c['pairs']);top=savedtop(ROOT/c['top'])
            emit(frame,top,c['model'],c['method'],c['schedule'],c['pairs'])
            if 'grade' in frame:
                assert np.array_equal(frame.grade.to_numpy(),frame.label.map(OLD_GAINS[ds]).to_numpy()),c
            # Explicit reproduction of highest-label Recall and original coverage.
            if c['method']=='raw':
                oldq=pd.read_csv(ROOT/f"phase2/results/phase2_per_query/{ds}_{c['model']}_native_{c['schedule']}.csv").set_index('query_id').loc[ids]
                newq=output[-1].set_index('query_id').loc[ids]
                for col in ['HighestRecall@20','JudgedCoverage@10','JudgedCoverage@20']:
                    target=col.replace('HighestRecall','Recall')
                    np.testing.assert_allclose(oldq[col],newq[target],rtol=0,atol=1e-14)
                audits.append(dict(dataset=ds,model=c['model'],method='raw_saved_reuse',schedule=c['schedule'],all_judged_rank_mismatches=0,highest_rank_mismatches=0,recall20_and_coverage_saved_exact=True,all_support_ids_exact=True))
        bm=np.load(ROOT/f'phase4/results/{ds}/BM25_scores.npy');bo,br=ranks(bm)
        check(pairread(ROOT/f'phase4/results/{ds}/BM25_pairs.parquet'),makepairs(br,bm),dict(dataset=ds,model='lexical',method='BM25_reconstruction',schedule='fixed'),strict=True)
        assert np.array_equal(bo[:,:1000],np.load(ROOT/f'phase4/results/{ds}/BM25_top1000.npy'))
        sparsebytes=read(ROOT/f'phase4/results/{ds}/BM25_cost.json')['bytes']
        # Canonical serialization is checked over all incoming schedules, without encoding.
        for rule in RULES:
            text=[]
            for item in p:
                attrs=item['attributes'];can=canonical(attrs,rule)
                variants=[attrs,list(reversed(attrs))]+[reps._random_order(attrs,str(item['product_id']),seed) for seed in range(20260911,20260916)]
                assert Counter(can)==Counter(attrs)
                assert all(canonical(v,rule)==can for v in variants)
                text.append(reps._plain(item,can))
            textsha=hashlib.sha256('\0'.join(text).encode()).hexdigest()
            structural.append(dict(dataset=ds,rule=rule,products=len(p),incoming_schedules=7,identical_canonical_input=True,source_text_sha256=textsha,VI=0,VI_basis='structural: identical fixed inputs, frozen embeddings and deterministic full branch ranks'))
            for model in MODELS:
                c=next(x for x in conditions if x['dataset']==ds and x['model']==model and x['method']==rule)
                src=read(OLD/f'experiments/{ds}_{model}_{rule}_source.json');ep=ROOT/c['embedding'];pm=read(ep.with_suffix('.json'))
                qp=ROOT/src['query_embedding']['relative_path'];qv=np.load(qp);qm=read(qp.with_suffix('.json'))
                assert src['query_ids']==q.query_id.tolist() and src['catalog_size']==len(p)
                assert textsha==src['text_source_sha256']==pm['source_sha256']
                assert sha(qp)==src['query_embedding']['sha256']
                assert qm['source_sha256']==hashlib.sha256('\0'.join(q['query'].fillna('').astype(str)).encode()).hexdigest()
                assert pm['model']==qm['model'] and pm['profile']==qm['profile']
                assert provenance[(provenance.dataset==ds)&(provenance.model==model)].query_embedding.nunique()==1
                assert qp==ROOT/provenance[(provenance.dataset==ds)&(provenance.model==model)].iloc[0].query_embedding
                t=time.perf_counter();pv=np.load(ep,mmap_mode='r');assert pv.dtype==qv.dtype==np.float32
                score=qv@pv.T
                savedscore=ROOT/f'phase4/results/{ds}/{model}_canonical_raw_scores.npy'
                if rule=='lexical_ascending' and savedscore.exists():score=np.load(savedscore)
                order,rank=ranks(score);scoring=time.perf_counter()-t
                old=pairread(ROOT/c['pairs']);calc=makepairs(rank,score)
                check(old,calc,dict(dataset=ds,model=model,method=rule,schedule='fixed'),strict=True)
                top=tops(order);emit(old,top,model,rule,'fixed',c['pairs'])
                np.savez_compressed(HERE/f'data/{ds}_{model}_{rule}_top1000.npz',catalog_indices=order[take,:1000],query_ids=np.array(ids))
                if model=='gte_modernbert':continue
                t=time.perf_counter();fusion=1/(60+br.astype('float32'))+1/(60+rank.astype('float32'))
                assert fusion.dtype==np.float32
                fo,fr=ranks(fusion);f=makepairs(fr,fusion);method='canonical_hybrid_'+rule
                dest=HERE/f'data/{ds}_{model}_{method}_pairs.parquet';f.to_parquet(dest,index=False)
                np.savez_compressed(dest.with_name(dest.stem.replace('_pairs','_top1000')+'.npz'),catalog_indices=fo[take,:1000],query_ids=np.array(ids))
                emit(f,tops(fo),model,method,'fixed',dest.relative_to(ROOT).as_posix())
                generated.append(dict(dataset=ds,model=model,method=method,query_embedding_sha256=sha(qp),product_embedding_sha256=sha(ep),dense_full_rank_sha256=hashlib.sha256(rank.tobytes()).hexdigest(),bm25_full_rank_sha256=hashlib.sha256(br.tobytes()).hexdigest(),rrf_full_rank_sha256=hashlib.sha256(fr.tobytes()).hexdigest(),pairs_sha256=sha(dest),full_catalog=True))
                costs.append(dict(dataset=ds,model=model,method=method,catalog_size=len(p),dimension=pv.shape[1],dtype='float32',requested_view_budget=1,index_vectors=len(p),vectors_per_product=1,dense_array_bytes=pv.nbytes,sparse_array_bytes=sparsebytes,total_index_array_bytes=pv.nbytes+sparsebytes,dense_scores_per_query=len(p),lexical_branches=1,dense_branches=1,full_catalog_branch_sorts=2,fusion_sorts=1,historical_dense_encoding_seconds=pm.get('seconds'),new_encoding_seconds=0,load_score_dense_sort_seconds=scoring,fusion_sort_evaluate_save_seconds=time.perf_counter()-t,timing_scope='one local batch; includes audit/I/O, not online latency; exact scoring; no ANN',excluded_storage='tokenizer/model weights, sparse vocabulary and metadata, saved evaluation artifacts'))
                del fusion,fo,fr,f
        # Reconstruct existing seven-schedule raw hybrid with its frozen full-rank protocol.
        for model in ['minilm','bge_base']:
            prov=provenance[(provenance.dataset==ds)&(provenance.model==model)].set_index('schedule')
            qv=np.load(ROOT/prov.loc['C0','query_embedding'])
            historical=sortedpairs(pairread(COMPLETE/f'data/{ds}_{model}_hybrid_seven_pairs.parquet'))
            rawhist=sortedpairs(pairread(COMPLETE/f'data/{ds}_{model}_reconstructed_raw_seven_pairs.parquet'))
            for s in STEMS:
                pv=np.load(ROOT/prov.loc[s,'product_embedding'],mmap_mode='r')
                score=np.load(ROOT/f'phase4/results/{ds}/{model}_Original_scores.npy') if s=='C0' else qv@pv.T
                order,rank=ranks(score);calc=makepairs(rank,score)
                old=pairread(ROOT/prov.loc[s,'rank_source'])
                audit=check(old,calc,dict(dataset=ds,model=model,method='raw_reconstructed_for_hybrid',schedule=s))
                hh=sortedpairs(calc[calc.label.eq(highest(ds))&calc.query_id.isin(ids)])
                assert hh.index.equals(rawhist.index) and np.array_equal(hh['rank'],rawhist[s])
                assert audit['highest_membership_mismatches_K20']==audit['highest_membership_mismatches_K100']==0
                fusion=1/(60+br.astype('float32'))+1/(60+rank.astype('float32'));fo,fr=ranks(fusion);f=makepairs(fr,fusion)
                hh=sortedpairs(f[f.label.eq(highest(ds))&f.query_id.isin(ids)])
                assert hh.index.equals(historical.index) and np.array_equal(hh['rank'],historical[s]),(ds,model,s,'hybrid highest rank')
                topold=np.load(COMPLETE/f'data/{ds}_{model}_hybrid_{s}_top1000.npz');indices=[qi[int(x)] for x in topold['query_ids']]
                assert np.array_equal(fo[indices,:1000],topold['catalog_indices']),(ds,model,s,'hybrid top1000')
                if s=='C0':
                    check(pairread(ROOT/f'phase4/results/{ds}/{model}_hybrid_pairs.parquet'),f,dict(dataset=ds,model=model,method='raw_hybrid_C0_all_judged',schedule=s),strict=True)
                    assert np.array_equal(fusion,np.load(ROOT/f'phase4/results/{ds}/{model}_hybrid_scores.npy'))
                audit.update(hybrid_highest_ranks_exact=True,hybrid_top1000_exact=True)
                dest=HERE/f'data/{ds}_{model}_raw_hybrid_{s}_pairs.parquet';f.to_parquet(dest,index=False)
                emit(f,tops(fo),model,'raw_hybrid',s,dest.relative_to(ROOT).as_posix())
                del score,order,rank,calc,fusion,fo,fr,f
        # Save completed dataset incrementally for recovery and inspection.
        pd.concat([f for f in output if f.dataset.iloc[0]==ds],ignore_index=True).to_parquet(HERE/f'data/{ds}_per_query.parquet',index=False)
        pd.DataFrame(audits).to_csv(HERE/'qa/rank_reuse_audit.csv',index=False)
        pd.DataFrame(differences).to_csv(HERE/'qa/rank_precision_differences.csv',index=False)
    pq=pd.concat(output,ignore_index=True);pq.to_parquet(HERE/'data/per_query.parquet',index=False);pq.to_csv(HERE/'data/per_query.csv.gz',index=False)
    pd.DataFrame(labelaudit).to_csv(HERE/'qa/label_grade_audit.csv',index=False)
    pd.DataFrame(structural).to_csv(HERE/'qa/canonical_input_invariance.csv',index=False)
    pd.DataFrame(costs).to_csv(HERE/'tables/canonical_hybrid_costs.csv',index=False)
    dump(HERE/'data/canonical_hybrid_sources.json',generated)
    dump(HERE/'qa/evaluation_complete.json',dict(condition_count=len(pq[['dataset','model','method','schedule']].drop_duplicates()),per_query_records=len(pq),encoding_performed=False,elapsed_seconds=time.perf_counter()-started,recall_and_Hq_gain_swap_invariance_verified_every_record=True,all_judged_support_verified_every_record=True,protocol_sha256=sha(HERE/'PROTOCOL.json')))
    print('COMPLETE',len(pq),'records',flush=True)

if __name__=='__main__':main()
