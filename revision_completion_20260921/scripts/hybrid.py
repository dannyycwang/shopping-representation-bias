"""Full-catalog FP32 RRF; no truncation to a top-K union, no encoding."""
from common import *
import time

def main():
    provenance=pd.read_csv(OLD/'data/encoder_profiles_and_rank_sources.csv')
    audit=[];differences=[];timings=[];bs=[]
    for ds in ['wands','esci']:
        p,q,j=p4.load(ds);pids=[str(x['product_id']) for x in p]
        pi={v:i for i,v in enumerate(pids)};qi={int(v):i for i,v in enumerate(q.query_id)}
        h=relevant(j.assign(product_id=j.product_id.astype(str)),ds)
        a=np.array([qi[int(v)] for v in h.index.get_level_values(0)])
        b=np.array([pi[str(v)] for v in h.index.get_level_values(1)])
        savedbm=np.load(ROOT/f'phase4/results/{ds}/BM25_scores.npy')
        bm,size=p4.bm25([p4._plain(x) for x in p],q['query'].fillna('').astype(str).tolist())
        assert bm.dtype==np.float32 and np.array_equal(bm,savedbm)
        assert size==read(ROOT/f'phase4/results/{ds}/BM25_cost.json')['bytes']
        _,rb=p4.ranks(bm)
        oldbm=relevant(pairread(ROOT/f'phase4/results/{ds}/BM25_pairs.parquet'),ds)
        assert h.index.equals(oldbm.index) and np.array_equal(rb[a,b],oldbm['rank'])
        bs.append(dict(dataset=ds,bm25_full_scores_exact=True,bm25_highest_ranks_exact=True,sparse_array_bytes=size,catalog_size=len(p)))
        for model in ['minilm','bge_base']:
            rec=provenance[(provenance.dataset==ds)&(provenance.model==model)].set_index('schedule')
            qpath=ROOT/rec.loc['C0','query_embedding'];qv=np.load(qpath)
            qm=read(qpath.with_suffix('.json'))
            assert qm['source_sha256']==hashlib.sha256('\0'.join(q['query'].fillna('').astype(str)).encode()).hexdigest()
            out=h[['label']].copy(); recraw=out.copy()
            for s in STEMS:
                t=time.perf_counter();path=ROOT/rec.loc[s,'product_embedding'];pv=np.load(path,mmap_mode='r');pm=read(path.with_suffix('.json'))
                assert pv.dtype==qv.dtype==np.float32 and len(pv)==len(p) and len(qv)==len(q)
                assert pm['model']==qm['model'];assert pm['profile']==qm['profile']
                with gzip.open(ROOT/f'phase2/data/representations/{ds}/{s}.jsonl.gz','rt',encoding='utf8') as f: lines=list(f)
                reps=list(map(json.loads,lines))
                assert [str(x['product_id']) for x in reps]==pids
                # Phase II run_dense hashes uncompressed JSONL lines, preserving order.
                textsha=hashlib.sha256(''.join(lines).encode()).hexdigest()
                assert pm['source_sha256']==textsha,(ds,model,s,'text ordering/hash')
                reconstructed=qv@pv.T
                dense=reconstructed
                fullscoreerror=None
                if s=='C0':
                    dense=np.load(ROOT/f'phase4/results/{ds}/{model}_Original_scores.npy')
                    fullscoreerror=float(np.max(np.abs(dense-reconstructed)))
                scoring=time.perf_counter()-t;t=time.perf_counter()
                order,rd=p4.ranks(dense)
                raw=relevant(pairread(ROOT/rec.loc[s,'rank_source']),ds)
                assert raw.index.equals(out.index)
                r=rd[a,b];recraw[s]=r
                mismatch=r!=raw['rank'].to_numpy()
                for ix in np.flatnonzero(mismatch):
                    differences.append(dict(dataset=ds,model=model,schedule=s,query_id=int(h.index[ix][0]),product_id=h.index[ix][1],saved_rank=int(raw['rank'].iloc[ix]),reconstructed_rank=int(r[ix]),saved_score=float(raw.score.iloc[ix]),reconstructed_score=float(dense[a[ix],b[ix]])))
                fusion=1/(60+rb.astype('float32'))+1/(60+rd.astype('float32'))
                assert fusion.dtype==np.float32
                ho,hr=p4.ranks(fusion);out[s]=hr[a,b]
                # A stable permutation provides K unique product IDs; validate explicitly.
                eligible=sorted(set(a.tolist()))
                assert all(len(set(row))==1000 for row in ho[eligible,:1000])
                top=ho[eligible,:1000]
                np.savez_compressed(DATA/f'{ds}_{model}_hybrid_{s}_top1000.npz',catalog_indices=top,query_ids=q.query_id.to_numpy()[eligible])
                c0_exact=None;c0_full_scores=None
                if s=='C0':
                    old=relevant(pairread(ROOT/f'phase4/results/{ds}/{model}_hybrid_pairs.parquet'),ds)
                    assert old.index.equals(out.index)
                    c0_exact=bool(np.array_equal(out[s],old['rank']))
                    c0_full_scores=bool(np.array_equal(fusion,np.load(ROOT/f'phase4/results/{ds}/{model}_hybrid_scores.npy')))
                    assert c0_exact and c0_full_scores
                    assert np.array_equal(ho[:,:1000],np.load(ROOT/f'phase4/results/{ds}/{model}_hybrid_top1000.npy'))
                row=dict(dataset=ds,model=model,schedule=s,highest_label_pairs=len(out),saved_rank_mismatches=int(mismatch.sum()),max_rank_difference=int(np.max(np.abs(r-raw['rank']))),max_highest_score_difference=float(np.max(np.abs(dense[a,b]-raw.score.to_numpy()))),C0_saved_fullscore_reconstruction_max_error=fullscoreerror,C0_hybrid_rank_exact=c0_exact,C0_hybrid_fullscore_exact=c0_full_scores,score_source='Phase IV saved Original full scores' if s=='C0' else 'FP32 full-matrix dot product of historical embeddings',query_embedding=rec.loc[s,'query_embedding'],product_embedding=rec.loc[s,'product_embedding'],catalog_size=len(p),dtype=str(fusion.dtype),ties='fixed catalog index')
                for k in [20,100]:row[f'raw_membership_mismatches_K{k}']=int(np.sum((r<=k)!=(raw['rank'].to_numpy()<=k)))
                audit.append(row);timings.append(dict(dataset=ds,model=model,schedule=s,score_and_load_seconds=scoring,full_rank_fusion_sort_seconds=time.perf_counter()-t,scope='one completion execution; includes Python and I/O; not cold encoding or ANN latency'))
                print('HYBRID',ds,model,s,'rank mismatches',mismatch.sum(),'membership',row['raw_membership_mismatches_K20'],row['raw_membership_mismatches_K100'],flush=True)
            out.reset_index().to_parquet(DATA/f'{ds}_{model}_hybrid_seven_pairs.parquet',index=False)
            recraw.reset_index().to_parquet(DATA/f'{ds}_{model}_reconstructed_raw_seven_pairs.parquet',index=False)
    pd.DataFrame(audit).to_csv(DATA/'hybrid_reconstruction_audit.csv',index=False)
    pd.DataFrame(differences,columns=['dataset','model','schedule','query_id','product_id','saved_rank','reconstructed_rank','saved_score','reconstructed_score']).to_csv(DATA/'raw_precision_differences.csv',index=False)
    pd.DataFrame(timings).to_csv(DATA/'hybrid_execution_timing.csv',index=False)
    pd.DataFrame(bs).to_csv(DATA/'bm25_reconstruction_audit.csv',index=False)
    dump(DATA/'hybrid_execution.json',dict(encoding_performed=False,canonical_rerun=False,raw_rank_policy='historical raw ranks remain authoritative; reconstructed raw saved separately for matched fusion comparison',membership_compatible=all(r['raw_membership_mismatches_K20']==r['raw_membership_mismatches_K100']==0 for r in audit),runtime=read(HERE/'qa/runtime.json')))

if __name__=='__main__':main()
