"""Compatible target-score computation; rankings always retain the full catalog.

Keep the complete query matrix, FP32 and four BLAS threads. Certify each
256-column source block against EVERY full-matrix A score, without tolerance.
Numerically incompatible blocks use the original full-catalog multiplication.
Only one row is replaced even when computing the target column. This is not
retrieval over the block: every target rank uses the complete baseline array.
"""
from core import *

BLOCK=256

def target_column(queries,one_target_replaced,full_baseline,index,gate):
    start=(index//BLOCK)*BLOCK;end=min(start+BLOCK,len(one_target_replaced))
    if str(start) in gate['bitwise_compatible_blocks']:
        local=queries@one_target_replaced[start:end].T;offset=index-start
        assert np.array_equal(local[:,:offset],full_baseline[:,start:index])
        assert np.array_equal(local[:,offset+1:],full_baseline[:,index+1:end])
        return local[:,offset].copy(),'certified_column_block'
    full=queries@one_target_replaced.T
    assert np.array_equal(full[:,:index],full_baseline[:,:index])
    assert np.array_equal(full[:,index+1:],full_baseline[:,index+1:])
    return full[:,index].copy(),'original_full_catalog_matrix'

def main():
    from run import cached_forward
    freeze_check();t=Timer();plan=read(HERE/'cache_plan_harrier.json')
    cat=np.load(ROOT/plan['files']['product']['path'],mmap_mode='r')[:plan['source_catalog_slice'][1]]
    q=np.load(ROOT/plan['files']['query']['path']);work=np.array(cat,copy=True,order='C');base=q@work.T
    good=[];records=[]
    for start in range(0,len(work),BLOCK):
        part=q@work[start:start+BLOCK].T;ref=base[:,start:start+BLOCK];same=np.array_equal(part,ref)
        if same:good.append(str(start))
        records.append(dict(start=start,end=min(start+BLOCK,len(work)),bitwise_equal=same,compared_scores=ref.size,
            mismatches=int(np.count_nonzero(part!=ref)),max_abs_error=float(abs(part-ref).max())))
    gate=dict(block_size=BLOCK,bitwise_compatible_blocks=good)
    full=pd.read_parquet(HERE/'data/full_inputs.parquet');smoke=frame(HERE/'data/smoke_pairs.csv',dtype={'product_id':str});checks=[]
    def missing(*args):raise AssertionError('Original-companion smoke vectors are required')
    for r in full[full.product_id.isin(smoke.product_id)&full.variant.ne('A')].itertuples():
        loc=plan['product_map'][r.product_id];ix=loc['baseline_index'];texts=list(plan['catalog_batches'][loc['batch']]['texts']);texts[loc['offset']]=r.text
        arr,reused=cached_forward('harrier',missing,texts,'document',dict(batch=loc['batch'],variant=r.variant,product_id=r.product_id),'column_scoring_validation');assert reused
        try:
            work[ix]=arr[loc['offset']]
            result,mode=target_column(q,work,base,ix,gate);reference=(q@work.T)[:,ix]
            checks.append(dict(product_id=r.product_id,variant=r.variant,mode=mode,query_scores=len(q),
                bitwise_equal=np.array_equal(result,reference),max_abs_error=float(abs(result-reference).max())))
        finally:work[ix]=cat[ix]
    passed=all(r['bitwise_equal'] for r in checks)
    csv('qa/column_scoring_baseline_blocks.csv',records);csv('qa/column_scoring_smoke.csv',checks)
    selected=frame(HERE/'data/selected_pairs.csv',dtype={'product_id':str}).drop_duplicates('product_id')
    fallback=selected[~selected.catalog_index.map(lambda i:str((int(i)//BLOCK)*BLOCK)).isin(good)]
    csv('qa/column_scoring_fallback_products.csv',fallback)
    dump('qa/column_scoring_gate.json',dict(**gate,passed=passed,enabled=passed,baseline_score_cells=base.size,
        smoke_variant_query_score_cells=len(checks)*len(q),fallback_products=len(fallback),
        sample_manifest_sha256=sha(HERE/'sample_manifest.json'),cache_plan_sha256=sha(HERE/'cache_plan_harrier.json'),
        script_sha256=sha(Path(__file__)),timing=t.result(),
        criterion='All source A cells certify each usable block; all 50 smoke variants match full-catalog multiplication for all 308 queries bitwise. Uncertified blocks use the original operation. No sample or score-effect selection.'))
    assert passed,'Optimization rejected: continue with original full-matrix implementation'
    print('COLUMN SCORING GATE PASSED',len(good),'/',len(records),'blocks;',len(fallback),'fallback products',flush=True)

if __name__=='__main__':main()
