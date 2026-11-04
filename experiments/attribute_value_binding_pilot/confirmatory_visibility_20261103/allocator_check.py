"""Validate releasing unused GPU allocator cache between native forwards."""
from core import *
from run import initialize,cached_forward

def main():
    t=Timer();plan,engine,forward,ids,truncated,cat,q=initialize('harrier')
    full=pd.read_parquet(HERE/'data/full_inputs.parquet');smoke=frame(HERE/'data/smoke_pairs.csv',dtype={'product_id':str});rows=[]
    def missing(*args):raise AssertionError('Existing smoke reference required')
    try:
        for r in full[full.product_id.isin(smoke.product_id)].itertuples():
            loc=plan['product_map'][r.product_id];texts=list(plan['catalog_batches'][loc['batch']]['texts']);texts[loc['offset']]=r.text
            context=dict(batch=loc['batch'],variant='A') if r.variant=='A' else dict(batch=loc['batch'],variant=r.variant,product_id=r.product_id)
            reference,reused=cached_forward('harrier',missing,texts,'document',context,'allocator_validation');assert reused
            ft=Timer();actual=forward(texts,'document');timing=ft.result()
            before=engine.torch.cuda.memory_reserved();allocated=engine.torch.cuda.memory_allocated()
            engine.torch.cuda.empty_cache();after=engine.torch.cuda.memory_reserved()
            rows.append(dict(product_id=r.product_id,variant=r.variant,all_batch_rows_bitwise_equal=np.array_equal(actual,reference),
                max_error=float(abs(actual-reference).max()),reserved_before=before,allocated=allocated,reserved_after=after,
                awake_forward_seconds=timing['awake_elapsed_seconds']))
            if len(rows)%11==0:print('ALLOCATOR REPLAY',len(rows),'/',55,'reserved MiB',round(after/2**20),flush=True)
    finally:engine.close()
    passed=all(r['all_batch_rows_bitwise_equal'] for r in rows);csv('qa/allocator_validation.csv',rows)
    dump('qa/allocator_gate.json',dict(passed=passed,enabled=passed,sample_manifest_sha256=sha(HERE/'sample_manifest.json'),
        script_sha256=sha(Path(__file__)),checked_forward_batches=len(rows),timing=t.result(),
        operation='torch.cuda.empty_cache() after completed forward returns CPU arrays; model weights and all arithmetic/input/batch settings unchanged'))
    assert passed;print('ALLOCATOR VALIDATION PASSED',flush=True)

if __name__=='__main__':main()
