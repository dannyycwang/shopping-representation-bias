"""Same-shape independent-row packing, gated by exact native replay.

No model/pooling/precision settings change. Keep original B and padded L by
retaining an original maximum-length guard row. Every usable row position
must reproduce A bitwise; smoke must additionally reproduce every C/O vector.
"""
from core import *

def packed_product(key,forward,inputids,context,rows,baseline,stage):
    from run import cached_forward
    b=len(context['texts']);original_lengths=[len(x) for x in inputids(context['texts'],'document')]
    guard=context['texts'][int(np.argmax(original_lengths))];pad=max(original_lengths)
    texts=rows.text.tolist();lengths=[len(x) for x in inputids(texts,'document')]
    reason=None
    if b<3:reason='source_batch_too_small'
    elif max(lengths)>pad:reason='would_change_padded_length'
    if reason:return None,dict(eligible=False,reason=reason,batch_size=b,padded_length=pad)
    a=rows[rows.variant.eq('A')].iloc[0];pid=a.product_id;slots=b-1
    check,_=cached_forward(key,forward,[a.text]*slots+[guard],'document',dict(packing='A_all_slots',product_id=pid,batch_size=b,padded_length=pad),stage)
    if not all(np.array_equal(check[i],baseline) for i in range(slots)):
        return None,dict(eligible=False,reason='A_slot_replay_not_bitwise_equal',batch_size=b,padded_length=pad,
            max_error=float(abs(check[:slots]-baseline).max()))
    variants=rows[~rows.variant.eq('A')];out={}
    for start in range(0,len(variants),slots):
        part=variants.iloc[start:start+slots];batch=part.text.tolist()+[guard]*(b-len(part))
        assert len(batch)==b and max(len(x) for x in inputids(batch,'document'))==pad
        arr,_=cached_forward(key,forward,batch,'document',dict(packing='same_shape',product_id=pid,variants=part.variant.tolist(),batch_size=b,padded_length=pad),stage)
        for i,r in enumerate(part.itertuples()):out[(pid,r.variant)]=arr[i]
    return out,dict(eligible=True,reason='all_A_positions_bitwise_equal',batch_size=b,padded_length=pad,packed_forward_batches=1+int(np.ceil(10/slots)))

def main():
    import argparse
    from run import initialize,cached_forward
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['gte','harrier'],required=True);key=p.parse_args().model
    freeze_check();t=Timer();plan,engine,forward,inputids,truncated,catalog,qemb=initialize(key)
    smoke=frame(HERE/'data/smoke_pairs.csv',dtype={'product_id':str});full=pd.read_parquet(HERE/'data/full_inputs.parquet');records=[]
    try:
        for pid in smoke.product_id.unique():
            loc=plan['product_map'][pid];context=plan['catalog_batches'][loc['batch']];rows=full[full.product_id.eq(pid)]
            packed,audit=packed_product(key,forward,inputids,context,rows,catalog[loc['baseline_index']],'packing_validation')
            exact=True;error=0
            if packed is not None:
                for r in rows[~rows.variant.eq('A')].itertuples():
                    texts=list(context['texts']);texts[loc['offset']]=r.text
                    original,_=cached_forward(key,forward,texts,'document',dict(batch=loc['batch'],variant=r.variant,product_id=pid),'packing_validation')
                    diff=float(abs(original[loc['offset']]-packed[(pid,r.variant)]).max());error=max(error,diff)
                    exact=exact and np.array_equal(original[loc['offset']],packed[(pid,r.variant)])
            records.append(dict(product_id=pid,**audit,all_variants_match_original_context=exact,max_variant_error=error))
    finally:engine.close()
    csv(f'qa/packing_{key}.csv',records)
    passed=all(r['all_variants_match_original_context'] for r in records)
    dump(f'qa/packing_{key}_gate.json',dict(passed=passed,enabled=passed and any(r['eligible'] for r in records),
        sample_manifest_sha256=sha(HERE/'sample_manifest.json'),packing_script_sha256=sha(Path(__file__)),timing=t.result(),
        criterion='bitwise equality of ALL smoke C/O embeddings plus A at every packed slot; no score/rank-dependent selection'))
    print('PACKING GATE',key,passed,records,flush=True)

if __name__=='__main__':main()
