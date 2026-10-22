"""Production-policy throughput estimate from development inputs only."""
from common import *
from adapters import NativeAdapter
import argparse,time

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);args=p.parse_args()
    design=protocol() if (HERE/'PROTOCOL.json').exists() else read(HERE/'audit/protocol_candidate.json');spec=design['models'][args.model]
    products,_,j=load('wands');ids=set(j.loc[j.query_id.isin(SPLITS['wands_dev'])&j.label.eq('Exact'),'product_id'])
    chosen=sorted((p for p in products if str(p['product_id']) in ids),key=lambda p:digest('recent-model-tech|'+str(p['product_id'])))[:32]
    texts=list(dict.fromkeys(t for s in STEMS for t in texts_for(chosen,s)))
    adapter=NativeAdapter(spec)
    try:
        lengths=adapter.lengths(texts,'document');order=sorted(range(len(texts)),key=lambda i:(lengths[i],digest(texts[i])));plan=[];offset=0
        while offset<len(order):
            candidate=order[offset:offset+spec['batching']['base']];longest=max(lengths[i] for i in candidate)
            size=min(spec['batching']['base'],next(b for lim,b in spec['batching']['limits'] if longest<=lim));plan.append(order[offset:offset+size]);offset+=size
        repeats=[];timings=[];adapter.torch.cuda.reset_peak_memory_stats()
        for repeat in range(3):
            adapter.torch.cuda.synchronize();start=time.perf_counter();values=np.empty((len(texts),spec['dimension']),dtype=np.float32)
            for batch in plan:values[batch]=adapter.encode([texts[i] for i in batch],'document')
            adapter.torch.cuda.synchronize();timings.append(time.perf_counter()-start);repeats.append(values)
        seconds=float(np.median(timings));estimated_inputs=sum(v['serialized_views'] for v in design['populations'].values())
        dump(HERE/f'smoke/{args.model}/throughput_estimate.json',dict(status='completed',label='ESTIMATE from WANDS development-only technical inputs; no held-out effectiveness',
            inputs=len(texts),batches=len(plan),batch_sizes=sorted(set(map(len,plan))),seconds=timings,inputs_per_second=len(texts)/seconds,
            raw_two_dataset_estimate_hours=estimated_inputs*seconds/len(texts)/3600,canonical_three_rule_extra_estimate_hours=3*(42994+10076)*seconds/len(texts)/3600,
            estimated_input_upper_bound=estimated_inputs,peak_allocated_bytes=adapter.torch.cuda.max_memory_allocated(),
            max_repeat_component_range=float(np.ptp(np.stack(repeats),axis=0).max()),profile_fingerprint=fingerprint(spec)))
        print('ESTIMATE',len(texts)/seconds,'inputs/sec',estimated_inputs*seconds/len(texts)/3600,'hours',flush=True)
    finally:adapter.close()

if __name__=='__main__':main()

