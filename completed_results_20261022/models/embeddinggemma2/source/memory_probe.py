"""Technical-only memory diagnosis after the original SDPA profile failed."""
from common import *
from adapters import NativeAdapter
import argparse,gc,time

def main():
    p=argparse.ArgumentParser();p.add_argument('--batch',type=int,default=1);p.add_argument('--backend',default='sdpa');p.add_argument('--math-only',action='store_true');p.add_argument('--steps',type=int,default=200);a=p.parse_args()
    spec=protocol()['models']['embeddinggemma2'].copy();spec['attention_backend']=a.backend
    adapter=NativeAdapter(spec);torch=adapter.torch
    if a.math_only:
        torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_cudnn_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    rows=[];start=time.time()
    for i in range(a.steps):
        # Deterministic synthetic lengths; no held-out judgment or retrieval outcome.
        text='Technical product record. '+'color: blue | '* (8+i)
        array=adapter.encode([text]*a.batch,'document')
        torch.cuda.synchronize()
        if i%10==0:
            row=dict(step=i,batch=a.batch,tokens=adapter.lengths([text],'document')[0],allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved(),free=torch.cuda.mem_get_info()[0],elapsed=time.time()-start)
            rows.append(row);print(json.dumps(row),flush=True)
            dump(HERE/f'audit/memory_probe_{a.backend}_{a.batch}_{a.math_only}.json',rows)
    adapter.close()

if __name__=='__main__':main()
