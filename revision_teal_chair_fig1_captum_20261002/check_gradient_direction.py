"""Four bounded midpoint derivative sanity checks, not another IG integrator.

Recreates the same FP16-rounded parameters, verifies their saved parameter hash
and endpoint scores, then compares autograd to finite differences. This does not
change, replace, or validate the failed quadrature results.
"""
from pathlib import Path
import argparse, hashlib, os
os.environ['HF_HUB_OFFLINE']='1'
import numpy as np
import torch
from transformers import AutoModel
from run_teal_captum import read,dump,writecsv,ENDS

def main():
    ap=argparse.ArgumentParser();ap.add_argument('output',type=Path);args=ap.parse_args();out=args.output.resolve()
    p=read(out/'PROTOCOL.json');profile=read(out/'environment/model_profile.json')
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    m=AutoModel.from_pretrained(p['model']['name'],revision=p['model']['revision'],local_files_only=True,attn_implementation='sdpa').cuda().half().eval()
    m.requires_grad_(False);m.float();m.config.reference_compile=profile['reference_compile_diagnostic']
    h=hashlib.sha256()
    for name,par in m.named_parameters():h.update(name.encode());h.update(par.detach().cpu().numpy().tobytes())
    assert h.hexdigest()==profile['fp32_rounded_parameter_hash'],'Different diagnostic parameters'
    q=torch.from_numpy(np.load(out/'fixed_query_embedding.npy')).cuda();bc=read(out/'baseline_score_checks.json');rows=[]
    for b in ['PAD','Zero']:
        for s in ENDS:
            data=np.load(out/f'inputs/{s}_{b}_embeddings.npz');x=torch.from_numpy(data['input_embeddings']).cuda();ref=torch.from_numpy(data['baseline_embeddings']).cuda();mask=torch.from_numpy(data['attention_mask']).cuda()
            def f(z):return torch.nn.functional.normalize(m(inputs_embeds=z,attention_mask=mask).last_hidden_state[:,0].float(),dim=1)@q
            with torch.no_grad():fx=float(f(x).item());fb=float(f(ref).item())
            scores=next(z for z in bc if z['baseline']==b)
            assert abs(fx-scores['F_'+s])<=1e-6 and abs(fb-scores['F_b_'+s])<=1e-6
            alpha=.5;z=(ref+alpha*(x-ref)).detach().requires_grad_(True);grad=torch.autograd.grad(f(z).sum(),z)[0]
            deriv=float((grad.double()*(x-ref).double()).sum().item())
            for step in [.001,.0003,.0001]:
                with torch.no_grad():lo=float(f(ref+(alpha-step)*(x-ref)).item());hi=float(f(ref+(alpha+step)*(x-ref)).item())
                fd=(hi-lo)/(2*step);rows.append({'baseline':b,'schedule':s,'alpha':alpha,'step':step,'autograd_directional_derivative':deriv,
                   'finite_difference_derivative':fd,'absolute_difference':abs(deriv-fd),'endpoint_score_error':abs(fx-scores['F_'+s]),
                   'baseline_score_error':abs(fb-scores['F_b_'+s]),'gradient_finite':bool(torch.isfinite(grad).all())})
            print(b,s,'autograd',deriv,'finite differences',[r['finite_difference_derivative'] for r in rows[-3:]],flush=True)
    writecsv(out/'qa/gradient_direction_checks.csv',rows)
    dump(out/'qa/gradient_direction_checks.json',{'scope':'midpoint finite-difference sanity check only; no IG rerun, no quadrature extension, no attribute moves',
        'parameter_hash_match':True,'fp32_parameter_hash':h.hexdigest(),'all_endpoint_baseline_scores_agree_with_IG_run':True,
        'not_a_completeness_or_stability_override':True,'rows':rows})
if __name__=='__main__':main()
