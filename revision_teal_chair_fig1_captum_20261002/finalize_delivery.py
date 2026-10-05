"""Add delivery-level status and an index manifest without changing numerical results."""
from pathlib import Path
import argparse, datetime, shutil, subprocess
import pandas as pd
from run_teal_captum import read,dump,sha

def main():
    ap=argparse.ArgumentParser();ap.add_argument('output',type=Path);args=ap.parse_args();out=args.output.resolve()
    r=read(out/'run_report.json');assert r['overall_status']!='running'
    original=out/'run_report_inference.json'
    if not original.exists():shutil.copy2(out/'run_report.json',original)
    reasons=[]
    for b,data in r['baselines'].items():
        if not data.get('passed'):
            reasons.append(f'{b}: finite Gauss-Legendre ladder ended at {data.get("nodes")} nodes without jointly passing endpoint completeness (A), paired contrast accounting (B), and contrast-vector stability (C).')
    r['failure_reasons']=reasons
    r['paper_usability']={'native_panel':r['native_ready'],'PAD_attribution_panel':r['PAD_plot_ready'],
       'baseline_robust_claim':False,'independent_attribute_causal_claim':False,
       'scope':'Saved native observation only; failed numerical attributions cannot support attribute-level interpretation.' if not r['PAD_plot_ready'] else 'Selected-case attribution redistribution only, with stated baseline limitations.'}
    r['native_precision_limitation']='Cached ranks reconstruct exactly. Current singleton and restored original length-bucket forwards differ from saved vectors under both reference_compile settings. Historical kernel/compile state is incompletely recorded. Do not claim bitwise execution-profile identity.'
    r['source_model_tokenizer_revision']=read(out/'PROTOCOL.json')['model']['revision']
    r['attribution_attempts_retained']=len(list((out/'attempts').glob('*/*/*/qa.json')))
    r['validation_audit']=read(out/'qa/artifact_and_numeric_audit.json')['status'] if (out/'qa/artifact_and_numeric_audit.json').exists() else 'not_yet_run'
    r['visual_qa']=read(out/'qa/visual_qa.json')['status'] if (out/'qa/visual_qa.json').exists() else 'not_yet_run'
    r['delivery_index']='INDEX.txt'
    dump(out/'run_report.json',r)
    summaries=[]
    if (out/'completeness_attempts.csv').exists():
        a=pd.read_csv(out/'completeness_attempts.csv',float_precision='round_trip')
        for (b,s),g in a.groupby(['baseline','schedule'],sort=False):
            last=g.sort_values('nodes').iloc[-1]
            summaries.append({'baseline':b,'schedule':s,'nodes':int(last.nodes),'absolute_completeness_residual':abs(float(last.residual)),
              'completeness_tolerance':float(last.tolerance),'endpoint_passed':bool(last.completeness_passed),
              'paired_status':r['baselines'][b]['status']})
    dump(out/'qa/four_IG_final_status.json',summaries)
    files={str(p.relative_to(out)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(out.rglob('*'))
           if p.is_file() and p.name!='DELIVERY_MANIFEST.json' and '__pycache__' not in p.parts}
    dump(out/'DELIVERY_MANIFEST.json',{'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'output':str(out),'overall_status':r['overall_status'],'files':files})
    print('Delivery indexed:',len(files),'files;',r['overall_status'])
if __name__=='__main__':main()
