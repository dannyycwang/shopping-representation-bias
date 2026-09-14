"""Sequential local GPU queue. Never chooses strategies using evaluation labels."""
from pathlib import Path
import sys,subprocess,time,json
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'phase4/results';S=ROOT/'phase4/scripts'
def run(name,args):
    print('START',name,flush=True)
    with (OUT/f'{name}.log').open('w',encoding='utf8') as f:subprocess.run([sys.executable,str(S/args[0]),*args[1:]],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
    print('DONE',name,flush=True)
if __name__=='__main__':
    while not (ROOT/'phase4/config/selection.json').exists():time.sleep(5)
    for ds,model in [('wands','bge_base'),('esci','bge_base'),('wands','minilm'),('esci','minilm')]:
        run(f'eval_{ds}_{model}',['run_methods.py','--dataset',ds,'--model',model])
    run('comparisons',['analyze_comparisons.py'])
    run('saturation',['run_saturation.py'])
    run('new_queries_prepare',['prepare_new_queries.py'])
    run('new_queries_evaluate',['run_methods.py','--dataset','esci_new','--model','bge_base'])
    run('new_queries_summary',['analyze_new_queries.py'])
    run('canonical_reranking',['run_canonical_reranking.py'])
    run('timing',['time_retrieval.py'])
    (OUT/'queue_completed.json').write_text(json.dumps({'completed':True,'time':time.time()}))
