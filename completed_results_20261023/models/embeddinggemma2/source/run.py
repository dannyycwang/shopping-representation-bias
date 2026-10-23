"""Stage dispatcher. A blocked model never silently falls back to a different model."""
from common import *
import argparse,traceback

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['preflight','freeze','encode','evaluate','boundary','canonical','analyze','report','all']);p.add_argument('--model',choices=['qwen3_4b','embeddinggemma2','all'],default='all');p.add_argument('--dataset',choices=['wands','esci','all'],default='all');a=p.parse_args()
    if a.stage in ['preflight','freeze','analyze','report']:
        module=__import__(a.stage);getattr(module,'main',getattr(module,'run',None))();return
    design=protocol();active=design.get('active_models',list(design['models']));models=active if a.model=='all' else [a.model];datasets=['wands','esci'] if a.dataset=='all' else [a.dataset]
    if any(model not in active for model in models):p.error('Model deferred by user; explicit resumption and scope amendment are required')
    errors=[]
    for model in models:
        for ds in datasets:
            try:
                if a.stage in ['encode','all']:
                    import encode;encode.run(model,ds)
                if a.stage in ['evaluate','all']:
                    import evaluate;evaluate.run(model,ds)
                if a.stage in ['boundary','all']:
                    import boundary;boundary.run(model,ds)
                if a.stage=='canonical':
                    import encode,evaluate;encode.run(model,ds,True);evaluate.run(model,ds,True)
            except Exception as exc:
                entry=dict(stage=a.stage,model=model,dataset=ds,exception=type(exc).__name__,reason=str(exc),traceback=traceback.format_exc(),time=now())
                dump(HERE/f'audit/failure_{model}_{ds}_{a.stage}_{int(__import__("time").time())}.json',entry)
                status(a.stage,model,ds,state='blocked',reason=str(exc));traceback.print_exc();errors.append(entry)
    if a.stage=='all':
        import analyze;analyze.run()
    if errors:sys.exit(2)

if __name__=='__main__':main()
