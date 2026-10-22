"""Sequential active-model queue; adopt the preserved worker without duplication."""
from common import *
import time,psutil,traceback
PREFIX='revision_recent_models_20261016/profiles/batch1_20261017/scripts/'

def verified(path):
    if not path.is_file():return False
    x=read(path)
    return x.get('status')=='completed' and all(Path(f['path']).is_file() and sha(f['path'])==f['sha256'] for f in x.get('outputs',[]))

def stage_complete(model,ds,stage):
    record=HERE/f'status/{model}_{ds}_{stage}.json'
    if not record.exists() or read(record)['status']!='completed':return False
    base=HERE/'runs'/model/ds
    if stage=='encode':
        return all((base/tag/'descriptor.json').exists() and valid(base/tag/'complete.json',read(base/tag/'descriptor.json')) for tag in ['queries','raw'])
    if stage in ['evaluate','canonical']:
        return verified(base/'evaluation'/('raw' if stage=='evaluate' else 'canonical')/'complete.json')
    if stage=='boundary':return verified(record) and verified(base/'evaluation/raw/boundary_audit.json')
    raise ValueError(stage)

def matching_process(entry):
    try:
        p=psutil.Process(entry['pid'])
        if abs(p.create_time()-entry['create_time'])>.01 or p.cmdline()!=entry['command']:return None
        return p if p.is_running() else None
    except (psutil.NoSuchProcess,psutil.AccessDenied):return None

def adopt_existing_worker(design):
    receipt=HERE/'audit/adopt_existing_gemma_worker.json'
    if not receipt.exists() or (HERE/'audit/adoption_completed.json').exists():return
    saved=read(receipt);assert saved['model'] in design.get('active_models',list(design['models']))
    while True:
        live=[p for entry in saved['workers'] if (p:=matching_process(entry)) is not None]
        if not live:break
        dump(HERE/'audit/active_process.json',{**saved['active_receipt'],'pid':live[0].pid,'supervisor_pid':os.getpid(),'adopted':True,'updated_utc':now()})
        time.sleep(15)
    dump(HERE/'audit/adoption_completed.json',dict(time=now(),workers=saved['workers'],reason='Preserved worker exited; verify completion before reuse or resume'))

def ensure_no_gpu_worker():
    for p in psutil.process_iter(['pid','name','cmdline']):
        if p.info['pid']==os.getpid() or 'python' not in (p.info['name'] or '').lower():continue
        args=p.info['cmdline'] or [];normalized=[x.replace('\\','/') for x in args]
        ours=any(x.endswith(PREFIX+n) for x in normalized for n in ['run.py','smoke.py','benchmark.py','memory_probe.py'])
        gpu=any(x in args for x in ['encode','boundary','canonical','all']) or any(x.endswith(('smoke.py','benchmark.py','memory_probe.py')) for x in normalized)
        assert not (ours and gpu),('Existing GPU worker: refusing duplication',p.info['pid'],args)

def child(command,log,gpu=False):
    if gpu:ensure_no_gpu_worker()
    start=now();log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('a',encoding='utf8') as f:
        f.write('\nEXEC '+json.dumps(command)+'\n');f.flush()
        process=subprocess.Popen([sys.executable,'-u',*command],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        dump(HERE/'audit/active_process.json',dict(status='running',pid=process.pid,command=command,log=str(log),started_utc=start,supervisor_pid=os.getpid()))
        code=process.wait()
    dump(HERE/f'audit/process_{process.pid}.json',dict(status='completed' if code==0 else 'failed',exit_code=code,pid=process.pid,command=command,log=str(log),started_utc=start,finished_utc=now()))
    return code

def refresh():
    child([PREFIX+'run.py','analyze'],HERE/'audit/analyze.log')
    child([PREFIX+'run.py','report'],HERE/'audit/report.log')

def main():
    design=protocol();models=design.get('active_models',list(design['models']));assert models
    assert not (set(models)&set(design.get('deferred_models',{})))
    dump(HERE/'audit/supervisor.json',dict(status='running',pid=os.getpid(),started_utc=now(),source_sha256=sha(Path(__file__)),active_models=models,deferred_models=list(design.get('deferred_models',{})),order=[m+'/'+ds for m in models for ds in ['esci','wands']]))
    adopt_existing_worker(design);results=[]
    for model in models:
        validation=HERE/f'smoke/{model}/validation.json'
        if not validation.exists():child([PREFIX+'smoke.py','--model',model],HERE/f'audit/smoke_{model}_queued.log',gpu=True)
        passed=validation.exists() and read(validation)['status']=='completed'
        if passed:
            v=read(validation);passed=v['profile_fingerprint']==fingerprint(design['models'][model]) and v['adapter_sha256']==sha(HERE/'scripts/adapters.py')
        if not passed:
            for ds in ['esci','wands']:status('encode',model,ds,state='blocked',reason='Matching official adapter validation has not completed')
            continue
        for ds in ['esci','wands']:
            for stage in ['encode','evaluate','boundary']:
                if stage_complete(model,ds,stage):
                    results.append(dict(model=model,dataset=ds,stage=stage,reused_verified_completion=True));continue
                status(stage,model,ds,state='running',started_utc=now())
                code=child([PREFIX+'run.py',stage,'--model',model,'--dataset',ds],HERE/f'audit/{model}_{ds}_{stage}.log',gpu=stage in ['encode','boundary'])
                results.append(dict(model=model,dataset=ds,stage=stage,exit_code=code))
                if code:break
            refresh()
    primary_done=all(stage_complete(m,ds,stage) for m in models for ds in ['esci','wands'] for stage in ['evaluate','boundary'])
    if primary_done:
        for model in models:
            for ds in ['esci','wands']:
                if stage_complete(model,ds,'canonical'):continue
                status('canonical',model,ds,state='running',started_utc=now())
                code=child([PREFIX+'run.py','canonical','--model',model,'--dataset',ds],HERE/f'audit/{model}_{ds}_canonical.log',gpu=True)
                results.append(dict(model=model,dataset=ds,stage='canonical',exit_code=code))
    refresh()
    done=primary_done and all(stage_complete(m,ds,'canonical') for m in models for ds in ['esci','wands'])
    dump(HERE/'audit/supervisor.json',dict(status='finished' if done else 'stopped_with_incomplete_stages',pid=os.getpid(),finished_utc=now(),results=results,all_active_experiments_completed=done,active_models=models,deferred_models=list(design.get('deferred_models',{}))))
    dump(HERE/'audit/active_process.json',dict(status='idle',updated_utc=now()))

if __name__=='__main__':
    try:main()
    except Exception as exc:
        dump(HERE/'audit/supervisor_failure.json',dict(status='blocked',time=now(),reason=str(exc),traceback=traceback.format_exc()))
        raise
