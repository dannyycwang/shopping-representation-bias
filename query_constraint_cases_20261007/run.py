"""One-command deterministic analysis from the committed frozen design/selection."""
import datetime, os, subprocess, sys, time
from common import HERE, ROOT, dump, git, sha

def main():
    env=dict(os.environ,PYTHONIOENCODING='utf-8',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    logs=[];start=datetime.datetime.now(datetime.timezone.utc).isoformat()
    for script in ['prepare.py','test_semantics.py','scan.py','audit_cases.py','verify.py','report.py']:
        command=[sys.executable,str(HERE/script)];now=time.perf_counter()
        print('RUN',script,flush=True)
        result=subprocess.run(command,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,encoding='utf8')
        log=HERE/'qa/run_logs'/script.replace('.py','.log');log.parent.mkdir(parents=True,exist_ok=True);log.write_text(result.stdout,encoding='utf8')
        logs.append(dict(argv=command,cwd=str(ROOT),exit_code=result.returncode,seconds=round(time.perf_counter()-now,3),output=log.relative_to(HERE).as_posix()))
        dump('qa/execution_log.json',dict(started_utc=start,source_head=git('rev-parse','HEAD'),source_branch=git('branch','--show-current'),commands=logs))
        print(script,'exit',result.returncode,'seconds',logs[-1]['seconds'],flush=True)
        if result.returncode:
            print(result.stdout);raise SystemExit(result.returncode)
    files=[p for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p!=HERE/'qa/output_manifest.json']
    dump('qa/output_manifest.json',[dict(path=p.relative_to(HERE).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(files)])
    print('Complete. Frozen search and three audits finished; prior files unchanged. See REPORT.md.',flush=True)

if __name__=='__main__':main()
