"""Complete this current batch after all frozen retrieval conditions are saved."""
from common import *
import time, subprocess, zipfile
from datetime import datetime, timezone

required=[HERE/'experiments'/f'{ds}_{model}_{rule}_source.json'
          for ds in ['wands','esci'] for model in ['minilm','bge_base','gte_modernbert'] for rule in RULES]
last=''
while not all(p.exists() for p in required):
    done=sum(p.exists() for p in required)
    log=HERE/'experiments/canonical_run.log'
    raw=log.read_bytes() if log.exists() else b''
    text=raw.decode('utf-16' if raw[:2] in [b'\xff\xfe',b'\xfe\xff'] else 'utf8',errors='replace')
    lines=[x.strip() for x in text.splitlines() if x.strip()]
    current=f'{done}/18 controls available; '+(lines[-1] if lines else 'starting')
    if current!=last:print(current,flush=True);last=current
    time.sleep(30)

for name in ['analyze_canonical.py','write_canonical_table.py','write_evidence_report.py','validate_outputs.py']:
    print('FINALIZING',name,flush=True)
    subprocess.run([sys.executable,str(HERE/'scripts'/name)],cwd=ROOT,check=True)
manifest=json.loads((HERE/'DELIVERY_MANIFEST.json').read_text(encoding='utf8'))
archive=HERE/'evidence_package.zip'
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for rel in [*manifest['files'],'DELIVERY_MANIFEST.json']:
        z.write(HERE/rel,arcname='revision_evidence_20260917/'+rel)
with zipfile.ZipFile(archive,'r') as z:
    assert z.testzip() is None, 'Archive CRC validation failed'
dump(HERE/'FINALIZATION_COMPLETE.json',dict(completed_utc=datetime.now(timezone.utc).isoformat(),
    controls=len(required),archive=archive.name,archive_sha256=sha(archive),bytes=archive.stat().st_size,archive_crc_checked=True,
    status='All mandatory evidence, figures, audits and frozen canonical controls complete; optional 16/32 expansion not run.'))
print('COMPLETE',archive,archive.stat().st_size,flush=True)
