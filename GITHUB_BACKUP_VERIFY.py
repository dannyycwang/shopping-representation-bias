"""Verify restored GitHub snapshot. Run from any directory; no dependencies."""
from pathlib import Path
import hashlib,json,sys
root=Path(__file__).resolve().parent
manifest=json.loads((root/'GITHUB_BACKUP_MANIFEST.json').read_text(encoding='utf8'))
failed=[]
for i,row in enumerate(manifest['files'],1):
 p=root/row['path']
 if not p.is_file():failed.append((row['path'],'missing'));continue
 if p.stat().st_size!=row['size']:failed.append((row['path'],'size'));continue
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 if h.hexdigest()!=row['sha256']:failed.append((row['path'],'sha256'))
 if i%500==0:print('Checked',i,flush=True)
for path,reason in failed:print(reason,path)
print('Files:',len(manifest['files']),'failed:',len(failed))
sys.exit(bool(failed))
