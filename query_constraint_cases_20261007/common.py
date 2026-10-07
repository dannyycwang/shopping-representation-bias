"""Read-only source helpers; all writes stay in this new analysis directory."""
from pathlib import Path
import gzip, hashlib, json, subprocess
import pandas as pd
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
INPUTS=set()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def src(path):
    p=ROOT/path;INPUTS.add(p)
    if not p.exists():raise FileNotFoundError(p)
    return p
def read(path):return json.loads(src(path).read_text(encoding='utf8'))
def dump(path,obj):
    p=HERE/path;assert p.resolve().is_relative_to(HERE.resolve());p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False,default=lambda x:x.item() if hasattr(x,'item') else str(x))+'\n',encoding='utf8')
def save(path,rows):
    p=HERE/path;p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',encoding='utf8',newline='\n') as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
def lines(path):
    with (HERE/path).open(encoding='utf8') as f:return list(map(json.loads,f))
def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT).decode('utf8').strip()
def load():
    with gzip.open(src('phase2/data/processed/wands_products.jsonl.gz'),'rt',encoding='utf8') as f:products=list(map(json.loads,f))
    q=pd.read_csv(src('phase2/data/processed/wands_queries.csv'))
    j=pd.read_csv(src('phase2/data/processed/wands_judgments.csv'),dtype={'product_id':str})
    return products,q,j
def inventory():return [dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(INPUTS)]
