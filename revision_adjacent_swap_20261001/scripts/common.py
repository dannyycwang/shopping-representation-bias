"""Shared, outcome-independent definitions for the adjacent-entry experiment."""
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
from pathlib import Path
import collections, datetime, gzip, hashlib, importlib.util, json, platform, re, subprocess, sys
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
threadpool_limits(4)
HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
HARM = ROOT / 'revision_section6_20260924/harmonized'
MODELS = ['minilm', 'bge_base']
DATASETS = ['wands', 'esci']
MODEL_SPECS = {x['key']: x for x in json.loads((ROOT/'phase2/config/phase2.json').read_text())['models']}
spec = importlib.util.spec_from_file_location('adjacent_frozen_serializer', ROOT/'phase2/src/representations.py')
REP = importlib.util.module_from_spec(spec)
spec.loader.exec_module(REP)

def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def digest(text): return hashlib.sha256(text.encode('utf8')).hexdigest()
def canonical_json(value): return json.dumps(value, ensure_ascii=False, separators=(',', ':'), sort_keys=True)
def selection_hash(purpose, *ids): return digest(canonical_json(['20261001', purpose, *[str(x) for x in ids]]))
def read(path): return json.loads(Path(path).read_text(encoding='utf8'))
def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''): h.update(b)
    return h.hexdigest()
def dump(path, value):
    path = Path(path)
    assert path.resolve().is_relative_to(HERE.resolve())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False,
        default=lambda x: x.item() if isinstance(x, np.generic) else str(x))+'\n', encoding='utf8')
def csv(frame, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format='%.17g')
def git(*args): return subprocess.check_output(['git', *args], cwd=ROOT).decode('utf8').strip()
def load(ds):
    with gzip.open(ROOT/f'phase2/data/processed/{ds}_products.jsonl.gz', 'rt', encoding='utf8') as f:
        products = list(map(json.loads, f))
    queries = pd.read_csv(ROOT/f'phase2/data/processed/{ds}_queries.csv')
    judgments = pd.read_csv(ROOT/f'phase2/data/processed/{ds}_judgments.csv', dtype={'product_id':str})
    return products, queries, judgments
def support(ds): return read(ROOT/'revision_graded_20260922/PROTOCOL.json')['populations'][ds]['eligible_ids']
def input_hash(enc, i):
    return digest(json.dumps({k:enc[k][i] for k in ['input_ids','token_type_ids','attention_mask'] if k in enc},
        sort_keys=True, separators=(',', ':')))
def bucket(length, cap): return next(b for b in [32,64,128,256,512] if b >= min(length, cap))
def variants(product):
    """Deduplicate by complete serialization; preserve every 1-based source position."""
    attrs = product['attributes']; original = REP._plain(product); out = {}
    noops = 0
    for i in range(len(attrs)-1):
        if attrs[i] == attrs[i+1]: noops += 1; continue
        a = list(attrs); a[i], a[i+1] = a[i+1], a[i]
        text = REP._plain(product, a)
        if text == original: noops += 1; continue
        if text not in out: out[text] = dict(text=text, positions=[], attributes=a)
        out[text]['positions'].append(i+1)
    return list(out.values()), noops
def audit(product, variant):
    a = product['attributes']; b = variant['attributes']; original = REP._plain(product)
    checks = {}
    checks['entry_multiset'] = collections.Counter(a) == collections.Counter(b)
    checks['keys_values'] = collections.Counter(x.partition(':') for x in a) == collections.Counter(x.partition(':') for x in b)
    checks['numeric_multiset'] = collections.Counter(re.findall(r'\d+(?:\.\d+)?', original)) == collections.Counter(re.findall(r'\d+(?:\.\d+)?', variant['text']))
    checks['fixed_text'] = [(k,v) for k,v in REP._blocks(product,a) if k != 'attributes'] == [(k,v) for k,v in REP._blocks(product,b) if k != 'attributes']
    checks['adjacent_transposition'] = all(b[:pos-1] == a[:pos-1] and b[pos+1:] == a[pos+1:] and
        b[pos-1:pos+1] == a[pos-1:pos+1][::-1] and a[pos-1] != a[pos] for pos in variant['positions'])
    checks['other_relative_order'] = all(b[:pos-1]+b[pos+1:] == a[:pos-1]+a[pos+1:] for pos in variant['positions'])
    checks['identity'] = REP._plain(product, list(a)) == original
    checks['canonical'] = REP._plain(product, sorted(a, key=lambda x:(x.casefold(),x))) == REP._plain(product, sorted(b, key=lambda x:(x.casefold(),x)))
    old = REP.audit(product, 'C1_reverse_attribute_order', variant['text'], original)
    checks['repository_audit'] = old['pass']
    checks['pass'] = all(checks.values())
    return checks
def dot(vectors, query): return np.sum(vectors * query[None,:], axis=1, dtype=np.float32)
def replacement(base, indices, values):
    """Exact rank after removing the original target; stable catalog-index ties."""
    order = np.argsort(-base, kind='stable')
    keys = np.empty(len(base), dtype=[('negative','f4'),('index','i8')])
    keys['negative'] = -base[order]; keys['index'] = order
    needles = np.empty(len(values), dtype=keys.dtype)
    needles['negative'] = -values; needles['index'] = indices
    return 1 + np.searchsorted(keys, needles) - (base[indices] > values).astype(int)
def boundary(base, target_index, k):
    order = np.argsort(-base, kind='stable'); competitors = order[order != target_index]
    j = int(competitors[k-1]); return j, base[j]
def verify_freeze():
    frozen = read(HERE/'freeze_manifest.json')
    for p,h in frozen['files'].items(): assert sha(HERE/p) == h, p
    for p,h in frozen['source_files'].items(): assert sha(ROOT/p) == h, p
    return frozen
