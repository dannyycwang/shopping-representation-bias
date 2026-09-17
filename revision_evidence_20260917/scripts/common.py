"""Read-only access to inherited experiments; new outputs stay in this package."""
from pathlib import Path
import gzip, hashlib, json, sys
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
threadpool_limits(4)
HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
DATA = HERE / 'data'
FIG = HERE / 'figures'
for p in (DATA, FIG, HERE/'qa', HERE/'experiments'):
    p.mkdir(parents=True, exist_ok=True)
STEMS = ['C0', 'C1', 'C2s1', 'C2s2', 'C2s3', 'C2s4', 'C2s5']
CFG = json.loads((ROOT/'phase2/config/phase2.json').read_text())
SPLITS = json.loads((ROOT/'phase4/config/splits.json').read_text())
PROVENANCE = json.loads((ROOT/'phase3/results/target_only_permutations/provenance.json').read_text())
SEED, DRAWS = 2026091701, 10000
sys.path.insert(0, str(ROOT/'phase2'))
from src.representations import build, _plain, _random_order

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()

def dump(p, x):
    Path(p).write_text(json.dumps(x, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf8')

def source(p):
    p = Path(p)
    return dict(path=str(p.resolve()), relative_path=p.relative_to(ROOT).as_posix(), bytes=p.stat().st_size, sha256=sha(p))

def products(ds):
    with gzip.open(ROOT/f'phase2/data/processed/{ds}_products.jsonl.gz', 'rt', encoding='utf8') as f:
        return list(map(json.loads, f))

def reps(ds, stem):
    with gzip.open(ROOT/f'phase2/data/representations/{ds}/{stem}.jsonl.gz', 'rt', encoding='utf8') as f:
        return list(map(json.loads, f))

def pairread(path):
    f = pd.read_parquet(path)
    f['product_id'] = f.product_id.astype(str)
    f['query_id'] = f.query_id.astype(int)
    assert not f.duplicated(['query_id', 'product_id']).any(), path
    return f

def highest_label(ds): return 'Exact' if ds == 'wands' else 'E'

def rawwide(ds, model):
    keys = ['query_id', 'product_id', 'label']
    frames = [pairread(ROOT/f'phase2/results/phase2_pair_ranks/{ds}_{model}_native_{s}.parquet') for s in STEMS]
    base = frames[0].set_index(keys).sort_index()
    out = base[[]].copy()
    for s, f in zip(STEMS, frames):
        f = f.set_index(keys).sort_index()
        assert f.index.equals(base.index), (ds, model, s, 'alignment')
        assert (f['rank'] >= 1).all() and np.equal(f['rank'], f['rank'].astype(int)).all()
        out[s] = f['rank']
    out = out.reset_index()
    return out[out.label.eq(highest_label(ds))].reset_index(drop=True)

def populations(ds):
    q = pd.read_csv(ROOT/f'phase2/data/processed/{ds}_queries.csv')
    full = sorted(map(int, q.query_id))
    primary = sorted(SPLITS['wands_heldout']) if ds == 'wands' else sorted(SPLITS['esci_old'])
    return [('historical_full', full), ('primary_existing_evaluation', primary)]

def weights(qids):
    # Identical query IDs => identical paired draws for every K, encoder and contrast.
    n = len(qids)
    rng = np.random.default_rng(SEED)
    w = np.zeros((DRAWS, n), dtype=np.float64)
    for start in range(0, DRAWS, 250):
        ix = rng.integers(0, n, size=(min(250,DRAWS-start), n))
        np.add.at(w, (np.arange(start,start+len(ix))[:,None], ix), 1)
    return w

def ci(x):
    x = np.asarray(x); good = x[np.isfinite(x)]
    return tuple(map(float, np.quantile(good, [.025,.975]))) if len(good) else (np.nan,np.nan)

def canonical_attrs(p, rule):
    key = lambda v: (v.casefold(), v)
    if rule == 'lexical_ascending': return sorted(p['attributes'], key=key)
    if rule == 'lexical_descending': return sorted(p['attributes'], key=key, reverse=True)
    if rule == 'field_priority_type':
        def priority(v):
            field = v.split(':',1)[0].lower() if ':' in v else v[:40].lower()
            return (0 if any(t in field for t in ['producttype','type','style']) else 1, v.casefold(), v)
        return sorted(p['attributes'], key=priority)
    raise KeyError(rule)

def canonical_text(p, rule): return _plain(p, canonical_attrs(p,rule))

RULES = ['lexical_ascending', 'lexical_descending', 'field_priority_type']
