"""Invalidate retrieval checkpoints when inputs, code, packages or settings change."""
import hashlib
import importlib.metadata
import json

def signature(root, config):
    paths=[root/'data/raw'/f for f in ['product.csv','query.csv','label.csv']]
    paths += [root/p for p in ['src/representations.py','src/retrieval.py','src/evaluation.py','src/cache.py','scripts/run_phase1.py','scripts/run_order_control.py']]
    content={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    versions={x:importlib.metadata.version(x) for x in ['numpy','pandas','scipy','torch','transformers']}
    return hashlib.sha256(json.dumps({'config':config,'inputs':content,'versions':versions},sort_keys=True).encode()).hexdigest()

def tag(path, fingerprint):
    path.with_suffix('.meta.json').write_text(json.dumps({'signature':fingerprint},indent=2))

def ready(root,path,fingerprint):
    metadata=path.with_suffix('.meta.json')
    siblings=[root/f'results/per_query/{path.stem}.csv',path.with_name(path.stem+'_exact.csv'),path.with_name(path.stem+'_top100.csv')]
    return path.exists() and metadata.exists() and all(p.exists() for p in siblings) and json.loads(metadata.read_text()).get('signature')==fingerprint
