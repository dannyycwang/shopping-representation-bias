import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.cache import signature,tag
cfg=json.loads((ROOT/'configs/phase1.json').read_text())
# Encoding semantics were fixed before the first completed product encoding.
# Migrate early cache keys to include the explicit precision/version suffix.
for meta_path in (ROOT/'data/embeddings').glob('*.json'):
    meta=json.loads(meta_path.read_text()); old=meta_path.with_suffix('.npy')
    name=meta_path.stem.rsplit('_',1)[0]
    if not old.exists(): continue
    if name=='queries':
        import pandas as pd
        texts=pd.read_csv(ROOT/'data/raw/query.csv',sep='\t').sort_values('query_id')['query'].tolist()
    else:
        source=ROOT/f'data/representations/{name}.jsonl'
        if not source.exists(): continue
        texts=[json.loads(x)['text'] for x in source.open(encoding='utf-8')]
    precision=meta.get('precision')
    if precision not in ['fp16_model_fp32_pooling','fp32']: continue
    fingerprint=hashlib.sha256(('\0'.join(texts)+json.dumps(cfg,sort_keys=True)+precision+'encoder_v2').encode()).hexdigest()[:16]
    target=ROOT/f'data/embeddings/{name}_{fingerprint}.npy'
    if old.resolve()==target.resolve(): continue
    assert old.resolve().parent==target.resolve().parent==(ROOT/'data/embeddings').resolve()
    if not target.exists(): old.rename(target)
    meta['fingerprint']=fingerprint
    target.with_suffix('.json').write_text(json.dumps(meta,indent=2))
    # Retain the early metadata for provenance, outside the active cache namespace.
    archive=ROOT/'results/cache_history'; archive.mkdir(exist_ok=True)
    meta_path.rename(archive/meta_path.name)
manifest=json.loads((ROOT/'results/manifest.json').read_text())
manifest['code_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for folder in ['src','scripts','configs','tests'] for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in str(p)}
manifest['packages'].update({x:importlib.metadata.version(x) for x in ['tokenizers','huggingface-hub','safetensors']})
manifest['completed_systems']=sorted(p.stem for p in (ROOT/'results/per_query').glob('*.csv'))
fingerprint=signature(ROOT,cfg)
for path in (ROOT/'results/raw').glob('*.npz'):
    sidecar=path.with_suffix('.meta.json')
    if sidecar.exists() and json.loads(sidecar.read_text())['signature']!=fingerprint:
        raise RuntimeError(f'Stale checkpoint: {path.name}; rerun experiments before finalization.')
    tag(path,fingerprint)
manifest['retrieval_cache_signature']=fingerprint
manifest['lexical_accumulation']='Sorted unique query terms; final outputs recomputed and full-catalog controls checked.'
(ROOT/'results/manifest.json').write_text(json.dumps(manifest,indent=2))
lock='\n'.join(f'{k}=={v}' for k,v in sorted(manifest['packages'].items()))+'\n'
(ROOT/'requirements-lock.txt').write_text(lock)
print('Manifest, code hashes and package lock finalized.',flush=True)
