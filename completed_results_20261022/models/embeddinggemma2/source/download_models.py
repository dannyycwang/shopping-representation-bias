"""Download only pinned official snapshots into this revision's cache."""
from pathlib import Path
import json, os, sys, time, traceback
HERE = Path(__file__).resolve().parents[1]
os.environ['HF_HOME'] = str(HERE / 'cache/huggingface')
os.environ['HF_HUB_DISABLE_XET'] = '1'
from huggingface_hub import snapshot_download

MODELS = {'qwen3_4b': 'Qwen/Qwen3-Embedding-4B', 'embeddinggemma2': 'google/embeddinggemma-2'}
for key in sys.argv[1:] or MODELS:
    name = MODELS[key]
    api = json.loads((HERE / 'audit/models' / name.split('/')[-1] / 'api.json').read_text(encoding='utf8'))
    started = time.time()
    result = dict(model=name, revision=api['sha'], started_unix=started)
    try:
        path = snapshot_download(name, revision=api['sha'], local_dir=str(HERE/'models'/key),
                                 max_workers=2, ignore_patterns=['*.md', '.gitattributes'])
        result.update(status='completed', path=path)
    except Exception as exc:
        result.update(status='blocked', exception=type(exc).__name__, reason=str(exc))
        traceback.print_exc()
    result['seconds'] = time.time()-started
    (HERE/f'audit/download_{key}_{int(started)}.json').write_text(json.dumps(result, indent=2), encoding='utf8')
    if result['status'] == 'completed':
        (HERE/f'audit/download_{key}.json').write_text(json.dumps(result, indent=2), encoding='utf8')
    print(json.dumps(result), flush=True)
