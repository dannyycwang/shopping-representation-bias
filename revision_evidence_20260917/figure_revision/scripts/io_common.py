"""Figure revision helpers. Write exclusively within this revision folder."""
from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parents[1]
BASE = HERE.parent
ROOT = BASE.parent
DATA = HERE / 'data'
FIG = HERE / 'figures'
STEMS = ['C0', 'C1', 'C2s1', 'C2s2', 'C2s3', 'C2s4', 'C2s5']
MODELS = ['minilm', 'bge_base', 'gte_modernbert']
NAMES = dict(zip(MODELS, ['MiniLM', 'BGE', 'GTE']))

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def dump(path, value):
    path = Path(path)
    assert path.resolve().is_relative_to(HERE.resolve())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')

def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def source(path):
    path = Path(path)
    return {'path': path.relative_to(ROOT).as_posix(), 'bytes': path.stat().st_size, 'sha256': sha(path)}
