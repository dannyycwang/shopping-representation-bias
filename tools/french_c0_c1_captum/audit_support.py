"""Read-only provenance checks for the supplied single-case Captum runner."""
from pathlib import Path
import hashlib
import importlib.metadata as metadata
import json
import platform
import sys


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def audit_original_inputs(repo, out, source):
    manifest_path = repo / 'revision_final_strengthening_20260922/INPUT_MANIFEST.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    names = {
        'phase2/data/processed/wands_products.jsonl.gz',
        'phase2/data/processed/wands_queries.csv',
        'phase2/results/phase2_pair_ranks/wands_minilm_native_C0.parquet',
        'phase2/results/phase2_pair_ranks/wands_minilm_native_C1.parquet',
        'phase3/results/target_only_permutations/wands_minilm_pairs.parquet',
    }
    stems = ['wands_minilm_native_C0_6ffb5b340d35d66f2b33',
             'wands_minilm_native_C1_3ee547664d76abcf96a3',
             'wands_minilm_native_queries_dd4ec9d25c9825426d5d']
    files = []
    for row in manifest['files']:
        name = row['path']
        if ('models--sentence-transformers--all-MiniLM-L6-v2' not in name
                and name not in names and not any(stem in name for stem in stems)):
            continue
        path = Path(name) if Path(name).is_absolute() else repo / name
        actual = sha(path) if path.is_file() else None
        files.append(dict(path=name, exists=path.is_file(), sha256=actual,
                          expected_sha256=row['sha256'], passed=actual == row['sha256']))
    result = dict(manifest_sha256=sha(manifest_path), files=files,
                  passed=len(files) == 17 and all(row['passed'] for row in files))
    (out / 'original_input_hash_checks.json').write_text(
        json.dumps(result, indent=2) + '\n', encoding='utf-8')
    if not result['passed']:
        raise ValueError('Original input missing or changed; see original_input_hash_checks.json')
    versions = {name: metadata.version(name) for name in
                ['torch', 'transformers', 'captum', 'numpy', 'pandas', 'matplotlib', 'threadpoolctl']}
    before = json.loads((source / 'environment_before.json').read_text(encoding='utf-8'))
    unchanged = {name: versions[name] == before['packages'][name]
                 for name in versions if name != 'captum'}
    if not all(unchanged.values()):
        raise ValueError('Existing research dependency changed during Captum installation')
    return dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                versions=versions, existing_dependencies_unchanged=unchanged)
