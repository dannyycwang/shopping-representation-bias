"""Record sources and outputs without modifying prior experiment artifacts."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'phase3/results/target_only_permutations'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


if __name__ == '__main__':
    provenance = json.loads((OUT / 'provenance.json').read_text())
    sources = set()
    for entry in provenance['sources']:
        sources.update(ROOT / entry[key] for key in ['query_embedding', 'product_embedding'])
        ds, model = entry['dataset'], entry['model']
        sources.add(ROOT / f'phase2/data/representations/{ds}/C0.jsonl.gz')
        sources.add(ROOT / f'phase2/data/processed/{ds}_queries.csv')
        sources.update((ROOT / 'phase2/results/phase2_pair_ranks').glob(f'{ds}_{model}_native_C*.parquet'))
    outputs = set(OUT.glob('*')) - {OUT / 'file_hashes.json'}
    outputs.update((ROOT / 'paper_www2027/figures').glob('figure1_target_only*'))
    records = {kind: {str(p.relative_to(ROOT)): digest(p) for p in sorted(paths) if p.is_file()}
               for kind, paths in [('inputs', sources), ('outputs', outputs)]}
    (OUT / 'file_hashes.json').write_text(json.dumps(records, indent=2))
    print('Recorded', len(records['inputs']), 'inputs and', len(records['outputs']), 'outputs')
