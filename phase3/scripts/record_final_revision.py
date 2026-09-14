"""Audit unchanged legacy results and record the final manuscript revision."""
from pathlib import Path
import json
from hash_target_only_artifacts import digest

ROOT = Path(__file__).resolve().parents[2]
old = json.loads((ROOT / 'phase3/results/final_manifest.json').read_text())
checked = []
for entry in old['files']:
    path = entry['path']
    if path.startswith(('phase2/results/', 'phase3/results/')):
        p = ROOT / path
        assert digest(p) == entry['sha256'], f'Legacy result changed: {path}'
        checked.append(path)
files = set((ROOT / 'paper_www2027').glob('*.tex'))
files.update((ROOT / 'paper_www2027/sections').glob('*.tex'))
files.update((ROOT / 'paper_www2027/figures').glob('figure1_target_only*'))
files.update(ROOT / p for p in ['paper_www2027/main.pdf', 'paper_www2027/references.bib',
    'paper_www2027/RESULT_PROVENANCE.md', 'FINAL_REVISION_NOTES.md',
    'phase3/results/target_only_permutations/file_hashes.json',
    'phase3/scripts/analyze_target_only_permutations.py',
    'phase3/scripts/make_target_only_figure.py', 'phase3/tests/test_target_only_permutations.py'])
record = {'revision': 'focused-target-only-control', 'date': '2026-09-06',
          'unchanged_legacy_result_files': checked,
          'files': [{'path': str(p.relative_to(ROOT)), 'sha256': digest(p)} for p in sorted(files)]}
(ROOT / 'paper_www2027/final_revision_manifest.json').write_text(json.dumps(record, indent=2))
print(f'Verified {len(checked)} unchanged legacy result files; recorded {len(files)} revision files.')
