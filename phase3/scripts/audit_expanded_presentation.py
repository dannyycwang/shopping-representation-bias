"""Check presentation sources and preserve legacy result checksums."""
from pathlib import Path
import json, subprocess
import pandas as pd
from hash_target_only_artifacts import digest
ROOT=Path(__file__).resolve().parents[2];P=ROOT/'paper_www2027'
for source,snapshot in [('phase3/results/target_only_permutations/summary.csv','figure5_target_cutoffs_source.csv'),('phase3/results/phase3_tables/robustness_relevance_pareto.csv','figure6_effectiveness_intervals_source.csv')]:
    pd.testing.assert_frame_equal(pd.read_csv(ROOT/source),pd.read_csv(P/'figures'/snapshot))
case=pd.read_parquet(ROOT/'phase3/results/target_only_permutations/wands_minilm_pairs.parquet')
case=case[(case.query_id==162)&(case.product_id.astype(str)=='34536')].iloc[0]
for s,rank,whole in zip(['C0','C1','C2s1','C2s2','C2s3','C2s4','C2s5'],[486,476,1527,28,582,5,5],[486,295,1303,21,484,3,4]):
    assert case[s]==rank and case['whole_'+s]==whole and case['delta_'+s]==rank-486
for ds,counts in [('wands',[25470,12240,8384,3856]),('esci',[4434,4039,3995,44])]:
    r=pd.read_csv(ROOT/f'phase3/results/phase3_tables/{ds}_reranker_pipeline.csv').iloc[0]
    assert [r.highest_pairs,r.ever_in_top100,r.common_top100,r.ever_in_top100-r.common_top100]==counts
legacy=json.loads((ROOT/'phase3/results/final_manifest.json').read_text())
checked=[]
for item in legacy['files']:
    if item['path'].startswith(('phase2/results/','phase3/results/')):
        assert digest(ROOT/item['path'])==item['sha256'];checked.append(item['path'])
text=subprocess.check_output(['pdftotext','-layout','main.pdf','-'],cwd=P).decode('utf8')
pages=text.split('\f');pages=[p for p in pages if p.strip()]
assert len(pages)==11 and 'Conclusion' in pages[9] and 'References' in pages[10]
files=list(P.glob('*.tex'))+list((P/'sections').glob('*.tex'))+list((P/'figures').glob('*.pdf'))+list((P/'figures').glob('*source.csv'))
files += [P/'main.pdf',P/'supplementary.pdf',P/'RESULT_PROVENANCE.md',ROOT/'EXPANDED_PRESENTATION_NOTES.md']
record={'revision':'expanded-presentation','body_pages':10,'reference_pages':1,'unchanged_legacy_results':checked,
        'files':[{'path':str(p.relative_to(ROOT)),'sha256':digest(p)} for p in files]}
(P/'expanded_revision_manifest.json').write_text(json.dumps(record,indent=2))
print('Verified figure sources, full case, coverage counts, 17 legacy files, and 10+1 PDF pages.')
