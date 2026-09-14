"""Record final reviewed PDF checks and complete delivery documentation."""
from pathlib import Path
import hashlib, json, re, datetime, subprocess

root = Path(__file__).resolve().parents[2]
out = root/'phase4/results'
paper = root/'paper_www2027'
log = (paper/'main.log').read_text(errors='replace')
assert not re.search(r'Overfull|undefined|Warning', log)
info = subprocess.check_output(['pdfinfo', str(paper/'main.pdf')]).decode(errors='replace')
assert re.search(r'Pages:\s+11\b', info)
pages = (out/'main_layout.txt').read_text(encoding='utf8').split('\f')
refs = [i+1 for i,p in enumerate(pages) if re.search(r'^\s*\d*\s+References\s{2,}',p,re.M)]
assert refs == [9], refs
aux = (paper/'main.aux').read_text(errors='replace')
assert len(re.findall(r'\\newlabel\{tab:',aux)) == 14
assert len(re.findall(r'\\newlabel\{fig:',aux)) == 4
assert '9 passed' in (out/'integrity_tests.log').read_text()
record = dict(checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),pages=11,main_pages=8,references_start_page=9,official_max_total=12,compile_success=True,undefined_references=False,overfull_boxes=False,table_labels=14,figure_labels=4,integrity_tests_passed=9,visual_review='All final pages reviewed via contact sheets and key individual renders; no clipping or overlap; first Introduction paragraph is uninterrupted.',render_directory='phase4/results/pdf_final_review',pdf_sha256=hashlib.sha256((paper/'main.pdf').read_bytes()).hexdigest(),official_source='https://www2027.thewebconf.org/research-track-papers/')
(out/'pdf_validation.json').write_text(json.dumps(record,indent=2),encoding='utf8')
p = root/'phase4/scripts/write_report.py'
s = p.read_text(encoding='utf8').replace('BGE Top-100 下，WANDS','在全部歷史查詢（不是僅 held-out queries）的 BGE Top-100 下，WANDS').replace('最終實際頁數與排版檢查見','本次 PDF 實際為 8 頁主文、合計 11 頁；排版檢查見')
p.write_text(s,encoding='utf8')
subprocess.run([str(root/'phase2/.venv/Scripts/python.exe'),str(p)],check=True,cwd=root)
p=root/'REVIEWER_RESPONSE_MAP.md'
s=p.read_text(encoding='utf8').replace('This map is finalized with `REVISION_REPORT.md`; status must follow actual artifacts, not planned experiments.','Completed 2026-09-07. This map accompanies the final Chinese `REVISION_REPORT.md` and the verified 8-main-page / 11-total-page PDF.').replace('reranking K=50/100/200 |','reranking K=50/100/200/500, with the K=500 trigger recorded in `reranking_extension_decision.json` |')
p.write_text(s,encoding='utf8')
p=root/'FINAL_REVISION_NOTES.md'
s=p.read_text(encoding='utf8')
header='''# Completed retrieval-first extension — 2026-09-07

- All scheduled experiments and conditional K=500 canonical reranking are complete. The integrated PDF has 8 main pages and 11 total pages; nine integrity tests pass.
- Figure 1 retains the verified primary-family target-only example: the same Exact-relevant turquoise chair ranks 5 versus 1,527 while all competitors remain in C0 (target C0 rank 486). These are distinct from whole-catalog ranks 3 and 1,303. The new selected-rule target-only net Recall@100 is −0.10 percentage points on WANDS and +0.12 on ESCI; the latter is small and not a broad practical-benefit claim.
- Figure 2 was replotted at native column size using unchanged validated macro VI and confidence intervals. The opening retains the broad Web visibility motivation, and Figure 1 does not interrupt its first paragraph. Detailed sensitivity statistics are in the integrated appendix.
- Added recall-first baselines, fixed-selection transfer, view-count/cost checks, candidate transitions, canonical reranking, and qualified diagnostics. ESCI gain repair and the historical/shared query-cache distinction are documented; original artifacts remain archived.
- Main interpretation: representation-induced instability is supported, but the development-selected field rule does not consistently improve held-out recall. Hybrid is a strong comparator; zero order sensitivity alone does not establish adequate recall or effectiveness equivalence.
- Remaining reviewer risks: novelty relative to tabular representational stability, post-hoc historical evaluation, English offline benchmarks, limited model/domain scope, and no demonstrated commercial exposure or user outcome.

See `REVISION_REPORT.md`, `REVIEWER_RESPONSE_MAP.md`, and `paper_www2027/RESULT_PROVENANCE.md` for the final results and source map.

---

## Historical focused-revision notes (preserved)

'''
if not s.startswith('# Completed retrieval-first'):p.write_text(header+s,encoding='utf8')
p=root/'phase4/scripts/finalize_provenance.py'
s=p.read_text(encoding='utf8').replace('integrates an appendix in one PDF.','integrates an appendix in one PDF: 8 main pages, 11 total pages.').replace('Existing `paper_www2027/figures/figure2_cross_model_vi.pdf`','Regenerated native-column `paper_www2027/figures/figure2_cross_model_vi.pdf` (same validated values)')
needle='| Offline encoding cost |'
extra='''| Estimated processing-cost curve | `phase4/results/recall_cost_frontier.csv`; query encoding + exact retrieval + K times amortized pair inference; first-stage recall, not final recall or online latency |
| Main multi-view cost table | `phase4/results/saturation_summary.csv` and `*_exact_timing.csv`; fixed m=2/4/7, no winner reselection |
| Shared-query numerical audit | `phase4/results/shared_query_cache_audit.csv`; historical reconstruction versus common-query extension condition |
| Query/product transfer scope and field coverage | `phase4/results/generalization_scope.json` and `selected_rule_coverage.csv`; no designed product-disjoint evaluation |
| Reranking budget contrasts | `phase4/results/reranking_budget_contrasts.csv`; paired query differences and bootstrap intervals |
| Hardware, tests, and final layout | `phase4/config/hardware.json`, `phase4/results/integrity_tests.log`, `pdf_validation.json`; 9 tests, 8 main / 11 total pages |
'''
if '| Shared-query numerical audit |' not in s:s=s.replace(needle,extra+needle)
s=s.replace("'REVIEWER_RESPONSE_MAP.md'])","'REVIEWER_RESPONSE_MAP.md','FINAL_REVISION_NOTES.md'])")
p.write_text(s,encoding='utf8')
subprocess.run([str(root/'phase2/.venv/Scripts/python.exe'),str(p)],check=True,cwd=root)
print('Final delivery checks and documents complete.')
