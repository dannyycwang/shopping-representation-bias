from common import *
import datetime
assert (OUT/'queue_completed.json').exists()
paper=ROOT/'paper_www2027'
section='''

## Retrieval-first extension (2026-09-07): authoritative current mapping

Earlier entries describe archived manuscript versions. The current manuscript follows the official 8-main/12-total research-track limit and integrates an appendix in one PDF: 8 main pages, 11 total pages. Old result files are preserved. ESCI effectiveness below supersedes historical effectiveness **only in the revised presentation**, because the old Substitute/Complement gains were reversed relative to the benchmark paper.

| Current claim/table/figure | Artifact relative to repository root; fields and scope |
|---|---|
| Abstract / RQ1 general-encoder catalog VI range | `phase2/results/phase2_tables/table1_representation_sensitivity_native.csv`, `VI@20_micro` min/max, six cells, seven serializations |
| Cross-model visualization | Regenerated native-column `paper_www2027/figures/figure2_cross_model_vi.pdf` (same validated values); Phase II sensitivity table and Phase III `ecommerce_retriever_sensitivity.csv`, macro VI and query CI; no effectiveness values used |
| Introduction / Figure 1 / all case ranks | `phase3/results/target_only_permutations/wands_minilm_pairs.parquet`, query 162 product 34536; figure1 target-only source CSV; `phase4/results/illustrative_case.json` raw title and atoms |
| Target-only primary-family aggregate | `phase3/results/target_only_permutations/summary.csv`, VI@20 micro/macro; generated `sensitivitydetail.tex` |
| RQ1 always/sometimes/never counts and appendix | `phase4/results/{wands,esci}_visibility_states.csv`, K/state/pairs/macro/CI; all historical queries. WANDS 13,230 never=25,470−12,240; ESCI 395=4,434−4,039 |
| Rank-conditioned Figure and RQ2 differences | `phase4/results/*_rank_controlled_contrasts.csv`, `delta_macro_vi20`, CI and common_queries; query common support within original rank band |
| RQ2 full-input and truncated strata | `phase4/results/wands_diagnostic_strata.csv`, factor fully_fits, label Exact, level True/False, macro_vi20/CI/pairs/queries; underlying `*_diagnostic_pairs.parquet` |
| Other factor diagnostics | `phase4/results/*_diagnostic_strata.csv`, `additional_diagnostic_strata.csv`; margin, length, multiplicity, query length and literal overlap. Display bins are descriptive, not strategy tuning |
| Development selection / all unsuccessful screening configurations | `phase4/results/development_selection.csv`, frozen `phase4/config/selection.json`; Original/hybrid development per-query Recall@100 means |
| Main baseline recall/effectiveness table | `phase4/results/{dataset}_{encoder}_retrieval_summary.csv`, mean/CI/query eligibility; WANDS held-out only; ESCI standard gains; generated recall_main.tex |
| Transfer table and result differences | `phase4/results/*_primary_contrasts.csv`, method_a/method_b, Recall@100 delta/CI; `presentation_generalization.csv`; fixed selected rule transferred unchanged |
| Selected rule target-only net gain/crossing | `phase4/results/*_target_only_summary.csv`, K=100, net_macro_recall/CI/crossing_micro; rank-level selected_rule_target_only.parquet |
| Set mean and canonical comparisons / negative late variants | `phase4/results/repaired_legacy_mitigation.csv`; cNDCG@10 and paired delta/CI. Direct WANDS comparison has 383 positive-IDCG queries. No equivalence claim |
| Gain repair | `phase4/results/esci_gain_repair.csv` and per-query parquet, 79 historical rank artifacts; benchmark source https://arxiv.org/pdf/2206.06588 Section 3.1; pinned official code discrepancy in BRIEF_AUDIT.md |
| Recall-budget curve / Original budget needed | `phase4/results/*_retrieval_summary.csv` Recall@K, K=20/50/100/200/500/1000; `recall_equivalent_candidate_budget.csv` discrete grid match, not exact cost matching |
| Main cost table / full method timing | `phase4/results/*_exact_timing.csv` actual four-thread score/sort times and vector/posting bytes; `*_query_encoding_timing.csv`; summaries presentation_cost.csv |
| Estimated processing-cost curve | `phase4/results/recall_cost_frontier.csv`; query encoding + exact retrieval + K times amortized pair inference; first-stage recall, not final recall or online latency |
| Main multi-view cost table | `phase4/results/saturation_summary.csv` and `*_exact_timing.csv`; fixed m=2/4/7, no winner reselection |
| Shared-query numerical audit | `phase4/results/shared_query_cache_audit.csv`; historical reconstruction versus common-query extension condition |
| Query/product transfer scope and field coverage | `phase4/results/generalization_scope.json` and `selected_rule_coverage.csv`; no designed product-disjoint evaluation |
| Reranking budget contrasts | `phase4/results/reranking_budget_contrasts.csv`; paired query differences and bootstrap intervals |
| Hardware, tests, and final layout | `phase4/config/hardware.json`, `phase4/results/integrity_tests.log`, `pdf_validation.json`; 9 tests, 8 main / 11 total pages |
| Offline encoding cost | `phase4/results/encoding_costs.csv` from content-addressed embedding metadata; historical cached build times are not claimed as matched cold builds |
| Unique view counts / saturation | `phase4/results/unique_view_summary.csv`, underlying product-level parquet; `saturation_summary.csv` from complete exact ranks, no new selection |
| Canonical reranker table / appendix | `phase4/results/*_canonical_reranking_summary.csv`; per-query CSV and Top-20 parquet; paired contrasts `canonical_reranking_paired_contrasts.csv`; timing JSON records measured batched throughput |
| K=500 extension decision | `phase4/results/reranking_extension_decision.json`: frozen development threshold and measured local cost; final K set is visible in reranker summary |
| Legacy common-pool reranker VI | `phase3/results/phase3_tables/{wands,esci}_reranker_pipeline.csv`, common_query_macro fields; varying-input condition, conditional population; no new fixed-text result substituted |
| Legacy candidate reconciliation | `phase4/results/legacy_candidate_reconciliation.csv` and per-query detail; zero highest-relevant membership changes, despite numerical/order differences |
| Rescued/newly missed/net | `phase4/results/transition_summary.csv` and per-query `*_transitions.csv`, same query highest population, K specified |
| Additional baseline contrasts | `phase4/results/descriptive_baseline_contrasts.csv`; explicitly descriptive beyond frozen primary contrasts; no selection or p-value screening |
| Historical dataset counts | `phase2/results/dataset_statistics.json`; new data counts `phase4/data/catalog_manifest.json`; metric-specific eligibility `evaluation_denominators.csv` |
| Split sizes/seeds/rules | `phase4/config/splits.json`, frozen protocol, source hashes; 96 development / 384 held-out WANDS and 100 newly sampled ESCI queries |

Generation chain: `complete_evidence.py` → `build_presentation.py` → `write_manuscript.py` → `write_results.py` → `write_appendix.py`, followed by reviewed main.tex and LaTeX compilation. Table values are rounded from the linked CSVs. Narrative values are checked against the same artifacts, with historical facts retained only where the audited intervention/denominator matches. SHA-256 inventory: `phase4/results/revision_file_hashes.json` (excludes itself and large embedding arrays; each embedding metadata file contains its input-content hash and model fingerprint). Original archived paper and prior provenance remain available under `phase4/archive/`.
'''
path=paper/'RESULT_PROVENANCE.md';old=path.read_text(encoding='utf8');marker='\n\n## Retrieval-first extension (2026-09-07): authoritative current mapping'
old=old.split(marker)[0];path.write_text(old+section,encoding='utf8')
files=[]
for base in [P4/'scripts',P4/'config',P4/'data',OUT,paper]:
    files.extend(p for p in base.rglob('*') if p.is_file() and p.suffix in {'.py','.md','.json','.csv','.parquet','.tex','.bib','.pdf','.gz'} and p.name!='revision_file_hashes.json')
files.append(P2/'config/phase2.json')
files.append(OUT/'integrity_tests.log')
for ds in ['wands','esci']:
    files.extend(P2/'data/processed'/f'{ds}_{name}' for name in ['products.jsonl.gz','queries.csv','judgments.csv'])
files.extend(ROOT/name for name in ['BRIEF_AUDIT.md','EXPERIMENT_PROTOCOL.md','REVISION_REPORT.md','REVIEWER_RESPONSE_MAP.md','FINAL_REVISION_NOTES.md'])
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
record={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':{str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in sorted(set(files))}}
(OUT/'revision_file_hashes.json').write_text(json.dumps(record,indent=2))
print('Provenance and',len(record['files']),'file hashes written')
