"""Trace raw labels and downstream metric consumers; preserve historical files."""
from common import *

def main():
    verify_protocol();inputs=[];checks=[]
    def record(p):
        p=ROOT/p;inputs.append(dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p),bytes=p.stat().st_size))
    path='phase2/data/esci_repo/shopping_queries_dataset/shopping_queries_dataset_examples.parquet';record(path)
    raw=pd.read_parquet(ROOT/path,columns=['query_id','product_id','esci_label'],filters=[('product_locale','==','us'),('small_version','==',1),('split','==','test')])
    raw=raw.rename(columns={'esci_label':'label'});raw.product_id=raw.product_id.astype(str)
    _,q,j=load('esci');raw=raw[raw.query_id.isin(q.query_id)];a=sortedpairs(raw);b=sortedpairs(j)
    assert a.index.equals(b.index) and a.label.equals(b.label)
    checks.append(dict(dataset='esci',raw_source=path,queries=len(q),judgments=len(j),raw_labels_equal_processed=True,grade_mapping='stored E/C/S/I=1/.1/.01/0; correction reconstructed from E/S/C/I labels'))
    path='data/raw/label.csv';record(path)
    raw=pd.read_csv(ROOT/path,sep='\t',dtype={'product_id':str});conflicts=raw.groupby(['query_id','product_id']).label.nunique();bad=conflicts[conflicts>1].index
    raw=raw[~raw.set_index(['query_id','product_id']).index.isin(bad)].drop_duplicates(['query_id','product_id'])
    _,q,j=load('wands');a=sortedpairs(raw);b=sortedpairs(j)
    assert a.index.equals(b.index) and a.label.equals(b.label)
    checks.append(dict(dataset='wands',raw_source=path,queries=len(q),judgments=len(j),raw_labels_equal_processed=True,conflicting_pairs_removed=len(bad),grade_mapping='unchanged Exact/Partial/Irrelevant=3/1/0'))
    sources=['phase2/scripts/analyze_mitigations.py','phase2/scripts/analyze_sensitivity.py','phase2/scripts/run_dense.py','phase2/scripts/run_mitigations.py','phase2/src/encoding.py','phase3/scripts/audit_metrics.py','phase3/scripts/analyze_official_esci.py','phase3/scripts/run_invariant_method.py','phase3/scripts/run_ecommerce_retriever.py','phase3/scripts/analyze_pareto.py','phase4/scripts/audit_and_diagnose.py','phase4/scripts/common.py','phase4/scripts/run_methods.py','phase4/results/esci_gain_repair.csv','phase5_mitigation/screen.py','phase6_corrected/PHASE6_CONFIG.json','revision_completion_20260921/tables/strategy_costs.csv','revision_completion_20260921/data/structural_invariance_audit.csv']
    # Ascending canonical fusion consumes these authoritative full-score arrays.
    # Their expected hashes were already present in the frozen completion manifest.
    prior={e['path']:e.get('sha256') for e in read(COMPLETE/'data/source_manifest.json')['files']}
    for ds in ['wands','esci']:
        for model in ['minilm','bge_base']:
            path=f'phase4/results/{ds}/{model}_canonical_raw_scores.npy'
            assert sha(ROOT/path)==prior[path]
            sources.append(path)
    for path in sources:record(path)
    historical=pd.read_csv(ROOT/'phase4/results/esci_gain_repair.csv')
    historical.to_csv(HERE/'data/historical_gain_repair_inventory.csv',index=False)
    costs=pd.read_csv(COMPLETE/'tables/strategy_costs.csv');costs.to_csv(HERE/'tables/inherited_strategy_costs.csv',index=False)
    dump(HERE/'qa/raw_label_trace.json',checks)
    dump(HERE/'data/supplemental_audit_inputs.json',dict(files=inputs,scope='read-only label trace, consumer audit, and inherited cost/state validation; no design change'))
    # Capture every already modified tracked file explicitly (status whitespace is not parsed).
    modified=git('diff','--name-only').splitlines()
    dump(HERE/'qa/preexisting_tracked_changes.json',{path:sha(ROOT/path) for path in modified})
    consumers=[
      ('phase2/scripts/prepare_data.py','materializes grade','ESCI numeric S/C grades reversed; raw labels unchanged'),
      ('phase2/src/evaluation.py','cNDCG@10/20/50; RelevanceVisibilitySpearman','affected graded outputs; Recall and RelevantRecall (grade>0) unchanged'),
      ('phase2/scripts/run_dense.py','frozen dense retrieval plus evaluator','graded reporting affected; embeddings and scores do not consume gains'),
      ('phase2/scripts/run_mitigations.py','frozen text controls, evaluator, target support grade>0','graded reporting affected; positive set unchanged; no label-driven encoder training'),
      ('phase2/scripts/analyze_mitigations.py','SAR=cNDCG-lambda*VI; paired graded comparisons; success flags','ESCI graded comparisons/SAR/derived classifications affected; lambda grid not tuned'),
      ('phase2/scripts/analyze_sensitivity.py','aggregates historical cNDCG','ESCI graded summaries affected; highest-label sensitivity unchanged'),
      ('phase3/scripts/audit_metrics.py','stored-grade condensed metrics and SAR audit','affected where ESCI old grades retained'),
      ('phase3/scripts/analyze_official_esci.py','judged-candidate pool NDCG from stored grade','ESCI candidate-pool graded metric affected; not catalog-wide nDCG'),
      ('phase3/scripts/run_invariant_method.py','calls phase2 evaluator; alpha selection by WANDS dev cNDCG','ESCI reporting affected; WANDS selection gains unchanged'),
      ('phase3/scripts/run_ecommerce_retriever.py','frozen Marqo evaluation via phase2 evaluator','ESCI reporting affected, not retriever training'),
      ('phase3/scripts/analyze_pareto.py','comparisons of saved cNDCG','historical graded frontier/contrasts inherit old gains'),
      ('phase4/scripts/audit_and_diagnose.py','old/new gain reconciliation across saved rank artifacts','already corrected from labels; this audit predates the new package'),
      ('phase4/scripts/common.py','label-mapped corrected cNDCG@10 and highest-label Recall','already correct E/S/C/I=1/.1/.01/0'),
      ('phase4/scripts/run_methods.py','WANDS development Recall100, cNDCG tiebreak, simplicity','unaffected WANDS selection; frozen pipelines have no new training'),
      ('phase5_mitigation/screen.py','WANDS Exact/Irrelevant triples','unaffected by ESCI mapping'),
      ('phase6_corrected/PHASE6_CONFIG.json','WANDS graded training and dev selection','uses WANDS 3/1/0; unaffected by ESCI mapping'),
    ]
    pd.DataFrame(consumers,columns=['source','consumer','scope']).to_csv(HERE/'tables/historical_metric_consumers.csv',index=False)
    print('AUDIT',len(checks),'raw label traces;',len(historical),'historical repair rows;',len(inputs),'supplemental hashes',flush=True)

if __name__=='__main__':main()
