"""Read-only provenance and supervised-data coverage audit."""
from screen import *
def main():
    c=setup();p,q,j,pi,qi=data();triples=json.loads((HERE/'triples.json').read_text())
    training_products={x[k] for x in triples for k in ['positive','negative']}
    rows=[]
    for split in ['train','validation','test']:
        g=j[j.query_id.isin(c[split])];h=g[g.label.eq('Exact')]
        rows.append({'split':split,'queries':len(c[split]),'eligible_exact_queries':h.query_id.nunique(),'highest_pairs':len(h),'unique_highest_products':h.product_id.nunique(),'highest_pairs_with_training_product':int(h.product_id.astype(str).isin(training_products).sum())})
    pd.DataFrame(rows).to_csv(HERE/'data_coverage.csv',index=False)
    sources=[HERE/'README.md',HERE/'config.json',HERE/'triples.json',ROOT/'phase4/config/splits.json',ROOT/'phase4/results/wands_product_features.csv',ROOT/'phase3/results/target_only_permutations/wands_bge_base_pairs.parquet']
    sources += list((ROOT/'phase2/data/processed').glob('wands*'))
    sources += [ROOT/f'phase2/results/phase2_pair_ranks/wands_bge_base_native_{s}.parquet' for s in STEMS]
    sources += [ROOT/f'phase4/results/wands/bge_base_{s}_pairs.parquet' for s in ['canonical_raw','set_mean']]
    sources += list((ROOT/'paper_www2027').rglob('*.tex'))+[ROOT/'paper_www2027/main.pdf',ROOT/'paper_www2027/references.bib']
    dump(HERE/'source_audit.json',[{'path':str(f.relative_to(ROOT)),'sha256':sha(f)} for f in sources if f.is_file()])
    print(pd.DataFrame(rows).to_string(index=False),flush=True)
if __name__=='__main__':main()
