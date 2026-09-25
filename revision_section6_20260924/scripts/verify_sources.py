"""Focused source identity checks, without altering any historical payload."""
from common import *
from collections import Counter

def main():
    rows=[]
    for ds in ['wands','esci']:
        p,q,j=load(ds);expected=[str(x['product_id']) for x in p]
        for s in STEMS:
            path=ROOT/f'phase2/data/representations/{ds}/{s}.jsonl.gz'
            with gzip.open(path,'rt',encoding='utf8') as f:
                n=0
                for product,line in zip(p,f,strict=True):
                    stored=json.loads(line);assert str(stored['product_id'])==str(product['product_id']);assert stored['text']==text_for(product,s)
                    assert Counter(attrs(product,s))==Counter(product['attributes']);n+=1
            rows.append(dict(dataset=ds,schedule=s,products=n,source=path.relative_to(ROOT).as_posix(),sha256=sha(path),original_strings_exact=True,whole_entry_multiplicities=True,fixed_nonattribute_text=True))
        # Every raw schedule uses identical judged pair support and labels.
        for model in MODELS:
            base=None
            for s in STEMS:
                f=pairread(ROOT/f'phase2/results/phase2_pair_ranks/{ds}_{model}_native_{s}.parquet').set_index(['query_id','product_id']).sort_index()
                if base is None:base=f.label
                assert base.equals(f.label)
    csv(pd.DataFrame(rows),HERE/'qa/original_serialized_strings.csv')
    prof=pd.read_csv(OLD/'data/encoder_profiles_and_rank_sources.csv',keep_default_na=False);meta=[]
    for r in prof.itertuples():
        record=r._asdict()
        for role,key in [('product','product_embedding'),('query','query_embedding')]:
            p=ROOT/record[key];side=p.with_suffix('.json');j=read(side)
            meta.append(dict(dataset=r.dataset,model=r.model,schedule=r.schedule,role=role,payload=record[key],payload_exists=p.exists(),payload_sha256=sha(p),metadata=side.relative_to(ROOT).as_posix(),metadata_sha256=sha(side),implementation=j.get('implementation','unavailable'),precision=j.get('precision','unavailable'),profile=json.dumps(j.get('profile',{})),model_revision=j.get('model',{}).get('revision','unavailable'),transformers=j.get('transformers',j.get('versions',{}).get('transformers','unavailable in per-array sidecar')),prefix='empty',catalog_ordering=f'revision_graded_20260922/data/{r.dataset}_catalog_ordering.csv',query_ordering=f'revision_graded_20260922/data/{r.dataset}_queries_ordering.csv'))
    csv(pd.DataFrame(meta),HERE/'data/historical_execution_matrix.csv')
    start=read(HERE/'qa/starting_state.json');revision=source_revision()
    for p,h in start['preexisting_files'].items():assert sha(ROOT/p)==h,p
    dump(HERE/'qa/source_integrity.json',dict(time_utc=now(),exact_saved_serializations=sum(x['products'] for x in rows),all_labels_unchanged=True,stable_catalog_mapping=True,historical_array_records=len(meta),preexisting_changes_preserved=True,**revision))
    print('SOURCE INTEGRITY PASS',sum(x['products'] for x in rows),flush=True)
if __name__=='__main__':main()
