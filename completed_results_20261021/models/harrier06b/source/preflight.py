"""Check complete catalogs, inherited strings, support, and immutable input sources."""
from common import *
from collections import Counter
import importlib.metadata, platform, shutil

def main():
    (HERE/'audit').mkdir(exist_ok=True)
    if not (HERE/'audit/initial_git_status.txt').exists():
        (HERE/'audit/initial_git_status.txt').write_text(git('status','--porcelain=v1','-uall')+'\n',encoding='utf8')
        modified={p:sha(ROOT/p) for p in git('diff','--name-only').splitlines() if (ROOT/p).is_file()}
        dump(HERE/'audit/preexisting_changes.json',modified)
    sources=[];populations={}
    for ds in ['wands','esci']:
        p,q,j=load(ds);pp,eligible,jj=support(ds);h=jj[jj.label.eq(highest(ds))]
        expected=(42994,480,308,21299) if ds=='wands' else (10076,500,499,4434)
        actual=(len(p),len(q),len(eligible),len(h));assert actual==expected,(ds,actual,expected)
        assert not j.duplicated(['query_id','product_id']).any()
        assert len(set(str(x['product_id']) for x in p))==len(p)
        assert set(j.product_id)<=set(str(x['product_id']) for x in p)
        old=read(ROOT/'revision_graded_20260922/PROTOCOL.json')['populations'][ds]
        assert eligible.query_id.tolist()==old['eligible_ids']
        oldh=pd.read_csv(ROOT/f'revision_graded_20260922/data/{ds}_highest_support.csv',dtype={'product_id':str})
        assert set(map(tuple,h[['query_id','product_id','label']].values))==set(map(tuple,oldh[['query_id','product_id','label']].values))
        catalog=pd.read_csv(ROOT/f'revision_graded_20260922/data/{ds}_catalog_ordering.csv',dtype={'catalog_id':str})
        assert catalog.catalog_id.tolist()==[str(x['product_id']) for x in p]
        if ds=='wands':
            dev=SPLITS['wands_dev'];held=SPLITS['wands_heldout'];assert (len(dev),len(held))==(96,384)
            assert not set(dev)&set(held) and set(dev)|set(held)==set(q.query_id)
        else: assert set(SPLITS['esci_old'])==set(q.query_id)
        distinct=[set() for _ in p]
        for stem in STEMS:
            path=ROOT/f'phase2/data/representations/{ds}/{stem}.jsonl.gz';sources.append(source(path))
            with gzip.open(path,'rt',encoding='utf8') as stream:records=list(map(json.loads,stream))
            texts=texts_for(p,stem);assert len(records)==len(p)
            for i,(product,record,text) in enumerate(zip(p,records,texts)):
                assert str(record['product_id'])==str(product['product_id']) and record['text']==text
                attrs=attributes(product,stem);assert Counter(attrs)==Counter(product['attributes'])
                base=serializer._blocks(product);changed=serializer._blocks(product,attrs)
                assert [k for k,_ in base]==[k for k,_ in changed]
                assert [(k,v) for k,v in base if k!='attributes']==[(k,v) for k,v in changed if k!='attributes']
                distinct[i].add(digest(text))
            print('SERIALIZATION',ds,stem,len(p),flush=True)
        rows=[]
        for product,views in zip(p,distinct):
            attrs=product['attributes']
            for rule in RULES:
                canon=canonical_attrs(product,rule);assert Counter(canon)==Counter(attrs)
                for stem in STEMS:assert canonical_attrs({**product,'attributes':attributes(product,stem)},rule)==canon
            rows.append(dict(product_id=str(product['product_id']),unique_serialized_views=len(views),attributes=len(attrs),duplicate_occurrences=len(attrs)-len(set(attrs)),all_raw_strings_exact=True,all_canonical_rules_invariant=True))
        pd.DataFrame(rows).to_csv(HERE/f'audit/{ds}_serialization.csv',index=False)
        h[['query_id','product_id','label']].sort_values(['query_id','product_id']).to_csv(HERE/f'audit/{ds}_highest_support.csv',index=False)
        pd.DataFrame({'catalog_index':range(len(p)),'product_id':[str(x['product_id']) for x in p]}).to_csv(HERE/f'audit/{ds}_catalog.csv',index=False)
        eligible.to_csv(HERE/f'audit/{ds}_eligible_queries.csv',index=False)
        populations[ds]=dict(catalog_size=len(p),source_queries=len(q),eligible_queries=len(eligible),highest_pairs=len(h),judged_pairs=len(jj),query_ids=eligible.query_id.astype(int).tolist(),serialized_views=sum(map(len,distinct)))
        for suffix in ['products.jsonl.gz','queries.csv','judgments.csv']:sources.append(source(ROOT/f'phase2/data/processed/{ds}_{suffix}'))
    for rel in ['phase2/config/phase2.json','phase2/src/representations.py','phase2/src/encoding.py','phase2/scripts/prepare_data.py','phase3/scripts/analyze_target_only_permutations.py','phase4/config/splits.json','revision_evidence_20260917/scripts/common.py','revision_evidence_20260917/scripts/audit_fitting.py','revision_evidence_20260917/scripts/analyze_membership.py','revision_graded_20260922/scripts/metrics.py','revision_graded_20260922/scripts/common.py','revision_graded_20260922/PROTOCOL.json','revision_final_strengthening_20260922/PROTOCOL.json','revision_section6_20260924/scripts/analyze_saved.py','revision_section6_20260924/scripts/reports.py']:
        sources.append(source(ROOT/rel))
    import torch
    versions={}
    for name in ['torch','transformers','sentence-transformers','huggingface-hub','numpy','pandas','pyarrow','pytest','accelerate','tokenizers']:
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]=None
    gpu=subprocess.run(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv,noheader'],capture_output=True,text=True).stdout.strip()
    dump(HERE/'audit/preflight.json',dict(completed_utc=now(),git_commit=git('rev-parse','HEAD'),populations=populations,source_files=sources,hardware=dict(gpu=gpu,cuda_available=torch.cuda.is_available(),cuda_runtime=torch.version.cuda,bf16_supported=torch.cuda.is_bf16_supported(),disk_free=shutil.disk_usage(HERE).free),software=versions,python=sys.version,platform=platform.platform(),agents_files=[]))
    status('preflight',outputs=[source(HERE/'audit/preflight.json')])

if __name__=='__main__':main()
