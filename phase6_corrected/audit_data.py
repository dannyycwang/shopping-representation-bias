"""Inspect training qrels and freeze a complete historical archive before corrected training."""
import json
from pathlib import Path
import pandas as pd
from common_corrected import HERE,ROOT,Data,RELATIONS,dump,sha,now
from sampling import BalancedSampler


def main():
    destination = HERE/'PHASE6_DATA_AUDIT.json'
    if destination.exists():
        print('Existing data audit retained')
        return
    old = json.loads((ROOT/'PHASE6_CONFIG.json').read_text(encoding='utf8'))
    cfg = {k:old[k] for k in ['train','validation','test']}
    d = Data(cfg)
    train = d.judgments[d.judgments.query_id.isin(cfg['train'])]
    counts = pd.crosstab(train.query_id,train.label).reindex(cfg['train'],fill_value=0)
    for label in ['Exact','Partial','Irrelevant']:
        if label not in counts:
            counts[label] = 0
    counts['Exact_and_Partial'] = (counts.Exact>0)&(counts.Partial>0)
    counts['Exact_and_Irrelevant'] = (counts.Exact>0)&(counts.Irrelevant>0)
    counts['Exact_and_lower'] = (counts.Exact>0)&((counts.Partial+counts.Irrelevant)>0)
    counts['eligible'] = counts.index.isin(d.eligible)
    counts.reset_index().to_csv(HERE/'training_query_labels.csv',index=False)
    pools = {str(q):g for q,g in d.groups.items()}
    dump(HERE/'training_pools.json',pools)
    dump(HERE/'field_vocabulary.json',{'UNK':0,'native_keys':d.vocab,'source':'all products judged for TRAIN query IDs; no dev/test query labels'})
    sampling_rows=[]
    for seed in [42,43,44]:
        sampler=BalancedSampler(d.groups,seed,64)
        presentations=[]
        for epoch in range(1,21):
            rows=sampler.epoch(epoch)
            frame=pd.DataFrame(rows)
            exposure=frame.groupby('query_id').size()
            assert exposure.min()==exposure.max()==64
            presentations.append(frame)
            sampling_rows.append(dict(seed=seed,epoch=epoch,examples=len(rows),queries=len(exposure),min_count=64,max_count=64,ratio=1.))
        (HERE/'sampling').mkdir(exist_ok=True)
        pd.concat(presentations,ignore_index=True).to_parquet(HERE/'sampling'/f'seed{seed}.parquet',index=False)
    pd.DataFrame(sampling_rows).to_csv(HERE/'planned_query_exposure.csv',index=False)
    protected=set()
    for folder in ['phase6','phase6b']:
        protected.update(p for p in (ROOT/folder).rglob('*') if p.is_file() and not {'__pycache__','.pytest_cache'}&set(p.parts))
    protected.update(p for p in ROOT.glob('PHASE6_*') if p.is_file())
    protected.add(ROOT/'FINAL_DELIVERY_MANIFEST.json')
    # Validate every VI-B delivery hash, then archive pilot source/result hashes.
    manifest=json.loads((ROOT/'phase6b/FINAL_DELIVERY_MANIFEST.json').read_text())
    for e in manifest['outputs']:
        assert sha(ROOT/'phase6b'/e['path'])==e['sha256'],e['path']
    for e in manifest['historical_sources']:
        assert sha(ROOT/e['path'])==e['sha256'],e['path']
        protected.add(ROOT/e['path'])
    dump(HERE/'historical_sources.json',[dict(path=str(p.relative_to(ROOT)),sha256=sha(p),bytes=p.stat().st_size) for p in sorted(protected)])
    summary=dict(frozen_utc=now(),train_queries=len(cfg['train']),judged_pairs=len(train),
                 queries_with_labels={s:int((counts[s]>0).sum()) for s in ['Exact','Partial','Irrelevant']},
                 label_counts={s:dict(pairs=int((train.label==s).sum()),unique_products=int(train.loc[train.label.eq(s),'product_id'].nunique())) for s in ['Exact','Partial','Irrelevant']},
                 queries_Exact_and_Partial=int(counts.Exact_and_Partial.sum()),queries_Exact_and_Irrelevant=int(counts.Exact_and_Irrelevant.sum()),
                 queries_Exact_and_lower=int(counts.Exact_and_lower.sum()),eligible_queries=d.eligible,
                 excluded_queries=sorted(set(cfg['train'])-set(d.eligible)),relations=[dict(higher=a,lower=b,weight=w) for a,b,w in RELATIONS],
                 presentations_per_query_per_epoch=64,examples_per_epoch=64*len(d.eligible),query_max_min_ratio=1.,
                 field_keys=len(d.vocab),UNK_id=0,catalog_products=len(d.products),attribute_count_q75=float(d.features.attribute_count.quantile(.75)),
                 no_dev_test_training=True,no_unjudged_negatives=True,unmodified_raw_atoms=True,
                 sampling='each epoch has 64 independently shuffled complete query cycles; equal exposure among eligible queries; relation and both grade-product pools rotate across epochs',
                 limitation='Only queries with a valid strict judged grade relation can train; relation types are cycled equally when available and loss weights encode grade gap.')
    dump(destination,summary)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
