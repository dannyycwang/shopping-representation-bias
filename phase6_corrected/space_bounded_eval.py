"""Storage-only amendment: actual forwards twice, retained C0/checkpoints/ranks, no permutation arrays."""
import argparse
import hashlib
import json
import shutil
import time
import numpy as np
import pandas as pd
import torch
from common_corrected import HERE,ROOT,CONFIG,Data,STEMS,model_for,measures,dump,sha,now,validate,cost,prior
from baselines import BASELINES,baseline_frame


def digest(array):return hashlib.sha256(array.tobytes(order='C')).hexdigest()


def setup():
    cfg=validate();lock=json.loads((HERE/'selection.json').read_text())
    path=HERE/'EVALUATION_STORAGE_AMENDMENT.json'
    if not path.exists():
        assert not (HERE/'evaluation_complete.json').exists()
        dump(path,dict(frozen_utc=now(),primary_config_sha256=sha(CONFIG),selection_sha256=sha(HERE/'selection.json'),
                       script_sha256=sha(HERE/'space_bounded_eval.py'),reason='C drive capacity: estimated archived vectors exceed available space before rank outputs',
                       probe=json.loads((HERE/'storage_probe.json').read_text()),free_bytes=shutil.disk_usage(HERE).free,
                       changes='Retain C0 vectors and all checkpoints/ranks. Run every full-catalog permutation before test, then recompute forward arrays for held-out ranking and release arrays. Store array digests and numerical summaries instead of XOR archives.',
                       training_and_selection_changed=False,heldout_metrics_seen=False))
    amendment=json.loads(path.read_text())
    assert amendment['primary_config_sha256']==sha(CONFIG) and amendment['selection_sha256']==sha(HERE/'selection.json')
    assert amendment['script_sha256']==sha(HERE/'space_bounded_eval.py')
    assert not lock['test_used_for_selection']
    return cfg,lock,Data(cfg)


def deviations(base,other):
    diff=np.linalg.norm(base-other,axis=1)
    a,b=base.astype('f8'),other.astype('f8')
    cosine=np.abs(1-(a*b).sum(1)/(np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1)))
    assert np.isfinite(other).all() and diff.max()<=1e-5
    return dict(max_L2=float(diff.max()),mean_L2=float(diff.mean()),max_cosine_difference=float(cosine.max()),mean_cosine_difference=float(cosine.mean()))


def audit():
    cfg,lock,d=setup();records={}
    for method,selected in lock['selected'].items():
        for seed in cfg['seeds']:
            key=f'{method}_s{seed}';folder=HERE/'results'/key;folder.mkdir(parents=True,exist_ok=True)
            if (folder/'pretest_audit.json').exists():
                record=json.loads((folder/'pretest_audit.json').read_text())
                assert record['selection_sha256']==sha(HERE/'selection.json')
                records[key]=record;continue
            checkpoint=selected['checkpoints'][str(seed)]
            assert sha(HERE/checkpoint['path'])==checkpoint['sha256']
            model=model_for(d,method,seed)
            model.load_state_dict(torch.load(HERE/checkpoint['path'],map_location=d.device,weights_only=True))
            sample=d.invariance(model,key+'_selected')
            start=time.monotonic();base,stats,fields=d.generate(model,diagnostics=True);cost(key,'final_catalog_generation',start)
            np.save(folder/'products.npy',base)
            stats.to_parquet(folder/'field_weights.parquet',index=False);fields.to_csv(folder/'field_keys.csv',index=False)
            summary={col:dict(mean=float(stats[col].mean()),min=float(stats[col].min()),p05=float(stats[col].quantile(.05)),
                              median=float(stats[col].median()),p95=float(stats[col].quantile(.95)),max=float(stats[col].max()))
                     for col in ['entropy','normalized_entropy','max_weight','effective_fields','fields','weight_sum']}
            multi=stats[stats.fields>1]
            summary['concentration']={str(t):dict(all_products=float(stats.max_weight.gt(t).mean()),multiple_fields=float(multi.max_weight.gt(t).mean())) for t in [.5,.9]}
            dump(folder/'field_weight_summary.json',summary)
            if method=='C4':
                vocabulary=json.loads((HERE/'field_vocabulary.json').read_text())['native_keys'];angles=model.theta.detach().cpu().numpy()
                pd.DataFrame([dict(field_key='<UNK>',field_id=0,theta=float(angles[0]))]+[dict(field_key=k,field_id=i,theta=float(angles[i])) for k,i in vocabulary.items()]).to_csv(folder/'field_angles.csv',index=False)
            full=[]
            for s in STEMS[1:]:
                start=time.monotonic();other=d.generate(model,s)
                full.append(dict(schedule=s,products=42994,**deviations(base,other),forward_sha256=digest(other),actual_forward=True))
                cost(key,'full_catalog_vector_audit',start,schedule=s)
            record=dict(method=method,seed=seed,epoch=selected['epoch'],sample=sample,full_catalog=full,passed=True,
                        products_sha256=sha(folder/'products.npy'),selection_sha256=sha(HERE/'selection.json'),
                        storage_amendment_sha256=sha(HERE/'EVALUATION_STORAGE_AMENDMENT.json'),completed_utc=now(),test_evaluated=False)
            dump(folder/'pretest_audit.json',record);records[key]=record
            dump(HERE/'PHASE6_INVARIANCE.json',dict(models=records))
            print('PRETEST INVARIANCE',key,'maxL2',max(x['max_L2'] for x in full),flush=True)
    dump(HERE/'PHASE6_INVARIANCE.json',dict(models=records))
    dump(HERE/'all_models_pretest_verified.json',dict(passed=True,models=list(records),completed_utc=now(),selection_sha256=sha(HERE/'selection.json')))


def evaluate():
    cfg,lock,d=setup();gate=json.loads((HERE/'all_models_pretest_verified.json').read_text())
    assert gate['passed'] and len(gate['models'])==12 and gate['selection_sha256']==sha(HERE/'selection.json')
    invariant=json.loads((HERE/'PHASE6_INVARIANCE.json').read_text())
    for name in BASELINES:
        folder=HERE/'baselines'/name;folder.mkdir(parents=True,exist_ok=True)
        if not (folder/'test_graded_pairs.parquet').exists():
            f=baseline_frame(d,name,'test')
            if name=='Set-Attention':dump(folder/'graded_reconciliation.json',f.attrs['historical_exact_reconciliation'])
            f.to_parquet(folder/'test_graded_pairs.parquet',index=False)
    for method,selected in lock['selected'].items():
        for seed in cfg['seeds']:
            key=f'{method}_s{seed}';folder=HERE/'results'/key
            if (folder/'evaluation.json').exists():continue
            checkpoint=selected['checkpoints'][str(seed)];assert sha(HERE/checkpoint['path'])==checkpoint['sha256']
            model=model_for(d,method,seed);model.load_state_dict(torch.load(HERE/checkpoint['path'],map_location=d.device,weights_only=True))
            start=time.monotonic();base=np.load(folder/'products.npy');record=invariant['models'][key]
            assert sha(folder/'products.npy')==record['products_sha256']
            f=d.judged_frame(base,'test');f.to_parquet(folder/'catalog_graded_pairs.parquet',index=False)
            regenerated=f.copy();audit=[];target_method=method==lock['best_method']
            exact=f[f.label.eq('Exact')].copy().reset_index(drop=True)
            target=exact.copy() if target_method else None
            qids=sorted(f.query_id.unique());qi={q:i for i,q in enumerate(qids)};qv=d.queries_v[[d.qi[q] for q in qids]]
            base_scores=qv@base.T if target_method else None
            for s in STEMS[1:]:
                other=d.generate(model,s);numbers=deviations(base,other)
                g=d.judged_frame(other,'test');assert g[['query_id','product_id']].equals(f[['query_id','product_id']])
                regenerated[s]=g.C0;hi=f.label.eq('Exact')
                previous=next(x for x in record['full_catalog'] if x['schedule']==s)
                audit.append(dict(schedule=s,**numbers,forward_sha256=digest(other),matches_pretest_forward_sha=digest(other)==previous['forward_sha256'],
                                  all_judged_rank_changes=int((g.C0!=f.C0).sum()),Exact_rank_changes=int((g.loc[hi,'C0']!=f.loc[hi,'C0']).sum()),
                                  Exact_inclusion20_changes=int(((g.loc[hi,'C0']<=20)!=(f.loc[hi,'C0']<=20)).sum()),
                                  Exact_inclusion100_changes=int(((g.loc[hi,'C0']<=100)!=(f.loc[hi,'C0']<=100)).sum())))
                if target_method:
                    scores=qv@other.T;ar=np.empty(len(exact),dtype='i4')
                    for qid,group in exact.groupby('query_id',sort=False):
                        ids=np.array([d.pi[p] for p in group.product_id]);ar[group.index]=prior.rank_target(base_scores[qi[qid]],ids,scores[qi[qid],ids])
                    target[s]=ar
            regenerated.to_parquet(folder/'regenerated_graded_pairs.parquet',index=False)
            m,eq,gq=measures(f);rm,req,rgq=measures(regenerated)
            eq.to_csv(folder/'catalog_exact_queries.csv');gq.to_csv(folder/'catalog_graded_queries.csv')
            req.to_csv(folder/'regenerated_exact_queries.csv');rgq.to_csv(folder/'regenerated_graded_queries.csv')
            if target_method:
                for qid,group in exact.groupby('query_id',sort=False):
                    ids=np.array([d.pi[p] for p in group.product_id]);assert np.array_equal(prior.rank_target(base_scores[qi[qid]],ids,base_scores[qi[qid],ids]),group.C0)
                exact.to_parquet(folder/'target_pairs.parquet',index=False);target.to_parquet(folder/'regenerated_target_pairs.parquet',index=False)
            record.update(rank_audit=audit,regenerated_metrics=rm,target_only=target_method,test_evaluated=True)
            dump(folder/'invariance.json',record);dump(HERE/'PHASE6_INVARIANCE.json',invariant)
            dump(folder/'evaluation.json',dict(completed_utc=now(),pretest_gate_utc=gate['completed_utc'],selection_sha256=sha(HERE/'selection.json'),
                    metrics=m,regenerated_metrics=rm,target_only=target_method,Exact_pairs=len(exact),Exact_queries=len(eq),graded_queries=int(gq.positive_idcg.sum())))
            cost(key,'heldout_forward_and_ranking_all_schedules',start)
            print('TEST',key,'R100',round(100*m['Recall@100'],4),'cNDCG',round(m['cNDCG@10'],6),'VI20',rm['VI@20'],flush=True)
    dump(HERE/'evaluation_complete.json',dict(completed_utc=now(),selection_sha256=sha(HERE/'selection.json'),models=gate['models']))


if __name__=='__main__':
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['audit','evaluate'])
    {'audit':audit,'evaluate':evaluate}[parser.parse_args().action]()
