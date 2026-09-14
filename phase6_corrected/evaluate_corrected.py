"""Audit every selected value-only model before any corrected held-out ranking."""
import argparse
import json
import time
import numpy as np
import pandas as pd
import torch
from common_corrected import HERE,ROOT,CONFIG,Data,STEMS,METHODS,model_for,measures,dump,sha,now,validate,cost,prior
from baselines import BASELINES,baseline_frame,historical_exact


def restore_schedule(folder,base,schedule):
    if schedule=='C0':return base
    with np.load(folder/f'{schedule}_xor.npz') as archive:
        bits=np.bitwise_xor(base.view('u4'),archive['xor'])
    return bits.view('f4')


def audit():
    cfg=validate();lock=json.loads((HERE/'selection.json').read_text())
    assert lock['config_sha256']==sha(CONFIG) and not lock['test_used_for_selection']
    d=Data(cfg);records={}
    for method,selected in lock['selected'].items():
        for seed in cfg['seeds']:
            key=f'{method}_s{seed}';folder=HERE/'results'/key;folder.mkdir(parents=True,exist_ok=True)
            if (folder/'pretest_audit.json').exists():
                record=json.loads((folder/'pretest_audit.json').read_text())
                assert record['selection_sha256']==sha(HERE/'selection.json')
                records[key]=record;continue
            weights=selected['checkpoints'][str(seed)]
            assert sha(HERE/weights['path'])==weights['sha256']
            model=model_for(d,method,seed)
            model.load_state_dict(torch.load(HERE/weights['path'],map_location=d.device,weights_only=True))
            sampled=d.invariance(model,key+'_selected')
            start=time.monotonic();base,stats,fields=d.generate(model,diagnostics=True)
            cost(key,'final_catalog_generation',start)
            np.save(folder/'products.npy',base)
            stats.to_parquet(folder/'field_weights.parquet',index=False)
            fields.to_csv(folder/'field_keys.csv',index=False)
            summary={col:{'mean':float(stats[col].mean()),'min':float(stats[col].min()),'p05':float(stats[col].quantile(.05)),
                          'median':float(stats[col].median()),'p95':float(stats[col].quantile(.95)),'max':float(stats[col].max())}
                     for col in ['entropy','normalized_entropy','max_weight','effective_fields','fields','weight_sum']}
            multi=stats[stats.fields>1]
            summary['concentration']={str(t):{'all_products':float(stats.max_weight.gt(t).mean()),'multiple_fields':float(multi.max_weight.gt(t).mean())} for t in [.5,.9]}
            dump(folder/'field_weight_summary.json',summary)
            if method=='C4':
                vocab=json.loads((HERE/'field_vocabulary.json').read_text())['native_keys']
                theta=model.theta.detach().cpu().numpy()
                pd.DataFrame([dict(field_key='<UNK>',field_id=0,theta=float(theta[0]))]+[dict(field_key=k,field_id=i,theta=float(theta[i])) for k,i in vocab.items()]).to_csv(folder/'field_angles.csv',index=False)
            full=[];vd=base.astype('f8');vn=np.linalg.norm(vd,axis=1)
            for s in STEMS[1:]:
                start=time.monotonic();other=d.generate(model,s)
                diff=np.linalg.norm(base-other,axis=1)
                od=other.astype('f8');cos=np.abs(1-(vd*od).sum(1)/(vn*np.linalg.norm(od,axis=1)))
                assert np.isfinite(other).all() and diff.max()<=1e-5,'STOP: full-catalog structural invariance failed before test'
                # Bitwise XOR stores exact FP32 forwards compactly, without recomputation or quantization.
                np.savez_compressed(folder/f'{s}_xor.npz',xor=np.bitwise_xor(base.view('u4'),other.view('u4')))
                restored=restore_schedule(folder,base,s)
                assert np.array_equal(restored.view('u4'),other.view('u4'))
                full.append(dict(schedule=s,products=len(base),max_L2=float(diff.max()),mean_L2=float(diff.mean()),
                                 max_cosine_difference=float(cos.max()),mean_cosine_difference=float(cos.mean()),
                                 vector_sha256=sha(folder/f'{s}_xor.npz'),lossless_forward_archive=True))
                cost(key,'full_catalog_vector_audit',start,schedule=s)
            record=dict(method=method,seed=seed,epoch=selected['epoch'],sample=sampled,full_catalog=full,passed=True,
                        products_sha256=sha(folder/'products.npy'),selection_sha256=sha(HERE/'selection.json'),completed_utc=now(),test_evaluated=False)
            dump(folder/'pretest_audit.json',record);records[key]=record
            dump(HERE/'PHASE6_INVARIANCE.json',dict(models=records))
            print('PRETEST INVARIANCE',key,'maxL2',max(x['max_L2'] for x in full),flush=True)
    dump(HERE/'PHASE6_INVARIANCE.json',dict(models=records))
    dump(HERE/'all_models_pretest_verified.json',dict(passed=True,models=list(records),completed_utc=now(),selection_sha256=sha(HERE/'selection.json')))


def evaluate():
    cfg=validate();lock=json.loads((HERE/'selection.json').read_text())
    gate=json.loads((HERE/'all_models_pretest_verified.json').read_text())
    assert gate['passed'] and gate['selection_sha256']==sha(HERE/'selection.json')
    assert len(gate['models'])==12
    d=Data(cfg)
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
            if (folder/'evaluation.json').exists():
                meta=json.loads((folder/'evaluation.json').read_text())
                assert meta['selection_sha256']==sha(HERE/'selection.json')
                continue
            start=time.monotonic();base=np.load(folder/'products.npy')
            assert sha(folder/'products.npy')==invariant['models'][key]['products_sha256']
            f=d.judged_frame(base,'test');f.to_parquet(folder/'catalog_graded_pairs.parquet',index=False)
            regenerated=f.copy();audit=[]
            target_method=method==lock['best_method']
            exact=f[f.label.eq('Exact')].copy().reset_index(drop=True)
            target=exact.copy() if target_method else None
            qids=sorted(f.query_id.unique());qi={q:i for i,q in enumerate(qids)}
            qv=d.queries_v[[d.qi[q] for q in qids]]
            base_scores=qv@base.T if target_method else None
            for s in STEMS[1:]:
                other=restore_schedule(folder,base,s)
                g=d.judged_frame(other,'test')
                assert g[['query_id','product_id']].equals(f[['query_id','product_id']])
                regenerated[s]=g.C0
                hi=f.label.eq('Exact')
                audit.append(dict(schedule=s,all_judged_rank_changes=int((g.C0!=f.C0).sum()),Exact_rank_changes=int((g.loc[hi,'C0']!=f.loc[hi,'C0']).sum()),
                                  Exact_inclusion20_changes=int(((g.loc[hi,'C0']<=20)!=(f.loc[hi,'C0']<=20)).sum()),
                                  Exact_inclusion100_changes=int(((g.loc[hi,'C0']<=100)!=(f.loc[hi,'C0']<=100)).sum())))
                if target_method:
                    scores=qv@other.T
                    ar=np.empty(len(exact),dtype='i4')
                    for qid,group in exact.groupby('query_id',sort=False):
                        ids=np.array([d.pi[p] for p in group.product_id])
                        ar[group.index]=prior.rank_target(base_scores[qi[qid]],ids,scores[qi[qid],ids])
                    target[s]=ar
            regenerated.to_parquet(folder/'regenerated_graded_pairs.parquet',index=False)
            m,eq,gq=measures(f);rm,req,rgq=measures(regenerated)
            eq.to_csv(folder/'catalog_exact_queries.csv');gq.to_csv(folder/'catalog_graded_queries.csv')
            req.to_csv(folder/'regenerated_exact_queries.csv');rgq.to_csv(folder/'regenerated_graded_queries.csv')
            if target_method:
                for qid,group in exact.groupby('query_id',sort=False):
                    ids=np.array([d.pi[p] for p in group.product_id])
                    assert np.array_equal(prior.rank_target(base_scores[qi[qid]],ids,base_scores[qi[qid],ids]),group.C0)
                exact.to_parquet(folder/'target_pairs.parquet',index=False)
                target.to_parquet(folder/'regenerated_target_pairs.parquet',index=False)
            record=invariant['models'][key]
            record.update(rank_audit=audit,regenerated_metrics=rm,target_only=target_method,test_evaluated=True)
            dump(folder/'invariance.json',record)
            dump(HERE/'PHASE6_INVARIANCE.json',invariant)
            dump(folder/'evaluation.json',dict(completed_utc=now(),pretest_gate_utc=gate['completed_utc'],selection_sha256=sha(HERE/'selection.json'),
                    metrics=m,regenerated_metrics=rm,target_only=target_method,Exact_pairs=len(exact),Exact_queries=len(eq),graded_queries=int(gq.positive_idcg.sum())))
            cost(key,'heldout_ranking_all_schedules',start)
            print('TEST',key,'R100',round(100*m['Recall@100'],4),'cNDCG',round(m['cNDCG@10'],6),'VI20',rm['VI@20'],flush=True)
    dump(HERE/'evaluation_complete.json',dict(completed_utc=now(),selection_sha256=sha(HERE/'selection.json'),models=gate['models']))


if __name__=='__main__':
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['audit','evaluate'])
    {'audit':audit,'evaluate':evaluate}[parser.parse_args().action]()
