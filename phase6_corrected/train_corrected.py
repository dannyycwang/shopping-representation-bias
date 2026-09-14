"""Freeze -> neutral guard -> three-seed common configuration/epoch selection. Never test-select."""
import argparse
import json
import time
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from common_corrected import HERE,ROOT,CONFIG,Data,METHODS,STEMS,model_for,measures,dump,sha,now,validate,cost,pilot
from baselines import BASELINES,baseline_frame


def score_better(a,b):
    if b is None:return True
    return a['Recall@100']>b['Recall@100']+1e-12 or (abs(a['Recall@100']-b['Recall@100'])<=1e-12 and a['cNDCG@10']>b['cNDCG@10']+1e-12)


def atomic_torch(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp')
    torch.save(value,temporary)
    temporary.replace(path)


def prepare():
    if CONFIG.exists():
        return validate()
    old=json.loads((ROOT/'PHASE6_CONFIG.json').read_text(encoding='utf8'))
    audit=json.loads((HERE/'PHASE6_DATA_AUDIT.json').read_text())
    local=['PROTOCOL.md','USER_INSTRUCTIONS.txt','value_models.py','common_corrected.py','sampling.py','audit_data.py',
           'baselines.py','train_corrected.py','evaluate_corrected.py','test_corrected.py','PHASE6_DATA_AUDIT.json',
           'historical_sources.json','training_pools.json','training_query_labels.csv','field_vocabulary.json','planned_query_exposure.csv']
    sources={HERE/p for p in local}|set((HERE/'sampling').glob('*.parquet'))
    sources.update(ROOT/e['path'] for e in old['sources'])
    sources.update([ROOT/'phase4/scripts/common.py',ROOT/'phase5_mitigation/screen.py',ROOT/'phase6/models.py',ROOT/'phase6/run.py'])
    sources.update(ROOT/f'phase2/results/phase2_pair_ranks/wands_bge_base_native_{s}.parquet' for s in STEMS)
    sources.update(ROOT/f'phase4/results/wands/bge_base_{s}_pairs.parquet' for s in ['canonical_raw','set_mean'])
    cfg=dict(phase='VI corrected',frozen_utc=now(),track='raw_WANDS_BGE',**{k:old[k] for k in ['train','validation','test','model']},
             seeds=[42,43,44],methods=METHODS,hidden=128,heads=4,blocks=1,dimension=768,vectors_per_product=1,
             learning_rates=[1e-4,3e-5],weight_decay=.01,gradient_clip=1.,batch_size=16,temperature=.05,max_epochs=20,patience=4,
             presentations_per_query_per_epoch=64,eligible_queries=audit['eligible_queries'],examples_per_epoch=audit['examples_per_epoch'],
             neutral_initialization='zero scalar scores, uniform original BGE values',epoch0_selectable=False,
             common_epoch_and_lr_across_seeds=True,selection='mean dev Exact R100; mean graded cNDCG10 tie-break; earlier epoch, lower LR',
             formula='normalize(.5*non_attribute + .5*normalize(sum_i softmax(scores)_i * frozen_BGE_atom_i))',
             value_projection=False,embedding_residual=False,mixing_coefficient_learned=False,field_keys=audit['field_keys'],
             graded_gains={'Exact':3,'Partial':1,'Irrelevant':0},graded_weights={'Exact>Irrelevant':1.,'Exact>Partial':.5,'Partial>Irrelevant':.5},
             catalog_products=42994,schedules=STEMS,field_cache_reused=True,bootstrap_draws=10000,bootstrap_seed=20260963,
             long_attribute_threshold=audit['attribute_count_q75'],environment={'torch':torch.__version__,'numpy':np.__version__,'cuda':torch.version.cuda,
             'device':torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'},
             sources=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in sorted(sources)])
    dump(CONFIG,cfg)
    d=Data(cfg)
    dev_baselines={}
    for name in BASELINES:
        f=baseline_frame(d,name,'validation')
        folder=HERE/'baselines'/name;folder.mkdir(parents=True,exist_ok=True)
        f.to_parquet(folder/'dev_graded_pairs.parquet',index=False)
        m,eq,gq=measures(f)
        eq.to_csv(folder/'dev_exact_queries.csv');gq.to_csv(folder/'dev_graded_queries.csv')
        dev_baselines[name]=m
    dump(HERE/'development_baselines.json',dev_baselines)
    # Independently construct the exact equal-weight original-value path on the full catalog.
    mean=np.empty_like(d.non)
    with torch.inference_mode():
        for offset in range(0,len(d.products),128):
            ids=d.infer_order[offset:offset+128].tolist()
            atoms,mask,non,_=d.batch(ids)
            attr=F.normalize((atoms*mask[:,:,None]).sum(1)/mask.sum(1).clamp(min=1)[:,None],dim=1)
            mean[ids]=F.normalize(.5*non+.5*attr,dim=1).cpu().numpy()
    rows=[]
    for method in METHODS:
        model=model_for(d,method,42)
        inv=d.invariance(model,method+'_neutral')
        vectors=d.generate(model)
        error=float(np.linalg.norm(vectors-mean,axis=1).max())
        assert error<=1e-5
        f=d.judged_frame(vectors,'validation')
        m,_,_=measures(f)
        assert abs(m['Recall@100']-dev_baselines['Set-Mean']['Recall@100'])<=.0005
        rows.append(dict(method=method,seed=42,epoch=0,max_L2_to_independent_SetMean=error,**m,passed=True))
    pd.DataFrame(rows).to_csv(HERE/'initialization.csv',index=False)
    dump(HERE/'initialization_passed.json',dict(passed=True,utc=now(),config_sha256=sha(CONFIG),test_read=False))
    print('NEUTRAL CHECKS PASSED',flush=True)


def train_group(d,method,lr):
    name=f'{method}_lr{lr:g}';folder=HERE/'checkpoints'/name
    if (folder/'selected.json').exists():return json.loads((folder/'selected.json').read_text())
    folder.mkdir(parents=True,exist_ok=True)
    models={seed:model_for(d,method,seed) for seed in d.config['seeds']}
    opts={seed:torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.01) for seed,model in models.items()}
    plans={s:pd.read_parquet(HERE/'sampling'/f'seed{s}.parquet') for s in models}
    epoch_rows,step_rows,exposure_rows=[],[],[]
    best,best_epoch,stale=None,None,0
    start_epoch=0
    resume=folder/'resume.pt'
    if resume.exists():
        saved=torch.load(resume,map_location=d.device,weights_only=False)
        assert saved['config_sha256']==sha(CONFIG)
        for seed in models:
            models[seed].load_state_dict(saved['models'][seed]);opts[seed].load_state_dict(saved['optimizers'][seed])
        epoch_rows,step_rows,exposure_rows=saved['epoch_rows'],saved['step_rows'],saved['exposure_rows']
        best,best_epoch,stale=saved['best'],saved['best_epoch'],saved['stale']
        start_epoch=saved['epoch']+1
    for epoch in range(start_epoch,d.config['max_epochs']+1):
        if epoch and stale>=4:break
        frames,metrics_by_seed={},{}
        for seed,model in models.items():
            losses,grads=[],[]
            start=time.monotonic()
            if epoch:
                model.train()
                examples=plans[seed][plans[seed].epoch.eq(epoch)].to_dict('records')
                for offset in range(0,len(examples),16):
                    group=examples[offset:offset+16]
                    ids=sorted({d.pi[e[k]] for e in group for k in ['positive','negative']})
                    local={p:i for i,p in enumerate(ids)}
                    vectors=model(*d.batch(ids))
                    p=vectors[[local[d.pi[e['positive']]] for e in group]]
                    n=vectors[[local[d.pi[e['negative']]] for e in group]]
                    q=d.qt[[d.qi[e['query_id']] for e in group]]
                    weights=torch.tensor([e['weight'] for e in group],device=d.device)
                    loss=(weights*F.softplus(((q*n).sum(1)-(q*p).sum(1))/.05)).mean()
                    opts[seed].zero_grad(set_to_none=True);loss.backward()
                    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                    opts[seed].step()
                    losses.append((float(loss),len(group)));grads.append(float(norm))
                    step_rows.append(dict(seed=seed,epoch=epoch,step=offset//16,loss=float(loss),gradient_norm_before_clip=float(norm),presentations=len(group)))
                used=pd.DataFrame(examples)
                for qid,g in used.groupby('query_id'):
                    exposure_rows.append(dict(seed=seed,epoch=epoch,query_id=qid,presentations=len(g),weight_sum=float(g.weight.sum()),
                                              unique_higher=g.positive.nunique(),unique_lower=g.negative.nunique(),
                                              Exact_Irrelevant=int(((g.higher_label=='Exact')&(g.lower_label=='Irrelevant')).sum()),
                                              Exact_Partial=int(((g.higher_label=='Exact')&(g.lower_label=='Partial')).sum()),
                                              Partial_Irrelevant=int(((g.higher_label=='Partial')&(g.lower_label=='Irrelevant')).sum())))
                cost(name,'training',start,seed=seed,epoch=epoch)
            start=time.monotonic()
            inv=d.invariance(model,f'{name}_s{seed}_e{epoch}')
            cost(name,'sampled_invariance',start,seed=seed,epoch=epoch)
            start=time.monotonic();vectors=d.generate(model);cost(name,'dev_catalog_generation',start,seed=seed,epoch=epoch)
            start=time.monotonic();f=d.judged_frame(vectors,'validation');m,eq,gq=measures(f)
            cost(name,'dev_ranking',start,seed=seed,epoch=epoch)
            with torch.inference_mode():
                _,w,_=model.components(*d.batch(d.sample))
                entropy=-(w*w.clamp(min=1e-30).log()).sum(1)
                concentration=float(w.max(1).values.mean())
            row=dict(method=method,group=name,lr=lr,seed=seed,epoch=epoch,train_loss=sum(x*n for x,n in losses)/sum(n for _,n in losses) if losses else None,
                     gradient_mean=float(np.mean(grads)) if grads else 0.,gradient_max=max(grads,default=0.),
                     sample_entropy=float(entropy.mean()),sample_max_weight=concentration,**m,invariance_max_L2=inv['max_L2'])
            epoch_rows.append(row);frames[seed]=f;metrics_by_seed[seed]=m
        aggregate={key:float(np.mean([m[key] for m in metrics_by_seed.values()])) for key in ['Recall@100','cNDCG@10']}
        if epoch and score_better(aggregate,best):
            best,best_epoch,stale=aggregate,epoch,0
            for seed,model in models.items():
                sf=folder/f'seed{seed}';sf.mkdir(exist_ok=True)
                atomic_torch(sf/'weights.pt',model.state_dict())
                frames[seed].to_parquet(sf/'dev_pairs.parquet',index=False)
                _,eq,gq=measures(frames[seed]);eq.to_csv(sf/'dev_exact_queries.csv');gq.to_csv(sf/'dev_graded_queries.csv')
        elif epoch:stale+=1
        pd.DataFrame(epoch_rows).to_csv(folder/'epochs.csv',index=False)
        if step_rows:pd.DataFrame(step_rows).to_csv(folder/'steps.csv',index=False)
        if exposure_rows:pd.DataFrame(exposure_rows).to_csv(folder/'query_presentations.csv',index=False)
        atomic_torch(resume,dict(config_sha256=sha(CONFIG),epoch=epoch,best=best,best_epoch=best_epoch,stale=stale,
                                models={s:m.state_dict() for s,m in models.items()},optimizers={s:o.state_dict() for s,o in opts.items()},
                                epoch_rows=epoch_rows,step_rows=step_rows,exposure_rows=exposure_rows))
        print(name,'epoch',epoch,'mean dev R100',round(100*aggregate['Recall@100'],4),'cNDCG',round(aggregate['cNDCG@10'],6),'best',best_epoch,'stale',stale,flush=True)
    assert best_epoch>=1
    selected=dict(name=name,method=method,lr=lr,epoch=best_epoch,seeds=list(models),epochs_run=max(r['epoch'] for r in epoch_rows),
                  dev_mean=best,parameters=sum(p.numel() for p in models[42].parameters()),config_sha256=sha(CONFIG),completed_utc=now(),
                  checkpoints={str(s):dict(path=str((folder/f'seed{s}/weights.pt').relative_to(HERE)),sha256=sha(folder/f'seed{s}/weights.pt')) for s in models})
    for seed,model in models.items():atomic_torch(folder/f'seed{seed}/last_weights.pt',model.state_dict())
    dump(folder/'selected.json',selected)
    return selected


def train():
    cfg=validate()
    if (HERE/'selection.json').exists():
        print('Existing selection locked; training retained');return
    assert json.loads((HERE/'initialization_passed.json').read_text())['passed']
    d=Data(cfg);groups=[];selected={}
    for method in METHODS:
        candidates=[train_group(d,method,lr) for lr in cfg['learning_rates']]
        groups.extend(candidates)
        selected[method]=sorted(candidates,key=lambda c:(-round(c['dev_mean']['Recall@100'],12),-round(c['dev_mean']['cNDCG@10'],12),c['epoch'],c['lr']))[0]
        dump(HERE/'selection_progress.json',selected)
    winner=sorted(selected,key=lambda m:(-round(selected[m]['dev_mean']['Recall@100'],12),-round(selected[m]['dev_mean']['cNDCG@10'],12),selected[m]['parameters'],m))[0]
    dump(HERE/'selection.json',dict(locked_utc=now(),config_sha256=sha(CONFIG),groups=groups,selected=selected,best_method=winner,
                                    test_used_for_selection=False,epoch0_selectable=False,common_config_across_seeds=True))
    print('DEVELOPMENT SELECTION LOCKED',winner,flush=True)


if __name__=='__main__':
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','train'])
    {'prepare':prepare,'train':train}[parser.parse_args().action]()
