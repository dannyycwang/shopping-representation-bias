"""Independent delivery audit: histories, graded sampling, common selection, actual ranks, uncertainty."""
import json
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import torch
from common_corrected import HERE,ROOT,CONFIG,STEMS,METHODS,GAINS,validate,sha,dump,now,measures,prior


def main():
    cfg=validate();lock=json.loads((HERE/'selection.json').read_text())
    assert not lock['test_used_for_selection'] and lock['common_config_across_seeds'] and not lock['epoch0_selectable']
    assert len(lock['groups'])==8 and len(lock['selected'])==4 and cfg['seeds']==[42,43,44]
    assert not cfg['value_projection'] and not cfg['embedding_residual'] and not cfg['mixing_coefficient_learned']
    test_xml=ET.parse(HERE/'unit_tests.xml').getroot()
    suites=list(test_xml.iter('testsuite'))
    assert sum(int(s.attrib['tests']) for s in suites)==10
    assert all(int(s.attrib['errors'])==int(s.attrib['failures'])==0 for s in suites)
    initial=json.loads((HERE/'initialization_passed.json').read_text())
    assert initial['passed'] and not initial['test_read'] and initial['config_sha256']==sha(CONFIG)
    judgments=pd.read_csv(ROOT/'phase2/data/processed/wands_judgments.csv',dtype={'product_id':str}).set_index(['query_id','product_id']).label
    plans={}
    for seed in cfg['seeds']:
        plan=pd.read_parquet(HERE/'sampling'/f'seed{seed}.parquet')
        assert set(plan.query_id)==set(cfg['eligible_queries'])<=set(cfg['train'])
        assert not set(plan.query_id)&(set(cfg['validation'])|set(cfg['test']))
        assert plan.groupby(['epoch','query_id']).size().eq(64).all()
        higher=judgments.loc[pd.MultiIndex.from_frame(plan[['query_id','positive']])].to_numpy()
        lower=judgments.loc[pd.MultiIndex.from_frame(plan[['query_id','negative']])].to_numpy()
        assert np.array_equal(higher,plan.higher_label) and np.array_equal(lower,plan.lower_label)
        assert np.all(pd.Series(higher).map(GAINS).to_numpy()>pd.Series(lower).map(GAINS).to_numpy())
        weights=np.array([cfg['graded_weights'][a+'>'+b] for a,b in zip(higher,lower)])
        assert np.array_equal(weights,plan.weight)
        plans[seed]=plan
    for group in lock['groups']:
        folder=HERE/'checkpoints'/group['name']
        epochs=pd.read_csv(folder/'epochs.csv')
        means=epochs.groupby('epoch')[['Recall@100','cNDCG@10']].mean()
        best=means.loc[means.index>=1].reset_index().sort_values(['Recall@100','cNDCG@10','epoch'],ascending=[False,False,True]).iloc[0]
        assert int(best.epoch)==group['epoch']>=1
        assert np.isclose(best['Recall@100'],group['dev_mean']['Recall@100'])
        assert np.isclose(best['cNDCG@10'],group['dev_mean']['cNDCG@10'])
        assert epochs.groupby('epoch').seed.nunique().eq(3).all()
        assert epochs.invariance_max_L2.max()<=1e-5
        steps=pd.read_csv(folder/'steps.csv');presentations=pd.read_csv(folder/'query_presentations.csv')
        assert np.isfinite(steps[['loss','gradient_norm_before_clip']]).all().all()
        assert steps.loss.between(0,40.01).all()
        assert presentations.presentations.eq(64).all()
        assert set(presentations.query_id)==set(cfg['eligible_queries'])
        assert steps.groupby(['seed','epoch']).presentations.sum().eq(cfg['examples_per_epoch']).all()
        for seed in cfg['seeds']:
            checkpoint=group['checkpoints'][str(seed)]
            assert sha(HERE/checkpoint['path'])==checkpoint['sha256']
            state=torch.load(HERE/checkpoint['path'],map_location='cpu',weights_only=True)
            assert sum(value.numel() for value in state.values())==group['parameters']
            assert all(torch.isfinite(value).all() for value in state.values())
            final_score=state['scorer.2.weight'] if group['method'] in ['C1','C4'] else state['scorer.weight']
            assert torch.count_nonzero(final_score)>0, 'Selected trained scoring layer must differ from neutral epoch 0'
            sf=(HERE/checkpoint['path']).parent
            actual=measures(pd.read_parquet(sf/'dev_pairs.parquet'))[0]
            expected=epochs[(epochs.seed==seed)&(epochs.epoch==group['epoch'])].iloc[0]
            for metric in actual:assert np.isclose(actual[metric],expected[metric])
            assert (sf/'last_weights.pt').exists()
        for (seed,epoch),used in presentations.groupby(['seed','epoch']):
            plan=plans[seed][plans[seed].epoch.eq(epoch)]
            w=plan.groupby('query_id').weight.sum()
            assert np.allclose(used.set_index('query_id').weight_sum.loc[w.index],w)
    for method,selected in lock['selected'].items():
        choices=[g for g in lock['groups'] if g['method']==method]
        expected=sorted(choices,key=lambda c:(-round(c['dev_mean']['Recall@100'],12),-round(c['dev_mean']['cNDCG@10'],12),c['epoch'],c['lr']))[0]
        assert selected==expected
    winner=sorted(lock['selected'],key=lambda m:(-round(lock['selected'][m]['dev_mean']['Recall@100'],12),
                  -round(lock['selected'][m]['dev_mean']['cNDCG@10'],12),lock['selected'][m]['parameters'],m))[0]
    assert winner==lock['best_method']
    gate=json.loads((HERE/'all_models_pretest_verified.json').read_text())
    assert gate['passed'] and len(gate['models'])==12
    gate_time=pd.Timestamp(gate['completed_utc'])
    assert gate_time>pd.Timestamp(lock['locked_utc'])
    amendment_path=HERE/'EVALUATION_STORAGE_AMENDMENT.json'
    amendment=json.loads(amendment_path.read_text())
    assert amendment['primary_config_sha256']==sha(CONFIG)
    assert amendment['selection_sha256']==sha(HERE/'selection.json')
    assert amendment['script_sha256']==sha(HERE/'space_bounded_eval.py')
    assert not amendment['training_and_selection_changed'] and not amendment['heldout_metrics_seen']
    assert pd.Timestamp(lock['locked_utc'])<pd.Timestamp(amendment['frozen_utc'])<gate_time
    invariant=json.loads((HERE/'PHASE6_INVARIANCE.json').read_text())['models']
    assert len(invariant)==12
    target_count=0
    for key,record in invariant.items():
        folder=HERE/'results'/key
        meta=json.loads((folder/'evaluation.json').read_text())
        pretest=json.loads((folder/'pretest_audit.json').read_text())
        assert pretest['passed'] and not pretest['test_evaluated']
        assert pd.Timestamp(amendment['frozen_utc'])<pd.Timestamp(pretest['completed_utc'])<=gate_time
        assert pretest['storage_amendment_sha256']==sha(amendment_path)
        assert pretest['full_catalog']==record['full_catalog']
        assert pd.Timestamp(meta['completed_utc'])>gate_time
        assert meta['selection_sha256']==sha(HERE/'selection.json')
        assert record['sample']['changed_input_products']>=100 and record['sample']['products']>=128
        assert len(record['full_catalog'])==6 and [x['schedule'] for x in record['full_catalog']]==STEMS[1:]
        base=np.load(folder/'products.npy')
        assert base.shape==(42994,768) and base.dtype==np.float32 and np.isfinite(base).all()
        assert sha(folder/'products.npy')==record['products_sha256']
        primary=pd.read_parquet(folder/'catalog_graded_pairs.parquet')
        regenerated=pd.read_parquet(folder/'regenerated_graded_pairs.parquet')
        assert primary[['query_id','product_id','label']].equals(regenerated[['query_id','product_id','label']])
        assert primary.C0.equals(regenerated.C0)
        assert len(record['rank_audit'])==6
        for numerical,rank_record in zip(record['full_catalog'],record['rank_audit']):
            s=numerical['schedule'];assert rank_record['schedule']==s
            assert numerical['actual_forward'] and numerical['products']==42994
            for actual in [numerical,rank_record]:
                assert 0<=actual['mean_L2']<=actual['max_L2']<=1e-5
                assert 0<=actual['mean_cosine_difference']<=actual['max_cosine_difference']<=1e-5
                assert len(actual['forward_sha256'])==64
            same_digest=numerical['forward_sha256']==rank_record['forward_sha256']
            assert rank_record['matches_pretest_forward_sha']==same_digest
            if same_digest:
                for metric in ['max_L2','mean_L2','max_cosine_difference','mean_cosine_difference']:
                    assert np.isclose(numerical[metric],rank_record[metric],rtol=1e-6,atol=1e-12)
            hi=primary.label.eq('Exact')
            assert int((regenerated[s]!=primary.C0).sum())==rank_record['all_judged_rank_changes']
            assert int((regenerated.loc[hi,s]!=primary.loc[hi,'C0']).sum())==rank_record['Exact_rank_changes']
            for k in [20,100]:
                assert int(((regenerated.loc[hi,s]<=k)!=(primary.loc[hi,'C0']<=k)).sum())==rank_record[f'Exact_inclusion{k}_changes']
        for metric,value in measures(regenerated)[0].items():assert np.isclose(value,record['regenerated_metrics'][metric])
        stats=pd.read_parquet(folder/'field_weights.parquet')
        assert len(stats)==stats.product_id.nunique()==42994
        assert stats.max_weight.between(0,1+1e-6).all()
        assert (stats.entropy>=-1e-6).all() and (stats.normalized_entropy<=1+1e-5).all()
        nonempty=stats.fields>0
        assert np.allclose(stats.loc[nonempty,'weight_sum'],1,atol=1e-5)
        assert (stats.effective_fields<=stats.fields+1e-3).all()
        expected_target=record['method']==winner
        assert (folder/'target_pairs.parquet').exists()==expected_target
        if expected_target:target_count+=1
    assert target_count==3
    boundary=pd.read_csv(HERE/'numerical_boundary_changes.csv')
    for key,record in invariant.items():
        for rank_record in record['rank_audit']:
            for cutoff in [20,100]:
                actual=boundary[(boundary.Method==record['method'])&(boundary.Seed==record['seed'])&
                                (boundary.Schedule==rank_record['schedule'])&(boundary.Cutoff==cutoff)]
                assert len(actual)==rank_record[f'Exact_inclusion{cutoff}_changes']
                assert ((actual.C0_rank<=cutoff)!=(actual.Regenerated_rank<=cutoff)).all()
    results=pd.read_csv(HERE/'PHASE6_RESULTS.csv');seeds=pd.read_csv(HERE/'PHASE6_SEEDS.csv')
    for _,row in results.iterrows():
        if row.Split=='test':
            assert (row.Exact_pairs,row.Exact_queries)==((6797,239) if 'fully_fitting' in row.Mode else (21299,308))
        assert np.isclose(row['VI@100']+row['Robust@100']+row['Never@100'],100)
        if row.Method in METHODS:
            z=seeds[(seeds.Method==row.Method)&(seeds.Split==row.Split)&(seeds.Mode==row.Mode)]
            assert len(z)==3 and z.Epoch.eq(lock['selected'][row.Method]['epoch']).all() and z.LR.eq(lock['selected'][row.Method]['lr']).all()
            metric='Inclusion@100' if 'target' in row.Mode else 'Recall@100'
            assert np.isclose(z[metric].mean(),row[metric]) and np.isclose(z[metric].std(),row[metric+'_seed_sd'])
    ci=pd.read_csv(HERE/'PHASE6_BOOTSTRAP.csv')
    assert ci.Draws.eq(10000).all() and ci.Bootstrap_seed.eq(20260963).all()
    for _,row in ci[ci.Seed.eq('seed_mean')].iterrows():
        a=results[(results.Method==row.Method)&(results.Split==row.Split)&(results.Mode==row.Mode)].iloc[0]
        b=results[(results.Method==row.Reference)&(results.Split==row.Split)&(results.Mode==row.Mode)].iloc[0]
        assert np.isclose(a[row.Metric]-b[row.Metric],row.Delta_pp)
    # Recompute one paired interval independently from saved per-seed Exact rankings.
    pq=[measures(pd.read_parquet(HERE/'results'/f'{winner}_s{s}'/'catalog_graded_pairs.parquet'))[1]['Recall@100'] for s in cfg['seeds']]
    a=sum(pq)/3
    old=pd.read_parquet(ROOT/'phase5_mitigation/results/Original/catalog_pairs.parquet')
    b=prior.metric_frame(old[old.query_id.isin(cfg['test'])])[1]['Recall@100']
    expected=np.array(prior.bootstrap(a-b.loc[a.index],seed=20260963,samples=10000))*100
    row=ci[(ci.Method==winner)&ci.Reference.eq('Original')&ci.Seed.eq('seed_mean')&ci.Split.eq('test')&ci.Mode.eq('catalog')&ci.Metric.eq('Recall@100')].iloc[0]
    assert np.allclose(expected,row[['Delta_pp','CI_low_pp','CI_high_pp']].to_numpy(dtype=float))
    report=(HERE/'PHASE6_REPORT.md').read_text(encoding='utf8')
    assert report.startswith('# PHASE VI — Corrected Structural Permutation-Invariant Product Representations')
    assert all(f'**{i}.' in report for i in range(1,12))
    required=['PHASE6_REPORT.md','PHASE6_RESULTS.csv','PHASE6_BOOTSTRAP.csv','PHASE6_SEEDS.csv','PHASE6_TRAINING_CURVES.csv',
              'PHASE6_INVARIANCE.json','PHASE6_FIELD_WEIGHTS.csv','PHASE6_DATA_AUDIT.json','PHASE6_CONFIG.json']
    assert all((HERE/name).exists() for name in required)
    verification=dict(passed=True,verified_utc=now(),historical_files_unchanged=True,unit_tests_passed=10,
                      train_only_graded_sampling_verified=True,query_presentation_ratio=1.,common_lr_epoch_across_seeds_verified=True,
                      candidate_groups=8,candidate_seed_runs=24,selected_seed_models=12,target_seed_models=3,
                      selected_scoring_layers_updated_from_neutral=True,numerical_boundary_rows_verified=len(boundary),
                      all_actual_vector_audits_before_test=True,storage_only_amendment_verified=True,
                      pretest_and_ranking_forward_digests_verified=True,regenerated_ranks_verified=True,
                      paired_interval_independently_reproduced=True,result_rows=len(results),seed_rows=len(seeds),bootstrap_rows=len(ci),
                      selection_sha256=sha(HERE/'selection.json'))
    dump(HERE/'verification.json',verification)
    paths=sorted(p for p in HERE.rglob('*') if p.is_file() and not {'__pycache__','.pytest_cache'}&set(p.parts)
                 and p.name!='FINAL_DELIVERY_MANIFEST.json' and not p.name.endswith('.tmp'))
    manifest=dict(phase='VI corrected',status='complete',completed_utc=now(),output_directory=str(HERE),
                  required_files=required+['FINAL_DELIVERY_MANIFEST.json'],verification=verification,
                  decision=json.loads((HERE/'decision.json').read_text()),historical_sources=json.loads((HERE/'historical_sources.json').read_text()),
                  outputs=[dict(path=str(p.relative_to(HERE)),bytes=p.stat().st_size,sha256=sha(p)) for p in paths])
    dump(HERE/'FINAL_DELIVERY_MANIFEST.json',manifest)
    for entry in manifest['outputs']:assert sha(HERE/entry['path'])==entry['sha256']
    print(json.dumps(verification,indent=2));print('Verified',len(paths),'corrected output hashes')


if __name__=='__main__':main()
