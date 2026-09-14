"""Write the definitive corrected-phase report from measurements, without changing selection."""
import json
import numpy as np
import pandas as pd
from common_corrected import HERE,ROOT,CONFIG,METHODS,sha,dump,now


def tab(frame,columns=None,digits=3):
    frame=frame[columns] if columns else frame
    frame=frame.copy()
    for column in frame:
        frame[column]=frame[column].map(lambda value:'—' if pd.isna(value) else
            (format(value,'.0e' if column=='LR' else '.6g' if 'L2' in column else f'.{digits}f') if isinstance(value,(float,np.floating)) else value))
    return frame.to_markdown(index=False,disable_numparse=True)


def main():
    cfg=json.loads(CONFIG.read_text());lock=json.loads((HERE/'selection.json').read_text())
    decision=json.loads((HERE/'decision.json').read_text())
    results=pd.read_csv(HERE/'PHASE6_RESULTS.csv');seeds=pd.read_csv(HERE/'PHASE6_SEEDS.csv')
    bootstrap=pd.read_csv(HERE/'PHASE6_BOOTSTRAP.csv');curves=pd.read_csv(HERE/'PHASE6_TRAINING_CURVES.csv')
    audit=json.loads((HERE/'PHASE6_DATA_AUDIT.json').read_text());init=pd.read_csv(HERE/'initialization.csv')
    inv=json.loads((HERE/'PHASE6_INVARIANCE.json').read_text())['models']
    field=pd.read_csv(HERE/'PHASE6_FIELD_WEIGHTS.csv');keys=pd.read_csv(HERE/'frequent_field_weights.csv')
    categories=pd.read_csv(HERE/'category_analysis.csv');query_deltas=pd.read_csv(HERE/'per_query_gains_losses.csv')
    lengths=pd.read_csv(HERE/'length_analysis.csv');costs=pd.read_csv(HERE/'cost_summary.csv')
    exposure=pd.read_csv(HERE/'actual_query_presentations.csv')
    best=lock['best_method'];best_label=METHODS[best]
    main_table=results[(results.Split=='test')&(results.Mode=='catalog')].copy()
    dev=results[(results.Split=='dev')&(results.Mode=='catalog')].copy()
    fitting=results[(results.Split=='test')&(results.Mode=='catalog_fully_fitting')].copy()
    targets=results[(results.Split=='test')&(results.Mode=='target')].copy()
    def delta(method,reference,metric='Recall@100',split='test',mode='catalog'):
        row=bootstrap[(bootstrap.Method==method)&(bootstrap.Reference==reference)&(bootstrap.Metric==metric)&
                      (bootstrap.Split==split)&(bootstrap.Mode==mode)&bootstrap.Seed.eq('seed_mean')].iloc[0]
        return f"{row.Delta_pp:+.3f} pp (95% CI {row.CI_low_pp:+.3f} to {row.CI_high_pp:+.3f})"
    # C1 is included explicitly even when another family wins the development selection.
    def observed(method,reference,split='test',metric='Recall@100',mode='catalog'):
        table=results[(results.Split==split)&(results.Mode==mode)].set_index('Method')
        return float(table.loc[method,metric]-table.loc[reference,metric])
    def seed_direction(method):
        baseline=float(dev[dev.Method=='Set-Attention']['Recall@100'].iloc[0])
        values=seeds[(seeds.Method==method)&(seeds.Split=='dev')&(seeds.Mode=='catalog')]['Recall@100']
        return int((values>baseline+1e-10).sum())
    def threshold_methods(reference,strict=False):
        reference_value=float(main_table[main_table.Method==reference]['Recall@100'].iloc[0])
        learned=main_table[main_table.Method.isin(METHODS)]
        qualifying=learned[learned['Recall@100']>reference_value+1e-10] if strict else learned[learned['Recall@100']>=reference_value-1e-10]
        return ', '.join(f"{r.Label} ({r['Recall@100']:.3f}%)" for _,r in qualifying.iterrows()) or 'none'
    audit_labels=pd.DataFrame([dict(Label=k,Training_queries=audit['queries_with_labels'][k],**v) for k,v in audit['label_counts'].items()])
    selected=pd.DataFrame([dict(Method=METHODS[m],LR=x['lr'],Epoch=x['epoch'],Params=x['parameters'],Mean_dev_R100=100*x['dev_mean']['Recall@100'],
                                Mean_dev_cNDCG10=100*x['dev_mean']['cNDCG@10'],Dev_seeds_above_SetAttention=seed_direction(m)) for m,x in lock['selected'].items()])
    groups=[]
    for group in lock['groups']:
        c=curves[curves.group.eq(group['name'])]
        last=c[c.epoch.eq(group['epochs_run'])]
        groups.append(dict(Group=group['name'],Selected_configuration=group['name']==lock['selected'][group['method']]['name'],
                           Selected_epoch=group['epoch'],Epochs_run=group['epochs_run'],Best_mean_dev_R100=100*group['dev_mean']['Recall@100'],
                           Last_mean_dev_R100=last['Recall@100'].mean(),Last_mean_train_loss=last.train_loss.mean()))
    invrows=[]
    for key,record in inv.items():
        invrows.append(dict(Model=key,Sample_max_L2=record['sample']['max_L2'],Sample_mean_L2=record['sample']['mean_L2'],
                            Full_max_L2=max(x['max_L2'] for x in record['full_catalog']),Full_mean_L2=np.mean([x['mean_L2'] for x in record['full_catalog']]),
                            Full_max_cosine_difference=max(x['max_cosine_difference'] for x in record['full_catalog']),
                            Repeated_forward_digest_matches=sum(x['matches_pretest_forward_sha'] for x in record['rank_audit']),
                            Changed_exact_ranks=sum(x['Exact_rank_changes'] for x in record['rank_audit']),
                            Regenerated_VI20=100*record['regenerated_metrics']['VI@20'],Regenerated_VI100=100*record['regenerated_metrics']['VI@100']))
    invtable=pd.DataFrame(invrows);invtable.to_csv(HERE/'invariance_summary.csv',index=False)
    architecture=pd.DataFrame([
        ['Set-Mean','no','no','no','yes','yes'],['Set-Attention','no','implicit content','no','yes','yes'],
        ['Contextual DeepSets','global context','no explicit ID','no','yes','yes'],['Set Transformer','pairwise','no explicit ID','no','yes','yes'],
        ['Set Transformer + Field-ID','pairwise','native learned ID','no','yes','yes'],['Circular Attention','global context','circular native ID','no','yes','yes']],
        columns=['Method','Cross-field interaction','Field identity','Learned replacement values','Uses original BGE values','Permutation invariant'])
    architecture.to_csv(HERE/'TABLE_VI_2.csv',index=False)
    columns=['Label','Params','Recall@20','Recall@100','cNDCG@10','Delta_R100_vs_Set-Attention','Delta_R100_vs_Canonical','VI@20','VI@100','Robust@100','Never@100','Recall@100_seed_sd']
    table1=main_table[columns].rename(columns={'Label':'Method','Recall@100_seed_sd':'Seed SD','Delta_R100_vs_Set-Attention':'Delta R100 vs Set-Attention','Delta_R100_vs_Canonical':'Delta R100 vs Canonical'})
    table1.to_csv(HERE/'TABLE_VI_1.csv',index=False)
    weighted=exposure.groupby(['Group','seed','epoch']).agg(Min_presentations=('presentations','min'),Max_presentations=('presentations','max'),
                                                        Min_weight_sum=('weight_sum','min'),Max_weight_sum=('weight_sum','max')).reset_index()
    collapse=field[field.Metric.eq('max_weight')].copy()
    len_summary=lengths.groupby(['Method','Split','Subset']).agg(Recall100=('Recall100','mean'),Delta_vs_SetAttention=('Delta_R100_vs_SetAttention','mean'),
                                                               Pairs=('Pairs','first'),Queries=('Queries','first')).reset_index()
    current=main_table.set_index('Method').loc[best]
    seed_sd_text='; '.join(f"{m}: {main_table.set_index('Method').loc[m,'Recall@100_seed_sd']:.3f} pp" for m in METHODS)
    seed_direction_text='; '.join(f'{m}: {seed_direction(m)}/3' for m in METHODS)
    relation_exp=exposure[['Exact_Irrelevant','Exact_Partial','Partial_Irrelevant']].sum().reset_index()
    relation_exp.columns=['Relation','Actual_presentations_all_candidate_runs']
    report=[
        '# PHASE VI — Corrected Structural Permutation-Invariant Product Representations','',
        f"Completed {now()}. All corrected artifacts live in `phase6_corrected/`. The development-selected architecture is **{best_label} ({best})**.",'',
        f"**Final decision: {decision['code']}. {decision['title']}.** The decision uses the operational development/seed/paired-uncertainty rules frozen in PROTOCOL.md before corrected held-out evaluation.",'',
        f"Its held-out seed-mean Recall@100 is **{current['Recall@100']:.3f}%**, seed SD **{current['Recall@100_seed_sd']:.3f} pp**; cNDCG@10 is **{current['cNDCG@10']:.3f}%**. Compared with frozen Set-Attention, Recall@100 changes by {delta(best,'Set-Attention')}. {seed_direction(best)} of3 seeds improve on Set-Attention's development Recall@100.",'',
        '## 1. Corrected scientific question and preserved history','',
        'This experiment asks whether contextual field weighting improves invariant retrieval while preserving frozen BGE values. Earlier Phase VI/VI-B pilots motivate removing random replacement-value projections and unconstrained embedding residuals. Their learned held-out results are not primary evidence or tuning inputs for this corrected phase. Their code, checkpoints, logs and reports remain archived unchanged under their original paths, verified by historical_sources.json. The raw representation track is separate from labeled PI-FT.','',
        'The catalog,72/24/384 query split, original atom inventory with duplicate occurrences, seven schedules, pinned frozen BGE/query/field/non-attribute embeddings, FP32 scores, stable catalog-index tie rule and candidate budget are reused. No new dataset, encoder, architecture variant or manuscript edit is part of this run.','',
        '## 2. Training-data audit and query balancing','',
        f"The fixed training split has {audit['train_queries']} queries and {audit['judged_pairs']:,} judged query-product pairs. {len(audit['eligible_queries'])} queries have a valid strict grade comparison; excluded query IDs are {audit['excluded_queries']}. These exclusions are eligibility constraints, not missing queries silently sampled at lower frequency.",'',
        tab(audit_labels),'',
        f"Queries with Exact+Partial: {audit['queries_Exact_and_Partial']}; Exact+Irrelevant: {audit['queries_Exact_and_Irrelevant']}; Exact+any lower grade: {audit['queries_Exact_and_lower']}. training_query_labels.csv preserves Exact/Partial/Irrelevant counts for every training query. The saved pools and20-epoch plans contain only TRAIN judgments.",'',
        'Relations are Exact>Irrelevant with loss weight1, Exact>Partial with weight.5, and Partial>Irrelevant with weight.5. No unjudged product or another query’s positive is automatically a negative. Duplicate product presentations remain query-specific judgments; duplicate attribute atoms remain separate field occurrences.','',
        f"Every epoch has64 randomly ordered complete query cycles: {audit['examples_per_epoch']:,} comparisons and exactly64 per eligible query. The observed presentation max/min ratio is {float((weighted.Max_presentations/weighted.Min_presentations).max()):.3f} across all completed candidate epochs. Relation types and independently shuffled grade pools rotate across epochs. actual_query_presentations.csv records every query/seed/configuration/epoch, relation counts and unique higher/lower products. The per-query weighted exposure ranges from {weighted.Min_weight_sum.min():.1f} to {weighted.Max_weight_sum.max():.1f}; equality of presentation counts does not cancel the required grade-gap loss weights.",'',
        tab(relation_exp),'',
        'The same planned comparison stream is paired across architectures/LRs within each seed. Epochs stop at different times, so lifetime exposures can differ across configurations; equal query exposure holds inside each completed epoch. No uniform sampling from the global judgment table and no multi-negative diagnostic was used.','',
        '## 3. Value-preserving architectures','',
        '```text','h_i = existing frozen BGE embedding of the original field atom','scores_i = contextual scalar scoring network(fields)','w_i = softmax_i(scores), with padding removed','v_attr = normalize(sum_i w_i * h_i)','v_product = normalize(0.5 * v_non_attribute + 0.5 * v_attr)','```','',
        'Only weights depend on learned contextual states. Every nonempty attribute pool has nonnegative weights summing to one before vector normalization. There is no learned768-D replacement value, direct embedding residual, learned mixing coefficient or serialization-position input. Within-field sequence semantics and the original cached BGE values remain unchanged. Empty sets fall back to the normalized non-attribute vector.','',
        f"C1 uses a128-D DeepSets mean context and a shared scalar scorer of concat(original h_i,context). C2 uses one128-D, four-head self-attention/residual/LayerNorm/feed-forward block to produce scalar scores. C3 adds a128-D native field embedding only to that attention input. C4 uses a128-D projection plus cos/sin of trainable native-key angles, shared field MLPs and mean context for scalar scores. The vocabulary has {audit['field_keys']:,} TRAIN-product keys plus explicit UNK0; no key is assigned by serialization position. C4 angles start evenly spaced in lexical ID order. All duplicate keys/atoms survive.",'',
        '### Table VI-2 — What each method learns','',tab(architecture),'',
        'The original BGE atom text can itself contain a field name, even where no explicit identity parameter is added. Circular versus C3 changes contextual architecture as well as identity parameterization; its contrast is exploratory and cannot establish a causal benefit of circular geometry.','',
        '## 4. Neutral initialization, training and common selection','',
        'All final scalar-score weights/biases start at exactly zero, producing uniform weights. The independent full-catalog equal-weight reconstruction and actual permutation checks passed before optimization. Only development initialization metrics were read before selection. Epoch0 is a sanity check and was excluded from selectable checkpoints; a learned success must come from epoch≥1.','',
        tab(init.assign(**{m:100*init[m] for m in ['Recall@20','Recall@100','cNDCG@10','VI@20','VI@100']}),['method','epoch','max_L2_to_independent_SetMean','Recall@20','Recall@100','cNDCG@10','VI@20','VI@100'],digits=6),'',
        'Optimizer: AdamW, LR{1e-4,3e-5}, weight decay.01, clip1, batch16 query-balanced comparisons, temperature.05, maximum20 epochs, patience4. Loss is mean(weight × softplus((q·v(lower)−q·v(higher))/.05)). Only scoring-network parameters train. Every architecture/LR runs seeds42/43/44 together. Epochs are chosen from their mean dev Exact R100, then mean graded cNDCG10; a single LR and a single epoch applies to all three seeds. Ties prefer earlier epoch, then lower LR. The architecture winner uses the same dev metrics, then parameter count and fixed method order. No held-out metric enters selection.','',
        tab(selected),'',
        'All LR groups, including negative runs:','',tab(pd.DataFrame(groups)),'',
        '![Development learning curves](PHASE6_TRAINING.png)','',
        'The figure shows mean±one sample SD across three seeds at each architecture’s selected common LR. Vertical dotted lines mark its common checkpoint. Epoch0 is displayed but cannot win. PHASE6_TRAINING_CURVES.csv contains both LRs and all seed curves, losses, gradients and sample weight-concentration diagnostics. Shared stopping keeps the three seeds on common epochs; no missing-seed curve is imputed.','',
        '## 5. Metric definitions and distinct query supports','',
        'Recall/VI/Robust/Never use Exact pairs within query and query-macro averaging. Robust means inclusion under every schedule; VI means sometimes included; Never means never included. The reported worst-schedule R100 averages each query’s minimum schedule recall. minimum_macro_schedule_R100 additionally gives the minimum aggregate catalog-schedule recall.','',
        'cNDCG@10 retains all judged labels with WANDS gains3/1/0, removes unjudged products before discounting, uses1/log2(rank+1), and normalizes by all ideal judged gains within the evaluated support. Every positive-IDCG query is eligible even without Exact labels. The implementation is checked against the current repository’s common.pair_metrics. This is condensed full-catalog cNDCG, not returned-list NDCG or an assumption that unjudged products are irrelevant. All table metrics below are percentages; differences and seed SD are percentage points. Raw training logs retain fractions.','',
        tab(main_table,['Label','Exact_pairs','Exact_queries','cNDCG_queries']),'',
        'Original’s seven frozen raw schedule ranks, Canonical raw ranks, Set-Mean ranks and all four controls’ frozen Exact metrics are reused. The missing Set-Attention graded metric is derived from its archived product vectors without training/encoding; baselines/Set-Attention/graded_reconciliation.json discloses any small Exact-rank arithmetic differences, while frozen Exact ranks remain authoritative.','',
        '## 6. Table VI-1 — Corrected invariant representations','',tab(table1),'',
        'Primary one-vector deployment metrics use a fixed product vector for invariant methods. Their independent actual-forward numerical schedule results are reported in Section9 and in regenerated modes of PHASE6_RESULTS.csv. Zero cached VI alone is not the invariance verification.','',
        tab(main_table,['Label','worst_schedule@100','minimum_macro_schedule_R100']),'',
        'Development evidence at the same selected checkpoints:','',tab(dev,['Label','Recall@20','Recall@100','cNDCG@10','Recall@100_seed_sd']),'',
        'Individual selected seeds (test):','',tab(seeds[(seeds.Split=='test')&(seeds.Mode=='catalog')&seeds.Method.isin(METHODS)],['Label','Seed','Epoch','LR','Recall@100','cNDCG@10','Robust@100','Never@100']),'',
        '## 7. Paired uncertainty and architectural contrasts','',
        'Intervals use10,000 aligned paired query-bootstrap draws, seed20260963, percentile95% limits and no multiplicity correction. Primary rows average per-query performance across the fixed three seeds before query resampling; this is neither a vector ensemble nor three times as many independent queries. Seed SD describes training variation, separately from query uncertainty. Individual-seed intervals and all requested metrics/supports remain in PHASE6_BOOTSTRAP.csv. Crossing zero or overlapping CIs does not establish equivalence.','',
        tab(bootstrap[(bootstrap.Split=='test')&(bootstrap.Mode=='catalog')&bootstrap.Seed.eq('seed_mean')&bootstrap.Metric.isin(['Recall@100','cNDCG@10','Robust@100','Never@100'])],
            ['Method','Reference','Metric','Queries','Delta_pp','CI_low_pp','CI_high_pp']),'',
        '## 8. Common fully-fitting raw support','',
        'The unchanged phase4 fully_fits product mask gives exactly6,797 held-out Exact targets across239 queries, with all42,994 catalog competitors retained. For the newly requested graded metric, the same product mask filters judged targets before both condensed ranking and ideal computation; its positive-IDCG query count is reported independently. The complement and attribute-length strata are descriptive, not a causal truncation decomposition or a guarantee that all independently encoded atoms are untruncated.','',
        tab(fitting,['Label','Exact_queries','cNDCG_queries','Recall@20','Recall@100','cNDCG@10','VI@20','VI@100','Robust@100','Never@100','worst_schedule@100']),'',
        '## 9. Actual-forward invariance and numerical ranking audit','',
        'Each candidate epoch tested the same128 products under all seven original schedules, checking preserved atom multisets and the existing raw serializer. Every selected checkpoint then passed a full42,994-product audit before any corrected held-out ranking. C0 vectors and all checkpoints are retained. Alternate schedules were actually forwarded across the full catalog before test, with numerical summaries and SHA-256 array digests recorded, then forwarded again for held-out ranking; repeated-forward digest agreement is recorded for every model/schedule. Alternate arrays are released after use without rounding or quantization.','',
        'EVALUATION_STORAGE_AMENDMENT.json froze this storage-only procedure after model selection and before any corrected held-out metrics. A storage probe showed that retaining all compressed permutation arrays would exceed then-available disk space; the user subsequently freed sufficient space while the amended audit was already running. The procedure was retained consistently. Training, selection, models, schedules and ranking are unchanged. The original evaluator and primary configuration remain preserved with their original hashes; space_bounded_eval.py is separately frozen by the amendment.','',
        invtable.to_markdown(index=False,floatfmt='.9g'),'',
        'Max L2 must be≤1e-5. Float64 cosine comparisons characterize FP32 output differences. Tiny exact-rank changes near ties and any cutoff crossings are preserved in PHASE6_INVARIANCE.json and the regenerated pair files; they are distinguished from the mathematical permutation-invariant architecture.','',
        'numerical_boundary_changes.csv identifies every Exact target whose Top-20 or Top-100 inclusion changes under a separately forwarded schedule, including both ranks. The fixed-vector deployment table has zero VI for learned methods by construction; the numerical audit table above is the evidence for their actual implementation stability.','',
        '## 10. Target-only interventions','',
        f"The development winner {best_label}, all three seeds, is the only corrected architecture receiving target-only evaluation. Frozen Original, Canonical and Set-Attention controls are reused. Each intervention replaces one target against that method’s own C0 catalog, with stable target removal/reinsertion. Inclusion is not a common mixed-index Recall. The regenerated_target modes additionally expose floating-point schedule differences.",'',
        tab(targets,['Label','Inclusion@20','Inclusion@100','VI@20','VI@100','Robust@100','Never@100']),'',
        '## 11. Field-weight concentration and schema diagnostics','',
        'PHASE6_FIELD_WEIGHTS.csv gives per-seed catalog distributions of Shannon entropy, normalized entropy, max weight and exp(entropy) effective fields. Every selected model preserves the per-product data. Normalized entropy is entropy/log(m) for m>1; singleton/empty conventions are documented in code, and concentration is also reported only among multiple-field products. These weights are associations, not causal explanations.','',
        tab(field[field.Metric.isin(['entropy','normalized_entropy','effective_fields'])],['Method','Seed','Metric','mean','p05','median','p95']),'',
        tab(collapse,['Method','Seed','mean','median','p95','Fraction_max_weight_gt_0_5','Fraction_max_weight_gt_0_9','Multiple_field_fraction_max_gt_0_9']),'',
        'Frequent native field keys (at least100 occurrences; top15 per identity method by occurrence count). Means are per occurrence, preserving duplicate keys. Unseen model keys share UNK parameters, but reporting retains their original key strings.','',
        tab(keys.sort_values('Occurrences',ascending=False).groupby('Method',sort=False).head(15),['Method','field_key','Occurrences','Mean_weight','Seed_SD']),'',
        '## 12. Failure diagnostics and heterogeneous effects','',
        'Query-type strata compare the seed-mean learned R100 with Set-Attention on the same Exact queries. Small strata and selected extremes are descriptive; no category or threshold changes the models. per_query_gains_losses.csv retains all queries, including the largest gains and losses.','',
        tab(categories.sort_values(['Split','Method','Delta_R100_pp']),['Method','Split','Query_type','Queries','Delta_R100_pp']),'',
        f"Long products use the pretraining, unlabeled catalog75th-percentile threshold: attribute_count≥{cfg['long_attribute_threshold']:g}. Length and fit supports keep all catalog competitors, with query-macro recall computed within each target subset.",'',
        tab(len_summary[len_summary.Split.eq('test')]),'',
        'These diagnostics distinguish changes in access, weight concentration and support composition. They do not prove why an architecture gains or loses retrieval effectiveness, and no architecture is modified after these observations.','',
        '## 13. Compute, deployment and limitations','',
        f"Completed {len(lock['groups'])} architecture/LR groups ×3 seeds, with {int(sum(g['epochs_run'] for g in lock['groups'])*3)} trained seed-epochs. Instrumented stages sum to {costs.seconds.sum():.3f} wall seconds; training, dev generation, final generation, full-catalog audit and held-out ranking are separated below. These measurements exclude interpreter/import startup and report-writing time, and are not online latency benchmarks.",'',
        tab(costs.groupby('stage').seconds.sum().reset_index()),'',
        'Every deployed method stores one768-D FP32 vector/product:132,077,568 payload bytes for42,994 products, plus the same index overhead as before. Query encoding, dot-product scoring, candidate slots and stable ranking are unchanged. Multiple seeds are experimental replications, each with its own single-vector index. BGE encoding is entirely reused.','',
        'Limitations: one historical dataset and one encoder; only24 development queries with Exact and cNDCG eligibility differing; three training seeds; a finite query-balanced comparison plan and grade-weighted exposure; shared early stopping on a coarse Exact metric; train-product field vocabulary; descriptive concentration and support analyses; uncorrected multiple comparisons. Neural attention weights can still discard useful values even though replacement-value and direct-residual confounds are removed.','',
        'Set-Attention retains its historical training protocol and single archived seed, whereas the corrected architectures use the new graded, query-balanced protocol and three seeds. Their comparison therefore changes training as well as architecture; the value path and deployment geometry are controlled, but an isolated causal architecture effect is not identified. This is a new frozen evaluation on a historical test set that the project has used before, so any positive solution claim would still require external confirmation.','',
        'The frozen training logger calls float(loss) after backward for scalar recording, which emits a PyTorch warning about converting a requires-grad tensor. This does not enter the loss, optimizer or selection computation; stored finite losses/gradients and parameter updates are checked. The experimental source was retained unchanged after freezing.','',
        '## 14. Explicit answers to the eleven questions','',
        f"**1. Does global contextual weighting improve over independent Set-Attention?** No held-out improvement was observed. C1’s test R100 difference is {delta('C1','Set-Attention')} and its dev difference is {observed('C1','Set-Attention','dev'):+.3f} pp; {seed_direction('C1')}/3 dev seeds improve. The development gain does not carry over to held-out access.",'',
        f"**2. Does true field-to-field self-attention improve independent weighting?** The small positive point estimate is inconclusive. C2−Set-Attention test R100 is {delta('C2','Set-Attention')}; dev direction is {observed('C2','Set-Attention','dev'):+.3f} pp with {seed_direction('C2')}/3 improving seeds. This does not establish a consistent benefit here or imply that Set Transformers are generally ineffective.",'',
        f"**3. Does field identity help once values are preserved?** No consistent advantage is established. C3−C2 test R100 is {delta('C3','C2')}; the corresponding dev difference is {observed('C3','C2','dev'):+.3f} pp. The configurations are independently selected across the same three seeds, so this is the requested practical architecture comparison, not an isolated parameter intervention.",'',
        f"**4. Does circular identity outperform ordinary ID?** No supported advantage is established. C4−C3 test R100 is {delta('C4','C3')}, with dev difference {observed('C4','C3','dev'):+.3f} pp. A circular-geometry causal claim is also unsupported because global versus pairwise context differs.",'',
        f"**5. Can a learned invariant method outperform Set-Attention?** Corrected architectures whose historical test seed-mean exceeds Set-Attention: {threshold_methods('Set-Attention',strict=True)}. These point estimates do not replace the development winner. The frozen best method’s contrast is {delta(best,'Set-Attention')}; its all-seed development condition is {decision['all_seeds_improve_dev_vs_SetAttention']}. The prespecified learned-benefit criterion is {decision['qualified_learned_benefit']}. The development table gives the consistency counts for every architecture.",'',
        f"**6. Can a method match Canonical?** Corrected architectures whose test seed-mean reaches Canonical: {threshold_methods('Canonical')}. The best-method mean contrast is {delta(best,'Canonical')}; all three of its test seeds reach Canonical: {decision['all_test_seeds_at_least_Canonical']}. This point-threshold statement is separate from statistical equivalence and the development qualification.",'',
        f"**7. Can a method match Original while retaining zero VI?** Corrected architectures whose test seed-mean reaches Original: {threshold_methods('Original')}. The best-method contrast is {delta(best,'Original')}; all three of its test seeds reach Original: {decision['all_test_seeds_at_least_Original']}. Actual numerical VI and vector deviations are disclosed in Section9. Strong-solution status also requires the frozen development and uncertainty conditions.",'',
        f'**8. Are results consistent across seeds?** Test R100 seed SDs are {seed_sd_text}. Counts improving development R100 over Set-Attention are {seed_direction_text}. C1/C4 have stable test results but lose to Set-Attention; C2/C3 do not establish a consistent improvement across development and test. Three seeds remain a small sample and are not pooled as independent queries.','',
        f"**9. Do weights collapse?** Across selected checkpoints, the multiple-field fraction with max weight>.9 ranges from {100*collapse.Multiple_field_fraction_max_gt_0_9.min():.3f}% to {100*collapse.Multiple_field_fraction_max_gt_0_9.max():.3f}%. Entropy and effective-field distributions above distinguish broad weighting from concentration; no weight-based causal claim is made.",'',
        f"**10. Does the fully-fitting subset behave differently?** For the development winner, its R100 difference from Set-Attention is {observed(best,'Set-Attention',mode='catalog_fully_fitting'):+.3f} pp on fully-fitting targets versus {observed(best,'Set-Attention'):+.3f} pp overall. The fit/complement and length tables report support sizes and directions. This is not a causal truncation decomposition.",'',
        ('**11. Should a learned architecture enter the WWW main paper?** No new learned mitigation claim is justified by the frozen decision rule. Keep the paper diagnosis-focused and retain corrected negative/uncertain evidence transparently.' if decision['code']=='A' else
         '**11. Should a learned architecture enter the WWW main paper?** The corrected evidence supports further consideration under the frozen decision, but external confirmation is needed before promoting it as the paper’s solution. Do not rewrite the manuscript from this historical WANDS/BGE result alone.'),'',
        '## 15. Final decision, stopping and reproducibility','',
        f"**{decision['code']}. {decision['title']}.** This conclusion concerns contextual field weighting in this fixed WANDS/BGE setting. It does not concern relearning BGE embeddings, and it does not generalize to all invariant architectures.",'',
        ('Stop learned mitigation work under this definitive corrected WANDS/BGE experiment. No additional architecture is proposed or run.' if decision['code']=='A' else
         'Recommend a separately frozen external confirmation on ESCI and at least one additional retriever before any solution claim. That confirmation is not run here.'),'',
        'Required artifacts: PHASE6_REPORT.md, PHASE6_RESULTS.csv, PHASE6_BOOTSTRAP.csv, PHASE6_SEEDS.csv, PHASE6_TRAINING_CURVES.csv, PHASE6_INVARIANCE.json, PHASE6_FIELD_WEIGHTS.csv, PHASE6_DATA_AUDIT.json, PHASE6_CONFIG.json, FINAL_DELIVERY_MANIFEST.json. The manifest includes every retained checkpoint, negative run, log, source hash and actual-forward audit. verification.json records final integrity and metric checks.','',
        '```powershell','phase2/.venv/Scripts/python.exe -X utf8 -m pytest phase6_corrected/test_corrected.py -q -o cache_dir=phase6_corrected/.pytest_cache',
        'phase2/.venv/Scripts/python.exe -X utf8 phase6_corrected/audit_data.py','phase2/.venv/Scripts/python.exe -X utf8 phase6_corrected/train_corrected.py prepare',
        'if (!(Test-Path phase6_corrected/selection.json)) { phase2/.venv/Scripts/python.exe -X utf8 phase6_corrected/train_corrected.py train }',
        'if (!(Test-Path phase6_corrected/all_models_pretest_verified.json)) { phase2/.venv/Scripts/python.exe -X utf8 phase6_corrected/space_bounded_eval.py audit }',
        'if (!(Test-Path phase6_corrected/evaluation_complete.json)) { phase2/.venv/Scripts/python.exe -X utf8 phase6_corrected/space_bounded_eval.py evaluate }',
        'phase2/.venv/Scripts/python.exe -X utf8 phase6_corrected/analyze_corrected.py',
        'phase2/.venv/Scripts/python.exe -X utf8 phase6_corrected/report_corrected.py','phase2/.venv/Scripts/python.exe -X utf8 phase6_corrected/verify_corrected.py','```','',
        'Run these commands from the repository root. The existence guards preserve completed selection and pretest timestamps; do not rerun the unguarded audit after test. Training progress checkpoints support resumption at completed common epochs. Analysis/report regeneration must be followed by final verification to refresh the delivery manifest. No pilot archive is overwritten.',''
    ]
    text='\n'.join(report)
    for original,replacement in {'of3 seeds':'of 3 seeds','catalog,72/24/384':'catalog, 72/24/384','and20-epoch':'and 20-epoch',
        'weight1':'weight 1','weight.5':'weight .5','epoch has64':'epoch has 64','exactly64':'exactly 64','learned768-D':'learned 768-D',
        'a128-D':'a 128-D','one128-D':'one 128-D','UNK0':'UNK 0','Epoch0':'Epoch 0','epoch0':'epoch 0','epoch≥1':'epoch ≥ 1',
        'LR{':'LR {','decay.01':'decay .01','clip1':'clip 1','batch16':'batch 16','temperature.05':'temperature .05',
        'maximum20':'maximum 20','patience4':'patience 4','seeds42/43/44':'seeds 42/43/44','gains3/1/0':'gains 3/1/0',
        'uses1/log2':'uses 1/log2','use10,000':'use 10,000','seed20260963':'seed 20260963','percentile95%':'percentile 95%',
        'exactly6,797':'exactly 6,797','across239':'across 239','all42,994':'all 42,994','same128':'same 128','full42,994':'full 42,994',
        'Section9':'Section 9','least100':'least 100','top15':'top 15','catalog75th':'catalog 75th','one768-D':'one 768-D',
        'product:132,077,568':'product: 132,077,568','for42,994':'for 42,994','only24':'only 24'}.items():
        text=text.replace(original,replacement)
    (HERE/'PHASE6_REPORT.md').write_text(text,encoding='utf8')
    print(decision['code'],decision['title'])


if __name__=='__main__':main()
