from common import *

def main():
    s=pd.read_csv(DATA/'canonical_membership_summary.csv');m=pd.read_csv(DATA/'canonical_mean_schedule_reference.csv')
    s=s[(s.population=='primary_existing_evaluation')&(s.aggregation=='query_macro')]
    m=m[(m.population=='primary_existing_evaluation')&(m.aggregation=='query_macro')]
    d=s.merge(m[['dataset','model','K','alternative','raw_mean_schedule_recall','delta_vs_mean_schedules','delta_vs_mean_schedules_ci_low','delta_vs_mean_schedules_ci_high']],
              on=['dataset','model','K','alternative'],validate='one_to_one')
    d.to_csv(DATA/'canonical_primary_comparison_table.csv',index=False)
    labels={'lexical_ascending':'Lexical ascending','lexical_descending':'Lexical descending','field_priority_type':'Type/style priority','historical_M2_template':'Historical M2 template'}
    lines=['# Canonical controls on primary query support','',
      'All three pure full-entry sorting rules were frozen before new retrieval and are reported without selecting a test-set winner. WANDS uses 308 eligible held-out queries / 21,299 Exact pairs; ESCI uses 499 eligible evaluation queries / 4,434 E pairs. Each gain/loss fraction uses all Hq products on that same support.',
      '',
      'Pure canonicalization maps all incoming attribute-entry permutations to one identical text, with duplicates, entry contents and fixed section placement preserved. With one fixed materialized index, VI=0 and mean-over-seven Recall equals C0 Recall for each canonical rule. Always equals the rule’s Recall and Never equals one minus that Recall. This property does not require effectiveness to decrease. Repeated independent floating-point encoder runs are a separate numerical issue.',
      '',
      'Values below are percentages; differences and interval endpoints are percentage points. Intervals are descriptive 95% paired query-cluster percentile intervals, with 10,000 draws and seed 2026091701. They are uncorrected and do not establish equivalence when they include zero.',
      '',
      'The four newly encoded BGE controls use a maximum execution batch size of 16, reduced from 48 for memory pressure under the recorded BGE execution addendum. The model, tokenizer cap, pooling, precision and sorting/evaluation design remain fixed; independent floating-point forwards are not claimed to be bitwise identical.',
      '',
      '| Dataset | Encoder | K | Pure rule | Recall | Delta vs C0 [95% CI] | Delta vs raw seven-order mean [95% CI] | Gain / loss |',
      '|---|---|---:|---|---:|---:|---:|---:|']
    for ds in ['wands','esci']:
      for model in ['minilm','bge_base','gte_modernbert']:
        for k in [20,100]:
          for rule in RULES:
            rows=d[(d.dataset==ds)&(d.model==model)&(d.K==k)&(d.alternative==rule)]
            if len(rows)==0:
                lines.append(f'| {ds.upper()} | {model} | {k} | {labels[rule]} | Pending | Pending | Pending | Pending |')
                continue
            r=rows.iloc[0]
            lines.append(f'| {ds.upper()} | {model} | {k} | {labels[rule]} | {100*r.alternative_recall:.3f} | {100*r.net_delta:+.3f} [{100*r.net_delta_ci_low:+.3f}, {100*r.net_delta_ci_high:+.3f}] | {100*r.delta_vs_mean_schedules:+.3f} [{100*r.delta_vs_mean_schedules_ci_low:+.3f}, {100*r.delta_vs_mean_schedules_ci_high:+.3f}] | {100*r.gain_fraction:.3f} / {100*r.loss_fraction:.3f} |')
    lines+=['','Historical M2 remains separate: it changes field framing and section placement. Its C0-relative outcomes are retained in the same machine-readable comparison file under `alternative=historical_M2_template`, not used as pure-order evidence. Exact saved canonical_raw transitions are reconciled against the inherited Phase IV gain/loss tables in `data/canonical_raw_transition_reconciliation.csv`.','',
      'The raw C0 and raw seven-order mean are different references. For instance, inherited WANDS/BGE lexical ascending at K=100 has 63.237% Recall: +0.066 pp versus raw C0 (63.171%), but -0.383 pp versus the raw seven-order mean (63.620%). Those contrasts must not be interchanged. Neither contrast alone establishes equivalence or a general effectiveness cost of invariance.']
    (HERE/'CANONICAL_CONTROLS.md').write_text('\n'.join(lines)+'\n',encoding='utf8')

if __name__=='__main__':main()
