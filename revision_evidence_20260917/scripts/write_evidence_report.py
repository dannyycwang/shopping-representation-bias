from common import *
from datetime import datetime, timezone

def main():
    allsummary=pd.read_csv(DATA/'membership_changes_summary.csv')
    s=allsummary[(allsummary.population=='primary_existing_evaluation')&(allsummary.aggregation=='query_macro')&(allsummary.reference=='C0')]
    pops=json.loads((DATA/'evaluation_populations.json').read_text(encoding='utf8'))
    table=[]
    for ds,model,k in s[['dataset','model','K']].drop_duplicates().itertuples(index=False):
        d=s[(s.dataset==ds)&(s.model==model)&(s.K==k)]
        table.append(dict(dataset=ds,model=model,K=k,change_min_pct=100*d.membership_change_fraction.min(),change_max_pct=100*d.membership_change_fraction.max(),
          delta_min_pp=100*d.net_delta.min(),delta_max_pp=100*d.net_delta.max(),
          exact_cancellation_min_pct=100*d.within_query_cancellation_fraction.min(),exact_cancellation_max_pct=100*d.within_query_cancellation_fraction.max()))
    pd.DataFrame(table).to_csv(DATA/'rq2_all_contrast_ranges.csv',index=False)
    available=pd.read_csv(DATA/'canonical_control_availability.csv') if (DATA/'canonical_control_availability.csv').exists() else None
    ncontrols=int(available.status.eq('available').sum()) if available is not None else 0
    canonical_claim='Canonical-control outcomes are reported in `CANONICAL_CONTROLS.md`; incomplete conditions are not imputed.'
    if ncontrols==18:
        c=pd.read_csv(DATA/'canonical_primary_comparison_table.csv')
        b=c[(c.dataset=='wands')&(c.model=='bge_base')&(c.K==100)&c.alternative.isin(RULES)]
        assert len(b)==3
        canonical_claim=(f'**Canonical effectiveness depends on the fixed rule and comparison reference; a uniform effectiveness loss is not supported.** For the pre-existing WANDS/BGE K=100 example, all three pure rules improve over raw C0 by {100*b.net_delta.min():.3f}–{100*b.net_delta.max():.3f} pp, yet their differences from the raw seven-order mean range from {100*b.delta_vs_mean_schedules.min():+.3f} to {100*b.delta_vs_mean_schedules.max():+.3f} pp. These are descriptive point estimates, not equivalence claims. `CANONICAL_CONTROLS.md` reports every rule, dataset, encoder and cutoff with paired intervals and matched gains/losses; no test-set winner is selected.')
    case=json.loads((DATA/'introduction_case_audit.json').read_text(encoding='utf8')) if (DATA/'introduction_case_audit.json').exists() else None
    statepath=DATA/'inclusion_states_summary.csv'
    line=['# Evidence status','',f'Updated {datetime.now(timezone.utc).isoformat()}. All paths below are relative to `{HERE}`. Original Chapters 1–2 and historical outputs were not edited.','',
      '| Status | Deliverable | Evidence / scope |','|---|---|---|',
      '| Verified | `data/membership_integrity.csv` | All 42 local raw rank files match the documented backup hashes. Seven schedules align exactly with the highest-label judgments. No backup restoration or raw reretrieval was required. |',
      '| Derived | `data/membership_changes_per_query.csv`, `data/membership_changes_summary.csv`, `data/membership_changes_sources.json` | Both K=20 and K=100, all six C0-relative contrasts and all 21 schedule pairs, full historical and primary query populations, macro and micro results kept separate. |',
      '| Derived | `data/rq2_cancellation_companion.csv`, `CANCELLATION_TABLE.md` | Exact integer gains = losses > 0, with all-query and changed-query denominators. Across-query cancellation is reported separately. |',
      '| Verified | `data/target_only_integrity.csv` (when present), `data/target_rank_reconstruction_audit.csv` | Saved target C0 and saved whole-catalog columns are checked against raw ranks. Independent old-entry-replacement tests include tied scores. |',
      f'| {"Verified" if case else "In progress"} | `data/all_seven_token_lengths.csv`, `data/introduction_case_audit.json` | Untruncated, encoder-specific counts including special tokens, using all seven complete serialized inputs. |',
      f'| {"Derived" if statepath.exists() else "Pending token audit"} | `data/inclusion_states_summary.csv`, `data/inclusion_states_per_query.csv` | Full and fully-fitting targets, raw C0 competitors, all three states under identical aggregation; full catalog retained. |',
      '| Created and visually checked | `figures/figure2_intervention_and_states.pdf`, `.svg`; `figures/rq2_membership_changes.pdf`, `.svg` | Schematic explicitly labeled; empirical figure retains all six C0-relative contrasts. The PDF has separate K=20 and K=100 pages. |',
      f'| {"Created" if (FIG/"rq1_target_fully_fitting.pdf").exists() else "Pending token audit"} | `figures/rq1_target_fully_fitting.pdf`, `.svg` | This six-cell, encoder-specific target support was not covered by the inherited BGE-only fitting visuals. |',
      f'| {"Complete" if ncontrols==18 else "In progress"} | `PROSPECTIVE_PROTOCOL.json`, `experiments/*_source.json`, `data/canonical_*` | {ncontrols}/18 dataset/encoder/rule conditions available. Ascending, descending and type/style-priority were frozen before new retrieval. All rules are retained; M2 stays separate. |',
      f'| {"Derived" if ncontrols==18 else "In progress"} | `CANONICAL_CONTROLS.md`, `data/canonical_primary_comparison_table.csv` | All three pure rules at both cutoffs, with matched gains/losses and paired intervals against raw C0 and the raw seven-order mean. Canonical VI, Always and Never are in `data/canonical_inclusion_states.csv`. |',
      '| Optional, not run | Nested 16/32 schedule expansion | Omitted to prioritize membership, fitting, and three canonical controls. No saturation or all-permutation guarantee is claimed. |','',
      '## Populations and definitions','',
      '| Dataset / population | Requested queries | Eligible queries | Highest-label pairs | Catalog |','|---|---:|---:|---:|---:|']
    for p in pops:
        if p['model']!='minilm':continue
        line.append(f"| {p['dataset'].upper()} / {p['population']} | {len(p['requested_query_ids'])} | {p['eligible_queries']} | {p['highest_relevance_pairs']:,} | {p['catalog_size']:,} |")
    line += ['',
      'WANDS primary uses the pre-existing 384-query held-out split (308 eligible queries). ESCI primary uses the pre-existing 500-query evaluation sample (499 eligible); it is not a newly held-out sample. ESCI historical-full and primary rows refer to the same IDs and must not be treated as independent evidence. `data/evaluation_populations.json` enumerates every requested and eligible query ID for each cell.',
      '',
      'Hq contains every distinct judged product with label Exact (WANDS) or E (ESCI). Counts include products absent in both lists. All fractions divide by |Hq|. Macro first averages within query, then over eligible queries; micro weights pairs equally. C0 Recall and mean-over-seven-schedules Recall are separate measures. Target-only mean inclusion averages separate counterfactual indexes and is not Recall from one shared index. Always + VI + Never = 1 on the same support.',
      '',
      'All intervals use 10,000 paired query-cluster draws, seed 2026091701, and 95% percentile endpoints. All products and schedule comparisons for a sampled query stay together. Intervals are descriptive and uncorrected; intervals containing zero do not establish equivalence.',
      '',
      'The baseline audit verifies that Phase VI WANDS/BGE Original Recall@100 is 63.620% averaged over seven schedules, whereas its C0 Recall@100 is 63.171%. At K=20 they are 36.038% and 36.105%, respectively. See `data/baseline_metric_reconciliation.csv`; these names are never interchanged.',
      '',
      'The raw schedule family is C0, C1, C2s1–C2s5, with product-ID-dependent random seeds 20260911–20260915. Mean inclusion over this family is not a uniform permutation probability. Exact model revisions, native pooling, tokenizer caps (256/512/8192), empty prefixes, tie rules, source hashes and embedding provenance are recorded in the JSON sources and frozen protocol.',
      '',
      '## Claims supported by the membership results','',
      'Membership changes occur in every dataset/encoder/K cell. Their magnitude can exceed the net Recall change because gains and losses cancel both within and across queries. The six-contrast ranges below preserve every scheduled comparison; they are descriptive summaries, not selected favorable permutations.','',
      '| Dataset | Encoder | K | Changed membership (%) | Net Recall change (pp) | Exact zero-net, changed queries (%) |',
      '|---|---|---:|---:|---:|---:|']
    for r in table:
        line.append(f"| {r['dataset'].upper()} | {r['model']} | {r['K']} | {r['change_min_pct']:.2f} to {r['change_max_pct']:.2f} | {r['delta_min_pp']:+.2f} to {r['delta_max_pp']:+.2f} | {r['exact_cancellation_min_pct']:.2f} to {r['exact_cancellation_max_pct']:.2f} |")
    line += ['',
      '**Narrow the introduction’s aggregate-stability claim.** The data do not establish stable aggregate effectiveness across all six contrasts or all cells. For example, WANDS/MiniLM at K=20 has net Recall changes from -2.60 to -0.95 pp alongside 8.23–10.52% changed relevant membership. A defensible statement is: “Net Recall changes can understate changes in relevant-product membership; within-query substitutions can leave Recall exactly unchanged.” Do not describe the magnitude as statistically significant without a stated multiple-comparison procedure.',
      '',
      canonical_claim,
      '',
      '## Target scope and numerical audit','',
      'Saved C0 target ranks exactly match saved C0 complete-catalog ranks. Current-runtime fp32 reconstruction differs for one WANDS/BGE/C2s3 pair: query 252, product 385, saved rank 479 versus reconstructed 480. Neither K=20 nor K=100 inclusion changes. All inherited ranks remain authoritative. Separately, Phase IV WANDS/BGE Original differs from Phase II for one Irrelevant pair at ranks 350/351; all highest-label ranks and both cutoffs agree. See `data/target_rank_precision_differences.csv` and reused-control source JSONs.',
      '',
      'Phase VI method-specific target sensitivity fixes competitors at that method’s own C0. This package’s raw target intervention fixes competitors at raw C0. Those conditions are not pooled. The old BGE 512-token fitting mask is not applied to MiniLM or GTE. Full-minus-fitting differences are descriptive population differences, not causal truncation effects.']
    if case:
        line+=['',f"The introduction example is verified: C2s4 rank 5 and C2s1 rank 1,527. Its seven MiniLM token counts are {', '.join(s+'='+str(case['token_counts_including_special'][s]) for s in STEMS)} (cap 256). **It {'fits' if case['fully_fits_all_orders'] else 'does not fit'} completely under all seven orders.** It remains a valid target-only illustration, with truncation explicitly disclosed."]
    if statepath.exists():
        line += ['',
          'Fully-fitting target-only crossing remains nonzero in every cell. At K=20 on held-out WANDS, query-macro VI is 10.59% for MiniLM (118 queries, 925 pairs), 7.16% for BGE (239 queries, 6,797 pairs), and 14.55% for GTE (308 queries, 21,299 pairs). All GTE inputs in both catalogs fit its 8,192-token limit, so its full and fitting populations coincide. A separately labeled, post-hoc fully-fitting illustrative case is provided in `data/posthoc_fully_fitting_illustrative_case.json`; it does not replace the population analysis or the Introduction example.']
    line+=['','## Preservation and limits','',
      'The transformations preserve recorded product content, entry multiplicity, fixed text and source relevance judgments. This is source preservation, not independent verification of real-world facts. Finite-family empirical stability is not a guarantee over all permutations. Pure deterministic canonicalization maps all input attribute permutations to one text by construction; effectiveness is evaluated separately for every rule. No new architecture or training phase is introduced.',
      '',
      'New retrieval outputs are isolated in `experiments/`. Raw RQ2 membership and target-state results are analyses of inherited ranks. Four existing pure ascending controls are reused; missing controls are newly encoded and ranked with frozen pretrained encoders and historical query embeddings. Software, device, precision, batching implementation and exact source hashes accompany each new condition.',
      '',
      '`data/encoder_profiles_and_rank_sources.csv` identifies raw MiniLM/BGE caches as `phase2-encoder-v1` and raw GTE as `phase2-encoder-v3-length-bucket-sdpa`. New encodings use the existing v3 implementation documented before retrieval, preserving the model revision, native pooling, tokenizer cap and fp16-model/fp32-pooling precision. Historical query vectors are reused. These execution details are disclosed rather than claiming bitwise identity of independent encoder runs.',
      '',
      '`BGE_EXECUTION_ADDENDUM.json` records a later execution adjustment before any of the four new BGE conditions completed: maximum batch size was reduced from 48 to 16 after memory pressure and a Windows sleep interruption. The model, tokenizer cap, pooling, precision, canonical rules and evaluation support were unchanged. Batch partitioning can alter floating-point rounding. See `experiments/EXECUTION_NOTES.md`; interrupted partial attempts contribute no ranks.',
      '',
      'Reproduction commands are in `REPRODUCE.md`; figure captions and accessible descriptions are in `CAPTIONS_AND_ACCESSIBILITY.md`. '+('All mandatory inputs and all 18 frozen canonical conditions are available: four inherited controls and 14 newly run conditions. No mandatory computation remains. The optional 16/32-family expansion was not run.' if ncontrols==18 else 'Remaining conditions are enumerated in `data/canonical_control_availability.csv`; the minimal remaining computation is encoding that rule’s full catalog and exact ranking against the saved query embeddings. No missing values are imputed.')]
    (HERE/'EVIDENCE_STATUS.md').write_text('\n'.join(line)+'\n',encoding='utf8')
    rq2=['# Suggested RQ2 results text','',
      'We compare each of the six non-source attribute-order schedules with C0 at K=20 and K=100, keeping product contents, relevance judgments, the encoder and catalog fixed. The primary WANDS comparison contains 308 eligible held-out queries and 21,299 Exact pairs; ESCI contains 499 eligible queries and 4,434 E pairs from the existing evaluation sample. For each query, all highest-relevance products are classified as retained, newly included, newly omitted or absent in both. We average fractions within query before macro aggregation.',
      '',
      'Net Recall change does not describe the full change in candidate membership. Across the six C0-relative schedules, WANDS at K=20 has 8.23–10.52% changed highest-relevance membership for MiniLM, 4.31–4.77% for BGE and 5.75–6.36% for GTE, while the corresponding net Recall changes range from -2.60 to -0.95, -0.46 to +0.14 and -0.92 to +0.14 percentage points. On ESCI the membership changes are smaller, but remain nonzero in every encoder/cutoff cell. Figure [rq2_membership_changes] reports every C0-relative contrast at both cutoffs; the supplement retains all 21 schedule pairs.',
      '',
      'Cancellation also occurs within individual queries. At K=20, the six WANDS contrasts have exactly equal nonzero gains and losses for 52–67 of 308 queries with MiniLM, 67–78 with BGE and 61–81 with GTE. These substitutions leave each affected query’s Recall exactly unchanged. The companion table reports each numerator and both denominators: all eligible queries and queries with any membership changes. Across-query cancellation is separately summarized using positive and negative query-level net changes. Thus changes in aggregate Recall can understate changes in which relevant products are retrieved. This does not establish equivalence of aggregate effectiveness across permutations.',
      '',
      'All intervals are descriptive 95% percentile intervals from 10,000 paired bootstrap draws of queries (seed 2026091701), retaining all products and schedule contrasts within a query. They are not adjusted for multiple comparisons. These membership results use complete-catalog pure attribute-entry permutations; rewriting and historical M2 template outcomes are not evidence for pure order effects.']
    (HERE/'RQ2_MANUSCRIPT_TEXT.md').write_text('\n'.join(rq2)+'\n',encoding='utf8')

if __name__=='__main__':main()
