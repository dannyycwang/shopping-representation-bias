# Section 6 audit

**PASS** for the recomputed saved evidence and accounting checks. **NEEDS_CORRECTION** for v11 presentation and expansion provenance wording. **UNRESOLVED**: final integrated manuscript pagination/references until the full matching LaTeX is available. PASS: harmonized supplement completed separately; historical mixed-execution estimates remain historical.

## Completed / reused / recomputed / unresolved

- Completed P0: support selection, stable catalog axes, complete-entry permutations, unchanged labels, complete-input fitting support and single-target replacement tie checks.
- Reused: historical seven-schedule ranks, all encoder/tokenizer pins, all three canonical rules, BM25, raw/canonical hybrids, Set-Mean, all centroid/multi-vector budgets and cost records; prior validation and numerical failure records retained.
- Recomputed: both-intervention full/fitting states at K20/K100; all 49,227 graded records; additional nDCG/cNDCG/Coverage@100 from saved judged ranks; 48,420 query/contrast/cutoff membership records; method effects with named paired comparators; six-cell transition checks.
- Completed presentation: compact vector Figure 3/4, main controls-table candidate, exactly one main transition table, hash-selected factual replacement case, full plotted CSVs and scripts.
- New forward passes: separately versioned harmonized expansion only. PASS: harmonized supplement completed separately; historical mixed-execution estimates remain historical.
- Unavailable: encoder/method cells listed in `tables/unavailable_method_cells.csv`; full matching integrated v11 LaTeX. These are not imputed.
- Deferred/out of scope: IG/attention repair, PI-FT reproduction, new architectures/data/user studies/agent evaluations/latency studies. Historical records remain available.

## Quantitative reconciliation

`section6_claim_ledger.csv` has 383 numerical assertion/cell/mark rows, including the actual 144 Table 1 cells and both copies of the 12-cell transition table. All checked manuscript numeric displays reproduce at their printed precision; four historical expanded-family assertions require a provenance-qualified replacement despite numerical agreement. Counts/ranges use their stated populations and are not adjusted to force agreement.

WANDS/BGE reversal, full precision: gained 2.357388986945%, lost 2.263461092622%, turnover 4.620850079566%, net Recall +0.093927894323 pp, 95% CI [-0.409261930858, 0.563251180737]. The displayed 2.357 + 2.263 versus 4.621 is ordinary rounding, not a failed identity.

Equal positive gains/losses occur in 67–78 of 308 WANDS/BGE queries per contrast (21.753247–25.324675%). Reverse specifically has 73/308 (23.701299% of all, 39.037433% of changed queries). Six contrasts remain separate; these counts cannot be summed into a union.

The illustrative query `anti fatigue mat` (ID 231) exchanges 5 Exact products in each direction. Both numerators are 20 over 307; exact Recall@20 is unchanged. Complete names/IDs/labels/ranks/orders are in `data/illustrative_replacement_case.csv`. The hash rule, eligible case list and selection time are disclosed; no all-fitting eligibility was imposed and no user-harm claim is made.

## Named common-support method contrasts

The factual replacement case uses catalog-wide Source versus Reverse rankings on the same 42,994-product catalog. It is separate from the single-target, fixed-competitor intervention in RQ1.

Raw hybrid minus raw dense seven-schedule means, pp [95% unadjusted paired CI]:

| Dataset / encoder | Recall@100 | VI@100 |
|---|---|---|
| WANDS / minilm | +8.935 [7.210, 10.819] | -10.814 [-12.935, -8.823] |
| WANDS / bge_base | +4.152 [2.551, 5.819] | -5.258 [-7.049, -3.557] |
| ESCI / minilm | +3.374 [2.180, 4.691] | -0.853 [-1.502, -0.304] |
| ESCI / bge_base | +0.569 [-0.205, 1.367] | +0.025 [-0.409, 0.572] |

WANDS/BGE ascending **canonical hybrid** minus **raw-hybrid seven-schedule mean** Recall@20 is -0.624 [-1.370, -0.084] pp. This names the comparator hidden by the abbreviated v11 sentence. Pure ascending dense versus raw source order is a different contrast, retained separately. Intervals spanning zero do not establish preservation.

All six transition cells, row margins and control inclusion/omission margins pass on identical query support. The WANDS/MiniLM ascending dense illustration includes both the 0.23% persistent-inclusion-to-omission cell and the 0.43% persistent-omission-to-inclusion cell. Their difference is not total Recall change, because crossing products contribute too. Crossing-to-omission is unconditional accounting mass, not a newly caused loss probability.

## Corrections and limits

Remove the duplicate Table 3 and p8 duplicate paragraph. Replace tDCG with the fully specified nDCG/cNDCG conventions. Clarify that fully-fitting support is a secondary target restriction, not the entire primary evaluation. Correct the canonical-rule cross-reference to the methods section. Remove attribution pointers when omitting the corresponding analysis from the proposed submission fragment; the old outputs/failures are untouched. Centroid construction uses multiple encodings but retains/scores one vector; multi-vector stores/scores M vectors. Historical timings are warm-cache CPU exact scoring, exclude query encoding, and are not ANN deployment latency.

The manuscript can support order sensitivity, membership information beyond averages, and empirically evaluated invariant controls. It does not establish a universal robustness/effectiveness trade-off, necessary canonicalization harm, architecture/length causation, saturation at 32, complete relevance judgments, or downstream user/agent harm.
