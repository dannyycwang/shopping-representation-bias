# B. Query-conditioned attribution

The seed-2026092201 hash panel has four crossing, four always-in and four always-out all-seven-fitting pairs per MiniLM/BGE, with distinct queries within each stratum where possible. No quota is short. French molding is an added post-hoc case: 25 pairs / 50 variants. All candidates, selections and failures are retained, without replacement after inspecting results.

Variants are minimum/maximum native-score schedules, ties by original order; these are extrema, not random views. Main crossing BGE q235/product7253 is nearest the selected stratum's median rank span. Stable control q260/product41359 minimizes the frozen token/attribute-count distance. Figures use the first eight source occurrence IDs plus an exact other-content sum, not largest-change entries.

## Scalar, adapter and accounting

F_q(x)=dot(cached normalized query vector, normalized product vector). The query stays fixed. MiniLM uses attention-mask mean pooling, BGE CLS. IG varies word embeddings only at nonspecial nonpadding positions, preserving sequence length, positions, types, mask and special/padding embeddings. Fixed title/description fields remain included. Serializer spans and tokenizer offsets preserve duplicate occurrence IDs and assign all tokens to entries, fixed fields, separators, boundaries or specials.

Native retrieval uses FP16 model weights with FP32 pooling/normalization. The gradient adapter represents the same FP16-rounded frozen parameters in FP32, eval mode, TF32 disabled. It is a numerical diagnostic, not new authoritative retrieval. Actual runtime is torch 2.11.0+cu128, transformers 4.46.2 on RTX 4060 Laptop; inherited metadata sometimes report transformers 4.55.4. Fresh FP16 and FP32 scores are separately compared with saved scores; no bitwise runtime compatibility is assumed.

Maximum input_ids/inputs_embeds error: 0; maximum diagnostic/saved score error: 0.000163912773; changed diagnostic inclusions: 0/50. Token and entry sums agree to 1e-10. Signed sums, absolute token magnitudes and absolute feature magnitudes are separately exported.

## Retained main-case limitations

The crossing maximum-order margin is 9.74535942078e-05, versus diagnostic/native discrepancy 2.72989273071e-05. Both inclusion decisions remain unchanged, but margin >5×error fails. Its native extrema ranks 20 and 39 document a crossing; IG cannot reliably explain it.

The stable control extrema ranks 19 and 14 stay included. Its maximum-order primary zero-baseline IG fails completeness at 256 nodes. The figure explicitly labels these limitations and retains native margins. Neither a changed baseline nor another case is substituted to obtain a favorable explanation.

## Completeness and baseline sensitivity

Primary reference: zero nonspecial word vectors. Sensitivity reference: PAD vectors at the same positions, not a natural product description. Gauss–Legendre 64→128→256, stopping at |sumIG−(F(input)−F(baseline))| <= max(1e-4,.01|score difference|). Completeness is checked separately for each variant/reference. Baseline scores differ across orders, so attribution-sum differences alone need not equal input-score differences.

100/100 runs completed; 98 pass, 2 retained failures. Node counts: {"128": 11, "256": 8, "64": 81}.

| model | query_id | product_id | variant | baseline | status | steps | completeness_residual | completeness_tolerance |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bge_base | 260 | 41359 | maximum | zero | completeness_failed | 256 | -0.004633 | 0.003794 |
| bge_base | 26 | 5296 | minimum | zero | completeness_failed | 256 | 0.005303 | 0.001873 |

data/xai contains every attempt, score, residual, token and occurrence value. qa/xai_failures.json retains failures. Aligned maximum-minus-minimum entry changes across both references are exported. Descriptive baseline comparisons for displayed cases:

| model | query_id | product_id | display | zero_pad_change_pearson | change_sign_agreement | zero_change_l1 | pad_change_l1 | both_variant_boundary_guard | all_completeness_passed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bge_base | 235 | 7253 | main_crossing | 0.287667 | 0.607843 | 0.131172 | 0.101612 | False | True |
| bge_base | 260 | 41359 | main_stable_control | 0.279923 | 0.585366 | 0.104894 | 0.096821 | True | False |
| minilm | 359 | 12575 | supplement_historical | 0.603924 | 0.857143 | 0.197523 | 0.148706 | True | True |

Across the selected panel, median change-sign agreement is 0.585 (range 0.350–0.857); median change correlation is 0.290. These are exploratory selected-panel diagnostics, not population prevalence. Correlation cannot rescue failed guards or completeness.

For the historical French-molding case, both baselines preserve positive maximum-minus-minimum changes for width (zero +.021282, PAD +.015322), product type (+.008897/+.008801) and material (+.018395/+.014716), and a negative change for molding use (−.012439/−.013483). Color changes sign (−.024187/+.000349), showing a baseline-sensitive detail. These exact occurrence-level changes are in xai_aligned_entry_changes.parquet; they describe attribution redistribution without identifying a unique mechanism.

Figures use one common signed color scale, no independent per-order normalization. Full serialized texts, spans, displayed values and all seven native margins/ranks accompany the plots.

[Integrated Gradients](https://proceedings.mlr.press/v70/sundararajan17a.html) motivates the method; see also [Jain–Wallace](https://aclanthology.org/N19-1357/) and [Wiegreffe–Pinter](https://aclanthology.org/D19-1002/) on attention interpretation. No raw attention study was added. IG describes a scalar, path and baseline, not a unique causal mechanism, human attention, seller harm, behavior or fairness.
