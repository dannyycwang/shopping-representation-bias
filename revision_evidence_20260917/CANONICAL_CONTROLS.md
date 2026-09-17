# Canonical controls on primary query support

All three pure full-entry sorting rules were frozen before new retrieval and are reported without selecting a test-set winner. WANDS uses 308 eligible held-out queries / 21,299 Exact pairs; ESCI uses 499 eligible evaluation queries / 4,434 E pairs. Each gain/loss fraction uses all Hq products on that same support.

Pure canonicalization maps all incoming attribute-entry permutations to one identical text, with duplicates, entry contents and fixed section placement preserved. With one fixed materialized index, VI=0 and mean-over-seven Recall equals C0 Recall for each canonical rule. Always equals the rule’s Recall and Never equals one minus that Recall. This property does not require effectiveness to decrease. Repeated independent floating-point encoder runs are a separate numerical issue.

Values below are percentages; differences and interval endpoints are percentage points. Intervals are descriptive 95% paired query-cluster percentile intervals, with 10,000 draws and seed 2026091701. They are uncorrected and do not establish equivalence when they include zero.

The four newly encoded BGE controls use a maximum execution batch size of 16, reduced from 48 for memory pressure under the recorded BGE execution addendum. The model, tokenizer cap, pooling, precision and sorting/evaluation design remain fixed; independent floating-point forwards are not claimed to be bitwise identical.

| Dataset | Encoder | K | Pure rule | Recall | Delta vs C0 [95% CI] | Delta vs raw seven-order mean [95% CI] | Gain / loss |
|---|---|---:|---|---:|---:|---:|---:|
| WANDS | minilm | 20 | Lexical ascending | 28.723 | -2.086 [-4.027, -0.221] | -0.747 [-2.046, +0.476] | 3.719 / 5.805 |
| WANDS | minilm | 20 | Lexical descending | 29.556 | -1.252 [-2.692, +0.060] | +0.086 [-1.125, +1.301] | 3.447 / 4.699 |
| WANDS | minilm | 20 | Type/style priority | 29.221 | -1.588 [-3.492, +0.221] | -0.249 [-1.679, +1.089] | 3.815 / 5.402 |
| WANDS | minilm | 100 | Lexical ascending | 56.049 | -1.834 [-3.168, -0.643] | -0.564 [-1.516, +0.312] | 3.518 / 5.353 |
| WANDS | minilm | 100 | Lexical descending | 56.162 | -1.722 [-2.972, -0.585] | -0.451 [-1.364, +0.426] | 3.932 / 5.653 |
| WANDS | minilm | 100 | Type/style priority | 57.187 | -0.697 [-2.124, +0.677] | +0.574 [-0.592, +1.768] | 4.245 / 4.941 |
| WANDS | bge_base | 20 | Lexical ascending | 36.072 | -0.032 [-0.530, +0.449] | +0.034 [-0.456, +0.528] | 2.097 / 2.129 |
| WANDS | bge_base | 20 | Lexical descending | 36.276 | +0.171 [-0.724, +1.165] | +0.238 [-0.499, +1.087] | 2.429 / 2.258 |
| WANDS | bge_base | 20 | Type/style priority | 36.483 | +0.379 [-0.084, +0.887] | +0.445 [+0.003, +0.932] | 2.367 / 1.989 |
| WANDS | bge_base | 100 | Lexical ascending | 63.237 | +0.066 [-1.474, +1.646] | -0.383 [-1.396, +0.559] | 3.199 / 3.133 |
| WANDS | bge_base | 100 | Lexical descending | 63.609 | +0.438 [-0.395, +1.472] | -0.011 [-0.609, +0.577] | 2.627 / 2.189 |
| WANDS | bge_base | 100 | Type/style priority | 63.565 | +0.394 [-0.968, +1.861] | -0.055 [-1.004, +0.880] | 3.256 / 2.862 |
| WANDS | gte_modernbert | 20 | Lexical ascending | 35.745 | -0.156 [-1.079, +0.869] | +0.142 [-0.469, +0.836] | 3.181 / 3.337 |
| WANDS | gte_modernbert | 20 | Lexical descending | 35.002 | -0.899 [-1.696, -0.150] | -0.601 [-1.198, -0.017] | 2.632 / 3.531 |
| WANDS | gte_modernbert | 20 | Type/style priority | 35.982 | +0.081 [-0.564, +0.678] | +0.380 [-0.178, +0.923] | 2.841 / 2.760 |
| WANDS | gte_modernbert | 100 | Lexical ascending | 62.630 | -0.218 [-1.269, +0.764] | +0.064 [-0.440, +0.606] | 2.832 / 3.050 |
| WANDS | gte_modernbert | 100 | Lexical descending | 61.891 | -0.958 [-2.286, +0.226] | -0.675 [-1.671, +0.156] | 2.819 / 3.776 |
| WANDS | gte_modernbert | 100 | Type/style priority | 62.633 | -0.216 [-1.196, +0.627] | +0.066 [-0.441, +0.560] | 2.711 / 2.927 |
| ESCI | minilm | 20 | Lexical ascending | 67.497 | -0.456 [-0.977, +0.049] | -0.080 [-0.491, +0.344] | 0.523 / 0.978 |
| ESCI | minilm | 20 | Lexical descending | 67.521 | -0.431 [-1.134, +0.296] | -0.055 [-0.420, +0.316] | 0.859 / 1.291 |
| ESCI | minilm | 20 | Type/style priority | 67.480 | -0.472 [-0.997, +0.032] | -0.096 [-0.512, +0.325] | 0.529 / 1.001 |
| ESCI | minilm | 100 | Lexical ascending | 86.412 | +0.276 [-0.045, +0.757] | +0.219 [+0.003, +0.468] | 0.384 / 0.109 |
| ESCI | minilm | 100 | Lexical descending | 86.381 | +0.246 [-0.223, +0.812] | +0.189 [-0.043, +0.455] | 0.573 / 0.328 |
| ESCI | minilm | 100 | Type/style priority | 86.357 | +0.221 [-0.092, +0.703] | +0.164 [-0.046, +0.404] | 0.361 / 0.140 |
| ESCI | bge_base | 20 | Lexical ascending | 73.842 | -0.263 [-0.654, +0.095] | +0.011 [-0.299, +0.326] | 0.536 / 0.799 |
| ESCI | bge_base | 20 | Lexical descending | 73.596 | -0.508 [-0.954, -0.117] | -0.234 [-0.586, +0.091] | 0.541 / 1.050 |
| ESCI | bge_base | 20 | Type/style priority | 73.819 | -0.286 [-0.676, +0.076] | -0.011 [-0.318, +0.301] | 0.534 / 0.819 |
| ESCI | bge_base | 100 | Lexical ascending | 91.045 | +0.068 [-0.039, +0.177] | +0.004 [-0.101, +0.114] | 0.177 / 0.109 |
| ESCI | bge_base | 100 | Lexical descending | 91.013 | +0.035 [-0.126, +0.190] | -0.028 [-0.130, +0.077] | 0.181 / 0.146 |
| ESCI | bge_base | 100 | Type/style priority | 91.086 | +0.108 [-0.012, +0.231] | +0.044 [-0.055, +0.152] | 0.217 / 0.109 |
| ESCI | gte_modernbert | 20 | Lexical ascending | 71.608 | -0.842 [-1.703, -0.031] | -0.207 [-0.790, +0.377] | 1.117 / 1.959 |
| ESCI | gte_modernbert | 20 | Lexical descending | 71.645 | -0.805 [-1.602, -0.062] | -0.169 [-0.745, +0.370] | 1.062 / 1.867 |
| ESCI | gte_modernbert | 20 | Type/style priority | 71.681 | -0.768 [-1.622, +0.039] | -0.133 [-0.711, +0.443] | 1.191 / 1.959 |
| ESCI | gte_modernbert | 100 | Lexical ascending | 88.804 | -0.321 [-0.625, -0.063] | +0.031 [-0.339, +0.444] | 0.160 / 0.481 |
| ESCI | gte_modernbert | 100 | Lexical descending | 88.709 | -0.416 [-1.179, +0.272] | -0.064 [-0.469, +0.335] | 0.412 / 0.828 |
| ESCI | gte_modernbert | 100 | Type/style priority | 88.830 | -0.295 [-0.605, -0.035] | +0.056 [-0.313, +0.466] | 0.185 / 0.481 |

Historical M2 remains separate: it changes field framing and section placement. Its C0-relative outcomes are retained in the same machine-readable comparison file under `alternative=historical_M2_template`, not used as pure-order evidence. Exact saved canonical_raw transitions are reconciled against the inherited Phase IV gain/loss tables in `data/canonical_raw_transition_reconciliation.csv`.

The raw C0 and raw seven-order mean are different references. For instance, inherited WANDS/BGE lexical ascending at K=100 has 63.237% Recall: +0.066 pp versus raw C0 (63.171%), but -0.383 pp versus the raw seven-order mean (63.620%). Those contrasts must not be interchanged. Neither contrast alone establishes equivalence or a general effectiveness cost of invariance.
