# Harmonized expansion supplement

PASS: harmonized supplement completed separately; historical mixed-execution estimates remain historical.

The WANDS cohort is the existing 128 query IDs and all 7,125 highest-label pairs (6,444 unique products): the original selection keeps the first 128 eligible IDs sorted by SHA256(`coverage-20260922|decimal query ID`), breaking ties numerically. The exact frozen IDs and pairs are reused. ESCI uses the existing 499 queries and 4,434 E pairs. No cohort, schedule, seed or product was selected using the new outcomes. Each encoder uses its own single pinned profile for every query, source-order competitor, original-seven target and extra order. MiniLM and BGE profiles are in `harmonized/<model>/profile.json`; every alias and vector is addressable in the adjacent manifests. This is a follow-up, not a new preregistration.

| Dataset | Encoder | Support | Family | Queries/pairs | VI@20 % [95% CI] |
|---|---|---|---|---|---|
| wands | minilm | full | 7 | 128/7125 | 20.83387 [16.83321, 25.28457] |
| wands | minilm | full | 16 | 128/7125 | 24.49014 [20.37770, 28.95771] |
| wands | minilm | full | 32 | 128/7125 | 28.01552 [23.57637, 32.81200] |
| wands | minilm | common_all_fitting | 7 | 48/354 | 11.36494 [5.84681, 17.63088] |
| wands | minilm | common_all_fitting | 16 | 48/354 | 15.34855 [8.80862, 22.59730] |
| wands | minilm | common_all_fitting | 32 | 48/354 | 16.61320 [9.89936, 23.90516] |
| esci | minilm | full | 7 | 499/4434 | 2.90920 [2.14281, 3.80542] |
| esci | minilm | full | exhaustive_target | 499/4434 | 3.24533 [2.46056, 4.13838] |
| esci | minilm | common_all_fitting | 7 | 460/2578 | 3.20467 [2.25395, 4.25457] |
| esci | minilm | common_all_fitting | exhaustive_target | 460/2578 | 3.49159 [2.53699, 4.56375] |
| wands | bge_base | full | 7 | 128/7125 | 11.41882 [8.97683, 14.27055] |
| wands | bge_base | full | 16 | 128/7125 | 14.99520 [12.04135, 18.30271] |
| wands | bge_base | full | 32 | 128/7125 | 17.50862 [14.39166, 20.89833] |
| wands | bge_base | common_all_fitting | 7 | 92/2186 | 8.75675 [6.00534, 11.97053] |
| wands | bge_base | common_all_fitting | 16 | 92/2186 | 13.06772 [9.16295, 17.48496] |
| wands | bge_base | common_all_fitting | 32 | 92/2186 | 17.56292 [13.09338, 22.61046] |
| esci | bge_base | full | 7 | 499/4434 | 2.41795 [1.75959, 3.16701] |
| esci | bge_base | full | exhaustive_target | 499/4434 | 3.14909 [2.34566, 4.04870] |
| esci | bge_base | common_all_fitting | 7 | 488/3587 | 2.54691 [1.86780, 3.30743] |
| esci | bge_base | common_all_fitting | exhaustive_target | 488/3587 | 3.35114 [2.51957, 4.28817] |

The all-32-fitting WANDS supports are held fixed across 7/16/32. The ESCI fitting support fits all distinct target orders. `tables/harmonized_increments.csv` reports paired 16-minus-7, 32-minus-16 and 32-minus-7 increments, and exhaustive-minus-7 for ESCI, at K20 and K100. Each estimate has the same complete per-query support as its comparator. `data/harmonized_added_crossing_identities.parquet` retains the affected identities, with each contrast labelled separately. Nested VI monotonicity follows structurally; the amounts and identities are empirical. There is no saturation claim.

| Dataset | Encoder | Support | K | Harmonized7 minus historical7 VI, pp [95% CI] |
|---|---|---|---|---|
| wands | minilm | full | 20 | -0.014984 [-0.039927, 0.000000] |
| wands | minilm | full | 100 | -0.005017 [-0.021883, 0.008830] |
| wands | minilm | common_all_fitting | 20 | +0.000000 [0.000000, 0.000000] |
| wands | minilm | common_all_fitting | 100 | +0.000000 [0.000000, 0.000000] |
| esci | minilm | full | 20 | +0.000000 [0.000000, 0.000000] |
| esci | minilm | full | 100 | +0.000000 [0.000000, 0.000000] |
| esci | minilm | common_all_fitting | 20 | +0.000000 [0.000000, 0.000000] |
| esci | minilm | common_all_fitting | 100 | +0.000000 [0.000000, 0.000000] |
| wands | bge_base | full | 20 | +0.018752 [-0.086658, 0.137018] |
| wands | bge_base | full | 100 | -0.028270 [-0.096578, 0.030449] |
| wands | bge_base | common_all_fitting | 20 | +0.013147 [-0.099000, 0.153583] |
| wands | bge_base | common_all_fitting | 100 | +0.027809 [-0.010148, 0.078047] |
| esci | bge_base | full | 20 | -0.005693 [-0.054655, 0.037575] |
| esci | bge_base | full | 100 | +0.016700 [0.000000, 0.050100] |
| esci | bge_base | common_all_fitting | 20 | -0.005822 [-0.055887, 0.038422] |
| esci | bge_base | common_all_fitting | 100 | +0.017077 [0.000000, 0.051230] |

`tables/historical_to_harmonized.csv` also compares persistent inclusion and omission. Exact per-schedule rank and cutoff differences are in `harmonized/<model>/original7_rank_differences.csv`; they are not silently attributed to order. Focused repeated-input and changed-decision forward checks are in each model directory. No score tolerance changes ranking or membership.

| Dataset | Encoder | K | Changed decisions / pair-schedule records | Affected queries | Affected target pairs |
|---|---|---|---|---|---|
| wands | minilm | 20 | 6/49875 | 6 | 6 |
| wands | minilm | 100 | 16/49875 | 10 | 16 |
| esci | minilm | 20 | 1/31038 | 1 | 1 |
| esci | minilm | 100 | 0/31038 | 0 | 0 |
| wands | bge_base | 20 | 31/49875 | 22 | 31 |
| wands | bge_base | 100 | 43/49875 | 27 | 41 |
| esci | bge_base | 20 | 8/31038 | 4 | 4 |
| esci | bge_base | 100 | 3/31038 | 1 | 1 |

These counts compare historical and harmonized execution of the **same** seven orders. They are record counts, not query-macro rates; `tables/harmonized_execution_differences.csv` also includes exact-rank changes. Additional-order increments use harmonized arrays throughout.

minilm: 40 focused checks, 40 vectors with exactly equal numeric components, maximum component difference 0; 0 changed repeated cutoff decisions.
bge_base: 40 focused checks, 40 vectors with exactly equal numeric components, maximum component difference 0; 0 changed repeated cutoff decisions.

These are focused sampled repeat checks, not an assertion that every possible batch shape or hardware environment produces identical output. The tested repeated forwards keep the documented batch shape, padding bucket and numerical settings. The recorded `bitwise_identical` field is defined by `numpy.array_equal`: it checks exact numeric component equality, without a tolerance, but does not distinguish signed zeros. Byte integrity of saved vector payloads is checked separately by SHA256; no stronger byte-level repeat claim is inferred from that field name.

Scoring consistently uses FP32 elementwise products and a fixed-axis FP32 sum for every source and target, followed by descending score and ascending catalog index. A target's original C0 entry is removed. The source and repeated target input resolve to the same cached vector. New source competitors and queries were encoded under the same profile; no historical vector is mixed into the harmonized family. This is a separately versioned supplement, not a replacement of the primary historical seven-schedule results.

ESCI exhaustive means every distinct order of one target with all competitors fixed, not all joint catalog configurations. Its exhaustive state calculation uses the distinct E schedules; duplicate original-seven observations do not create extra distinct orders. This package reports inclusion states, not a synthetic coherent-catalog Recall for target-only counterfactuals.

The MiniLM process survived the reported computer closure and resumed after a long pause. Wall time and the affected block's elapsed time include the pause, so these timings are execution records, not a latency benchmark. Historical cost tables remain the cost evidence.
