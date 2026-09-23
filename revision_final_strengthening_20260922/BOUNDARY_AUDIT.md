# A. Score boundary audit

All competing products remain at C0. The old target is removed before replacement, preserving catalog size 42,994 or 10,076. t_K is the Kth competitor excluding the target. A higher score wins; equality uses the smaller original catalog index. Exact saved FP32 comparisons are used with no epsilon; margin sign alone is not the tie rule.

C0 scores come from the complete cached normalized query/product matrix; alternative target scores are saved Phase II pair scores. Authoritative ranks are Phase III target-only ranks, never catalog-wide ranks. Separate replacements do not form one coherent query ranking. qa/boundary_algorithm_tests.json verifies the inherited algorithm against brute-force tied-score replacements.

| dataset | model | pairs | records | rank_mismatches | boundary_mismatches | exact_threshold_ties | elapsed_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| wands | minilm | 21299 | 298186 | 0 | 0 | 0 | 3.379201 |
| wands | bge_base | 21299 | 298186 | 1 | 0 | 0 | 7.162619 |
| wands | gte_modernbert | 21299 | 298186 | 0 | 0 | 0 | 6.511069 |
| esci | minilm | 4434 | 62076 | 0 | 0 | 0 | 1.277541 |
| esci | bge_base | 4434 | 62076 | 0 | 0 | 0 | 1.515991 |
| esci | gte_modernbert | 4434 | 62076 | 0 | 0 | 0 | 1.682241 |

## Precision reconciliation

The sole discrepancy, WANDS/BGE q252/product 385/C2s3, is saved rank 479 versus reconstructed 480, exactly the inherited target_rank_precision_differences.csv record. Target score .7454689145088196; competitor 30099 score .7454689741134644: one FP32 ULP (5.960464477539063e-8) higher under current matrix multiplication. Neighboring IDs/scores appear in qa/boundary_discrepancy_neighbors.csv. There is no exact tie or Top-20/100 change. The old computation's precise rounding source is not retroactively assigned; saved rank 479 remains authoritative. No tolerance is widened. All unresolved-boundary flags are false.

## Distributions

The full tables/boundary_distributions.csv covers K=20/100, three states, full/all-seven-fitting supports, score ranges, maximum C0 deviation, C0 margins, cosine distances, best/worst ranks and included-schedule counts. Query-macro weights each nonempty query equally and divides its mass among its pairs; conditioning on state then renormalizes those weights. It is not the equal-query average over only queries with that state. Pair-micro weights pairs equally. Cluster intervals keep queries whole.

Top-20 state masses (0–1) and conditional score-range summaries:

| dataset | model | support | queries | support_pairs | state | state_pairs | state_mass | state_mass_ci_low | state_mass_ci_high | conditional_mean | median | p95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| esci | bge_base | full | 499 | 4434 | persistent_inclusion | 2967 | 0.731922 | 0.704150 | 0.759287 | 0.005914 | 0.004409 | 0.017837 |
| esci | bge_base | full | 499 | 4434 | crossing | 140 | 0.024236 | 0.017583 | 0.031787 | 0.011805 | 0.009376 | 0.022050 |
| esci | bge_base | full | 499 | 4434 | persistent_omission | 1327 | 0.243841 | 0.217849 | 0.270150 | 0.005606 | 0.004458 | 0.015047 |
| esci | bge_base | all_seven_fitting | 488 | 3587 | persistent_inclusion | 2407 | 0.725897 | 0.696122 | 0.753329 | 0.006290 | 0.004924 | 0.017838 |
| esci | bge_base | all_seven_fitting | 488 | 3587 | crossing | 117 | 0.025527 | 0.018685 | 0.033156 | 0.012564 | 0.009376 | 0.026895 |
| esci | bge_base | all_seven_fitting | 488 | 3587 | persistent_omission | 1063 | 0.248576 | 0.222208 | 0.277540 | 0.006236 | 0.005287 | 0.015924 |
| esci | gte_modernbert | full | 499 | 4434 | persistent_inclusion | 2760 | 0.681296 | 0.651310 | 0.710050 | 0.009395 | 0.008185 | 0.025010 |
| esci | gte_modernbert | full | 499 | 4434 | crossing | 296 | 0.058833 | 0.048261 | 0.070191 | 0.016667 | 0.013807 | 0.034244 |
| esci | gte_modernbert | full | 499 | 4434 | persistent_omission | 1378 | 0.259870 | 0.232806 | 0.287483 | 0.011029 | 0.009569 | 0.029198 |
| esci | gte_modernbert | all_seven_fitting | 499 | 4434 | persistent_inclusion | 2760 | 0.681296 | 0.651310 | 0.710050 | 0.009395 | 0.008185 | 0.025010 |
| esci | gte_modernbert | all_seven_fitting | 499 | 4434 | crossing | 296 | 0.058833 | 0.048261 | 0.070191 | 0.016667 | 0.013807 | 0.034244 |
| esci | gte_modernbert | all_seven_fitting | 499 | 4434 | persistent_omission | 1378 | 0.259870 | 0.232806 | 0.287483 | 0.011029 | 0.009569 | 0.029198 |
| esci | minilm | full | 499 | 4434 | persistent_inclusion | 2719 | 0.661616 | 0.631472 | 0.690676 | 0.009718 | 0.006821 | 0.033472 |
| esci | minilm | full | 499 | 4434 | crossing | 138 | 0.029092 | 0.021428 | 0.038054 | 0.025358 | 0.019593 | 0.060047 |
| esci | minilm | full | 499 | 4434 | persistent_omission | 1577 | 0.309292 | 0.280991 | 0.338489 | 0.009185 | 0.005993 | 0.029707 |
| esci | minilm | all_seven_fitting | 460 | 2578 | persistent_inclusion | 1594 | 0.654982 | 0.622097 | 0.687536 | 0.011709 | 0.009242 | 0.035060 |
| esci | minilm | all_seven_fitting | 460 | 2578 | crossing | 93 | 0.032047 | 0.022540 | 0.042546 | 0.025519 | 0.018846 | 0.060047 |
| esci | minilm | all_seven_fitting | 460 | 2578 | persistent_omission | 891 | 0.312971 | 0.281288 | 0.345669 | 0.011887 | 0.010001 | 0.032632 |
| wands | bge_base | full | 308 | 21299 | persistent_inclusion | 1749 | 0.301765 | 0.263119 | 0.342270 | 0.015537 | 0.012765 | 0.032793 |
| wands | bge_base | full | 308 | 21299 | crossing | 1815 | 0.108317 | 0.092912 | 0.125476 | 0.019783 | 0.017850 | 0.038135 |
| wands | bge_base | full | 308 | 21299 | persistent_omission | 17735 | 0.589918 | 0.550201 | 0.628442 | 0.017008 | 0.014136 | 0.039279 |
| wands | bge_base | all_seven_fitting | 239 | 6797 | persistent_inclusion | 824 | 0.213326 | 0.177176 | 0.252597 | 0.010803 | 0.009692 | 0.021195 |
| wands | bge_base | all_seven_fitting | 239 | 6797 | crossing | 569 | 0.071604 | 0.057710 | 0.087316 | 0.016242 | 0.013907 | 0.032733 |
| wands | bge_base | all_seven_fitting | 239 | 6797 | persistent_omission | 5404 | 0.715070 | 0.673784 | 0.753490 | 0.014920 | 0.012488 | 0.034377 |
| wands | gte_modernbert | full | 308 | 21299 | persistent_inclusion | 1612 | 0.293814 | 0.254609 | 0.334653 | 0.020388 | 0.018463 | 0.037220 |
| wands | gte_modernbert | full | 308 | 21299 | crossing | 2345 | 0.145506 | 0.128443 | 0.163156 | 0.025699 | 0.023464 | 0.044651 |
| wands | gte_modernbert | full | 308 | 21299 | persistent_omission | 17342 | 0.560680 | 0.521802 | 0.598609 | 0.022502 | 0.020985 | 0.040135 |
| wands | gte_modernbert | all_seven_fitting | 308 | 21299 | persistent_inclusion | 1612 | 0.293814 | 0.254609 | 0.334653 | 0.020388 | 0.018463 | 0.037220 |
| wands | gte_modernbert | all_seven_fitting | 308 | 21299 | crossing | 2345 | 0.145506 | 0.128443 | 0.163156 | 0.025699 | 0.023464 | 0.044651 |
| wands | gte_modernbert | all_seven_fitting | 308 | 21299 | persistent_omission | 17342 | 0.560680 | 0.521802 | 0.598609 | 0.022502 | 0.020985 | 0.040135 |
| wands | minilm | full | 308 | 21299 | persistent_inclusion | 1107 | 0.201876 | 0.168347 | 0.237191 | 0.050117 | 0.046383 | 0.090520 |
| wands | minilm | full | 308 | 21299 | crossing | 2655 | 0.194465 | 0.171177 | 0.219435 | 0.074722 | 0.066250 | 0.144264 |
| wands | minilm | full | 308 | 21299 | persistent_omission | 17537 | 0.603659 | 0.565317 | 0.641406 | 0.054201 | 0.050686 | 0.110155 |
| wands | minilm | all_seven_fitting | 118 | 925 | persistent_inclusion | 142 | 0.185002 | 0.132266 | 0.241784 | 0.027505 | 0.026595 | 0.049313 |
| wands | minilm | all_seven_fitting | 118 | 925 | crossing | 96 | 0.105920 | 0.070150 | 0.145090 | 0.042991 | 0.041721 | 0.074387 |
| wands | minilm | all_seven_fitting | 118 | 925 | persistent_omission | 687 | 0.709078 | 0.645069 | 0.769225 | 0.036038 | 0.034164 | 0.069874 |

Saved compatible embeddings supply maximum cosine distance from C0. The margin/rank relationship is an algebraic validation identity; the distributions are empirical information. Complete records include catalog/threshold indices and IDs, scores, margins, saved/reconstructed ranks, tokens, fitting flags and precision status.

## Historical illustrations

data/historical_case_boundary.csv verifies MiniLM q359 “french molding” / product12575: ranks 15–174, source rank 19, 139 tokens including specials, all seven fit under 256. GTE q409 “teal chair” / product24318: ranks 11–939, source rank 11, 1,059 tokens under 8,192. These are explicitly post-hoc illustrations, not prevalence estimates. The author's Introduction figure is untouched.
