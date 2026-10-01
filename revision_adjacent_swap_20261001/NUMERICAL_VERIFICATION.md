## Numerical validity and preservation

All competitors and query representations stay fixed. Ranks use exact FP32 scores and ascending saved catalog index for ties; the original target is removed before insertion. Margins are signed differences between native target and fixed Kth competitor scores, with boundary indices and exact-tie flags saved. Diagnostic subtraction represents the exact difference of the two saved FP32 scores in float64; membership uses the original native score comparison. No epsilon changes membership.

- MiniLM: 1,353 no-op pair replays, 1,320 distinct inputs; maximum C0 score discrepancy 0; 0 replay rank discrepancies. 386 variants crossing either cutoff received three C0 and three swapped forward checks each (shared identical inputs reused across query checks). Maximum crossing score errors: C0 0, swap 0.
- BGE: 2,088 no-op pair replays, 2,070 distinct inputs; maximum C0 score discrepancy 0; 0 replay rank discrepancies. 613 variants crossing either cutoff received three C0 and three swapped forward checks each (shared identical inputs reused across query checks). Maximum crossing score errors: C0 0, swap 0.

Every observed crossing reproduced at its cutoff in all three repeats. There are zero unresolved observed variant-cutoff events; confirmed and observed estimates coincide.

All 59,983 distinct product-setting variants passed entry-multiset (including duplicates), key/value, numeric-substring, fixed-text, adjacent-transposition, other-relative-order, identity and no-truncation checks. The repository representation audit was reused. Zero audit failures. C0 and every adjacent swap map to exactly the same ascending canonical serialization; this invariance is structural and is not claimed as a new empirical result.

The installed PyTorch/Transformers/tokenizers versions, CUDA/GPU, model snapshots, precision, pooling, padding/batching and serializer matched the earlier harmonized C0 reference. Payload hashes and fresh input aliases were verified. No mixed-profile historical score arrays were used. Queries missing from that reference were encoded under the same profile. Exact compatible swap-input aliases, especially the already enumerated ESCI orders, also reuse vectors; this does not mean that every tested serialization required a new forward pass. Full repeat records and numerical reports retain native discrepancies.

