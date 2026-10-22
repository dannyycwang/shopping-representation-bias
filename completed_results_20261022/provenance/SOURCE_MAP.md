# Sources for the recent-model follow-up

The attached 12-page manuscript (version 49) was read in full. Pages 4 and 11 were also rendered to verify populations, formulas, seeds, and gain definitions. Its SHA is recorded separately in `audit/attachments.json`. Repository HEAD at entry: `f42a4435f1a9e39a4dc0e73333e4c2749692cfa0`. Existing modified manuscript, figure, and plotting files are preserved by hash. No applicable AGENTS.md was found in the repository or ancestor directories inspected.

| Decision | Actual authoritative function / payload | Reconciliation |
|---|---|---|
| Product records and null handling | `phase2/scripts/prepare_data.py::clean,wands,esci`; saved `phase2/data/processed/{dataset}_products.jsonl.gz` | Reuse the processed payload, never invent another null policy. WANDS keeps every full pipe-separated occurrence. ESCI keeps nonempty brand, color, and the complete bullet-point field as at most three entries. |
| Serializer and fixed sections | `phase2/src/representations.py::_blocks,_plain,build` | Loaded directly, without edits. Every one of 371,490 raw schedule/product strings is checked against the historical representation files. |
| Per-product shuffles | `phase2/src/representations.py::_random_order`; `phase2/config/phase2.json::attribute_permutation_seeds` | Seeds 20260911--20260915. SHA256(seed:product_id), first eight bytes as big-endian shuffle seed. No replacement permutations for duplicate strings. |
| Query split | `phase4/config/splits.json::wands_dev,wands_heldout,esci_old`; `phase4/scripts/freeze.py` | WANDS 96/384 query split. ESCI keeps the existing 500 IDs, never resamples. |
| Eligible highest-label support | `revision_graded_20260922/PROTOCOL.json::populations`; `revision_graded_20260922/data/*_highest_support.csv` | Independently derive from processed labels and split and compare exact ID pairs. WANDS 308/21,299; ESCI 499/4,434. Full catalogs 42,994/10,076. |
| Catalog index | `revision_graded_20260922/data/*_catalog_ordering.csv` and processed product record order | Exact equality checked. Ties use original catalog index, not an alternative lexical sort. |
| Corrected graded gain | `revision_graded_20260922/scripts/common.py::GAINS`; `scripts/metrics.py::query_metrics`; `revision_section6_20260924/scripts/reports.py` | Direct WANDS 3/1/0 and ESCI E/S/C/I 1/.1/.01/0. The Phase II ESCI S/C values were swapped and are NOT reused. |
| Standard / condensed nDCG | `revision_graded_20260922/scripts/metrics.py::query_metrics`; `revision_section6_20260924/scripts/analyze_saved.py` | Absolute full-catalog ranks for standard; only judged ranks for condensed. Same IDCG from all judged gains. The new implementation extends both to K=100 and is tested against the corrected historical function at K=20. |
| Target rank insertion | `phase3/scripts/analyze_target_only_permutations.py::rank_target` | Compile the original function AST without historical top-level directory-creation side effects. Strict FP32 scores, self-removal and stable ties. Exhaustive small-catalog tests compare explicit re-ranking. |
| Query macro and paired intervals | `revision_evidence_20260917/scripts/common.py::weights`; `analyze_membership.py::per_query,summarize`; manuscript §3.3/D.4 | Pair indicators averaged within query, then equally across queries. Same sorted query IDs get identical bootstrap resamples; seed 2026091701, 10,000 draws. |
| Fitting target support | `revision_evidence_20260917/scripts/audit_fitting.py`; manuscript §3.2 | New model-native complete input lengths, not the old BGE-only mask. All seven target inputs must fit. Full competitor catalog remains in retrieval. |
| Canonical rules | `revision_evidence_20260917/scripts/common.py::canonical_attrs` | Compile and reuse exact original function AST. Casefold plus original-string tie break; reverse lexical; type/style substring priority. All products checked under all seven incoming orders. |
| Native model encoding | New `scripts/adapters.py::NativeAdapter` and pinned official model configs | Existing `phase2/src/encoding.py` maps model-native pooling to mean or CLS and is unsuitable for Qwen last-token / Gemma projection. It remains unchanged. |
| Later evidence corrections | `revision_final_strengthening_20260922/PROTOCOL.json`; `revision_section6_20260924/scripts/analyze_saved.py,reports.py` | The 128-query/32-schedule follow-up is a different population and is not substituted for the primary 308-query WANDS support. |

`audit/preflight.json` contains exact source hashes. `audit/*_serialization.csv` preserves each product's unique seven-view count and duplicate-occurrence count. No historic model values are inserted into cross-model plots without a further support/provenance check; the default plot uses completed new-model cells only.

All six processed product/query/judgment payload hashes and the split JSON hash exactly match the historical graded-revision input manifest. The seven comparisons are saved in `audit/historical_data_hash_reconciliation.json`.

The root README and historical `paper_www2027/RESULT_PROVENANCE.md` are navigation aids, not sufficient authority for current evaluation. Historical significance/go thresholds are not part of this follow-up: low or zero VI is a valid result.

