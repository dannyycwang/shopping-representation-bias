# Figure revision notes

## Changes and placement

- **Figure 2, Section 3:** 7.1 x 3.35 in, down from 7.1 x 4.8 in (30.21% shorter). Standardized "Target-only intervention"; intact entry blocks; unchanged competitors marked with equals signs; mathematical rank ≤ K; 111/101/000 schematic states. Long explanations moved to captions/setup.
- **RQ1 main, Results/RQ1:** Top-20 query-macro VI with existing 95% intervals, full/fully-fitting markers, all numeric VI values, and query/pair counts. GTE's identical support and intervals are vertically separated and footnoted. WANDS and ESCI use different horizontal ranges, disclosed in the full caption.
- **RQ1 appendix:** both Top-20 and Top-100, Always/VI/Never. Every VI value is printed outside the bar; WANDS/BGE fitting 7.16% is explicit. Original stacked outputs remain untouched.
- **RQ2 main/appendix:** separate one-page Top-20 and Top-100 PDFs, each with all six dataset/encoder panels and all six C0-relative comparisons. Shared -7 to +7 scale; left omissions/right inclusions in percent; black net-Recall diamonds with existing paired intervals in percentage points.
- **RQ2 main small table:** six dataset/encoder rows summarizing exact within-query cancellation across six comparisons. WANDS denominator 308; ESCI 499. Min-max ranges are not CIs. Appendix CSV/Markdown/LaTeX preserves all 72 numerator/denominator rows and changed-query conditional percentages. No counts are summed into unique queries.
- **Figure 1 candidate, Introduction:** independent 3.45-inch candidate showing the requested existing case. The current turquoise-chair figure and manuscript references are preserved.

## Case audit: PASS

WANDS query 409 is `teal chair` and belongs to the existing held-out split, not the development split. Product 24318 retains the existing highest relevance label Exact. No relabeling or case search was performed.

| Schedule | Target-only rank | Catalog-wide rank, not used | Tokens including special tokens |
|---|---:|---:|---:|
| C0 | 11 | 11 | 1,059 |
| C1 | 468 | 357 | 1,059 |
| C2s1 | 123 | 135 | 1,059 |
| C2s2 | 301 | 313 | 1,059 |
| C2s3 | 677 | 685 | 1,059 |
| C2s4 | 435 | 444 | 1,059 |
| C2s5 | 939 | 956 | 1,059 |

All seven complete inputs contain 1,057 ordinary tokens plus 2 special tokens. The effective GTE limit is 8,192, verified against the pinned model's `max_position_embeddings` and `DenseEncoder._forward_native`, which passes `max_length=self.spec["max_tokens"]`. The tokenizer's `model_max_length` is the Hugging Face unset-length sentinel, so it is not used as the operational cap. This metadata distinction does not change the fitting result.

Every saved serialization exactly matches the frozen builder. All 104 complete attribute entries (104 unique strings), their multiplicities, fixed text, section order, and separators are preserved. The original record contains multiple color fields and an empty-name field; these are retained. The display excerpt uses the first four C0 entries, not newly invented or selected explanatory attributes. Their C2s5 positions are 16, 41, 85, and 98 in the displayed relative order; the complete record was encoded.

An independent count of higher-scoring competitors, excluding the old target and applying stable catalog-index tie breaking, reproduces every saved target-only rank. All 42,993 competitors remain raw C0, with 42,994 total catalog entries after replacement. Saved alternate target scores and existing C0 vectors are reused; no new encoding occurs. The candidate uses rank 939, not the catalog-wide rank 956. It is explicitly post-hoc illustrative, with no claim of typicality, deployment history, or general effect magnitude.

## Evidence and interpretation

Plot sources are byte-identical copies of `../data/rq1_target_fully_fitting_plot_data.csv`, `../data/rq2_membership_plot_data.csv`, and `../data/rq2_cancellation_companion.csv`. Filtered plot CSVs retain original decimal strings. No estimates, CI endpoints, or aggregation rules are changed. Source hashes and paths are in `data/plot_sources.json`; the case has its own complete provenance JSON.

RQ1 is Target-only intervention with the full raw C0 competitor catalog retained, including non-fitting competitors. Full/fitting differences do not isolate a causal truncation effect, and differing supports do not identify intrinsic encoder sensitivity. RQ2 is catalog-wide intervention. Its gains/losses figure illustrates aggregate cancellation; the exact integer-count table establishes within-query substitutions with unchanged Recall. Existing intervals are descriptive and uncorrected; an interval containing zero is not evidence of equivalence.

## Delivery and verification

All seven deliverables are single-page vector PDFs plus vector SVGs. Figure widths are 3.45 or 7.1 inches, fonts are at least 8 pt, and SVGs contain accessible title/description metadata. Short/full captions, ACM Description text, insertion snippets, and optional shared setup are included. A native LaTeX table is supplied as an alternative to the small vector table.

`qa/validation_report.json` records exact source preservation, 72 cancellation comparisons checked independently against existing per-query integer counts, case checks, page dimensions, embedded fonts, and absence of raster images. `qa/VISUAL_QA.md` records final 150-dpi manuscript-width inspection. All work is confined to this folder; no original figure, historical data, manuscript text, training result, model choice, or permutation family is changed.
