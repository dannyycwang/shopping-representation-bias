# Final evidence strengthening — 22 September 2026

**A–D completed**, with attribution failures and numerical ambiguity retained as results. No resource-blocked cells, training, new GTE forward passes, manuscript edits or default-branch merge. Dedicated branch: codex/final-strengthening-20260922.

- A: all 21,299 WANDS Exact pairs / 308 queries and 4,434 ESCI E pairs / 499 queries, three encoders, seven schedules, K=20/100: 1,080,786 boundary records. No Top-K reconstruction contradictions. One previously documented deep BGE rank discrepancy remains authoritative.
- B: 24 hash-stratified diagnostic pairs plus French molding; 50 variants, 100 baseline runs. 98 completeness passes, 2 retained failures. The main crossing fails its precision guard; the stable control has a primary-baseline completeness failure. Neither is replaced.
- C: 48 dense/hybrid comparisons, every canonical rule, both K values, all six identity transitions with query-macro/pair-micro intervals and conditional denominators.
- D: actual encoding of 324,540 new distinct model inputs. WANDS 128 frozen queries / 7,125 pairs, 7/16/32 schedules; ESCI all 499 queries / 4,434 E pairs, every distinct whole-entry target order. Full-support Top-20 crossing: MiniLM: 20.85% → 24.50% → 28.21%; BGE: 11.40% → 15.03% → 17.55%. ESCI: MiniLM: 2.91% → 3.25%; BGE: 2.42% → 3.13%.

P0/P1's 122 conditions / 49,227 records, corrected graded metrics, canonical hybrids, previous M=2/4/7 results and costs were read and reused, not rerun.

Read BOUNDARY_AUDIT.md, XAI_AUDIT.md, CONTROL_TRANSITIONS.md and PERMUTATION_COVERAGE.md for the four workstreams. CLAIM_VERDICTS.md and CHAPTER6_EVIDENCE_MAP.md provide review-ready conclusions, English insertions, captions and placements. REPRODUCE.md, manifests and qa/final_validation.json give reproducibility and completion gates. One main case figure and one compact table are proposed; coverage/extra QA are supplementary.

Reviewed baseline 29a699794746b4ecc916f69e75e7d05f737677f7 and starting HEAD/merge-base 914f5ec36a75e8bcfeeef5234c86b2468ff761dd have identical tracked trees; the baseline adds two merge-history commits. Existing manuscript/figure edits were hashed and preserved. No applicable AGENTS.md was found.

Protocol SHA256: ab16a31715b095d1c1d75860632b97d625d2607ee9875909246d25adf3eafbb6. This is a frozen follow-up after observed historical results, not whole-study preregistration. INPUT_MANIFEST.json lists 185 exact payload/source files. COVERAGE_EXECUTION_ADDENDUM.json documents input-identity deduplication before new outcomes without changing supports, schedules or target strings.

Git includes scientific pair/schedule scores/ranks, XAI data, figures and cache metadata. Large generated vectors/score arrays remain locally available but ignored; CACHE_MANIFEST.json inventories hashes. Some original arrays/model snapshots also require restoration from pinned hashes outside Git. Never relabel newly generated arrays as original artifacts.
