# Reviewer concern to evidence map

Completed 2026-09-07. This map accompanies the final Chinese `REVISION_REPORT.md` and the verified 8-main-page / 11-total-page PDF.

| Concern | Response / artifact | Remaining boundary |
|---|---|---|
| Zero VI can hide consistently missing relevant products | `phase4/results/*_visibility_states.csv`, recall-first evaluation | Judged relevance only; not user exposure |
| Whole-catalog changes confound a target effect with competitors | Existing C0/C1/C2 target-only artifacts plus selected-rule target-only outputs | Conditional on C0 competition; not market-wide merchant benefit |
| Original relevance gains may be misconfigured | `esci_gain_repair.csv`, `repaired_legacy_mitigation.csv`, official paper Section 3.1 | Official repository evaluation scripts are themselves inconsistent; both settings retained |
| Set mean versus canonical was never directly compared | Direct paired contrast in repaired mitigation and primary contrast outputs | Non-significance does not prove equivalence |
| Historical and new set-mean baseline numbers differ slightly | `shared_query_cache_audit.csv`: historical scores reproduce with their original query cache; extension shares one Original query matrix | Separate numerical conditions are labeled; old ranks preserved |
| Exact/Irrelevant difference may be distance-to-cutoff confounding | `*_rank_controlled_contrasts.csv`, common-query support counts | Broad rank strata leave residual score-margin confounding; no universal relevance gradient claimed |
| Truncation alone may explain everything | Seven-view maximum tokenizer lengths; fully fitting subset in `*_diagnostic_strata.csv` | Subset comparison is observational, not an isolated attention mechanism |
| A strong hybrid may make new methods unnecessary | BM25 and fixed RRF60 included; selected-method versus hybrid paired comparisons | No trained sparse retriever or industrial hybrid tuning |
| Multiple views receive more candidate slots | Max scores reduced by product before Top-K, same K unique IDs | More index vectors and dot products remain an explicit cost |
| Method was tuned on test labels | Frozen 96-query development IDs and selection JSON before new evaluation | Historical tests previously inspected; full study is not pristine confirmation |
| Query split is not product generalization | Fixed shared catalogs and explicit overlap manifests | No product-disjoint claim; no supervised category/product policy fitted |
| Transfer could be target-model retuning | BGE-selected rule/configuration transferred unchanged to MiniLM and ESCI | Two encoder families in the extension, not all retrieval paradigms |
| Field-priority transfer may mean different things across schemas | `selected_rule_coverage.csv` reports matches and differences from raw sorting | ESCI's unkeyed atoms make the literal rule close to raw sorting; no universal semantic-priority claim |
| Fixed-candidate reranking hides candidate omissions | K curves plus fixed canonical-text reranking and raw coverage | Offline exact retrieval and one cross-encoder only |
| Reranker input varied along with retrieval | Historical varying-input condition distinguished from new canonical input | Canonical 256-token input may itself truncate; no general reranker robustness claim |
| Cost gain could be just a larger candidate budget | Original K=20–1000 against selected method and hybrid; reranking K=50/100/200/500, with the K=500 trigger recorded in `reranking_extension_decision.json` | Local scoring/encoding timing, approximate downstream throughput costs, no commercial latency claim |
| New methods merely rediscover centroid, multi-vector or RRF | Explicit comparisons to Bhandari et al., CHARM and Cormack et al. | Contribution is controlled evaluation and evidence, not invention of pooling/fusion |
| Good example is cherry-picked | Full raw product/title/attributes and source IDs in `illustrative_case.json`, all seven ranks | Illustration is post-hoc, aggregate estimates support conclusions |
| Submission format contains placeholders | Official 8-main/12-total rule checked; archive prior 10-page body | No automatic submission; final author and publication metadata remain author-controlled |
