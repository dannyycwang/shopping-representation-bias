# Full Study Synthesis and Submission Map

## One-sentence paper

Fact-equivalent attribute orders can change whether a relevant product is retrieved; this persists across product benchmarks and encoder families, is only partly reduced by reranking, and can be eliminated at the measured intervention boundary by encoding attributes as an unordered set.

## Narrative order for the paper

The paper should be read as one argument, not as three project phases.

1. **Web-search problem.** Product catalogs are structured, but dense retrievers consume serialized strings. An arbitrary serializer can decide which products enter a visible or rerankable set.
2. **Controlled intervention.** Create seven strings that preserve the title and every attribute-value atom while changing attribute order only.
3. **Measurement.** VI@K records whether the same judged query-product pair crosses a cutoff. Query-macro VI and query bootstrap uncertainty prevent heavily judged queries from dominating.
4. **Reproduction.** The effect appears on WANDS and ESCI, across MiniLM, BGE, GTE, and a Marqo e-commerce encoder. Token-budget and official-candidate checks rule out narrow artifacts.
5. **Pipeline consequence.** A cross-encoder stabilizes candidates it sees, but cannot recover representation-dependent first-stage omissions.
6. **Mitigation.** Encode schema attributes independently and average them. This makes the attribute component invariant to order, drives measured VI to zero, and has a small cNDCG change whose paired interval includes zero on both evaluation datasets.
7. **Boundary.** The result concerns retrieval visibility. Seller fairness, user exposure, sales, and product quality require additional data.

## What belongs in the main paper

| Component | Main-paper role | Evidence file |
|---|---|---|
| Turquoise-chair example | Makes the intervention and consequence understandable | `phase3/results/phase3_figures/figure1_same_product.*` |
| Primary cross-model sensitivity | Central RQ1 evidence | `phase2/results/phase2_tables/table1_representation_sensitivity_native.csv` |
| E-commerce retriever | Domain-specific robustness check | `phase3/results/phase3_tables/ecommerce_retriever_sensitivity.csv` |
| Reranking/irrecoverable split | Central pipeline contribution | `phase3/results/phase3_tables/*_reranker_pipeline.csv` |
| Set-mean method | Central mitigation contribution | `phase3/results/phase3_tables/robustness_relevance_pareto.csv` |
| Metric reconciliation | Credibility statement and artifact detail | `METRIC_RECONCILIATION.md` |

## What belongs in the appendix or artifact

| Component | Reason |
|---|---|
| VI@10 and VI@50 full tables | Supports cutoff robustness but distracts from VI@20 narrative |
| All chunk/pooling cells | Needed to rule out truncation; summarize range in main text |
| Official ESCI table | Useful formulation check; cNDCG equality is mechanical under condensation |
| Late mean/max/top-3 ablations | All invariant, but none improves the WANDS relevance tradeoff |
| Heuristic query taxonomy | No blinded human annotation; exploratory only |
| Direct vs full-catalog intervention | Good mechanistic supplement; main pipeline experiment is stronger |
| Complete qualitative candidate list | Supports case selection and guards against cherry-picking |

## Experiment-by-experiment disposition

| Experiment | Original phase | Scientific purpose | Final result | Decision | Paper section |
|---|---|---|---|---|---|
| R0 original representation | I | Establish lexical, dense, and hybrid reference systems | Supplies the original MiniLM profile and the baseline used to understand later protocol changes | APPENDIX / provenance | Appendix and reconciliation |
| R2 generic shopping invitation | I | Test whether generic promotional framing changes retrieval | Dense cNDCG decreased by 0.0109; large case-level movements were exploratory | REMOVE | None |
| R3 factual field sentences | I | Test a readable, fact-preserving template | Exploratory dense/hybrid gains motivated the frozen M1 baseline | APPENDIX precursor | Mitigation appendix |
| R4 attribute bullets | I | Test formatting delimiters | No unique result needed for the final causal story | REMOVE | None |
| R5 sorted, labeled attributes | I | Explore deterministic normalization | Exploratory dense gain motivated canonical M2; large positive and negative movements | APPENDIX precursor | Mitigation appendix |
| R8 Dual View | I | Test concatenated original and normalized views | No R8-versus-R0 comparison survived Holm correction | REMOVE | None |
| C_order reverse-order control | I | Isolate ordering with identical token multiset | BM25 unchanged; dense Exact pairs crossed Top-20, motivating the paper | MAIN PAPER conceptual precursor | Introduction / formulation |
| C_repeat length control | I | Separate duplication/length from representation content | Repetition could reduce dense relevance; does not explain order-only effect | APPENDIX | Artifact only |
| Seven primary permutations | II | Estimate visibility across an equivalence set rather than one contrast | Nontrivial VI in every dataset-model cell | MAIN PAPER | RQ1 / Table 1 |
| Three general dense retrievers | II | Test model-family generality | WANDS micro VI@20 9.41--13.26%; ESCI 4.42--8.25% | MAIN PAPER | RQ1 / Figure 2 |
| ESCI union-catalog replication | II | Test dataset generality under a fixed shared catalog | Smaller rank ranges but consistent Top-20 crossing | MAIN PAPER | RQ1 |
| Chunk, overlap, pooling, long context | II | Test a short-context/truncation explanation | No tested encoding profile removes instability | MAIN PAPER summary | Experiments / RQ1; full table appendix |
| Relevance strata | II | Test whether highest-relevance products are affected | Strong Exact-minus-I contrast on WANDS; small and mixed ESCI contrasts | APPENDIX, with absolute highest-label VI in main | Artifact / limitations |
| Automatic query taxonomy | II | Explore which intents are sensitive | Broad patterns, but labels are unvalidated lexical heuristics | APPENDIX | Appendix only |
| Single-product intervention | II | Separate direct target movement from transforming the full catalog | Direct and catalog-wide estimands differ because competitors also move | APPENDIX | Appendix / artifact |
| M1 factual template | II | Test readable deterministic mitigation | Small, inconsistent VI change; retained as baseline | MAIN PAPER baseline | Table 3 / Figure 4 |
| M2 canonical attributes | II | Test invariance by deterministic sorting | VI=0; confirmed relevance harm in WANDS for key encoders | MAIN PAPER baseline | Table 3 / Figure 4 |
| P0 independent metric audit | III | Establish numerical integrity and explain cNDCG/SAR drift | Primary metrics reproduce to $1.11\times10^{-16}$; SAR join defect corrected | MAIN PAPER credibility statement | Experimental setup |
| Official ESCI candidates | III | Check whether the union-catalog construction creates the phenomenon | VI persists, especially at Top-10; condensed NDCG equality is mechanical | APPENDIX plus main summary | RQ1 / Appendix |
| Marqo e-commerce encoder | III | Test whether domain training removes sensitivity | WANDS VI falls but remains nonzero; ESCI remains comparable/high | MAIN PAPER | Table 1 / Figure 2 |
| Dense Top-100 plus cross-encoder | III | Measure downstream attenuation and unrecoverable loss | Reranker halves common-pool VI; WANDS macro Irrecoverable@100 is 12.52% | MAIN PAPER | RQ2 / Figure 3 / Table 2 |
| Set-mean invariant encoding | III | Remove arbitrary order while preserving relevance | VI=0; paired cNDCG CIs include zero on WANDS and ESCI | MAIN PAPER proposed method | RQ3 / Figure 4 / Table 3 |
| Late mean/max/top-3 aggregation | III | Test alternate invariant pooling rules | All invariant; each has a confirmed WANDS relevance loss | APPENDIX | Appendix ablation |

## What should be removed from the submission claim

- “best product” language: benchmark labels measure query-product relevance, not global quality;
- claims of seller fairness or realized exposure: no seller groups, impressions, clicks, or welfare data are observed;
- a universal relevance-gradient claim: WANDS and ESCI differ;
- a claim that the method is relevance-equivalent: confidence intervals include zero but do not prove equivalence;
- human-sounding query-class conclusions based on automatic lexical rules.

## Venue assessment

The completed study is credible for WWW, SIGIR, or CIKM review because it now includes a Web-relevant system consequence and a mitigation, rather than only an encoder curiosity. WWW fit depends on leading with catalog ingestion and candidate visibility in Web shopping systems. SIGIR fit depends on metric precision, behavioral IR framing, and strong retrieval baselines. CIKM is a realistic alternative if the contribution is judged more empirical than methodological.

The highest reviewer risks are limited dataset breadth, synthetic serialization variants, the indexing cost of per-attribute encoding, and no user-behavior validation. The current paper states each directly. A blinded semantic-equivalence audit is the most efficient additional study if time permits.

## Current deliverables

- Beginner guide: `PROJECT_GUIDE_ZH.md`
- Phase III report: `PHASE3_REPORT.md`
- Integrated LaTeX paper: `paper_www2027/main.tex`
- Result provenance: `paper_www2027/RESULT_PROVENANCE.md`
- Reproduction instructions: `paper_www2027/README.md`
