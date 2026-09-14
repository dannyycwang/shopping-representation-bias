# Phase III + Full Paper Integration

## Target Venue

**The Web Conference 2027 — Research Track**

Primary target track:

**Search, Recommendation, and Retrieval-Augmented AI**

Working paper title:

# Same Product, Different Visibility: Representation Robustness in Neural E-Commerce Retrieval

Alternative title if the final method becomes sufficiently strong:

# Aligning Relevance and Visibility: Representation-Robust Neural Product Retrieval

---

# 0. FINAL PROJECT DIRECTION

This project has evolved across Phase I and Phase II.

Do NOT organize the final research paper as:

Phase I
→ Phase II
→ Phase III.

Those phases are the internal research history.

The final paper must instead present one coherent scientific story:

> Modern e-commerce retrieval increasingly relies on neural representations of product information. However, structured product attributes are conceptually unordered facts, while neural retrievers typically consume them as ordered token sequences. We investigate whether arbitrary serialization choices can therefore change the visibility of otherwise identical products, whether these changes affect human-judged relevant products, whether the phenomenon survives realistic retrieval pipelines, and whether permutation-invariant product representations can mitigate the problem without sacrificing relevance.

The true research focus is:

1. **Representation sensitivity**
2. **Relevance–visibility alignment**
3. **Downstream retrieval consequences**
4. **Representation-robust mitigation**

Do NOT return to the old framing:

"How can a merchant rewrite a product to rank higher?"

Do NOT treat Dual View as the proposed method.

Do NOT optimize a target merchant's rank.

The paper should instead ask:

> Why should arbitrary serialization determine whether a relevant product is visible?

and:

> Can product retrieval be made invariant to semantically irrelevant representation choices while preserving relevance?

---

# 1. WEB CONFERENCE 2027 FORMAT

Create the final paper directly in ACM Web Conference research-paper format.

Use:

```latex
\documentclass[sigconf, anonymous, review]{acmart}
```

The submission must be anonymous.

Do not include:

* author names;
* affiliations;
* acknowledgments revealing identity;
* identifying repository links;
* identifying self-references.

Write in English.

Use double-column ACM format.

Target:

* maximum 8 pages of main paper;
* references and optional appendix may extend the total manuscript to at most 12 pages;
* the first 8 pages must be completely self-contained.

The first page MUST explicitly explain why the work addresses a Web research problem.

Do not merely say that e-commerce is on the Web.

Explain that Web-mediated product discovery is increasingly performed through neural search, recommendation, retrieval-augmented systems, and agentic interfaces, and that representation-induced visibility instability affects how Web content is surfaced.

---

# 2. EXPECTED PAPER OUTPUT

Create:

```text
paper_www2027/
    main.tex
    references.bib
    sections/
        introduction.tex
        related_work.tex
        problem.tex
        methodology.tex
        experiments.tex
        results.tex
        discussion.tex
        conclusion.tex
    figures/
    tables/
    appendix.tex
    README.md
```

The final output should compile.

Also produce:

```text
paper_www2027/main.pdf
```

if the environment supports LaTeX compilation.

If compilation is unavailable, still produce valid LaTeX and clearly report the missing dependency.

---

# 3. USE PHASE I–III AS ONE STUDY

Read ALL existing Phase I and Phase II artifacts before writing the paper.

Important sources include:

* Phase I report
* Phase II report
* frozen protocols
* configs
* per-query results
* pair-rank files
* confidence intervals
* qualitative cases
* representation-equivalence audits
* single-product intervention results
* figures
* numerical appendices

Phase III results should later be incorporated into exactly the same paper.

Do NOT mechanically reproduce every Phase I experiment.

Remove experiments that no longer contribute to the final scientific story.

The paper should contain only the strongest, most defensible evidence.

---

# 4. P0 — METRIC RECONCILIATION

THIS IS A HARD GATE.

DO NOT START NEW PHASE III MAIN EXPERIMENTS UNTIL THIS AUDIT PASSES.

There is currently an unresolved metric reconciliation issue.

## 4.1 Observed discrepancy

The Phase II report gives WANDS C0_original cNDCG@10 approximately:

```text
MiniLM            0.7793
BGE-base          0.8277
GTE-ModernBERT    0.8219
```

However:

`stability_adjusted_relevance_native.csv`

at lambda=0 appears to produce approximately:

```text
MiniLM            0.738094
BGE-base          0.795444
GTE-ModernBERT    0.789513
```

Furthermore, the earlier Phase I WANDS MiniLM R0 cNDCG@10 was approximately:

```text
0.7383
```

This must be fully explained.

Do NOT assume that any one version is correct.

---

# 5. P0.1 — TRACE EVERY METRIC

For every relevant metric implementation identify:

* relevance gain mapping;
* Exact / Partial / Irrelevant values;
* ESCI E / C / S / I values;
* treatment of unjudged products;
* condensed-ranking procedure;
* query eligibility;
* number of evaluated queries;
* micro vs macro aggregation;
* query averaging;
* candidate catalog;
* duplicate judgments;
* conflict removal;
* ties;
* serialization;
* embedding model;
* pooling;
* chunking;
* truncation.

Compare explicitly:

1. Phase I report;
2. Phase II report;
3. stability-adjusted relevance outputs;
4. per-query metric files;
5. source evaluation code.

---

# 6. P0.2 — RECOMPUTE FROM LOW-LEVEL RESULTS

Do not fix summary CSV values manually.

Recompute from stored rankings/scores and judgments:

* cNDCG@10;
* cNDCG@20;
* Recall@20;
* highest-relevance Recall@20;
* RelevantHidden@20;
* VI@20;
* SAR(lambda=0).

Implement a clean independent audit function if possible.

At lambda=0:

```text
SAR(lambda=0)
=
the exact relevance component used by SAR
```

must hold to floating-point tolerance.

Add an automated unit test.

If it fails, stop execution.

---

# 7. P0.3 — MICRO / MACRO CONFIDENCE INTERVAL AUDIT

There is another reporting ambiguity.

For example, Phase II tables appear to mix values such as:

```text
Micro VI@20 = 13.26%
Query-macro VI@20 = 19.17%
Query-macro CI = [17.09%, 21.27%]
```

Some previous outputs may display:

```text
13.26% [17.09%, 21.27%]
```

which would be statistically incoherent because the point estimate and confidence interval refer to different estimands.

Audit every table.

Use explicit columns:

```text
Micro VI@20
Query-Macro VI@20
Query-Macro VI@20 95% CI
```

Never attach a macro CI to a micro estimate.

---

# 8. P0.4 — REQUIRED RECONCILIATION OUTPUT

Create:

```text
METRIC_RECONCILIATION.md
```

containing:

## Table A — Metric definitions

```text
Metric
Dataset
Gain Mapping
Query Count
Unjudged Handling
Aggregation
Implementation
```

## Table B — Recomputed values

```text
Dataset
Retriever
Representation
Existing Value
Recomputed Value
Difference
Explanation
```

## Table C — Phase I vs Phase II

Identify the exact cause of every material difference.

Possible explanations may include:

* changed gain mapping;
* changed representation;
* changed model version;
* chunking;
* pooling;
* query eligibility;
* different metric implementation.

Do NOT use vague explanations such as:

"different evaluation protocol"

without identifying the exact difference.

---

# 9. P0 EXIT GATE

Phase III may begin only when:

* all important Phase II numbers are reproducible;
* lambda=0 consistency passes;
* micro/macro outputs are correct;
* Phase I–II differences are documented;
* no unexplained metric drift remains.

If any Phase II value is wrong:

1. correct the implementation;
2. regenerate dependent results;
3. regenerate dependent figures;
4. update PHASE2_REPORT;
5. preserve an audit log.

---

# 10. CORE EMPIRICAL CLAIM FROM PHASE II

After P0 reconciliation, preserve Phase II's strongest result:

Across multiple dense retrievers and two datasets, fact-equivalent attribute-order variants produce substantial Top-K visibility instability.

This is the main phenomenon.

The final paper should NOT claim:

"neural retrieval is generally sensitive to token order"

as a novel discovery.

Instead the e-commerce-specific claim is:

> Representation sensitivity can materially change the visibility of human-judged relevant products in neural product retrieval.

The paper should emphasize practical Top-K boundary crossing rather than only embedding differences.

---

# 11. FIGURE 1 — PRIMARY MOTIVATING CASE

Use the existing Phase II pair-rank artifacts.

The strongest current candidate is:

## Query

```text
turquoise chair
```

## WANDS product

Product ID:

```text
34536
```

Product title is approximately:

```text
adjustable swivel office chair computer chair task chair mesh chair, turquoise
```

Ground-truth relevance:

```text
Exact
```

Phase II primary-family MiniLM ranks were observed approximately as:

```text
C0 original               486
C1 reversed               295
C2 random permutation 1   1303
C2 random permutation 2   21
C2 random permutation 3   484
C2 random permutation 4   3
C2 random permutation 5   4
```

IMPORTANT:

Before using any of these numbers in the paper, independently verify them from the frozen pair-rank files.

Do not copy these values into the manuscript unless verification passes.

The key illustrative comparison should preferably be:

```text
Same product
Same query
Same relevance
Same underlying facts
Same attribute multiset

Permutation A → Rank 3
Permutation B → Rank 1303
```

If verified.

The visibility gap would therefore be approximately:

```text
1300 rank positions
```

without changing the underlying product facts.

---

# 12. FIGURE 1 DESIGN

Create a publication-quality Figure 1.

Suggested layout:

```text
              Query:
         "turquoise chair"

                    SAME PRODUCT
              Human relevance: Exact
                       |
          -------------------------
          |                       |
          |                       |

Representation A          Representation B
same facts                same facts
same attributes           same attributes
order A                    order B

          |                       |
          v                       v

      Rank #3                 Rank #1303

      VISIBLE                  HIDDEN
```

Include a clear center annotation:

```text
Only attribute serialization changed
```

And a footer:

```text
Same product · Same facts · Same query · Same model
```

Do NOT include:

* sales effects;
* purchase claims;
* agent recommendation claims;
* revenue claims.

Those are not yet supported.

---

# 13. SECOND SUPPORTING CASE

Also verify and retain as a scientific supporting example:

## Query

```text
e12/candelabra
```

Product ID approximately:

```text
7756
```

Ground-truth relevance:

```text
Exact
```

The product title explicitly contains the E12/candelabra specification.

Observed MiniLM primary-control result:

```text
Original attribute order     Rank 14
Reverse attribute order      Rank 1881
```

Again:

VERIFY FROM FROZEN PAIR-RANK FILES FIRST.

This case is scientifically useful because it involves a particularly simple intervention:

```text
original order
vs.
fully reversed attribute order
```

rather than selecting one favorable random seed.

Use this as:

* a secondary callout;
* qualitative results example;
* appendix example;

rather than necessarily Figure 1.

---

# 14. CROSS-MODEL FIGURE CANDIDATE

Also inspect the existing candidate:

```text
query: teal chair
```

because preliminary pair-rank inspection suggested large Top-K changes under more than one retriever.

Export complete metadata before deciding whether to use it.

A cross-model qualitative example may be scientifically stronger than the largest single-model rank swing.

Create:

```text
FIGURE1_CANDIDATES.md
figure1_candidates.csv
```

with at least 20 candidates.

Selection criteria:

1. highest relevance;
2. clear semantic match;
3. same source facts;
4. no fact additions/deletions;
5. Top-20 boundary crossing;
6. large rank range;
7. understandable query;
8. preferably replicated across models;
9. avoid pathological one-word queries.

Figure 1 selection must be evidence-driven.

---

# 15. P1 — OFFICIAL ESCI REPLICATION

The existing Phase II ESCI experiment uses a custom union catalog.

Keep it.

Do not discard it.

Add an official ESCI candidate-pool evaluation.

Purpose:

> Does representation sensitivity persist under the dataset's standard evaluation formulation?

Use only the primary fact-equivalent family:

```text
C0 Original
C1 Reverse attribute order
C2 Fixed random attribute permutations
```

Do not add new stylistic rewrites.

Report:

* official NDCG metrics;
* highest relevance retrieval;
* rank range;
* VI;
* Top-K crossing where applicable.

Clearly label:

```text
ESCI Full-Catalog
```

and:

```text
ESCI Official Candidate Pool
```

as different experiments.

---

# 16. P2 — E-COMMERCE-SPECIALIZED RETRIEVER

Add at least one retriever trained specifically or substantially on:

* product search;
* e-commerce retrieval;
* shopping queries;
* product matching.

The research question is:

> Does product-domain training naturally improve representation robustness?

Do NOT tune perturbations for this model.

Use the frozen primary permutations.

Compare against:

```text
MiniLM
BGE-base
GTE-ModernBERT
E-commerce-specialized model
```

Report:

* relevance;
* rank range;
* VI@10;
* VI@20;
* VI@50.

If the e-commerce-specific model is more stable, treat this as an important finding.

Do not hide it because the effect is smaller.

---

# 17. P3 — RERANKER ROBUSTNESS

Build a realistic two-stage ranking pipeline:

```text
Query
→ Dense Retrieval Top-100
→ Strong Cross-Encoder Reranker
→ Top-20
```

Use one good reranker.

Do not create a reranker zoo.

---

# 18. P3.1 — IRRECOVERABLE CANDIDATE LOSS

Measure whether a relevant product disappears from Top-100 solely because serialization changed.

Define:

```text
Irrecoverable@100
```

A product is irrecoverable if:

```text
relevant = true
```

and one fact-equivalent representation is retrieved into Top-100 while another is not.

Interpretation:

> What is not retrieved cannot be reranked.

This is potentially one of the strongest downstream implications.

---

# 19. P3.2 — RERANKER CORRECTION

Among products appearing in Top-100 under all equivalent representations:

compare:

* rank range before reranking;
* rank range after reranking;
* VI@20 before;
* VI@20 after.

Possible outcomes:

### Outcome A

Reranker fixes most instability.

Interpretation:

representation sensitivity is primarily a first-stage candidate-generation problem.

### Outcome B

Products disappear before reranking.

Interpretation:

candidate generation produces irreversible visibility loss.

### Outcome C

Instability remains after reranking.

Interpretation:

representation sensitivity propagates through the ranking pipeline.

All outcomes are meaningful.

---

# 20. P4 — MAIN MITIGATION CONTRIBUTION

Do NOT treat M2 alphabetical/canonical sorting as the final solution.

M2 remains a baseline.

Its limitation is:

```text
VI = 0
```

largely by construction, while relevance may decline.

The main method should be:

# Permutation-Invariant Product Encoding

Motivation:

> Product attributes are sets, not sentences.

---

# 21. P4.1 — SET-BASED ATTRIBUTE ENCODING

Represent structured product facts as:

$$
A_p =
\{(a_1,v_1),...,(a_n,v_n)\}.
$$

Encode each attribute independently:

$$
h_i = E(a_i : v_i).
$$

Aggregate using a permutation-invariant operation:

$$
h_A =
\frac{1}{n}
\sum_i h_i.
$$

Encode text separately:

$$
h_T = E(title + description).
$$

Combine:

$$
h_p =
\alpha h_T +
(1-\alpha)h_A.
$$

Normalize appropriately.

Select alpha using development data only.

Do not tune on final evaluation queries.

---

# 22. P4.2 — QUERY-AWARE ATTRIBUTE SCORING

Also implement a simple late-interaction variant.

Encode query:

$$
h_q=E(q).
$$

For each attribute:

$$
h_i=E(a_i:v_i).
$$

Compute:

$$
s_i=sim(h_q,h_i).
$$

Aggregate using an order-independent function:

```text
mean
max
Top-k mean
```

Then:

$$
Score(q,p)
=
\alpha sim(h_q,h_T)
+
(1-\alpha)Agg(\{s_i\}).
$$

Keep this architecture simple.

The goal is not architectural complexity.

The goal is:

```text
permutation robustness
+
relevance preservation
```

---

# 23. EXACT INVARIANCE TEST

For every proposed set-based method:

evaluate all C0/C1/C2 variants.

For the same query/product:

scores should be identical under pure attribute permutation within a predefined floating-point tolerance.

Report:

```text
maximum absolute score difference
mean absolute score difference
```

If mathematically invariant code produces material variation:

treat it as an implementation bug.

---

# 24. ROBUSTNESS–RELEVANCE OBJECTIVE

A method succeeds only if it achieves both:

## Stability

low:

* VI;
* rank variance;
* Top-K crossing.

AND:

## Retrieval quality

preserved or improved:

* NDCG;
* Recall;
* MRR;
* highest-relevance visibility.

Do NOT optimize only one metric.

Create a:

# Robustness–Relevance Pareto Frontier

Compare:

```text
Original serialization
Canonical sorting baseline
M1 factual representation
Permutation-invariant mean pooling
Permutation-invariant late interaction
```

Do not include obsolete weak rewrites in the main figure.

---

# 25. QUERY TAXONOMY AUDIT

The current automatic query taxonomy is heuristic.

If query-type analysis remains in the main paper:

perform a blinded manual validation on a stratified sample.

Report:

* sample size;
* agreement;
* confusion matrix;
* corrected taxonomy if required.

If the taxonomy remains noisy, move the analysis to appendix.

Do not let this secondary analysis consume excessive effort.

---

# 26. WHAT TO REMOVE FROM MAIN PAPER

Remove or strongly de-emphasize:

* R8 Dual View;
* generic marketing rewrite experiments;
* generic invitation text;
* large rewrite zoo;
* weak Phase I exploratory variants;
* proprietary shopping-agent API experiments;
* multi-agent simulations;
* conversion/revenue claims;
* claims about objective product quality.

Some may remain in appendix as pilot evidence.

The main paper must be narrow.

---

# 27. RELATED WORK POSITIONING

The paper should organize related work into approximately three areas.

## 27.1 Neural E-Commerce Retrieval

Discuss:

* lexical product retrieval;
* dense retrieval;
* hybrid retrieval;
* structured product attributes;
* e-commerce-specific retrieval representations.

End with the gap:

> Existing work primarily improves matching performance while generally treating the supplied product serialization as fixed.

---

## 27.2 Generative Engine / Product Visibility Optimization

Position:

GEO
→ e-commerce GEO / E-GEO
→ end-to-end GEO evaluation.

Distinguish this work explicitly.

E-GEO asks approximately:

> How can product representation be changed to increase a target product's visibility?

Our work asks:

> Why should fact-equivalent representations receive substantially different visibility in the first place?

Do not frame our contribution as merchant optimization.

---

## 27.3 Representation Robustness

Discuss recent work on representation / serialization stability in neural retrieval, including tabular retrieval.

Do NOT claim to be the first paper discovering that neural retrieval can be sensitive to serialization.

Instead state the narrower contribution:

> We study how representation instability manifests as product visibility instability in e-commerce search, quantify its effect on human-judged relevant products, and investigate relevance-preserving mitigation.

Verify all citations.

Do not invent references.

---

# 28. WEB RELEVANCE — MUST APPEAR ON PAGE 1

The Web Conference may desk-reject work without clear Web relevance.

The Introduction must explicitly argue:

1. Product discovery is a core Web search/recommendation activity.
2. Web merchants expose heterogeneous semi-structured product information.
3. Neural retrieval increasingly mediates what products Web users can discover.
4. Representation instability therefore changes which relevant Web content reaches users.
5. This becomes increasingly important as search, recommendation, RAG, and agentic interfaces mediate product discovery.

Avoid speculative claims that shopping agents already dominate commerce.

The Web relevance should stand even without the agent experiment.

---

# 29. PAPER STRUCTURE

Target approximately:

## 1. Introduction

~1 page.

Begin with the concrete problem.

Use Figure 1 near the introduction.

Core opening idea:

> Two representations can describe exactly the same product facts, yet a neural product retriever may expose one near the top of the ranking and effectively hide the other.

Explain why this matters.

Then introduce:

```text
representation-induced visibility instability
```

and:

```text
relevance–visibility alignment
```

Contributions should be concise.

---

## 2. Related Work

~0.75 page.

Only literature necessary for positioning.

Avoid long survey-style paragraphs.

---

## 3. Problem Formulation

~0.5–0.75 page.

Define:

product fact set;

serialization;

fact-equivalent representations;

human relevance;

visibility;

VI@K;

Top-K crossing;

relevance–visibility alignment.

Distinguish:

```text
representation sensitivity
```

from:

```text
ranking quality.
```

---

## 4. Experimental Framework

~1–1.25 pages.

Datasets:

WANDS

ESCI.

Retrievers.

Primary fact-equivalent family.

Controls.

Statistical protocol.

Single-product intervention.

Keep details concise.

Move exhaustive configurations to appendix.

---

## 5. Representation Sensitivity

~1 page.

This is the main empirical section.

Show:

* multiple models;
* multiple datasets;
* VI;
* rank range;
* highly relevant products affected;
* qualitative case.

Do not flood the main text with all Phase I experiments.

---

## 6. Relevance and Pipeline Consequences

~0.75–1 page.

Include:

* relevance-stratified effects;
* single-product intervention;
* official ESCI;
* reranker;
* Irrecoverable@100.

---

## 7. Robust Mitigation

~1 page.

Present:

Permutation-Invariant Product Encoding.

Compare:

relevance
vs.
robustness.

Include the Pareto view.

---

## 8. Discussion and Limitations

~0.5 page.

Explicitly state:

* relevance labels are not objective product quality;
* experiments measure retrieval visibility, not purchase behavior;
* dataset coverage limitations;
* neural retriever scope;
* incomplete judgments where applicable;
* agents/revenue are not directly evaluated unless Phase III later adds evidence.

Discuss implications for:

* Web product indexing;
* merchant information interfaces;
* retrieval robustness.

---

## 9. Conclusion

Very short.

---

# 30. FINAL CONTRIBUTIONS

The final paper should aim to support four or five concise contributions.

Example:

### C1 — Problem

Formalize representation-induced visibility instability in e-commerce retrieval.

### C2 — Empirical evidence

Show that fact-equivalent serialization changes Top-K visibility across multiple neural retrievers and datasets.

### C3 — Relevance consequence

Show that human-judged highly relevant products can cross practical visibility boundaries solely because serialization changes.

### C4 — Pipeline analysis

Measure whether reranking repairs this instability or whether relevant products become irrecoverable during candidate generation.

### C5 — Mitigation

Introduce a permutation-invariant product representation that reduces arbitrary serialization sensitivity while preserving relevance.

Do not claim any contribution unless final experiments actually support it.

---

# 31. ABSTRACT

Write the abstract only after all reconciled Phase III results exist.

Structure:

1. Context: neural product retrieval.
2. Problem: product facts are sets but consumed as sequences.
3. Finding: fact-equivalent serialization changes relevant-product visibility.
4. Scale/generalization: models + datasets.
5. Method: permutation-invariant encoding.
6. Result: robustness–relevance tradeoff.
7. Web implication.

Do not put unsupported numerical results into the abstract.

---

# 32. PHASE I–III SYNTHESIS

After Phase III completes, create:

```text
FULL_STUDY_SYNTHESIS.md
```

This document should map every experiment from Phase I–III into one of:

```text
MAIN PAPER
APPENDIX
REMOVE
```

For every experiment explain:

```text
Experiment
Original phase
Scientific purpose
Final result
Keep/remove decision
Paper section
```

The final paper should not expose the exploratory chronological process.

For example:

```text
Phase I R8 Dual View
→ REMOVE / appendix

Phase I C_order
→ MAIN PAPER conceptual precursor

Phase II multiple permutations
→ MAIN PAPER core evidence

Phase II multiple retrievers/datasets
→ MAIN PAPER generalization

Phase II M2
→ MAIN PAPER baseline only

Phase III set-based encoder
→ MAIN PAPER mitigation
```

---

# 33. RESULT INTEGRITY

Never combine numbers from different protocols without saying so.

Every table should originate from a single clearly defined evaluation protocol.

Keep:

```text
experiment_id
dataset
dataset snapshot
query set
representation
model
model revision
metric definition
seed
```

with every final aggregate.

Build a provenance table that maps every number in the main paper to the source result file.

Create:

```text
paper_www2027/RESULT_PROVENANCE.md
```

Each main-paper number should be traceable.

---

# 34. STATISTICAL REPORTING

Use query-paired analysis where appropriate.

Primary comparisons:

* paired bootstrap;
* 10,000 samples;
* 95% CI.

Use predefined permutation/sign tests where appropriate.

Correct multiple primary tests using Holm when needed.

Do not overuse hypothesis testing.

Report:

* effect size;
* uncertainty;
* direction;
* practical magnitude.

Do not describe tiny statistically significant changes as practically important.

---

# 35. FIGURES FOR FINAL PAPER

Aim for approximately 4 main figures.

## Figure 1

Same Product, Different Visibility.

Prefer verified `turquoise chair` example.

## Figure 2

Cross-model / cross-dataset VI@20.

## Figure 3

Pipeline consequence:

retrieval → reranking

including Irrecoverable@100.

## Figure 4

Robustness–Relevance Pareto frontier for mitigation.

Optional appendix:

* query types;
* rank-range distributions;
* additional qualitative cases;
* ablations.

---

# 36. MAIN TABLES

Keep main tables compact.

Suggested:

## Table 1

Representation sensitivity across datasets/retrievers.

## Table 2

Reranking / candidate loss.

## Table 3

Mitigation relevance–robustness results.

Move large numeric matrices to appendix.

---

# 37. FINAL GO / NO-GO CRITERIA

The paper is ready for WWW 2027 main-track submission if:

1. P0 metric reconciliation is complete.
2. WANDS and ESCI evidence remains after reconciliation.
3. multiple neural retrievers show nontrivial sensitivity.
4. official ESCI evaluation does not contradict the main phenomenon.
5. the e-commerce-trained retriever produces an interpretable result.
6. reranker analysis establishes a meaningful pipeline consequence.
7. the proposed invariant representation materially reduces instability.
8. mitigation does not cause unacceptable relevance degradation.
9. all claims remain within retrieval evidence.
10. the first page clearly establishes Web relevance.

If mitigation fails:

do NOT fabricate a successful solution.

Instead evaluate whether the empirical study alone is sufficiently strong for submission and state the robustness–relevance tradeoff clearly.

---

# 38. REQUIRED FINAL DELIVERABLES

Produce:

```text
METRIC_RECONCILIATION.md
PHASE3_REPORT.md
FULL_STUDY_SYNTHESIS.md
FIGURE1_CANDIDATES.md
figure1_candidates.csv
```

and:

```text
paper_www2027/
    main.tex
    main.pdf
    references.bib
    appendix.tex
    RESULT_PROVENANCE.md
    figures/
    tables/
```

Also retain:

```text
results/phase3_tables/
results/phase3_figures/
results/phase3_per_query/
results/phase3_reranking/
results/phase3_invariant_method/
results/phase3_audit/
```

---

# 39. FINAL INSTRUCTION

Do not merely produce another experiment report.

The final task is to turn the complete Phase I–III research process into one coherent, submission-ready scientific paper for The Web Conference 2027.

The paper should make this central argument:

> **Product visibility in neural e-commerce retrieval should reflect relevance, not arbitrary serialization choices.**

The final scientific question is:

> **Are neural e-commerce retrieval systems systematically sensitive to arbitrary serialization of identical product facts, does this sensitivity change the visibility of human-relevant products in realistic retrieval pipelines, and can permutation-invariant product representations remove that instability without sacrificing relevance?**

Run P0 first.

Then perform only the Phase III experiments that directly answer this question.

Finally, synthesize all valid Phase I–III evidence into the ACM Web Conference manuscript.

Do not preserve an experiment simply because it was expensive to run.

Preserve only evidence that strengthens the final scientific argument.
