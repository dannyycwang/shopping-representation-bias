"""Integrate completed evidence into the existing manuscript; preserve prior draft."""
from common import *
PAPER=ROOT/'paper_www2027';S=PAPER/'sections'
assert (OUT/'queue_completed.json').exists()
packet=json.loads((OUT/'paper_numeric_packet.json').read_text());choice=packet['selected'];selected=choice['method']
labels={'rule_type':'type-first','rule_appearance':'appearance-first','rule_use':'use-first','rule_dimension':'dimension-first'}
def name(m):return labels.get(m,m.replace('_',' '))
def put(f,t): (S/f).write_text(t.strip()+'\n',encoding='utf8')

put('introduction.tex',r'''
\section{Introduction}
Web users increasingly reach information through systems that decide what to retrieve before the user sees it. Search, recommendation, retrieval-augmented systems, and emerging agentic interfaces all depend on an initial selection of content. For a shopping agent to compare a relevant product, that product must first enter its candidate set. This motivates a retrieval question, without assuming that better retrieval necessarily improves recommendations or purchases: \emph{can irrelevant choices in how unchanged information is represented determine whether relevant content is found?}

Much of the Web is structured or semi-structured. Products, jobs, hotels, and events are stored as fields, yet neural retrieval often consumes flattened token sequences. Many underlying facts have no meaningful linear order. Product color, material, dimensions, and compatibility remain the same facts when their positions in a record change. This mismatch makes serialization a potential nuisance signal between semantic relevance and retrieval visibility.

E-commerce provides a controlled setting for studying that mismatch. We preserve product identity, all source facts, the query, the model, and the relevance labels while permuting complete attribute atoms. We distinguish a catalog-wide intervention, corresponding to a shared indexing change, from a target-only intervention that holds every competitor in its original representation. The latter supports the precise comparison: same product, same facts, same competitors, same query, same model; only the target serialization changes.

Figure~\ref{fig:example} illustrates the effect. For \emph{turquoise chair}, an Exact-relevant product moves between ranks 5 and 1,527 under different attribute orders with every competitor fixed. Its original rank is 486. The example is selected for interpretability and a large crossing; aggregate results, rather than this extreme case, establish the population pattern. The full title, source atoms, identifiers, and all tested ranks are included in the artifact and appendix.

\begin{figure}[H]
\centering\includegraphics[width=\columnwidth]{figures/figure1_target_only.pdf}
\caption{Target-only intervention, WANDS query 162 and product 34536, MiniLM. The same Exact-relevant product crosses Top-20 with competitors fixed in C0. This differs from catalog-wide reserialization.}
\Description{Identical product facts with different attribute orders yield ranks five and 1527, with the query, model and all competitor representations fixed.}
\label{fig:example}
\end{figure}

Order instability is important, but eliminating it is not sufficient. A relevant product that is never retrieved under any representation has zero crossing instability. We therefore deepen the robustness question into a recall-first evaluation: \emph{can fixed, fact-preserving representation strategies make relevant products more reliably retrievable on queries not used for strategy selection, and at what cost?} Highest-label Recall@100 is the primary strategy outcome, with relevance effectiveness, instability, and resource costs reported alongside it.

The available intervention also matters. A merchant can change one product's text, whereas a platform can change an entire index, aggregate embeddings, or store multiple vectors per product. A target-only gain does not establish a market-wide benefit, and a platform-only method is not a merchant-deployable rewriting rule. We compare these settings explicitly and prevent multi-view retrieval from receiving extra candidate slots: all methods return the same number of unique products.

Our contribution is an empirical diagnosis and controlled strategy evaluation, not a claim to invent centroid averaging, multi-vector retrieval, or rank fusion. We ask four questions: how equivalent representations affect relevant-product inclusion; where instability concentrates after accounting for rank and input length; whether development-selected strategies transfer; and how their recall and cost compare with simply enlarging the candidate pool. We combine cross-model sensitivity, single-product interventions, rank-conditioned diagnostics, strong lexical/hybrid controls, and fixed-text reranking. Negative strategy results and the distinction between stability and useful retrieval remain central to the interpretation.
''')
put('related_work.tex',r'''
\section{Related Work}
\subsection{Structured Product and Hybrid Retrieval}
Product search must match queries to heterogeneous catalog fields. Lexical retrieval such as BM25 provides a strong order-insensitive baseline under a fixed token multiset~\cite{robertson2009bm25}; dense matching and field-aware models address semantic mismatch~\cite{choi2020semantic}. Attribute extraction can supply explicit matching signals~\cite{loughnane2024attributes}. CHARM constructs hierarchical field representations with block-triangular attention and uses aggregated vectors for candidate retrieval followed by richer field-level matching~\cite{freymuth2025charm}. Its learned, field-aware hierarchy differs from our frozen-encoder views of complete, fact-equivalent product records. We use it to position multi-vector product retrieval, not as an implemented baseline.

Hybrid retrieval combines lexical and semantic signals. Reciprocal rank fusion combines rankings without requiring comparable score scales~\cite{cormack2009rrf}. We include fixed full-list RRF as a practical competitor because a representation strategy is less compelling if a simple hybrid already obtains better recall. Our comparisons hold candidate count fixed and separately report the extra storage and scoring cost of multiple product vectors. The purpose is to evaluate the representation decision, rather than propose another fusion algorithm.

\subsection{Content and Visibility Optimization}
GEO studies how content changes influence visibility in generative systems~\cite{aggarwal2024geo}; E-GEO specializes this objective to e-commerce~\cite{bagga2025egeo}, while SAGEO Arena evaluates search-augmented optimization in a broader retrieval-and-generation setting~\cite{kim2026sageo}. Ranking-incentivized content modification likewise studies strategic changes under quality constraints~\cite{goren2020ranking}. Our initial question is different: why should a semantically irrelevant representation choice alter visibility in the first place? Our subsequent strategy evaluation preserves raw facts and uses query-independent rules or platform indexing. We do not evaluate generated product claims, agent recommendations, purchases, or merchant welfare.

\subsection{Representation Robustness}
Serialization sensitivity is not a new discovery in neural retrieval. Bhandari et al. study alternative table serializations, centroid representations, and a residual adapter toward centroid targets~\cite{bhandari2026tabular}. Their centroids motivate our full-record averaging baseline, which is distinct from independent attribute encoding followed by set pooling. We extend the empirical question to judged-relevant product inclusion, target-only causal control, persistent omission, cross-encoder candidate limitations, and recall--cost tradeoffs. We do not claim that a finite mean over views is a new invariant-learning algorithm.

Order effects and behavioral retrieval testing also caution against treating aggregate effectiveness as a complete description of a neural system~\cite{abdou2022wordorder,macavaney2022abnirml}. Our task-defined equivalence class is deliberately narrower than arbitrary rewriting: complete raw attribute atoms are moved while their contents and multiplicities remain fixed. Generalization to other structured Web domains remains an empirical question.
''')
put('problem.tex',r'''
\section{Problem Formulation}
Let $F(p)$ denote the information underlying item $p$, and let $\mathcal R(F(p))$ contain representations that preserve that information for the task. Human relevance $\operatorname{Rel}(q,p)$ remains fixed across the class. A normalized dual encoder scores a representation by $z(q,r(p))=\langle E_q(q),E_p(r(p))\rangle$. A representation-robust retriever should ideally avoid substantial visibility differences induced solely by task-irrelevant changes in $r$.

For rank $\rho_r(q,p)$ and cutoff $K$, define
\begin{equation}
\mathrm{VI}@K(q,p)=\mathbb{1}\left[\min_{r\in\mathcal R}\rho_r(q,p)\le K<\max_{r\in\mathcal R}\rho_r(q,p)\right].
\end{equation}
VI is an any-in/any-out event over the tested family, not a single-update loss probability. We also partition relevant pairs into always retrieved, sometimes retrieved, and never retrieved. Sometimes retrieved equals crossing; never retrieved can be a serious recall failure despite VI=0. Rank range describes movement independently of a cutoff. VI need not increase with $K$, since a boundary can move beyond an item's entire rank interval.

Let $H_q$ be the judged highest-relevance products for query $q$. Our primary strategy metric is
\begin{equation}
\mathrm{Recall}@K(q)=\frac{|H_q\cap\operatorname{TopK}(q)|}{|H_q|}.
\end{equation}
Queries with $|H_q|=0$ have undefined recall and are excluded only from recall aggregation. Query-macro averages weight eligible queries equally; pair-micro statistics weight relevant pairs equally. For each method--Original comparison we retain rescued and newly missed relevant pairs; their difference, normalized by $|H_q|$, is the per-query recall change.

Relevance--visibility alignment requires useful retrieval as well as stability. We report condensed NDCG, which removes unjudged products before discounting, separately from complete-catalog recall and rank visibility~\cite{jarvelin2002cumulated,sakai2008alternatives}. Unknown labels are not treated as irrelevant. In a two-stage pipeline, representation-dependent Irrecoverable@100 counts products present in some first-stage pools but absent in others; persistent omission is reported separately. Reranker VI on common candidates is conditional on survival and cannot be compared directly with all-highest-pair VI.
''')
put('methodology.tex',r'''
\section{Interventions and Retrieval Strategies}
\subsection{Equivalent Inputs and Intervention Permissions}
The primary family contains C0 source order, C1 reversal, and five deterministic attribute permutations. WANDS atoms are raw key--value fields; ESCI uses available brand, color, and complete bullet-point strings as atoms, retaining the internal bullet text. Audits preserve complete raw atoms and multiplicity, non-attribute fields, source values, and character/lexical token multisets. Catalog-wide intervention changes all products. Target-only intervention changes one relevant target at a time with every competitor in C0:
\begin{equation}
\rho_s^{\rm target}(q,p)=1+\sum_{u\ne p}\mathbb{1}[z(q,C0(u))>z(q,s(p))],
\end{equation}
with stable catalog-order tie handling. The old target entry is removed before insertion. These interventions answer different questions; their crossing rates do not form an additive causal decomposition.

Merchant-side strategies must produce one complete product text without knowledge of a future query. We freeze four field-priority rules emphasizing type/style, appearance, use/compatibility, or dimensions/material. They move entire raw atoms, retain the original free text and section order, and use a canonical lexical fallback. Matching uses field keys where available and literal prefixes for unkeyed ESCI atoms. One global rule is chosen on development Recall@100; cNDCG and a fixed simplicity order break ties. No product-specific or category-specific supervised policy is fitted.

We distinguish pure raw-atom sorting from the historical canonical template M2, which also adds field framing and changes section placement. Both are retained as baselines. This avoids attributing the template's effect entirely to sorting. The selected priority rule is evaluated both catalog-wide and target-only; the latter is a diagnostic of a single record's effect, not a deployment policy that knows test relevance labels.

\subsection{Platform-Side Aggregation and Fusion}
For full-record multi-view methods, generate nested sets of $m=2,4,7$ views by shuffling a canonical raw-atom list using fixed product-ID-dependent seeds. Each complete record is encoded and normalized independently. Centroid retrieval uses
\begin{equation}
v_m(p)=\operatorname{norm}\left(\frac1m\sum_{j=1}^{m}E_p(r_j(p))\right),
\end{equation}
whereas multi-vector retrieval scores $p$ by $\max_j\langle E_q(q),E_p(r_j(p))\rangle$. Reduction is by product before selecting $K$ unique products. Equal $m$ is used for every product; actual distinct text and encoded-view counts are recorded. The canonical-anchored generator is independent of incoming attribute order. This does not make finite averaging an exhaustive orbit average or establish robustness to other semantic transformations.

The existing set-mean method instead encodes individual attributes. With normalized non-attribute vector $t(p)$, it computes
\begin{equation}
\begin{aligned}
a(p)&=\operatorname{norm}\left(\frac1{|A(p)|}\sum_{a\in A(p)}E_p(a)\right),\\
v(p)&=\operatorname{norm}(.5t(p)+.5a(p)).
\end{aligned}
\end{equation}
The attribute mean is normalized before interpolation, matching the implementation; the earlier draft omitted this intermediate step. Empty attribute sets contribute zero. This proof of concept removes order dependence but may lose useful cross-attribute interactions. It is evaluated for recall and effectiveness, not declared successful because VI is zero.

BM25 uses the full lowercase alphanumeric token multiset without positional features or truncation. Hybrid retrieval adds full-list reciprocal-rank scores with constant 60~\cite{cormack2009rrf}. These are platform controls. Multi-vector and centroid methods increase offline encoding; only the former retains multiple index vectors. None is represented as a merchant-editable text strategy.
''')
put('experiments.tex',r'''
\section{Experimental Setup and Audit}
\subsection{Data, Splits, and Evaluation History}
The historical WANDS catalog contains 42,994 products, 480 queries, and 231,859 deduplicated, noncontradictory judgments. Highest relevance is Exact, with declared gains 3/1/0 for Exact/Partial/Irrelevant. The historical ESCI experiment uses 500 US small-version test queries and a fixed union of 10,076 products with 10,134 judgments~\cite{chen2022wands,reddy2022shopping}. We report judged-relevant recall; incomplete judgment coverage prevents claims about every truly relevant product.

The extension freezes the existing WANDS development split: 96 queries selected by SHA256 order and 384 held-out queries. Historical test results have already been inspected, so these are post-hoc fixed-strategy evaluations, not pristine confirmation. Only development labels choose the global rule and the best added configuration among the rules, centroid2/4, and max2/4. The chosen configuration transfers unchanged to MiniLM and ESCI; m=7 is a saturation check and cannot replace the winner. All screening results, including losses, remain available.

A further 100 ESCI query IDs absent from the historical query set are frozen before retrieval. Their judged products are added to the historical union to define a new common catalog for all compared methods. This evaluates newly sampled queries, while acknowledging prior access to the source dataset. Absolute recall differences from the old catalog cannot be attributed to generalization alone because competition changes. No product-disjoint or unseen-category claim is made.

\subsection{Models and Reconciliation}
We retain pinned MiniLM, BGE, GTE, and the historical product-specific encoder for sensitivity evidence; exact model identifiers and revisions are in the artifact. New strategy selection uses frozen BGE; MiniLM tests transfer to another encoder family without retuning. BGE uses CLS and 512 tokens, MiniLM mean pooling and 256 tokens. Query prefixes remain empty, matching the historical condition; BGE v1.5 supports instruction-free use although its model card recommends an instruction for retrieval. CUDA weights are FP16 with FP32 pooling, normalized vectors, and exact stable catalog-order ranking. Models are not retrained.

The audit found that historical ESCI gains interchanged Substitute and Complement. The benchmark paper specifies gains 1, .1, .01, and 0 for E, S, C, and I~\cite{reddy2022shopping}. Its training code agrees, but the pinned repository's qrels/evaluation-shell combination is internally inconsistent with that definition. We therefore recompute effectiveness from unchanged ranks under the paper's gains and retain the historical values in a reconciliation artifact. Highest-label recall, ranks, and VI do not depend on that swap. The earlier mitigation join repair is preserved: cNDCG retains every positive-IDCG query, independently of highest-label availability.

\subsection{Uncertainty, Cost, and Reranking}
Primary strategy comparisons use query-macro Recall@100 and paired query bootstrap with 10,000 resamples. We report fixed contrasts against Original and hybrid, the selected rule against Original, and direct set-mean--canonical comparisons. Confidence intervals spanning zero do not establish equivalence; no non-inferiority margin is claimed. Diagnostic rank strata use common-query support and clustered query uncertainty, rather than treating all pairs as independent.

We reuse full rankings for $K=20,50,100,200,500,1000$. Local measurements use an Intel i7-13700H CPU and RTX 4060 Laptop GPU with 8 GB memory. Cost separates dense-vector/sparse-posting storage, query encoding, and exact scoring/sorting on four CPU threads. Repeated timing uses the first 50 queries, a warm-up, and five measured repetitions. Encoding costs and distinct-view counts are retained in the artifact. These are local measurements, not ANN or commercial-scale latency estimates.

The new reranker experiment uses the pinned MiniLM cross-encoder with one fixed M2 canonical text per product, joint 256-token input, $K=50,100,200$, and a final Top-20. Candidate-union caching avoids repeated inference. Returned-list condensed NDCG removes unjudged items from those final results but normalizes against all judged ideal gains; it is distinguished from full-catalog cNDCG. The historical experiment, which also varied reranker input, is retained as a separate condition. Its highest-relevance Top-100 memberships survive reconciliation even where floating-point and tie handling alter other candidate positions.
''')
path=S/'experiments.tex';body=path.read_text(encoding='utf8');new=packet['new_catalog']
counts=[]
for ds in ['wands','esci','esci_new']:
    pq=pd.read_csv(OUT/ds/'bge_base_Original_per_query.csv')
    if ds=='wands':pq=pq[pq.query_id.isin(json.loads((P4/'config/splits.json').read_text())['wands_heldout'])]
    counts.append((int(pq['Recall@100'].notna().sum()),int(pq['cNDCG@10'].notna().sum())))
extra=f" The new catalog contains {new['products']:,} products and {new['judgments']:,} judgments. Recall/cNDCG eligible query counts are {counts[0][0]}/{counts[0][1]} for held-out WANDS, {counts[1][0]}/{counts[1][1]} for historical ESCI, and {counts[2][0]}/{counts[2][1]} for the new queries."
body=body.replace('No product-disjoint or unseen-category claim is made.','No product-disjoint or unseen-category claim is made.'+extra);path.write_text(body,encoding='utf8')
print('Wrote framework and audited setup sections.')
