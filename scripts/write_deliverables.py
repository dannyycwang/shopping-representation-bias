"""Assemble authored reports with numerical tables read from completed experiments."""
import json
import hashlib
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from analyze import md

def main():
    out=ROOT/'results'; t=out/'tables'
    a=pd.read_csv(t/'aggregate.csv'); cmp=pd.read_csv(t/'paired_comparisons.csv'); h=pd.read_csv(t/'hidden.csv')
    assert len(a)==24, 'All 8 representations x 3 retrievers must complete before final reporting.'
    meta=json.loads((out/'manifest.json').read_text()); notes=json.loads((out/'interpretation.json').read_text(encoding='utf-8'))
    assert notes['aggregate_sha256']==hashlib.sha256((t/'aggregate.csv').read_bytes()).hexdigest(), 'Results changed: update the authored interpretation before exporting.'
    review=pd.read_csv(out/'qualitative/case_review.csv').fillna('')
    examples=pd.read_csv(out/'qualitative/selected_examples.csv').merge(review,on=['query_id','product_id','direction'],validate='one_to_one',how='left',indicator=True)
    assert len(examples)==40 and examples['_merge'].eq('both').all(), 'Every selected case must be reviewed before final export.'
    q=pd.read_csv(ROOT/'data/raw/query.csv',sep='\t').set_index('query_id')['query']
    p=pd.read_csv(ROOT/'data/raw/product.csv',sep='\t').set_index('product_id')['product_name']
    examples['query']=examples.query_id.map(q); examples['product']=examples.product_id.map(p)
    excols=['query','product_id','product','retriever','representation','rank_original','rank_alternative','review_zh']
    primary=cmp[cmp.role=='primary'][['retriever','delta','ci_low','ci_high','permutation_p','holm_primary_p']]
    dual=h[(h.k==20)&(h.representation=='R8')][['retriever','exact_pairs','rescued','harmed','RelevantHidden_micro','ReverseHidden_micro','net_macro','net_macro_ci_low','net_macro_ci_high']]
    order=h[(h.k==20)&(h.representation=='C_order')][['retriever','rescued','harmed','RelevantHidden_micro','ReverseHidden_micro']]
    controls=pd.read_csv(t/'length_control.csv'); fields=pd.read_csv(t/'field_statistics.csv')
    fidelity=pd.read_csv(t/'fidelity.csv'); baseline_f=fidelity[fidelity.representation=='R0']
    conflicts=int((baseline_f.conflicting_keys>0).sum()); malformed=int((baseline_f.malformed_features>0).sum())
    c=cmp[(cmp.representation=='C_order')&(cmp.metric=='cNDCG@10')][['retriever','delta','ci_low','ci_high','permutation_p']]
    maincols=['representation','retriever','cNDCG@10','cNDCG@20','Recall@10','Recall@20','Recall@50','MRR','ExactHit@20','JudgedCoverage@10']
    positive=md(examples[examples.direction=='positive'][excols],0)
    negative=md(examples[examples.direction=='negative'][excols],0)
    report=f'''# 第一版研究成果：相同商品資訊，是否造成不同曝光？

完成日期：2026-09-04。這是實際執行的 WANDS Phase 1 檢索實驗，不是模擬結果，也不是完整購物代理實驗。

## 先看研究判斷

{notes['decision_zh']}

{notes['key_findings_zh']}

## 這版實際完成的範圍

完整商品庫 **{meta['products']:,} 件商品、480 個查詢**；六種主表示 × BM25、Dense、Hybrid，共 18 組主實驗；另加重複原文及僅反轉屬性順序兩種對照，共 **24 組全商品庫實驗**。所有改寫不接收查詢、不讀相關性標籤，不使用付費 API。Dense 使用固定版本 MiniLM；全文分段編碼，不截掉後半段。

原始標註 {meta['raw_judgments']:,} 列，清除重複並排除 14 組衝突配對後，留下 **{meta['judgments']:,} 組唯一配對**，其中 **{meta['labels']['Exact']:,} 組 Exact**。ExactHit 的分母為 379 個有 Exact 標註的查詢；cNDCG、Recall 與 MRR 有效分母為 479 個至少有一件已知相關商品的查詢。

六種表示是確定性原型：R0 原文；R2 原文加一般購物邀請語；R3 加事實欄位句型，仍保留原描述；R4 保留相同詞彙、將屬性分隔符改為條列；R5 加欄位標籤、按屬性字串排序並保留原描述；R8 串接 R0 與 R5。R5 是有限的正規化原型，沒有將所有黏連的屬性名稱語義拆詞，也未推定單位。R2/R3 不等於完整的行銷／事實語氣改寫。

## 最重要的比較：Dual View 相對原文

下表是 cNDCG@10 的配對差值；正值代表 Dual View 較好。95% 信賴區間來自 10,000 次以查詢為單位的配對 bootstrap；另做 10,000 次 permutation test。三個主要檢索器比較採 Holm 校正。這是探索性 pilot，未對外預註冊。

{md(primary)}

**cNDCG 是 judged-condensed NDCG，不是完整商品庫前十名的標準 NDCG。** 未標註商品先移出評估排序，避免直接當作不相關；因此還必須看原始 Top-K 的已標註覆蓋率及 known-relevant recall。本文不將 cNDCG 分數與一般 NDCG leaderboard 直接比較。

![配對效果與信賴區間](figures/paired_effects.png)

## 相關商品被找回，也可能被隱藏

下表只計算 WANDS 已標註為 Exact 的配對。rescued：原文排名 >20、Dual ≤20；harmed：原文 ≤20、Dual >20。兩者採同一分母 25,470。百分比欄位為 0–1 比率；net_macro 是先算各查詢的淨變化比例再取平均。

{md(dual)}

{notes['hidden_interpretation_zh']}

![Dual View 曝光得失](figures/hidden_products.png)

## 長度與詞彙控制

**C_order** 只反轉原始屬性順序。42,994 件商品逐一確認：字元數、BM25 詞彙多重集合及模型 WordPiece 多重集合完全不變。原值、否定資訊與同鍵多值均保留。BM25 全部查詢—商品分數與排名完全不變，提供實作負對照。

{md(c)}

{md(order)}

**C_repeat** 是重複兩次原文。下表是 Dual 減去 C_repeat；這是重複與近似長度控制，並非每件商品都嚴格等長。

{md(controls)}

{notes['control_interpretation_zh']}

## 全部實驗數值

Recall 的分母只涵蓋已標註 Exact/Partial 商品；MRR 保留完整商品庫順位；ExactHit 是有 Exact 標註查詢的命中率；JudgedCoverage 是真正 Top-10 中有標註的比例。所有欄位完整信賴區間另存於 [confidence_intervals.csv](tables/confidence_intervals.csv)。

{md(a[maincols])}

![表示方式與檢索器](figures/representation_heatmap.png)

## 資料與忠實度檢查

{conflicts:,}/{meta['products']:,} 件商品存在同名屬性的多個不同值；{malformed:,} 件含空白或不完整屬性鍵。這些未被擅自修復。同鍵多值可能是合法的安裝位置或兼容尺寸，也可能來自多款式混合，不能一概當作資料錯誤，也不能當作已驗證的單一規格。

每個生成文字都核對來源片段、額外數字、非模板殘餘詞與確定性生成結果。全部 {len(fidelity):,} 個表示通過此限定範圍的來源保留檢查。這不是外部實體商品真實性驗證，也不是可用於任意 LLM 改寫的通用語義 auditor。評分／評論欄位只做資料檢查，所有檢索條件一致不使用，以避免新增資訊來源。

{md(fields[['field','missing','missing_pct','unique_nonmissing']])}

## 20 個最強正向案例

從主實驗中選擇跨越 Top-20 邊界且順位改善最大的 20 組不同查詢—商品配對。以下由本次研究助理逐例閱讀來源後註解，**不是新增的獨立人工 relevance 標註**。極端案例不代表平均效果；完整原文與屬性見 [positive_20.md](qualitative/positive_20.md)。

{positive}

## 20 個最強負向案例

採同樣規則，選擇跨出 Top-20 且下降最多的配對。完整來源見 [negative_20.md](qualitative/negative_20.md)。

{negative}

## 是否值得繼續，以及下一版要補什麼

{notes['next_steps_zh']}

這版只能宣稱來源等價條件下的全商品庫檢索曝光敏感性。不能宣稱購物代理推薦偏誤已被驗證，不能把 Exact 等同客觀最佳商品，也不能把整個商品庫同時改寫的結果當成單一商品改寫的獨立因果效應。Hybrid 共享同一 Dense 模型，並不代表兩個獨立神經模型的重現。

最新版相關工作與研究定位見 [RESEARCH_POSITIONING.md](../paper/RESEARCH_POSITIONING.md)。[E-GEO v2](https://arxiv.org/abs/2511.20867v2) 已有跨引擎改寫分析，[SAGEO Arena v2](https://arxiv.org/abs/2602.12187v2) 已處理完整檢索流程；本研究需要以全商品庫、忠實度與雙向曝光分析形成區別。

## 可交付檔案

* [英文論文初稿](../paper/PAPER_DRAFT.md)
* [完整數值附錄](NUMERICAL_APPENDIX.md)
* [實驗設定與重跑方法](../README.md)
* [配對統計表](tables/paired_comparisons.csv)
* [逐查詢結果](per_query/)
* [資料、套件與程式版本紀錄](manifest.json)
'''
    (out/'REPORT_zh.md').write_text(report,encoding='utf-8')
    paper=f'''# Same Product Information, Different Visibility: A Catalog-Wide Study of Representation Sensitivity in E-Commerce Retrieval

**First empirical manuscript draft — exploratory Phase 1, September 4, 2026.** Authors and affiliations to be supplied. This is not a submission-ready claim of shopping-agent recommendation effects.

## Abstract

We investigate whether product visibility depends on how unchanged source information is serialized. Using the complete WANDS catalog of 42,994 products and 480 queries, we evaluate six deterministic source-preserving representations under BM25, a chunk-pooled MiniLM dense retriever and full-list reciprocal rank fusion. Two additional controls isolate source repetition and attribute order. After deduplication and exclusion of contradictory judgments, evaluation uses 231,859 unique query-product labels, including 25,470 Exact pairs. We report judged-condensed NDCG, known-relevant recall and bidirectional Top-20 exposure transitions without labeling unknown products as irrelevant. {notes['abstract_result_en']} The results distinguish representation sensitivity from successful relevance improvement and motivate provenance-preserving evaluation before making end-to-end claims about shopping agents.

## 1. Introduction

Product discovery depends on the interface between product information and retrieval systems. A title, merchant description, taxonomy and attribute feed can describe the same source record in different ways. A shopping system's candidate set may consequently depend on serialization choices as well as product relevance. We ask whether source-equivalent catalog representations change the exposure of products judged relevant by humans.

Our focus is the retrieval stage. A downstream shopping assistant cannot choose an absent candidate, but improved retrieval does not guarantee improved recommendations. We therefore distinguish the motivation of agentic commerce from the measured behavior of retrieval models. We also distinguish exposure sensitivity from a successful mitigation: substantial movement can coexist with little or negative aggregate relevance change.

The study contributes a reproducible whole-catalog pilot, explicit accounting of both recovered and newly hidden Exact products, and deterministic controls for formatting, repetition and attribute order. It does not claim a new universal representation method or exhaustive coverage of language styles.

## 2. Related work and positioning

[E-GEO v2](https://arxiv.org/abs/2511.20867v2) studies product-description rewriting across generative engines with fixed candidate sets. [SAGEO Arena v2](https://arxiv.org/abs/2602.12187v2) extends visibility evaluation across retrieval, reranking and generation. Document-side modification itself is established prior art, including [Document Expansion by Query Prediction](https://arxiv.org/abs/1904.08375). Our narrower emphasis is source-preserving universal catalog treatment, human relevance labels, and both directions of exposure change. A broader novelty review remains necessary before submission.

## 3. Problem formulation

Let C be a catalog, F_p the source record for product p, Y(q,p) its fixed human relevance label, and r a query-independent serializer. Applying r to every product yields ranking L(q,r(C)). Define V_K(q,p;r)=1[rank(q,p;r(C))≤K]. A change in visibility under different source-preserving r indicates representation sensitivity at the catalog level.

Universal treatment changes competitors and corpus statistics together. These effects are not the isolated direct causal effect of rewriting only one product. We use **source-preserving** rather than assuming all catalog statements are correct: the data contain multiple values under the same attribute key, missing fields and malformed entries.

For Exact-labeled pairs, RelevantHidden@K is the fraction with original rank >K and alternative rank ≤K. ReverseHidden@K reverses those inequalities. Their shared denominator is all 25,470 Exact pairs. We also calculate query-macro net transition rates. Rank span is maximum minus minimum rank, rank ratio is maximum divided by minimum rank, and HitInstability@K records whether a pair crosses the boundary across representations.

## 4. Experimental design

### 4.1 Data

We use [WANDS](https://github.com/wayfair/WANDS), pinned to commit `{meta['dataset_commit']}`. The release contains 233,448 annotation rows. We collapse identical duplicate pair judgments and exclude 14 pairs with contradictory judgments, producing 231,859 unique pairs: 25,470 Exact, 145,670 Partial and 60,719 Irrelevant. The catalog and all 480 queries are retained. Only 379 queries have Exact judgments, and 479 have at least one known relevant product. Queries average 3.38 whitespace-delimited words; findings should not be generalized directly to long conversational requests.

We inspect every product field. Descriptions are absent for 13.97% of products, product class for 6.63%, category hierarchy for 3.62%, and rating/review fields for 21.98%. There are {conflicts:,} records with multiple distinct values under at least one nonempty normalized key, and {malformed:,} with malformed or blank feature keys. Multiple values may reflect product variants; the pilot does not resolve them.

### 4.2 Representations and fidelity

R0 concatenates original name, class, category, description and features. R2 adds generic shopping invitations. R3 introduces factual field framing but retains the source description. R4 replaces feature separators with bullets while preserving the BM25 token multiset. R5 labels fields, sorts raw feature entries and places the unchanged description after attributes. R8 concatenates R0 and R5. C_repeat duplicates R0. C_order reverses raw attribute entry order without changing characters or token multisets.

These are conservative serialization prototypes. R2 is not a comprehensive persuasive rewrite, R3 does not remove source marketing language, and R5 does not infer units or fully segment fused attribute keys. All conditions use identical source fields; ratings are excluded uniformly. The generator sees no queries or labels. Auditing checks source-atom retention, novel numeric values, unapproved residual words and deterministic output equality. All {len(fidelity):,} generated records pass these restricted checks. This verifies the transformations, not real-world truth or semantic equivalence of arbitrary paraphrases.

### 4.3 Retrievers

BM25 uses lowercased alphanumeric tokens, k1=1.2 and b=0.75, no stemming, and a fresh catalog-specific index per representation. Unique query terms are accumulated in sorted order. Dense retrieval uses the fixed `sentence-transformers/all-MiniLM-L6-v2` revision in the configuration. It is a compact general-purpose baseline. All WordPieces are encoded in non-overlapping 254-token chunks with special tokens added; chunk means are weighted by content length, averaged and L2-normalized. The implementation uses FP16 model weights and FP32 pooling on an RTX 4060-class GPU. It does not apply the model's default first-256-token truncation. For R0, 93.36% of products exceed 254 content WordPieces.

Hybrid retrieval uses equal-weight reciprocal rank fusion over complete lists, with constant 60. Full lists permit uncensored ranks for every Exact pair. Ties are resolved by ascending product ID, and boundary ties are reported. No model, hyperparameter or representation is optimized against WANDS relevance judgments.

### 4.4 Metrics and statistics

Unknown products remain unjudged. Following the distinction in [incomplete-assessment evaluation](https://link.springer.com/article/10.1007/s10791-008-9059-7), we report **judged-condensed NDCG** (cNDCG): unjudged products are removed before rank-based discounting, using gains 3, 1 and 0. This is not ordinary full-catalog NDCG and may be optimistic under incomplete pooling. Judgment coverage accompanies it.

Known-relevant Recall@K uses Exact and Partial judgments and full-catalog ranks. MRR is the reciprocal full-catalog rank of the first known relevant item. ExactHit is the fraction of Exact-bearing queries with an Exact result in top K. ExplicitIrrelevant@10 and Unjudged@10 are kept separate. Metrics with no positive judgments are undefined and excluded, rather than assigned a misleading success or failure.

We use 10,000 query-paired bootstrap replicates and paired sign-permutation tests with seed 20260903. The designated primary comparisons are R8 versus R0 on cNDCG@10, with Holm correction across the three retrievers. Other analyses are exploratory. All queries are used to measure this pilot; any subsequently selected method requires held-out evaluation. The study was not externally preregistered.

## 5. Results

### 5.1 Catalog-wide relevance

{md(a[maincols])}

Table 1. Complete results, including two controls. Recall measures recovery of known positives, not true recall over the entire incompletely judged catalog. ExactHit uses 379 eligible queries. cNDCG, Recall and MRR use 479 queries with known positives.

{notes['results_en']}

{md(primary)}

Table 2. Dual View minus Original on cNDCG@10; paired 95% bootstrap confidence intervals and permutation p values. Holm-adjusted values apply to the three designated primary comparisons.

### 5.2 Exposure redistribution

{md(dual)}

Table 3. Exact-pair Top-20 transitions under Dual View. Rescue and harm share a denominator of 25,470 pairs. Macro net rates average within query; their confidence intervals resample queries, not independent product pairs.

{notes['hidden_en']}

![Bidirectional Top-20 transitions](../results/figures/hidden_products.png)

### 5.3 Controls

{md(c)}

{md(order)}

Table 4. Order-only control: relevance changes and exposure transitions. Character counts, lexical token multisets and WordPiece multisets are verified identical for every product. Whole-catalog BM25 score and rank equality is verified separately.

{md(controls)}

Table 5. Dual View minus repeated Original. Repetition approximates length but is not a strict per-product length match. C_order provides the stricter length-and-content control.

{notes['controls_en']}

### 5.4 Qualitative evidence

We select 20 strongest improvements and 20 strongest degradations crossing the Top-20 boundary from the main representations, deduplicating query-product pairs. Original descriptions, features and ranks are retained in the qualitative appendix. The source inspection was conducted by the research assistant producing this pilot, not by an independent human annotation panel. Extreme cases are not representative samples and do not establish population-level mechanisms.

{notes['qualitative_en']}

## 6. Discussion

{notes['discussion_en']}

The appropriate unit of interpretation is the information environment. A product's exposure gain alone does not demonstrate that users see more relevant results. The bidirectional transition analysis and the repetition control guard against that inference. Standardizing representations can also enforce invariance to a restricted transformation class without improving mean relevance; these are separate properties to test.

## 7. Limitations and next experiments

This pilot uses one catalog, short queries, incomplete judgments and one neural encoder. Hybrid shares its dense component and is not an independent cross-model replication. cNDCG changes ranking semantics by removing unjudged items. Raw attribute feeds may mix variants; a human Exact label does not verify all source statements or certify the objectively best product. Deterministic wrappers do not establish results for arbitrary marketing language, factual prose, schema markup or LLM paraphrases. Full-list fusion and chunk pooling are specific design choices, and no reranker or shopping agent is evaluated.

A next study should replicate the order-only effect with a second strong neural retriever, compare single-product and universal interventions, obtain relevance judgments for newly exposed candidates, and separate normalization effects from text length and repetition on a held-out query set. A learned factual product compiler must be evaluated on both preservation and relevance. Expensive shopping-agent experiments should follow evidence that survives these checks.

## 8. Conclusion

{notes['conclusion_en']}

## Reproducibility

The repository contains deterministic serializers, sparse BM25, a configurable dense encoder, full-list fusion, per-query evaluation, Exact-pair ranks, 10,000-replicate uncertainty estimates, fidelity records, selected source examples and PNG/SVG figures. The manifest records dataset checksums, model revision, package versions, seeds and code hashes. No paid API calls or generated experimental numbers are used. See [README](../README.md), [numerical appendix](../results/NUMERICAL_APPENDIX.md) and [Chinese research report](../results/REPORT_zh.md).

## References

1. Chen et al. (2022). *WANDS: Dataset for Product Search Relevance Assessment*. ECIR. [Dataset and citation](https://github.com/wayfair/WANDS).
2. Bagga et al. (2026 version). *E-GEO: A Testbed for Generative Engine Optimization in E-Commerce*. [arXiv:2511.20867v2](https://arxiv.org/abs/2511.20867v2).
3. Kim et al. (2026). *SAGEO Arena: A Realistic Environment for Evaluating Search-Augmented Generative Engine Optimization*. [arXiv:2602.12187v2](https://arxiv.org/abs/2602.12187v2).
4. Nogueira et al. (2019). *Document Expansion by Query Prediction*. [arXiv:1904.08375](https://arxiv.org/abs/1904.08375).
5. Sakai and Kando (2008). *On information retrieval metrics designed for evaluation with incomplete relevance assessments*. [Publisher record](https://link.springer.com/article/10.1007/s10791-008-9059-7).
6. Sentence Transformers. *all-MiniLM-L6-v2 model card*. [Model documentation](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).
'''
    (ROOT/'paper/PAPER_DRAFT.md').write_text(paper,encoding='utf-8')
    print('Chinese report and English manuscript created from final tables.')

if __name__=='__main__': main()
