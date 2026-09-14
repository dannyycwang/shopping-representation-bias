# Query Taxonomy Decision

## Decision

The query taxonomy is **not used as a main-paper claim**. The current labels are deterministic lexical heuristics rather than blinded human annotations, so they are suitable for exploratory and appendix analysis only.

## Why

The central causal claim does not require semantic query classes: every experiment holds the query, product identity, facts, candidate set, and retriever fixed while changing only the order or surface form of product attributes. Visibility changes can therefore be measured directly at the query--product level.

A main-text statement such as “broad queries are more sensitive than attribute-seeking queries” would require an annotation protocol, multiple blinded annotators, agreement statistics, and an adjudication rule. Those data do not yet exist. Presenting automatic categories as ground truth would weaken an otherwise controlled study.

## What remains usable

The automatic categories may be shown in an appendix as hypothesis-generating evidence, with an explicit “heuristic” label and no causal interpretation. The released query-level tables retain query identifiers, so a later human audit can be added without rerunning retrieval.

## Publication implication

The paper is organized around four claims that are already supported without taxonomy labels:

1. fact-equivalent product serializations change dense-retrieval visibility;
2. the effect appears across two product-search benchmarks and four retriever families;
3. a downstream reranker reduces rank variability among retained candidates but cannot recover first-stage omissions; and
4. permutation-invariant attribute aggregation removes the measured ordering effect with a small, statistically inconclusive relevance change for the selected method.

