# Submission Readiness: Phase II

## Decision

**Continue toward a full The Web Conference / SIGIR submission.** The evidence is stronger than an idea-only paper: it has a frozen confirmatory protocol, two datasets, three encoder families, query-level inference, mechanism controls, direct interventions, and a mitigation result with a clear tradeoff. The current package is a strong first full-paper draft, not yet camera-ready.

## Strongest evidence

- All six primary dataset-model cells pass the preregistered visibility-instability gate.
- Highest-relevance micro VI@20 is 9.41–13.26% on WANDS and 4.42–8.25% on ESCI.
- WANDS median rank ranges are 81–207 positions; ESCI still shows threshold crossings with smaller rank ranges.
- Four token-budget/pooling controls retain the effect, and GTE-ModernBERT supplies a long-context control.
- WANDS Exact products are significantly more unstable than Irrelevant products for all models.
- Canonicalization exposes a publishable negative result: perfect stability can significantly reduce relevance.
- Direct target and full-catalog interventions distinguish two causal estimands.

## Claims to avoid

- Do not claim that the system fails to recommend the objectively “best” product. Labels measure query relevance.
- Do not call the result seller fairness or realized exposure without impression, seller, and outcome data.
- Do not describe M1 as a robust universal mitigation; its VI reductions are mostly small and nonsignificant.
- Do not treat zero VI from M2 as sufficient evidence of improvement; it is mechanically induced and can harm cNDCG.

## Highest-value work before submission

1. Conduct a blinded manual audit of the automatic query taxonomy and report agreement.
2. Add one strong sparse or late-interaction baseline if compute permits. This would clarify whether the phenomenon is specific to single-vector dense encoding.
3. Run a systematic literature review and replace the working bibliography with 25–35 verified, directly relevant sources.
4. Convert the draft to the current ACM proceedings template and measure the eight-page main-text budget.
5. Add a compact qualitative case study with the exact preserved fields, rank trajectory, and a short error analysis.
6. Release a minimal regeneration command and artifact card after anonymizing paths and metadata.

## Venue fit

The strongest Web Conference framing is web shopping infrastructure and search reliability: invisible serialization choices alter the retrieval opportunity of relevant products. SIGIR is also a natural fit if the paper foregrounds the diagnostic metric, retrieval analysis, and robust IR contribution. CIKM is a credible fallback with the present experimental breadth. Acceptance remains uncertain because novelty must be established against the full robustness, neural IR diagnostic, and product-retrieval literature.

