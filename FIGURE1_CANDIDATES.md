# Figure 1 Candidate Audit

All candidates are WANDS Exact pairs, cross Top-20 under the frozen seven-member primary family, and retain the same fact/token multiset.

## Verified specified cases

- `turquoise chair`, product 34536: MiniLM ranks C0=486, C1=295, C2s1=1303, C2s2=21, C2s3=484, C2s4=3, C2s5=4. BGE is rank 1 for every variant; GTE ranges 1–6. The 3↔1303 illustration is verified.
- `e12/candelabra`, product 7756: MiniLM C0=14 and C1=1881; the simple original-versus-reverse contrast is verified. Other models do not cross Top-20.

## Selection rule

Candidates are ordered by number of models showing a Top-20 crossing, then query length and largest rank range. The two prespecified examples are retained at the top for explicit audit. Full ranks and metadata are in `figure1_candidates.csv`.

## Recommendation

Use `turquoise chair` as Figure 1 because the fact-equivalent MiniLM swing is extreme and the same product remains highly ranked and stable under BGE/GTE, making model dependence visible. Use `e12/candelabra` as an appendix case because it uses the especially simple original-versus-reverse intervention.