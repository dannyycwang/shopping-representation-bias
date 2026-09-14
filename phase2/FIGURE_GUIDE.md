# Publication Figure Guide

All publication figures are exported as 300-dpi PNG and editable vector PDF in `results/publication_figures/`. Each figure has a source CSV in `results/publication_figures/plotting_data/`, so the plots can be redrawn in R, Illustrator, or the final ACM LaTeX workflow without reading experiment-internal files.

| Figure | Main claim | Final-paper role | Redraw notes |
|---|---|---|---|
| `figure_1_sensitivity` | Equivalent records cross top-20 boundaries in both datasets and all models | Main result, early in paper | Preserve query-bootstrap intervals, filled/open dataset encoding, and log rank-range scale |
| `figure_2_controls` | Chunking, overlap, pooling, and long-context evidence do not reduce the result to truncation | Mechanism/control section | Native column should remain visually marked; GTE long-context evidence belongs in caption/text because it has no matched profile sweep |
| `figure_3_mitigation_tradeoff` | Stability and relevance must be evaluated jointly | Main mitigation result | Keep paired panels and zero lines; do not label M1 reductions as confirmed when Holm-adjusted VI tests are nonsignificant |
| `figure_4_direct_vs_full` | Target-only and catalog-wide transformations estimate different effects | Causal interpretation | Circle=same target only; square=full catalog; connected points share model and mitigation |
| `figure_5_query_types` | Instability extends beyond brand/style proxies | Scope analysis | Add per-row query counts in the camera-ready version; retain warning that long/descriptive WANDS has only 10 queries |

The figure hierarchy follows common SIGIR/Web Conference presentation practice: one claim per figure, uncertainty on primary effects, direct visual comparison across models and datasets, and captions that state the estimand. The plotting script uses a color-blind-safe palette, vector fonts, readable model names, and no decorative 3-D effects.

The older exploratory figures remain under `results/phase2_figures/` for auditability. Use the publication figures in the manuscript.

