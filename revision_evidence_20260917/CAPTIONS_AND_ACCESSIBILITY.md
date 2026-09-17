# Captions and accessible descriptions

## Figure 2: intervention and inclusion states

**Caption.** Schematic illustration. (A) Catalog-wide changes permute intact attribute entries for every product; single-target changes replace one product’s source-order entry while all competitors retain raw C0 representations. Non-attribute text, entry contents and section placement are fixed. Different targets define separate indexes. (B) Over a finite family, inclusion rows [1,1,1], [1,0,1] and [0,0,0] illustrate persistent inclusion, cutoff crossing and persistent omission. These are schematic states, not measured frequencies.

**Accessible description.** Two diagrams compare the same three product records before and after reordering. In the catalog-wide diagram every record’s attribute blocks move. In the target-only diagram only the target blocks move; both competitor records stay fixed. Below, three rows of inclusion indicators show always included, sometimes included and always omitted. A value of one means rank at or above the candidate cutoff in the sense rank <= K; zero means rank > K.

## RQ2: relevant-product membership changes

**Caption.** Newly included and newly omitted highest-relevance products for each of the six raw C0-relative attribute-order contrasts. Each fraction is computed within query, then macro-averaged on identical support. Losses extend left; gains extend right; diamonds show net Recall change, with descriptive 95% paired query-bootstrap intervals (10,000 draws, seed 2026091701; uncorrected). WANDS uses 308 eligible queries from the existing held-out split and 21,299 Exact pairs; ESCI uses 499 eligible queries and 4,434 E pairs from its existing evaluation sample. PDF pages correspond to K=20 and K=100. The companion table reports exact nonzero within-query gain/loss equality and both eligible-query and changed-query denominators.

**Accessible description.** Six panels on each PDF page cross two datasets with three encoders. Every panel has six horizontal matched bars for C1 and C2s1 through C2s5 versus C0. All panels share the same horizontal scale. Teal bars show gains and rust bars show losses. A black diamond and interval show the net change. Gains and losses may both be much larger than their difference. The combined SVG contains both cutoff pages vertically; separate K-specific SVGs are provided for manuscript placement. Exact values and cancellation denominators are in the plotting and companion CSVs.

## RQ1: target-only full and fully-fitting support

**Caption.** Persistent inclusion (Always), cutoff crossing (VI) and persistent omission (Never) among highest-relevance targets, under independent target-only interventions against the full raw C0 competitor catalog. Full and fully-fitting target populations are shown for each encoder; fitting requires the complete serialized input, including special tokens, to fit under all seven schedules at the encoder’s own cap. All three states use the same query-macro aggregation and sum to 100%. Query and pair denominators are printed beside each bar. WANDS uses its existing held-out split and ESCI its existing evaluation sample. Differences between full and fitting populations are descriptive, not causal truncation effects.

**Accessible description.** Four panels cross dataset and cutoff. Each encoder has two horizontal stacked bars: all evaluated targets and targets fitting every tested order. Teal is Always, rust is VI and light gray is Never. The complete competitor catalog stays fixed in every bar. Query and highest-relevance pair counts identify the changing analyzed target support. Detailed macro/micro estimates and descriptive intervals accompany the figure in CSV files.

## Typography and source data

Figures use vector Matplotlib drawing and the manuscript’s Linux Libertine outlines, locally converted to TrueType for reliable PDF embedding. SVG glyphs are vector paths with an accessible title and description; PDFs retain embedded vector fonts. No generated raster imagery is used. Main figure width is 7.1 inches. Rendered 150-dpi manuscript-width proofs are kept under `qa/`.

Empirical plotting inputs: `data/rq2_membership_plot_data.csv`, `data/rq2_cancellation_companion.csv`, and `data/rq1_target_fully_fitting_plot_data.csv`. Schematic block identities and inclusion states are specified directly in `scripts/plot_figures.py`.
