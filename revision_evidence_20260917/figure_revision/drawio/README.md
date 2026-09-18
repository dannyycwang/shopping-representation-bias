# Editable draw.io figures

Open `all_seven_figures.drawio` to get all seven figures as separate pages in one file. The seven matching single-figure `.drawio` files are also included.

| Page | File |
|---|---|
| Figure 1 candidate | `figure1_teal_chair_candidate.drawio` |
| Figure 2 schematic | `figure2_intervention_and_states.drawio` |
| RQ1 main: Top-20 VI and intervals | `rq1_vi_top20_main.drawio` |
| RQ1 appendix: all three states, both cutoffs | `rq1_states_top20_top100_appendix.drawio` |
| RQ2 main: Top-20 membership changes | `rq2_membership_top20_main.drawio` |
| RQ2 appendix: Top-100 membership changes | `rq2_membership_top100_appendix.drawio` |
| RQ2 exact cancellation table | `rq2_cancellation_top20_table.drawio` |

Text, bars, blocks, point markers, diamonds, interval lines, and axes are native editable draw.io objects. Double-click a text label to edit it; select a shape to change its position, size, color, or line styling. These files contain no flattened PNG, embedded SVG image, or outlined-text substitute. Across the seven pages there are 1,297 editable objects, including 469 text objects. Some wrapped labels use separate text objects for each displayed line.

Page sizes match the original 3.45-inch / 7.1-inch figures, using 96 drawing units per inch. Native text sizes preserve the original point sizes (at least 8 pt). Times New Roman is used for reliable desktop/browser editing without installing the manuscript's custom font, so letter shapes differ slightly from the supplied Libertine PDF/SVG figures. Figure content, numerical estimates, confidence intervals, colors, scales, and geometry come from the same existing plotting functions and verified CSV/JSON inputs.

The original PDF/SVG files, data, manuscript, and delivery ZIP remain unchanged. No statistical analysis, model inference, training, or additional permutations are performed. Draw.io shapes are editable graphics, not a live statistical chart: if numerical results need changing, regenerate from the verified source data rather than moving bars or confidence intervals by eye.

## Reproduce

Run from the repository root with the original figure revision package present:

```powershell
python revision_evidence_20260917/figure_revision/drawio/build_drawio.py
```

The custom Matplotlib renderer converts the original figure functions directly to native mxGraph cells. It replaces only the in-memory export callback and never calls the original PDF/SVG writer or original `main()` function. It checks that existing deliverables remain byte-identical.

`qa/build_validation.json` records object counts, all editable text, original-file preservation, source-script hash, and minimum font sizes.

For rendering with the official draw.io application, run:

```powershell
python revision_evidence_20260917/figure_revision/drawio/validate_in_app.py
```

Then open `http://127.0.0.1:8767/`. The local proof host loads the official draw.io GraphViewer bundle, renders each diagram and round-trips its native model, saves proofs under `qa/`, and leaves the source diagrams unchanged. It needs internet access to load the renderer. Stop the server with Ctrl+C after verification. The optional `/embed` route also supports the [draw.io editor load/export protocol](https://www.drawio.com/docs/reference/embed-mode/); the file structure follows the [native diagram format](https://www.drawio.com/docs/reference/diagram-generation/).

After inspecting the regenerated proofs, run `python revision_evidence_20260917/figure_revision/drawio/finalize_drawio.py` to verify the model round-trip and rebuild the delivery ZIP and manifest.

The ZIP contains all eight `.drawio` files, these instructions, the regeneration/validation scripts, and validation reports. Opening or editing the diagrams does not require Python or the research repository.
