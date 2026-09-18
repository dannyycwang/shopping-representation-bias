# Draw.io visual verification

All seven final diagrams were decoded, rendered to SVG/PNG, and re-encoded using the official draw.io GraphViewer engine on 2026-09-18. Proof scale was 1.5625, equivalent to 150 dpi at the file's 96-units-per-inch page geometry. Only outer whitespace was cropped in the proof exports.

The seven latest PNG proofs were visually inspected. Native text baselines were adjusted for draw.io's five-unit text inset so letters, bar values, table rows, and markers remain aligned. No remaining text overlaps, clipping, missing markers, lost error bars, or broken glyphs were observed.

- Figure 1: rank 11 / 939, complete-input note, same colored entries, and post-hoc label remain clear.
- Figure 2: all intact blocks, fixed competitors, arrows/equalities, and the three inclusion states render correctly.
- RQ1 main: all 12 estimates/intervals, both marker types, counts, and separated GTE points are present.
- RQ1 appendix: all 24 VI values, including 7.16%, remain visible beside the stacked bars.
- RQ2 Top-20 and Top-100: six panels and all six comparisons per panel retain native bars, diamonds, interval lines, and common scales.
- Cancellation table: all six rows, denominators, and ranges are readable; rules clear the text.

Text, styles, and geometry are compared against the actual draw.io model round-trip in `app_validation.json`. Each combined-file page is also compared with its individual file. The figures consist of native text/shape/edge cells, with no embedded raster/SVG images. All source data and previous PDF/SVG deliverables remain unchanged.

Times New Roman is intentionally used for portable editing instead of the original custom Libertine font. The native files retain the original point sizes. Proof PNG/SVG and round-trip intermediates are excluded from Git and the ZIP; regenerate them with `validate_in_app.py` when changing a diagram.
