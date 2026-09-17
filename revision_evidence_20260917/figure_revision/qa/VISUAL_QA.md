# Final visual QA

Reviewed 2026-09-17 after final plotting. Each final PDF was rendered with Poppler at 150 dpi and inspected at its native manuscript width: 1,065 px for 7.1 inches, 518 px for 3.45 inches. The reviewed PDF hashes are in `validation_report.json`; fonts, page size, zero raster images, and page count were also checked programmatically.

| Figure | Native size (in) | Minimum font | Visual review |
|---|---|---:|---|
| Figure 1 candidate | 3.45 x 3.90 | 8 pt | Same four complete entries share colors across two independent cards. Both ranks and cutoff states are clear. Excerpt/full-record/token-count/post-hoc labels are readable. No photo or temporal arrow. |
| Figure 2 schematic | 7.10 x 3.35 | 8.5 pt | All intact blocks remain present; only the target moves in the right panel. Equals signs identify unchanged competitors. Three state rows and the rank inequality render correctly. |
| RQ1 Top-20 VI main | 7.10 x 3.25 | 8.5 pt | All 12 points and intervals, numeric VI values, and counts are readable. Open triangles and filled circles distinguish supports. Identical GTE estimates are separated vertically and footnoted. |
| RQ1 all-state appendix | 7.10 x 5.50 | 8 pt | All 24 VI values have a separate numeric column, with clear space after the bar; 7.16% is explicit. Counts, legends, and both cutoffs are legible. |
| RQ2 Top-20 main | 7.10 x 4.40 | 8.5 pt | Six panels and 36 comparisons retained; bars, diamonds, and interval caps fit the shared scale. Percent and percentage-point units are clear. |
| RQ2 Top-100 appendix | 7.10 x 4.40 | 8.5 pt | Six panels and 36 comparisons retained on the same scale as Top-20. Small ESCI bars remain visible; no rescaling disguises their size. |
| Cancellation main table | 3.45 x 2.10 | 8 pt | All six ranges and denominators are readable. Horizontal rules clear the text, and the min-max/not-CI note is visible. |

Corrections during review: moved appendix VI numbers farther outside bars; separated RQ1 panel headings from the legend; replaced unchanged-competitor arrows with equals signs; centered table body text between rules. The final render has no observed overlapping text, clipped labels, broken glyphs, or obscured legends. The small table and candidate were inspected at single-column width, not enlarged double-column width.

`latex_check.json` separately verifies two-pass compilation of the supplied insertion snippets, native main/appendix tables, and shared setup at 7.1-inch text width and 3.45-inch column width, with no overfull boxes, oversized floats, or missing-glyph warnings. This is an isolated syntax/fit check, not a manuscript edit.

PNG and LaTeX proof intermediates are excluded from Git and the delivery ZIP. Regenerate them using the README commands when editing a figure. Only the seven standalone vector figure PDFs in `figures/` are final PDF deliverables.
