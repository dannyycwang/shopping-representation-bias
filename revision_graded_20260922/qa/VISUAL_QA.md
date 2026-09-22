# Visual QA

Both standalone PNG figures were opened and inspected after generation. Panel titles, axis labels, six-schedule ECDF legend, threshold explanation, three-comparator interval legend and uncertainty caption are legible, without overlap or clipping. SVG and PDF exports use the same Matplotlib figure geometry. Editable sources are `scripts/report.py` and the complete underlying CSV/Parquet data.

The ECDF includes all six source-order contrasts for each dataset/encoder, uses a linear region near zero and a logarithmic scale above .001, retains zero-change observations, and shows the entire 0–1 nDCG-difference range. The forest plot retains every canonical rule and all three comparators; its x axis is the original 0–1 nDCG scale.
