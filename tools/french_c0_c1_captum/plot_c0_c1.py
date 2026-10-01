"""Plot verified Captum C1-minus-C0 results; never substitute older shuffle IG."""
from pathlib import Path
import argparse
import csv
import json
import math
import xml.etree.ElementTree as ET
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from native_drawio import DrawioRenderer, sha

BLUE, ORANGE, GREEN, RUST = "#24618C", "#C27328", "#147D70", "#AC533C"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results", type=Path)
    args = ap.parse_args()
    root = args.results
    report = json.loads((root / "run_report.json").read_text(encoding='utf-8'))
    if report.get("status") != "passed" or not report.get("plot_ready"):
        raise SystemExit("A successful C0/C1 Captum report is required.")
    if report.get("contrast") != "C1 minus C0":
        raise SystemExit("Wrong contrast; do not reuse C2s3/C2s2 values.")
    rows = list(csv.DictReader((root / "aligned_entry_changes.csv").open(encoding='utf-8')))
    expected = [(s, b) for s in ("C0", "C1") for b in ("zero", "pad")]
    if sorted((r["schedule"], r["baseline"]) for r in report["checks"]) != sorted(expected):
        raise ValueError("Expected four C0/C1 x zero/PAD checks")
    attr_ids = [f"attr_{i:03d}" for i in range(10)]
    labels = ["Width: 9.38", "Product weight: 0.19", "Product type: wall nice/cove",
              "Molding use: corners", "Height: 9.38", "Material: urethane",
              "Color: off-white", "Composite wood product: no", "Installation required: yes",
              "Country of origin: China", "Fixed text / boundaries"]
    series = {}
    for baseline in ("zero", "pad"):
        subset = {r["entry_id"]: r for r in rows if r["baseline"] == baseline}
        if not all(k in subset for k in attr_ids):
            raise ValueError("Missing complete attribute occurrence")
        for row in subset.values():
            if not math.isclose(float(row["C1"]) - float(row["C0"]),
                                float(row["C1_minus_C0"]), abs_tol=1e-12):
                raise ValueError("Signed difference mismatch")
        series[baseline] = [float(subset[k]["C1_minus_C0"]) for k in attr_ids] + [
            math.fsum(float(r["C1_minus_C0"]) for k, r in subset.items() if k not in attr_ids)]
    plt.rcParams.update({"font.family": "Arial", "font.size": 8,
                         "axes.labelsize": 8, "xtick.labelsize": 7.5,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
    fig = plt.figure(figsize=(7.05, 3.65))
    left = fig.add_axes([.145, .260, .277, .53])
    right = fig.add_axes([.754, .260, .222, .53])
    old = {r["schedule"]: r for r in report["native"]}
    left.axvline(0, color="#687078", lw=.9, ls=(0, (3, 3)))
    for y, schedule in [(1, "C0"), (0, "C1")]:
        row = old[schedule]
        color = GREEN if row["inclusion"] else RUST
        left.plot([0, row["signed_margin"]], [y, y], color=color, lw=1.4)
        left.plot(row["signed_margin"], y, "o" if row["inclusion"] else "x",
                  color=color, markersize=4.6, mew=1.2)
    left.set(yticks=[1, 0], yticklabels=[f"C0  ·  {old['C0']['saved_rank']}",
                                      f"C1  ·  {old['C1']['saved_rank']}"],
             xlim=(-.055, .012), ylim=(-.8, 1.8), xticks=[-.04, -.02, 0],
             xlabel="Native score − Top-20 threshold")
    left.text(-.04, 1.035, "Order · rank", transform=left.transAxes, ha="right",
              fontsize=7.4, color="#687078")
    left.text(.5, 1.10, "(a) Source-to-reverse inclusion", transform=left.transAxes,
              ha="center", fontsize=8.4)
    left.legend(handles=[Line2D([], [], marker="o", ls="", color=GREEN, label="Included"),
                          Line2D([], [], marker="x", ls="", color=RUST, label="Omitted")],
                loc="lower left", bbox_to_anchor=(-.17, -.37), ncol=2,
                frameon=False, fontsize=7.4, columnspacing=.8, handletextpad=.35)
    yy = list(range(10, -1, -1))
    right.axvline(0, color="#687078", lw=.8, ls=(0, (3, 3)))
    right.axhline(.5, color="#D5D9DC", lw=.7)
    for y, a, b in zip(yy, series["zero"], series["pad"]):
        right.plot([a, b], [y, y], color="#C6CBCD", lw=1)
    right.scatter(series["zero"], [y+.08 for y in yy], c=BLUE, s=15,
                   marker="o", label="Zero baseline", zorder=3)
    right.scatter(series["pad"], [y-.08 for y in yy], c=ORANGE, s=16,
                   marker="^", label="PAD baseline", zorder=3)
    values = series["zero"] + series["pad"] + [0]
    lo, hi = min(values), max(values)
    span = max(hi-lo, .01)
    right.set(yticks=yy, yticklabels=labels, ylim=(-.65, 10.65),
              xlim=(lo-.12*span, hi+.12*span), xlabel="IG(C1) − IG(C0)")
    right.tick_params(axis="y", labelsize=7.2, pad=4)
    right.locator_params(axis="x", nbins=4)
    right.text(-.45, 1.10, "(b) Attribution change after reversal", transform=right.transAxes,
                ha="center", fontsize=8.4)
    right.legend(loc="lower right", bbox_to_anchor=(1, -.37), frameon=False,
                 ncol=2, columnspacing=.8, handletextpad=.35, fontsize=7.4)
    for ax in (left, right):
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#A9ADB1")
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", color="#A9ADB1", length=3)
    fig.text(.5, .966, 'WANDS / MiniLM · “french molding” · C0 → C1',
             ha="center", va="top", fontsize=9.3)
    fig.text(.5, .914, "139 / 256 tokens · complete attribute reversal · 42,993 fixed competitors",
             ha="center", va="top", fontsize=7.6, color="#49515A")
    fig.text(.5, .018, "Target-only reversal · Captum IG · baseline-dependent attribution diagnostic",
             ha="center", va="bottom", fontsize=7.4, color="#49515A")
    stem = "french_molding_c0_c1_captum"
    for ext in ("pdf", "svg", "png"):
        fig.savefig(root / (stem + "." + ext), dpi=260)
    fig.canvas.draw()
    figure_rows = [dict(entry_id=key, label=label,
                       zero_C1_minus_C0=series['zero'][i], pad_C1_minus_C0=series['pad'][i],
                       zero_marker_display=right.transData.transform((series['zero'][i], yy[i]+.08)).tolist(),
                       pad_marker_display=right.transData.transform((series['pad'][i], yy[i]-.08)).tolist())
                   for i, (key, label) in enumerate(zip(attr_ids + ['fixed_text_and_boundaries'], labels))]
    renderer = DrawioRenderer(fig, stem)
    fig.draw(renderer)
    mxfile = ET.Element("mxfile", host="app.diagrams.net", type="device",
                       version="29.0.0", compressed="false")
    mxfile.append(renderer.diagram)
    ET.indent(mxfile, space="  ")
    ET.ElementTree(mxfile).write(root / (stem + ".drawio"), encoding="utf-8", xml_declaration=True)
    payload = dict(contrast='C1 minus C0', rows=figure_rows, native=report['native'],
                   source_hashes={p.name: sha(p) for p in [Path(__file__),
                                  Path(__file__).with_name('native_drawio.py'),
                                  root / 'aligned_entry_changes.csv', root / 'run_report.json']},
                   width_pixels=fig.bbox.width, height_pixels=fig.bbox.height, dpi=fig.dpi,
                   drawio_scale=renderer.factor, drawio_object_counts=renderer.counts,
                   figure_text=renderer.strings,
                   connector='Pairs the Zero and PAD reference choices; not a confidence interval')
    (root / 'figure_data.json').write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    plt.close(fig)
    print("Created PDF, SVG, PNG and editable draw.io. Visually inspect before manuscript use.")


if __name__ == "__main__":
    main()
