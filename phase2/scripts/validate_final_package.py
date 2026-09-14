from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


P2 = Path(__file__).resolve().parents[1]
T = P2 / "results" / "phase2_tables"
F = P2 / "results" / "publication_figures"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    t1 = pd.read_csv(T / "table1_representation_sensitivity_native.csv")
    t4 = pd.read_csv(T / "table4_mitigation_native.csv")
    direct = pd.read_csv(T / "single_product_summary_native.csv")
    assert len(t1) == 6 and t1["go_cell"].all()
    assert set(t1.dataset) == {"wands", "esci"}
    assert set(t1.retriever) == {"minilm", "bge_base", "gte_modernbert"}
    assert len(t4) == 12 and t4.filter(regex="^(delta|ci_)").notna().all().all()
    assert len(direct) == 24
    m2 = t4[t4.representation.eq("M2_normalized_attributes")]
    assert (m2["VI_reduction_confirmed"]).all()
    assert t4.groupby("dataset")["mitigation_success"].any().all()

    required = [P2 / "PAPER_DRAFT.md", P2 / "PAPER_DRAFT_SUBMISSION.md",
                P2 / "PHASE2_REPORT.md", P2 / "FIGURE_GUIDE.md", P2 / "SUBMISSION_READINESS.md"]
    required += [F / f"figure_{n}{ext}" for n in ["1_sensitivity", "2_controls", "3_mitigation_tradeoff", "4_direct_vs_full", "5_query_types"] for ext in [".png", ".pdf"]]
    for path in required:
        assert path.exists() and path.stat().st_size > 1000, path

    manifest = {
        "status": "complete",
        "primary_cells": len(t1),
        "primary_cells_passing": int(t1.go_cell.sum()),
        "mitigation_rows": len(t4),
        "files": {str(p.relative_to(P2)).replace("\\", "/"): digest(p) for p in required},
    }
    out = P2 / "results" / "submission_manifest.json"
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({**manifest, "files": len(manifest["files"])}, indent=2))


if __name__ == "__main__":
    main()
