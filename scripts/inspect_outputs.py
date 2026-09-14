from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"


def main() -> None:
    aggregate = TABLES / "aggregate.csv"
    print("AGGREGATE_SHA", hashlib.sha256(aggregate.read_bytes()).hexdigest())

    paired = pd.read_csv(TABLES / "paired_comparisons.csv")
    print("PAIRED_COLUMNS", list(paired.columns))
    selected = paired[
        paired["representation"].isin(
            ["R2", "R3", "R4", "R5", "R8", "C_repeat", "C_order"]
        )
        & paired["metric"].eq("cNDCG@10")
    ]
    print(selected.to_json(orient="records", indent=2))

    for name in [
        "hidden.csv",
        "clean_source_subgroup.csv",
        "length_control.csv",
        "sensitivity.csv",
    ]:
        frame = pd.read_csv(TABLES / name)
        print("FILE", name, list(frame.columns))
        print(frame.to_json(orient="records", indent=2))

    review = pd.read_csv(ROOT / "results" / "qualitative" / "case_review.csv")
    print("REVIEW_COLUMNS", list(review.columns))
    print("REVIEW_ROWS", len(review))
    for column in ["direction", "support_class", "source_assessment"]:
        if column in review:
            print("REVIEW_COUNTS", column, review[column].value_counts().to_dict())


if __name__ == "__main__":
    main()
