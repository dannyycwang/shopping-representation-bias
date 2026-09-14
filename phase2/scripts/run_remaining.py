"""Resume-safe Phase II experiment matrix.

Every underlying runner has result and embedding caches, so this script can be
restarted after interruption without invalidating completed cells.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


P2 = Path(__file__).resolve().parents[1]


def run(script: str, *arguments: str) -> None:
    command = [sys.executable, "-X", "utf8", "-u", str(P2 / "scripts" / script), *arguments]
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, cwd=P2.parent, check=True)


def main() -> None:
    # Finish prespecified native secondary controls. Primary results already on
    # disk are skipped by the runner.
    for dataset, model in [
        ("wands", "bge_base"),
        ("esci", "gte_modernbert"),
        ("wands", "gte_modernbert"),
    ]:
        run("run_dense.py", "--dataset", dataset, "--model", model,
            "--profile", "native", "--family", "all")
    run("analyze_sensitivity.py", "--profile", "native")

    # Full-catalog position and pooling controls on WANDS for MiniLM and BGE.
    for profile in ["mean_126_o32", "mean_254_o0", "mean_254_o64", "cls_254_o0"]:
        for model in ["minilm", "bge_base"]:
            run("run_dense.py", "--dataset", "wands", "--model", model,
                "--profile", profile, "--family", "primary")
        run("analyze_sensitivity.py", "--profile", profile,
            "--datasets", "wands", "--models", "minilm", "bge_base")

    # The Phase II-A gate was crossed before these are run. Phase II-B uses the
    # same three frozen retrievers and both datasets.
    for dataset in ["wands", "esci"]:
        for model in ["minilm", "bge_base", "gte_modernbert"]:
            run("run_mitigations.py", "--dataset", dataset, "--model", model,
                "--profile", "native")
    run("analyze_mitigations.py", "--profile", "native")


if __name__ == "__main__":
    main()
