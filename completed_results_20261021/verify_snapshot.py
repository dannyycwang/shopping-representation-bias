"""Verify the portable completed-results snapshot without model/GPU dependencies."""
from pathlib import Path, PureWindowsPath
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parent


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def verify():
    manifest = json.loads((ROOT / "SNAPSHOT_MANIFEST.json").read_text(encoding="utf-8"))
    failures = []
    paths = set()
    original_map = {}
    for item in manifest["files"]:
        path = (ROOT / item["path"]).resolve()
        if not path.is_relative_to(ROOT) or item["path"] in paths:
            failures.append((item["path"], "unsafe or duplicate path"))
            continue
        paths.add(item["path"])
        if not path.is_file():
            failures.append((item["path"], "missing"))
        elif path.stat().st_size != item["bytes"] or digest(path) != item["sha256"]:
            failures.append((item["path"], "size or SHA-256 mismatch"))
        if item.get("original_path"):
            original_map[item["original_path"]] = item
    omitted = {item["original_path"]: item for item in manifest["omitted_receipt_outputs"]}
    original_root = PureWindowsPath(manifest["original_workspace"])
    for name in manifest["receipt_files"]:
        path = ROOT / name
        if not path.is_file():
            continue
        receipt = json.loads(path.read_text(encoding="utf-8-sig"))
        if receipt.get("status") != "completed":
            failures.append((name, "receipt is not complete"))
        for output in receipt.get("outputs", []):
            key = PureWindowsPath(output["path"]).relative_to(original_root).as_posix()
            item = original_map.get(key) or omitted.get(key)
            if item is None or item["sha256"] != output["sha256"] or item["bytes"] != output["bytes"]:
                failures.append((name, "unmapped or inconsistent output: " + key))
    expected_primary = {m + "/" + d for m in ("embeddinggemma2", "harrier06b") for d in ("esci", "wands")}
    if set(manifest["completed_primary_cells"]) != expected_primary:
        failures.append(("scope", "expected four completed primary cells"))
    if manifest["completed_canonical_cells"] != ["embeddinggemma2/esci"] or manifest["full_experiment_complete"]:
        failures.append(("scope", "incorrect canonical or whole-study completion claim"))
    for model, expected in {
        "embeddinggemma2": "3f3004b166a855287ad7435ff0b9c278d983640bfb5f20a11bc2ec3ebeaa876c",
        "harrier06b": "7153798d3c03023dba8fb62df34d04fcefd0618474afe44602c056690d3903a4",
    }.items():
        path = ROOT / "models" / model / "PROTOCOL.json"
        if not path.is_file() or digest(path) != expected:
            failures.append((model, "frozen protocol mismatch"))
    for name, reason in failures:
        print("FAIL", name, reason)
    print(f"Verified {len(paths)} uploaded files and {len(manifest['receipt_files'])} completed receipts; failures={len(failures)}.")
    print(f"Explicitly omitted bulk outputs: {len(omitted)} (not locally verified by this portable check).")
    return int(bool(failures))


if __name__ == "__main__":
    sys.exit(verify())
