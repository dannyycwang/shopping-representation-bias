from common5 import *
import datetime


def file_sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


frozen = json.loads((CONFIG / "sample.json").read_text(encoding="utf8"))
generation_path = OUT / "generations.jsonl"
generations = [json.loads(line) for line in generation_path.open(encoding="utf8")]
primary = [x for x in generations if x["replicate"] == "primary"]
noise = [x for x in generations if x["replicate"].startswith("noise")]

checks = {}
checks["generation_total"] = len(generations) == 6736 + 384
checks["primary_total"] = len(primary) == 6736
checks["noise_total"] = len(noise) == 384
by_key = {}
for x in generations:
    by_key.setdefault(x["cache_key"], set()).add(x["output_sha256"])
checks["cache_key_single_output"] = all(len(v) == 1 for v in by_key.values())
task_ids = [(x["dataset"], x["product_id"], x["method"], x["stem"], x["replicate"]) for x in generations]
checks["unique_task_rows"] = len(task_ids) == len(set(task_ids))
expected_tasks = set()
for ds, spec in frozen["datasets"].items():
    for pid in spec["product_ids"]:
        for method in prompt_sources():
            for stem in STEMS + ["canonical"]:
                expected_tasks.add((ds, pid, method, stem, "primary"))
for item in frozen["noise_products"]:
    for method in prompt_sources():
        for replicate in ["noise1", "noise2", "noise3"]:
            for stem in STEMS + ["canonical"]:
                expected_tasks.add((item["dataset"], item["product_id"], method, stem, replicate))
checks["exact_frozen_task_set"] = set(task_ids) == expected_tasks
checks["input_hashes"] = all(len(x.get("input_sha256", "")) == 64 for x in generations)
checks["output_hashes"] = all(sha(x.get("output", "")) == x.get("output_sha256") for x in generations)
checks["model_revision"] = all(x.get("model_revision") == MODEL_REV for x in generations)
checks["fact_checks_present"] = all(isinstance(x.get("fact_check"), dict) and "pass" in x["fact_check"] for x in generations)
maps = {ds: record_map(ds) for ds in frozen["datasets"]}
prompts = prompt_sources()
source_ok = True
cache_ok = True
prompt_ok = True
for x in generations:
    record = maps[x["dataset"]][x["product_id"]]
    source = canonical_text(record) if x["stem"] == "canonical" else raw_text(record, x["stem"])
    source_ok &= sha(source) == x["input_sha256"]
    prompt = prompts[x["method"]]
    prompt_ok &= sha(prompt) == x["prompt_sha256"] and sha(SYSTEM_PROMPT) == x["system_sha256"]
    cache_ok &= cache_key(source, prompt, x["decoding"], x["replicate"]) == x["cache_key"]
checks["reconstructed_input_hashes"] = bool(source_ok)
checks["prompt_hashes"] = bool(prompt_ok)
checks["reconstructed_cache_keys"] = bool(cache_ok)

rank_counts = {}
for ds, spec in frozen["datasets"].items():
    for model in ["bge_base", "minilm"]:
        path = OUT / f"{ds}_{model}_target_ranks.parquet"
        frame = pd.read_parquet(path)
        expected = spec["query_product_pairs"] * 7 * 10
        rank_counts[f"{ds}_{model}"] = {"rows": len(frame), "expected": expected}
        checks[f"rank_rows_{ds}_{model}"] = len(frame) == expected
        checks[f"rank_unique_{ds}_{model}"] = not frame.duplicated(["query_id", "product_id", "method", "stem"]).any()
        checks[f"rank_population_{ds}_{model}"] = (
            frame[["query_id", "product_id"]].drop_duplicates().shape[0]
            == spec["query_product_pairs"]
        )
        raw = frame[frame.method.eq("Raw")]
        checks[f"raw_orders_{ds}_{model}"] = (
            set(raw.stem) == set(STEMS)
            and len(raw) == spec["query_product_pairs"] * len(STEMS)
        )

    path = OUT / f"{ds}_bge_base_noise_ranks.parquet"
    noise_frame = pd.read_parquet(path)
    checks[f"noise_rank_unique_{ds}"] = not noise_frame.duplicated(
        ["query_id", "product_id", "method", "stem", "replicate"]
    ).any()

critical = []
for base in [CONFIG, P5 / "scripts", P5 / "tests", OUT]:
    for path in sorted(base.rglob("*")):
        if path.is_file() and path.name not in {"artifact_manifest.json", "integrity_report.json"}:
            critical.append({
                "path": path.relative_to(ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": file_sha(path),
            })

report = {
    "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "checks": checks,
    "all_pass": all(checks.values()),
    "generation_counts": {
        "total": len(generations),
        "primary": len(primary),
        "noise": len(noise),
    },
    "rank_counts": rank_counts,
}
(OUT / "integrity_report.json").write_text(json.dumps(report, indent=2), encoding="utf8")
(OUT / "artifact_manifest.json").write_text(
    json.dumps({"created_utc": report["created_utc"], "files": critical}, indent=2),
    encoding="utf8",
)
if not report["all_pass"]:
    raise AssertionError({k: v for k, v in checks.items() if not v})
print(json.dumps(report, indent=2))
