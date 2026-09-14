# Optimization robustness extension

Run from the repository root with `phase2/.venv/Scripts/python.exe`. Existing Phase I–IV experiments are inputs and are not rerun.

```powershell
# Source review is METHOD_SOURCE_AUDIT.md. Development smoke was run first.
phase2/.venv/Scripts/python.exe phase5/scripts/smoke_and_budget.py
phase2/.venv/Scripts/python.exe phase5/scripts/freeze.py  # refuses overwrite

# Checkpointed local generation, then independent stochastic control.
phase2/.venv/Scripts/python.exe phase5/scripts/run_generation.py --mode primary --batch-size 6
phase2/.venv/Scripts/python.exe phase5/scripts/run_generation.py --mode noise --batch-size 1
phase2/.venv/Scripts/python.exe phase5/scripts/recheck_facts.py

# Reuse the same generated text for BGE and MiniLM target-only retrieval.
phase2/.venv/Scripts/python.exe phase5/scripts/run_retrieval.py --mode primary
phase2/.venv/Scripts/python.exe phase5/scripts/run_retrieval.py --mode noise
phase2/.venv/Scripts/python.exe phase5/scripts/analyze.py
phase2/.venv/Scripts/python.exe phase5/scripts/build_presentation.py
phase2/.venv/Scripts/python.exe -m pytest phase5/tests -q
phase2/.venv/Scripts/python.exe phase5/scripts/finalize_manifest.py
```

`generations.jsonl` is append-checkpointed. Its cache key covers actual input text, full prompt and system prompt, model ID/revision, decoding configuration, and replicate ID. A single output is retained per key, while a separate row is materialized for every dataset/product/method/order/replicate task. Never cache by product ID alone. `sample.json` and `protocol.sha256` are immutable after test generation starts. The retained execution log records an initial batch-8 out-of-memory event, a batch-4 recovery, and the final batch-6 continuation; prompts, model, decoding, sample, and cache keys did not change.

The intervention is target-only: each relevant target is independently removed from the C0 index and reinserted with one counterfactual score. Competitors remain C0. Consequently, the reported target-intervention inclusion rate is not a single deployed-index Recall@K. There is no full-catalog LLM rewrite in this bounded extension.

The `G1/G2` rows are operational adapted-generator outcomes. `Guard*` rows replace failed outputs with raw canonical text according to the frozen rule checker. `Canonical` is pure raw-atom sorting and must not be confused with historical M2 framing. The stochastic control computes order VI within each replicate and same-input variation across replicates separately.
