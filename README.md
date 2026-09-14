# Product representation sensitivity: WANDS Phase 1

This is a separate research pilot inside an existing workspace. It does not modify the parent application's source files. Outputs are real WANDS experiments, not simulated data. Start with `results/REPORT_zh.md` and `paper/PAPER_DRAFT.md` after a completed run.

## Run

Python 3.10 or newer, NumPy, pandas, SciPy, PyTorch, transformers, matplotlib. A CUDA GPU is recommended. No paid API is used. Raw data and model weights are downloaded from their official repositories, pinned to commits in `configs/phase1.json`.

```powershell
python -m unittest discover -s shopping-representation-bias/tests -v
python shopping-representation-bias/scripts/run_phase1.py --stage all
python shopping-representation-bias/scripts/run_order_control.py
python shopping-representation-bias/scripts/refresh_lexical.py
python shopping-representation-bias/scripts/audit_saved.py
python shopping-representation-bias/scripts/analyze.py
python shopping-representation-bias/scripts/write_deliverables.py
python shopping-representation-bias/scripts/finalize.py
```

Commands above run from the parent workspace; from this directory remove `shopping-representation-bias/` from each command. `--stage prepare` only downloads, inspects and builds representations. `--stage bm25` runs the lexical baseline. `--stage dense` also loads or computes the BM25 ranks needed for hybrid fusion. `--config` accepts a JSON configuration.

Embeddings are cached using content, configuration and encoder fingerprints. Full rank arrays are saved separately per condition/retriever. Do not mix results produced under different settings in one results directory: use a fresh project copy or archive results before changing configurations. A completed raw rank file is the checkpoint for a retrieval run. `results/manifest.json` records data checksums, package versions, the model revision, seeds and settings. `data/embeddings/*.json` records encoder timing, precision and sequence length statistics.

## Representations actually implemented

These are **deterministic lossless serialization proxies**, not fluent LLM rewrites. The original description is retained even when it contains marketing language. That deliberate limitation prevents guessing facts, but means this pilot cannot settle a strong marketing-versus-factual-language claim.

| ID | Implementation |
|---|---|
| R0 | Original name, class, category, description and raw features |
| R2 | R0 with generic shopping invitations, no new product-specific claims |
| R3 | Original fields framed as factual statements; all original description retained |
| R4 | Original fields with feature separators changed into bullets; identical BM25 token multiset |
| R5 | Labeled identity/category, alphabetically ordered raw attribute entries, original description at end |
| R8 | R0 plus the complete R5 serialization; a dual-view prototype, not a learned ARPC compiler |
| C_repeat | R0 concatenated with itself; controls information repetition and approximately length |
| C_order | Original raw attribute entries reversed; exactly same character count and lexical token multiset |

No attributes, units, brand, capacity or compatibility are inferred. Duplicate keys, different values under the same key, negative claims and unparseable entries are retained. Ratings are inspected but excluded from **all** Phase 1 representations so no treatment adds an information source missing from R0. `source_map.jsonl` maps each retained source atom to the original field or attribute position. Fidelity checks verify every source atom occurs in every output; they establish source preservation, not real-world truth or independent semantic entailment.

## Retrieval and evaluation

* BM25: custom sparse implementation of `log(1+(N-df+.5)/(df+.5))`, k1=1.2, b=.75; lowercased `[a-z0-9]+` tokens, no stemming or stopword removal, unique query terms. Each catalog gets its own statistics.
* Dense: pinned `sentence-transformers/all-MiniLM-L6-v2`, transformers mean pooling. This is a compact exploratory baseline, not a claim of state-of-the-art retrieval. All tokens are divided into non-overlapping 254-token chunks, with CLS and SEP added. Chunk token means are weighted by content length, averaged and L2 normalized. GPU weights are FP16 and pooling/accumulation FP32. Query and product representations use the same encoder. This custom long-document pooling differs from the model's default first-256-token truncation.
* Hybrid: equal-weight reciprocal rank fusion, constant 60, **full catalog lists**. There is no tuning on WANDS judgments. Ties use ascending product ID; boundary ties and zero-score examples are exposed.
* Data cleaning: collapse duplicate identical query-product labels; exclude pairs with contradictory labels, preserving them in `ambiguous_judgments.csv`.
* `cNDCG@10/20`: **judged-condensed** NDCG, gain `2^grade-1`, remove unjudged products before assigning positions. This is not ordinary full-catalog NDCG and can be optimistic with incomplete judgments. Report judged coverage alongside it.
* `Recall@K`: fraction of all **known** Exact or Partial products retrieved in the full-catalog top K. Not an estimate of recall over all truly relevant catalog products.
* `MRR`: reciprocal full-catalog rank of the first known Exact or Partial product. Unjudged items retain their positions, not an assumed negative label.
* `ExactHit@K`: fraction of queries with at least one labeled Exact product in top K. Queries without Exact judgments are undefined and excluded; denominators are in the CI table.
* `ExplicitIrrelevant@10`: explicitly labeled Irrelevant count / 10, with unjudged rate separately reported. Unknown is never silently relabeled as irrelevant.
* `RelevantHidden@K`: known Exact pair transitions from rank >K under R0 to <=K under the alternative. Reverse harm is reported symmetrically. Micro denominator is all known Exact pairs; query-macro net rates and paired uncertainty are supplied separately.
* Sensitivity: exact full ranks, max/min rank ratio, max-min span and top-K boundary crossing. Scores from independently rebuilt BM25 indexes are **not** treated as directly comparable score variances.
* Statistics: 10,000 query-paired bootstrap replicates and paired sign permutation replicates, seed 20260903. Primary comparisons are R8 versus R0 on cNDCG@10, with Holm correction across three retrievers. All other comparisons are exploratory. No product pairs are falsely treated as independent statistical samples.

## Scope of inference

Universal adoption changes competitors and corpus statistics too. These effects are catalog-level representation sensitivity, not isolated direct causal effects of changing only one product. WANDS annotations are human relevance judgments, not objective verification of every physical specification. The pilot contains no LLM shopping agents, no rerankers and no API rewriting. Do not claim recommendation bias, a validated ARPC method, cross-agent robustness or an end-to-end gain from these outputs.

## Main files

`src/representations.py`, `src/retrieval.py`, `src/evaluation.py`; experiment configurations in `configs/`; scripts in `scripts/`; per-query CSVs precede aggregates; full rank arrays and Exact-pair movements in `results/raw/`; figures in PNG and SVG; selected examples retain full source descriptions and attributes.

## Sources

* [WANDS dataset and documentation](https://github.com/wayfair/WANDS), ECIR 2022, MIT-licensed data repository.
* [MiniLM model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), Apache-2.0.
* [E-GEO v2, July 2026](https://arxiv.org/abs/2511.20867v2).
* [SAGEO Arena v2, August 2026](https://arxiv.org/abs/2602.12187v2).
