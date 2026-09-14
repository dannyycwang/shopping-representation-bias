# PI-FT source audit and WANDS adaptation

## Finding

The existing `phase5_mitigation/Perm-FT` is **permutation-only screening**, not a
reproduction of PI-FT. Its negative result must not be attributed to PI-FT.
The earlier user protocol explicitly excluded dropout; the present request
authorizes a separate source-aligned experiment. Old artifacts remain intact.

Primary paper: https://arxiv.org/html/2606.30473v1 (29 June 2026), Sections 5.2–5.6
and Appendix B. Author implementation:
https://github.com/avsolatorio/ai4data/tree/64437335afd37b5e8f32d453eeb8e93af61c271d/research/pift-toolkit
(`exp/pift-toolkit`, inspected 9 September 2026). Vendored files are unchanged.

| Component | Author implementation | Previous screening |
|---|---|---|
| Field sampling | fresh per access | deterministic product/seed order |
| Dropout | .15, protected title/facet | absent |
| Text | labeled fields | labeled attributes, unlabeled scalar sections |
| Objective | cached-MNRL, batch128 | pairwise softplus, accumulation8 |
| Training | 5 epochs, lr3e-5, warmup10% | 1 epoch, lr1e-5, no warmup |
| Control | matched no-permutation/no-dropout FT | absent |

The paper describes held-back loss early stopping; the pinned public trainer
does not implement it. Follow its five-epoch fixed-duration logic here. Do not
claim exact replication of the paper's reported scores or full original setup.

## Three requested checks

1. Missing dropout is confirmed; fresh per-access sampling and the loss also differ.
2. WANDS has 2,434,192 atoms; 2,434,182 contain a colon. The 10 others retain their
   complete value under `attributes`. Native attribute keys and duplicate fields
   are retained; native scalar labels `title/class/category/description` are added.
3. Average nDCG degradation and pair-level threshold crossing answer different
   questions. Report both on identical rankings; no inference that low nDCG
   degradation implies zero VI. Use WANDS relevance conventions, not the paper's
   single-positive benchmark numbers.

## Frozen new experiment (before new training/results)

Dataset/encoder: WANDS and pinned BGE-base, same historical query split, catalog,
labels, seven attribute schedules, cutoff/tie handling. Train only on the old
training queries/positive IDs. No hyperparameter selection or additional lambda.
Three rows: labeled zero-shot; matched standard FT; PI-FT adaptation. Both FT
rows start from the same base, use identical triples, seed42, five epochs,
cached-MNRL batch128, mini-batch4 for memory, scale20, lr3e-5, warmup10%, AdamW,
bf16 autocast, no field consistency loss. Only PI-FT samples all field orders and
dropout at each access. Test dropout is zero. Save seeds/input/output hashes.

Necessary adaptations, not hidden reproduction claims:

- BGE/WANDS replace the author's multilingual statistical benchmark/encoders.
- WANDS has no query-facet annotations. Per explicit user approval, protect only title in every document; all other fields are independently droppable at .15. Do not infer facets. This can remove relevance evidence from training positives and is an explicit limitation. Test-time dropout is zero.
- Mine from explicitly Irrelevant training-query products using existing frozen
  BGE scores, skip the top three, apply positive-document cosine <=.95 and sample
  three from the next 27. Exclude training rows with fewer than three candidates;
  record exclusions. This replaces the author's multilingual miner.
- Mask known Exact/Partial and same-product in-batch negatives except each row's
  own positive. Cached gradients and 128-row pool remain; effective valid negative
  counts are reported. Unjudged in-batch documents remain uncertain negatives.
- Preserve full fields, including duplicates; disable character-budget shortening
  to avoid changing test facts. Report tokenizer truncation, and common fully-fitting
  target support. This differs from the author's schema-specific budgeting.
- Evaluate the original seven attribute-order interventions in the labeled wrapper;
  scalar section order stays fixed at test. Train-time whole-field permutation is
  broader. The labeled zero-shot control separates formatting from training. Do
  not directly attribute differences from the old raw baseline to training alone.

This is **source-aligned adapted PI-FT on WANDS**, not faithful reproduction of
DevDataBench. It can test transfer of the mechanism, not refute the original paper.
The corpus/query split is historical, not pristine. Fixed five-epoch endpoints;
do not select checkpoint or modify protection based on test rankings.

## Execution

Prioritize this experiment before further legacy consistency trials, as requested.
Pause the legacy queue at a durable embedding boundary and account elapsed work.
Retain unfinished legacy results for later resumption. New execution has a separate
8-hour task-wall ceiling, no paid API, no new generated queries. Stop on failure
and report limitations rather than manufacture success.

Commands: `phase2/.venv/Scripts/python.exe phase5_pift/run.py freeze`, then `run`.
Source hashes, frozen triples/config, trained checkpoints, rank-level results and
generation-free augmentation traces stay under `phase5_pift/`. Manuscript untouched.

