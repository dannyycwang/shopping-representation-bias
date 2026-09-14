# Learned mitigation screening

This directory implements the newly requested Phase V. `phase5/` contains the
earlier rewriting audit and is not overwritten. The manuscript is not edited.

## Audit and prospective decisions

The repository has inference/evaluation infrastructure and a frozen 96-query
development / 384-query historical held-out split, but no WANDS BGE training
pipeline or training loss. Consequently, these are adapted small-budget training
baselines, not reproductions of an existing fine-tuning procedure. The 96 old
development queries are hash-partitioned into 72 training and 24 validation
queries. The old held-out set remains unchanged; it is not pristine confirmation.

Training uses a shared query/product BGE encoder and pairwise softmax loss over
one Exact product and one explicitly Irrelevant product from the same training
query, temperature 0.05. Partial and unjudged items are never sampled as negatives.
At most 16 Exact products per training query, chosen by hash independently of
ranks. One epoch, microbatch 2, accumulation 8, AdamW lr 1e-5, weight decay .01,
gradient norm cap 1, full encoder fine-tuning with activation checkpointing and
FP16 autocast. M3 adds consistency between two positive permutations; its second
positive never enters the negative pool. M2 and all three M3 trials use seed 42
and identical supervised examples. Both positive and negative products receive
complete-atom permutations. No dropout of fields. The retrieval loss is identical
across M2/M3/M4, but there is no unaugmented-training control; improvements cannot
be attributed uniquely to augmentation rather than domain fine-tuning.

M4 freezes BGE and uses a 768->128->1 tanh attention scorer over independent
attribute embeddings, without positional inputs. Weighted attribute means are
normalized before the existing equal-weight nonattribute/attribute interpolation
and final normalization. Empty attribute sets contribute zero, as in Set-Mean.
One epoch over the same training triples, batch 16, AdamW lr 1e-3. No architecture
search or query tuning. Its 98,561 parameters are the only trainable parameters.

## Frozen selection and budget

Run M2 seed 42, M3 lambda .05/.1/.5 seed 42, then M4 seed 42. Select M3 on the 24
validation queries only: candidates within 1 absolute percentage point of Original
seven-order mean Recall@100, then lowest query-macro VI@20, then highest robust
coverage@100, then smaller lambda. If none passes the recall screen, apply the
latter ordering among all and explicitly report failure. This threshold is an
engineering screening margin, not a statistical equivalence test.

Additional seeds 43/44 are allowed only after the first screening if a method has
at least 25% lower VI@20 and loses at most 1 recall point on validation, or is
invariant and improves over Set-Mean without losing more than 1 point to Original.
They must fit the total 4-hour GPU-task wall-time cap. The cap includes inference
and is checked between batches; interrupted runs remain incomplete, not failures.
No other datasets/models, text generation, paid API, or large sweeps.

All 42,994 catalog products remain candidates. Evaluate the exact seven original
serializations via the existing `build` function, CLS pooling, 512 tokens, unit
normalization, FP32 dot products, and stable catalog-index ties. Cache identity
includes checkpoint identity and actual text hashes. Reuse baseline ranks and
embeddings. Canonical means pure raw-atom sorting, not the M2 field template.

Recall is query-macro, averaged across seven catalog schedules (also retain C0
and individual schedules). VI/robust/never are first computed over seven ranks
per Exact pair, then query-macro; retain pair-micro as supplementary output.
Target-only uses the existing `rank_target` function with each model's own C0
competitors and separately inserts each target. Its average inclusion is not a
single-index recall. Bootstrap paired query deltas with 10,000 draws, seed 20260963.
Report the common fully-fitting target subset without shrinking the catalog;
tokenizer and inputs are shared by Original/M2/M3. Seven-order worst observed is
not an all-permutations bound. Zero VI alone is not success.

Commands (from repository root):

```
phase2/.venv/Scripts/python.exe phase5_mitigation/screen.py freeze
phase2/.venv/Scripts/python.exe phase5_mitigation/screen.py run
phase2/.venv/Scripts/python.exe phase5_mitigation/screen.py report
```

The machine-readable frozen config, IDs, hashes, checkpoints, training logs,
rank-level files, validation selection and figure are kept under this directory.
