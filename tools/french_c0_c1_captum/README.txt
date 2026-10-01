French molding: C0 -> C1 with Captum
Prepared 2026-10-01

STATUS
This is a ready-to-run supplement, NOT a completed C0/C1 attribution result.
The preparation environment lacked torch, transformers, captum, the model
weights and the original embedding caches. Model inference and final figure
rendering have not been run here. Python compilation and frozen serialization
checks were performed. The older C2s3-minus-C2s2 attribution values are not used.

SCOPE
WANDS / MiniLM, query 359 (french molding), product 12575.
C0 is source attribute order; C1 reverses the complete attribute-entry list.
Keep all entry contents, title/class/category text and section placement fixed.
Keep the cached query vector and 42,993 C0 competitors fixed.
The inherited native observations are C0 rank 19 -> C1 rank 58.
The attribution contrast is ALWAYS IG(C1) minus IG(C0).
This is an illustrative historical case, not a representative sample.

RUN
Use the original experiment's CUDA Python environment. Keep the pinned model
and the existing PyTorch/Transformers installation; install Captum into that
environment if absent, reviewing dependency changes rather than upgrading the
experiment stack implicitly. NumPy, pandas, threadpoolctl, matplotlib are also
required. This script does not download models or embeddings.

Unzip this folder anywhere, open a terminal in this folder, then run:

  python run_c0_c1_captum.py --repo "PATH_TO/shopping-representation-bias" --plot

The default output is a new repository folder:
  revision_french_c0_c1_captum_20261001/

It refuses to overwrite an existing output directory. For a rerun, specify a
fresh --output path and keep the first run's results and any failures.

REQUIRED EXISTING REPOSITORY INPUTS
  revision_final_strengthening_20260922/scripts/common.py
  revision_final_strengthening_20260922/PROTOCOL.json and PROTOCOL.sha256
  phase2/config/phase2.json
  phase2/src/representations.py
  phase2/data/processed/wands_products.jsonl.gz
  phase2/data/processed/wands_queries.csv
  revision_evidence_20260917/data/encoder_profiles_and_rank_sources.csv
  C0/C1 product embeddings and cached query embeddings named in that CSV
  Cached model/tokenizer:
    sentence-transformers/all-MiniLM-L6-v2
    revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41

METHOD
Official captum.attr.IntegratedGradients, multiply_by_inputs=True,
method=gausslegendre. The scalar is the dot product of the cached normalized
query and the normalized attention-mask-mean-pooled MiniLM product vector.
The model is frozen and in eval mode; no training occurs. Model weights are
rounded to FP16, then represented in FP32 for the gradient diagnostic, matching
the historical XAI adapter. This is disclosed separately from native retrieval.

Change only nonspecial, nonpadding word embeddings along the IG path.
Zero reference: zero those word vectors.
PAD reference: replace them with the model's PAD word embedding.
Keep sequence length, masks, position IDs, type IDs and special/padding word
vectors unchanged. Query encoding is not recomputed.

Try 64, then 128, then 256 Gauss-Legendre nodes only as needed.
Pass completeness when absolute(sum IG - (F(input)-F(reference))) is at most
max(1e-4, 0.01 * abs(F(input)-F(reference))). Save Captum convergence delta too.
All four combinations C0/C1 x zero/PAD must be saved, including failed checks.
The fresh native and diagnostic decisions must match historical inclusion.
The inherited margin must exceed five times diagnostic/native score discrepancy.
Never replace a failed case, increase tolerances or change baseline to hide it.

Aggregate SIGNED embedding-feature attributions into token attributions, then
sum tokens within stable attribute-occurrence IDs. Do not normalize each order,
take absolute values or choose only favorable entries. Preserve fixed fields,
separators, boundary-spanning tokens and special tokens in the exported tables.
Align C0/C1 by occurrence IDs, not by token position or merely attribute names.
Compute C1-C0 separately for Zero and PAD. Baseline scores are exported for both
orders; do not assume attribution-difference sums equal native score differences.

OUTPUTS
  C0_input.txt / C1_input.txt
  cache_checks.json
  C0_zero_qa.json / C0_pad_qa.json / C1_zero_qa.json / C1_pad_qa.json
  token_attributions.csv
  entry_attributions.csv (all four individual attribution sets)
  aligned_entry_changes.csv (C0, C1, C1_minus_C0)
  run_report.json, including versions, precision, source hashes and all checks

With --plot and only if checks pass:
  french_molding_c0_c1_captum.pdf
  french_molding_c0_c1_captum.png
  french_molding_c0_c1_captum.svg
  french_molding_c0_c1_captum.drawio (native editable objects)

Inspect final figures in PDF and draw.io before using them in the manuscript.
The figure keeps original native margins on the left and the new Captum
C1-minus-C0 diagnostic on the right. The ten attributes remain in source-ID
order; the last row sums fixed fields and boundaries. No confidence intervals
are implied by the line joining the two baseline markers.

If inference fails, run_failure.json records the cause. A failed diagnostic
retains CSV/QA results and exits with status 2; no publication figure is created.

SOURCE REFERENCES
Original result snapshot: 846b266a98940da02a4e5915973554a5293db977
Native observations and original attribute contents:
revision_final_strengthening_20260922/figures/supplement_french_molding_source.json
The accompanying case_inputs.json freezes only these inputs/native observations.

https://captum.ai/api/integrated_gradients.html
https://captum.ai/tutorials/Bert_SQUAD_Interpret
https://proceedings.mlr.press/v70/sundararajan17a.html
