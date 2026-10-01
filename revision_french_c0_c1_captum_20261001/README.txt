French molding: verified C0 -> C1 Captum supplement (2026-10-01)

USE THE FILES IN rerun_02/ FOR THE FINAL RESULT.
The files at this directory's top level are the retained first execution,
including its failed artifact_qa.json. All four first-run model calculations
passed completeness and cutoff checks. Export review found Windows CRLF input
files instead of the frozen LF bytes, and the original DejaVu font fell back
to a serif font in draw.io. These export issues were repaired in the source,
then the same single case was rerun into rerun_02/ without overwriting run 1.

Two checks in the first validator were also corrected: feature sums now use
the exact CUDA FP32 reduction used by the runner instead of NumPy's different
summation tree, and PDF text comparison handles extractor-inserted whitespace
inside kerned words. No attribution or completeness tolerance was relaxed.
The original failed QA is retained. The three attribution CSVs in rerun_02 are
byte-identical to run 1 (see rerun_numeric_comparison.json).

Final results: rerun_02/RESULTS.txt
Final automated QA: rerun_02/artifact_qa.json (129 checks passed)
Final visual review: rerun_02/visual_qa.json
Final figures: rerun_02/french_molding_c0_c1_captum.{pdf,png,svg,drawio}
Final data: rerun_02/{token_attributions,entry_attributions,aligned_entry_changes}.csv
Raw orders: rerun_02/attribute_sequences.csv and attribute_sequences.txt
Fixed inputs: rerun_02/C0_input.txt, C1_input.txt, *_tokenized_input.json,
              fixed_query_embedding.npy, fixed_competitors.csv
Four individual runs: rerun_02/{C0,C1}_{zero,pad}_qa.json and *_features.npz
Provenance: rerun_02/run_report.json, original_input_hash_checks.json,
            environment.json, environment_change_audit.json
Reproduction: rerun_02/REPRODUCE.txt
Full file/source manifest: DELIVERY_MANIFEST.json
Runner and original preparation package: ../tools/french_c0_c1_captum/

The original benchmark, historical C2s3/C2s2 attributions and existing paper
figures were not overwritten. This is a target-only reversal,
baseline-dependent attribution diagnostic, not independent attribute causal
effects or a population prevalence estimate.
