# GitHub artifact scope

This branch publishes the completed value-swap visibility pilot: code, configuration, frozen selection and complete A/C inputs, raw results, summaries, cases, and QA records. The experiment files and their original hashes are preserved.

The `cache/` directory is local-only. In particular, the full-catalog embedding array is larger than 100 MB. `baseline_manifest.json` and `delivery_manifest.json` retain its checksum and those of the forward-cache files. These manifests describe the original local experiment, including files intentionally omitted from Git; they are not inventories limited to the GitHub export.

Re-running inference requires the original WANDS data, the pinned model snapshot, and the compatible repository environment and prerequisite historical data/cache artifacts recorded in `selection_manifest.json` and `config.json`. Those pre-existing artifacts are not bundled by this commit. The README's rebuild command is for that prepared research environment; a fresh clone alone does not contain all required model/data artifacts.

`qa/preservation.json` documents checks against the original workspace, including earlier pilot files and pre-existing manuscript edits. Those unrelated files and the cancelled `round2_controlled` work are not part of this publication. This export does not alter them locally.

The root repository's `* -text` attribute preserves the exact bytes used by the experiment's checksum manifests. `git_export_manifest.json` lists the files in this GitHub export, excluding itself.
