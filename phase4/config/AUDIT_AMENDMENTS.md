# Amendments before new held-out evaluation

2026-09-07: Legacy M2 is not pure canonical sorting: it adds field framing and moves the unchanged description after attributes. Retain it as `canonical` / “Canonical template” for historical comparability. Add `canonical_raw` / “Raw sort” as an audit-required A0 baseline: sort raw atoms with the same canonical fallback as A1 and keep all other text/sections fixed. This comparator is not eligible for changing the already frozen strategy-selection rule. It isolates priority ordering from M2's other representation changes. Direct set-mean–legacy-canonical remains the frozen primary contrast; additional raw-sort comparisons are descriptive.

Batch size changes from 48 to 16 for BGE and 128 to64 for MiniLM are resource-only adjustments documented in BRIEF_AUDIT.md. No text, model, normalization, token budget, gains, selection criterion, query IDs, or view seeds changed.

Gain source cross-check: the original ESCI paper explicitly states E/S/C/I=1/.1/.01/0. The pinned repository training mapping agrees, but its qrels generator and evaluation shell together reverse S and C. “Official” in the extension means the paper's task definition, not replication of that inconsistent evaluation script.
