# User-authorized continuation

The user explicitly requested "繼續跑完 謝謝" after delivery of the four-hour
budget-limited report. This authorizes completion of the existing matrix, not
selection of new experiments based on the observed results.

The original config, README, sample, training examples, lambda grid, validation
selection rule and seed gating remain unchanged. `budget_amendment.json` records
the original config hash and extends the task wall-time ceiling from four to
twelve total hours (at most eight additional hours). This is a conservative
resource ceiling, not a requested amount to consume. Stop when the original
screening and already gated seeds are complete. No new datasets, encoders,
architectures, paid APIs or manuscript edits are authorized by this amendment.

The prior report, tables, figure, cost snapshot, logs and manifest are archived
in `archive/budget4h/`. All completed training checkpoints and embeddings are
reused. The .1 checkpoint and C0 embeddings are reused; the unfinished array is
recomputed. The next full trial is .5, followed by frozen-BGE Set-Attention.
Lambda selection still requires all three validation trials, and is never based
on test performance. Earlier observed test results remain historical evidence.

`setup()` reads this separate amendment after validating the original freeze.
Run exceptions now charge previously unrecorded elapsed time, preserving the
earlier interruption accounting and avoiding a silent budget reset on resume.

Final report writing must replace statements about a four-hour total cap with
the initial cap plus this explicit extension. The old report must not be
presented as the final outcome of the completed matrix.
