# Execution interruption audit

At 2026-09-09 17:22:13 Europe/Berlin, the original Python worker was absent.
The original tool session was also unavailable. The last log entry was a normal
embedding progress message at 17:21:04; no exception or normal completion was
recorded. The termination cause is unknown and must not be attributed to a model
or experimental failure.

M2 seed 42 completed all seven catalog and target-only evaluations. M3 lambda
0.05 seed 42 completed training and retained query/C0/C1/C2s1 embeddings.
The incomplete next embedding array had not been saved. The initial log is
preserved as `run_initial_interrupted.log`.

An additional 1,050 seconds was charged to `cost.jsonl`: wall time from the last
completed-stage cost write to detection of the missing process. This conservative
accounting includes possible idle time after termination. The 14,400-second cap
is unchanged. A hidden standalone worker was launched to resume checkpoints and
caches, writing `run.log` and `resume_errors.log`. No sample, model selection rule,
or scientific configuration changed.
