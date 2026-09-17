# Execution timing note

On 2026-09-17, Windows entered Modern Standby and sleep while the WANDS/BGE lexical-descending condition was active. The system event log records standby around 15:27 local time, sleep/resume around 16:28, and a lid-related resume around 17:06 (Europe/Berlin, UTC+2). The Python worker initially continued from its in-memory progress after resume.

After resume, GPU memory usage was about 7,937 MiB and progress was unusually slow. The task-owned worker was stopped before that condition produced any saved embedding/rank/source artifact. Memory usage then fell to 15 MiB. A first fresh-process retry at the original batch size of 48 again reached about 7,921 MiB and slowed sharply at longer capped inputs; that incomplete retry was also stopped without saved ranks.

The four unfinished BGE conditions were then run with `--models bge_base --batch-size-cap 16`. `BGE_EXECUTION_ADDENDUM.json` was recorded before any of these four conditions completed and records the original and effective batch sizes, reason, scope, timestamp and source hashes. The 14 previously completed or inherited conditions remained intact. No representation, model, tokenizer, pooling, precision, tie rule, schedule or evaluation support was changed. This was runtime recovery, not selection using test outcomes. Batch partitioning may change floating-point rounding; no bitwise identity of independent forward passes is claimed.

The interrupted attempt's progress-log times include suspended wall time and must not be interpreted as continuous encoding-speed measurements. The final BGE source JSONs record the fresh process's elapsed time, exact ranks, embedding metadata, hashes and completion timestamps. No partial result from the interrupted attempt is used.

Temporary process-scoped Windows display/system idle-sleep prevention requests were enabled after this interruption. They are automatically released as finalization and the authorized GitHub push finish; the final-delivery request also has a 30-minute limit. No persistent power-plan or lid-action setting was changed.
