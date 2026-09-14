# Execution repairs

2026-09-09: both Standard-FT and Adapted-PI-FT training completed and checkpoints
were saved. A Hugging Face connection closed during evaluation model loading,
before PI-FT ranking evaluation. The exception handler charged 43.548 seconds of
otherwise unrecorded wall time. Logs are preserved in `archive/network_failure/`.

All required pinned model files were already in local cache (Standard-FT evaluation
had completed). The runner now sets HF_HUB_OFFLINE and TRANSFORMERS_OFFLINE before
importing the model libraries. This changes network access only, not model revision,
weights, tokenizer, input, training or ranking. Resume reuses both trained checkpoints
and completed Standard-FT ranks. No training is repeated.
