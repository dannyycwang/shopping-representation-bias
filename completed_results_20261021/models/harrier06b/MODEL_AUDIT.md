# Harrier 0.6B model audit

Official model: [microsoft/harrier-oss-v1-0.6b](https://huggingface.co/microsoft/harrier-oss-v1-0.6b).
Pinned model/tokenizer revision: `f9b9dc8d367d443f2479d27aa5d8d2850c0774ee`. Raw official API, model card and configs are preserved under `audit/models/harrier-oss-v1-0.6b/`.

- Qwen3Model backbone; 28 layers; 1,024 native dimensions; no projection, slicing or quantization.
- Model card and model config support 32,768 tokens. Tokenizer metadata advertises 131,072, which is NOT the supported model limit; effective max_length is explicitly 32,768.
- BF16 weights and inference; SDPA; cache disabled; fixed left padding; deterministic execution; original 4-thread FP32 full-matrix scoring and stable index ties.
- Official ST modules are Transformer, last-token Pooling (include_prompt=true), Normalize. The adapter implements last nonpadding pooling independently and compares with the complete official SentenceTransformer.encode path before formal work.
- Query uses the official `web_search_query` instruction, exactly once. Document uses no prefix and no default prompt. Original serialized document content remains intact. No chat template or manual EOS is added: token IDs must equal official preprocessing and the direct tokenizer.
- Query prefix: `'Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: '`.
- Fixed batching: base 8; <=512 tokens: 8, <=1024: 4, <=2048: 2, longer: 1. Stable length/content-hash order. OOM stops this profile; no silent length or batch changes.
- Official reference and repeated-input GPU validation are pending until the existing Gemma queue releases the GPU. CPU/unit tests and download checks are not substitutes for that validation.

The user explicitly enabled this model AFTER Gemma/ESCI results were discussed. `USER_SCOPE_AMENDMENT.json` records that timing. This is an additional user-requested follow-up, not prospective activation of the original optional third model. Qwen 4B remains deferred. Gemma's frozen profile and caches are unchanged.

## Runtime compatibility correction, 2026-10-19

The first GPU setup attempt failed before encoding because ST 6.1 exposes `pooling_mode`, not `pooling_mode_lasttoken`. The inspection assertion is corrected without changing native pooling or numerical settings. The old protocol and failed technical evidence are preserved under `audit/protocol_history/before_st61_pooling_check_20261019/`. A CPU test loads the official pooling config in the installed library and verifies its BF16 pooled/normalized output; 26 tests pass. Full GPU reference validation still must run after the Gemma queue. The user clarified the original sequential order; Gemma resumes its 44,800 verified input checkpoints before Harrier. See RUNTIME_COMPATIBILITY_AMENDMENT.json.
