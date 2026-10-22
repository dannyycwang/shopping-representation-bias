# Model audit

Checked official model cards, API revisions, configs, ST modules, and the installed Transformers/ST implementation on 2026-10-17. Raw responses are under `audit/models/`; no revision is inferred or fabricated.

| Model | Pinned revision | Native text configuration |
|---|---|---|
| [Qwen/Qwen3-Embedding-4B](https://huggingface.co/Qwen/Qwen3-Embedding-4B) | `5cf2132abc99cad020ac570b19d031efec650f2b` | 2560 dimensions, last nonpadding token, no projection, left padding, BF16, supported embedding length 32768. The config position ceiling 40960 is separately recorded, not treated as the model-card embedding limit. |
| [google/embeddinggemma-2](https://huggingface.co/google/embeddinggemma-2) | `914f7f89142e33e77833254d9c9b90c3cef7303b` | Text-only towers, 768 dimensions, native per-token 512→768 projection then attention-mask mean including prompt. BF16; effective/native supported length 8192. Config maximum positions 262144 do not establish the supported embedding context length. |

Google's [official model card](https://ai.google.dev/gemma/docs/embeddinggemma/model_card_2) and the pinned Hugging Face card specify mean pooling and projection. The pinned card expressly prohibits FP16 because of activation range; BF16/FP32 are supported. `vision_config=None,audio_config=None` disables unused modalities without changing the text model. The installed `EmbeddingGemma2TextModel.forward` applies `embedding_projection` before returning `last_hidden_state`. The adapter asserts its 512→768 dimensions and does not add another projection.

Qwen uses the exact saved `query` instruction (`Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:`) and an empty document prefix. No prompt search is performed. Gemma uses `SearchQuery` (`task: search result | query: `) and `Document` (`title: none | text: `). Its entire inherited serialized record, including the existing title in its original location, is passed as document content. A separate title is not extracted, moved, or duplicated. Prefixes are passed once through the official preprocessing entry point; effective IDs are compared with independently prefixed tokenizer inputs.

The dedicated adapter shares the official frozen Transformer backbone and preprocessing code but implements last-token / mean pooling explicitly. Official end-to-end `SentenceTransformer.encode` is a separate validation path. Native BF16 normalization is followed by FP32 unit normalization for exact FP32 scoring. Reference comparisons include normalization at the same final precision. Long/short documents, queries, padding, batch sizes 1/2, effective IDs, and three identical executions are saved under `smoke/<model>/`. The source code is explicit about sharing the backbone; this is not claimed as an independent reimplementation of the Transformer architecture.

The isolated environment uses Transformers 5.20.1 and SentenceTransformers 6.1.0, with the existing CUDA-enabled PyTorch 2.11.0+cu128. The inherited global Transformers 4.46.2 is preserved. Official Gemma config metadata was created with Transformers 5.18.0.dev0 and ST 6.1.0. The actual execution environment, rather than that author metadata, is recorded in the protocol.

Execution uses SDPA, eval/inference mode, no quantization or dimension slicing, deterministic algorithms, TF32 disabled, and a fixed length-tier batching policy. Identical content is cached once within each collection without dropping any product or attribute occurrence. Cross-batch numerical differences are saved; reference-output agreement is tested within matching batches. Ranking uses no tolerance or epsilon.

Downloads initially encountered Windows path-length errors. Retained failure logs document these attempts. Short snapshot paths under `models/<key>/` resolve the issue, with already downloaded Qwen shard bytes reused. Final download receipts determine availability; initial failures are not falsely reported as final blockers. Harrier is disabled, fixed before any new held-out outcome.

