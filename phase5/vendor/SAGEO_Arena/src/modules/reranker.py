"""
Purpose:
    Neural Reranker using Qwen3-Reranker-4B with vLLM for efficient inference.
    Uses official Qwen3-Reranker model with generate + logprobs approach.
    
    Passage-centric: Direct passage ranking (no document aggregation).

Usage:
    from src.modules.reranker import QwenReranker, rerank_passages_batch
"""
import gc
import math
import re

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs.data import TokensPrompt

from src.modules.utils.jsonld_extractor import extract_jsonld_text


# Official Qwen3-Reranker model
MODEL_NAME = "Qwen/Qwen3-Reranker-4B"

# Default parameters
DEFAULT_BATCH_SIZE = 8  # vLLM handles batching internally

# Reranker prompt
DEFAULT_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"


def get_free_gpus(min_free_memory_gb: float = 40.0) -> list[int]:
    """Detect GPUs with sufficient free memory.
    
    Args:
        min_free_memory_gb: Minimum free memory in GB required.
    
    Returns:
        List of GPU indices with sufficient free memory.
    """
    import subprocess
    
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True
        )
        
        free_gpus = []
        for line in result.stdout.strip().split("\n"):
            if line:
                parts = line.split(",")
                gpu_idx = int(parts[0].strip())
                free_mb = float(parts[1].strip())
                free_gb = free_mb / 1024
                if free_gb >= min_free_memory_gb:
                    free_gpus.append(gpu_idx)
        
        return free_gpus
    except Exception as e:
        print(f"Warning: Could not detect free GPUs: {e}")
        # Fall back to all GPUs
        return list(range(torch.cuda.device_count()))


class QwenReranker:
    """Qwen3-Reranker-4B based neural reranker using vLLM."""

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        seed: int = 42,
        gpu_memory_utilization: float = 0.8,
        max_model_len: int = 10000,
    ):
        """Initialize the reranker model with vLLM.

        Args:
            model_name: HuggingFace model name.
            seed: Random seed for reproducibility.
            gpu_memory_utilization: Fraction of GPU memory to use.
            max_model_len: Maximum sequence length.
        
        Note: For multi-GPU data parallelism, set CUDA_VISIBLE_DEVICES to a single
        GPU before importing this module. Each worker process should use 1 GPU.
        """
        import atexit
        import signal
        
        self._cleaned_up = False
        self._child_pids = set()
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Use all visible GPUs (typically 1 per worker in data parallel mode)
        num_gpus = torch.cuda.device_count()
        
        print(f"  Initializing reranker with {num_gpus} GPU(s)")

        self.model = LLM(
            model=model_name,
            tensor_parallel_size=num_gpus if num_gpus > 0 else 1,
            seed=seed,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            enable_prefix_caching=True,
            enforce_eager=True,
            disable_log_stats=True,
        )
        
        self.instruction = DEFAULT_INSTRUCTION

        # Pre-compute token IDs
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.suffix_tokens = self.tokenizer.encode(self.suffix, add_special_tokens=False)
        self.true_token = self.tokenizer("yes", add_special_tokens=False).input_ids[0]
        self.false_token = self.tokenizer("no", add_special_tokens=False).input_ids[0]
        # Reserve 1 token for the output (yes/no) so prompt fits within max_model_len
        self.max_length = min(10000, max_model_len) - 1

        # Sampling params for yes/no classification
        self.sampling_params = SamplingParams(
            temperature=0,
            max_tokens=1,
            logprobs=20,
            allowed_token_ids=[self.true_token, self.false_token],
        )
        
        # Track vLLM's child processes for cleanup
        self._track_child_processes()
        
        # Register cleanup on exit and signals
        atexit.register(self.cleanup)
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def _track_child_processes(self):
        """Track child processes spawned by vLLM for cleanup."""
        import os
        import subprocess
        try:
            my_pid = os.getpid()
            result = subprocess.run(
                ["pgrep", "-P", str(my_pid)],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0 and result.stdout.strip():
                for line in result.stdout.strip().split("\n"):
                    try:
                        self._child_pids.add(int(line.strip()))
                    except ValueError:
                        pass
        except Exception:
            pass
    
    def _signal_handler(self, signum, frame):
        """Handle termination signals by cleaning up first."""
        self.cleanup()
        import sys
        sys.exit(128 + signum)

    def cleanup(self):
        """Properly shutdown vLLM engine and free GPU memory.

        Must be called when done with the reranker, especially in
        multiprocessing workers, to prevent GPU memory leaks from
        orphaned vLLM EngineCore subprocesses.
        """
        if self._cleaned_up:
            return
        self._cleaned_up = True
        
        import os
        import signal
        
        # Step 1: Try graceful vLLM shutdown
        try:
            from vllm.distributed.parallel_state import destroy_model_parallel
            if hasattr(self, 'model') and self.model is not None:
                # vLLM v1 has internal cleanup
                if hasattr(self.model, 'llm_engine'):
                    engine = self.model.llm_engine
                    if hasattr(engine, 'engine_core') and engine.engine_core is not None:
                        if hasattr(engine.engine_core, 'shutdown'):
                            engine.engine_core.shutdown()
                del self.model
                self.model = None
            destroy_model_parallel()
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
        except Exception as e:
            print(f"Reranker cleanup (vLLM shutdown): {e}")
        
        # Step 2: Kill any tracked child processes
        for pid in self._child_pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        
        # Step 3: Kill any remaining child processes of this process
        try:
            my_pid = os.getpid()
            import subprocess
            result = subprocess.run(
                ["pgrep", "-P", str(my_pid)],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0 and result.stdout.strip():
                for line in result.stdout.strip().split("\n"):
                    try:
                        child_pid = int(line.strip())
                        os.kill(child_pid, signal.SIGKILL)
                    except (ValueError, ProcessLookupError, PermissionError):
                        pass
        except Exception:
            pass
        
        # Step 4: Free GPU memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def __del__(self):
        self.cleanup()

    @staticmethod
    def format_passage(
        doc: dict,
        body_chunk: str,
        include_jsonld: bool = True,
        max_total_chars: int = 40000,  # ~10000 tokens at 4 chars/token
    ) -> str:
        """Format a document chunk as passage text for reranking.

        Combines document metadata fields with body text chunk into a single
        string suitable for the reranker.
        
        Output format:
            Title: {title}. Description: {meta_desc}. Headings: h1 | h2 | h3. Schema: {jsonld}. {body_chunk}

        Args:
            doc: Document dict with keys: title, meta_description, headings, jsonld.
            body_chunk: The body text chunk to include.
            include_jsonld: Whether to include JSON-LD schema text (default: True).
            max_total_chars: Maximum total characters for the passage (default: 40000).

        Returns:
            Formatted passage string.
        """
        MAX_CHARS_PER_HEADING = 500
        
        # Step 1: Build core parts (always included): title, description, body
        title = doc.get("title", "").strip()
        meta_desc = doc.get("meta_description", "").strip()
        
        core_parts = []
        if title:
            core_parts.append(f"Title: {title}")
        if meta_desc:
            core_parts.append(f"Description: {meta_desc}")
        
        # Calculate core size (title + desc + body + separators)
        core_text = ". ".join(core_parts) + ". " if core_parts else ""
        core_size = len(core_text) + len(body_chunk)
        
        # Step 2: Calculate remaining budget for headings and schema
        remaining_budget = max_total_chars - core_size
        
        # Step 3: Fill headings within remaining budget
        headings_part = ""
        headings = doc.get("headings", [])
        if headings and remaining_budget > 20:
            clean_headings = []
            total_chars = 0
            headings_prefix_len = len("Headings: . ")  # "Headings: " + ". " separator
            budget_for_headings = remaining_budget - headings_prefix_len
            
            for h in headings:
                text = re.sub(r"<[^>]+>", "", h).strip() if h else ""
                if text:
                    text = text[:MAX_CHARS_PER_HEADING]
                    separator_cost = 3 if clean_headings else 0  # " | "
                    if total_chars + len(text) + separator_cost <= budget_for_headings:
                        clean_headings.append(text)
                        total_chars += len(text) + separator_cost
                    else:
                        break
            
            if clean_headings:
                headings_part = "Headings: " + " | ".join(clean_headings)
                remaining_budget -= len(headings_part) + 2  # +2 for ". " separator
        
        # Step 4: Fill schema within remaining budget
        schema_part = ""
        if include_jsonld and remaining_budget > 20:
            jsonld = doc.get("jsonld", [])
            if jsonld:
                jsonld_text = extract_jsonld_text(jsonld)
                if jsonld_text:
                    schema_prefix_len = len("Schema: . ")
                    if len(jsonld_text) + schema_prefix_len <= remaining_budget:
                        schema_part = f"Schema: {jsonld_text}"
                    else:
                        # Truncate schema to fit
                        available = remaining_budget - schema_prefix_len
                        if available > 50:
                            schema_part = f"Schema: {jsonld_text[:available]}"
        
        # Step 5: Assemble final passage (order: title, desc, headings, schema, body)
        parts = core_parts.copy()
        if headings_part:
            parts.append(headings_part)
        if schema_part:
            parts.append(schema_part)
        
        prefix = ". ".join(parts) + ". " if parts else ""
        return prefix + body_chunk

    def _format_message(self, query: str, doc: str) -> list[dict]:
        """Format query-document pair as chat message."""
        return [
            {
                "role": "system",
                "content": 'Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".'
            },
            {
                "role": "user",
                "content": f"<Instruct>: {self.instruction}\n\n<Query>: {query}\n\n<Document>: {doc}"
            }
        ]

    def _process_inputs(self, query: str, passages: list[str]) -> list[TokensPrompt]:
        """Tokenize and format inputs for vLLM generate."""
        messages = [self._format_message(query, doc) for doc in passages]
        tokenized = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            enable_thinking=False
        )
        # Truncate and add suffix
        max_len = self.max_length - len(self.suffix_tokens)
        tokenized = [tokens[:max_len] + self.suffix_tokens for tokens in tokenized]
        return [TokensPrompt(prompt_token_ids=tokens) for tokens in tokenized]

    def _compute_scores(self, outputs) -> list[float]:
        """Compute relevance scores from logprobs."""
        scores = []
        for output in outputs:
            final_logits = output.outputs[0].logprobs[-1]

            # Get logprobs for yes/no tokens
            if self.true_token not in final_logits:
                true_logit = -10
            else:
                true_logit = final_logits[self.true_token].logprob

            if self.false_token not in final_logits:
                false_logit = -10
            else:
                false_logit = final_logits[self.false_token].logprob

            # Convert to probability score
            true_score = math.exp(true_logit)
            false_score = math.exp(false_logit)
            score = true_score / (true_score + false_score)
            scores.append(score)

        return scores

    def score_pairs(
        self,
        query: str,
        passages: list[str],
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> list[float]:
        """Score query-passage pairs.

        Args:
            query: Query string.
            passages: List of passage strings.
            batch_size: Ignored (vLLM handles batching internally).

        Returns:
            List of relevance scores (0-1, higher = more relevant).
        """
        if not passages:
            return []

        # Prepare tokenized inputs
        inputs = self._process_inputs(query, passages)

        # Run generation with logprobs
        outputs = self.model.generate(inputs, self.sampling_params, use_tqdm=False)

        # Compute scores from logprobs
        scores = self._compute_scores(outputs)
        return scores

    def score_multi_query(
        self,
        query_passage_pairs: list[tuple[str, str]],
        max_batch_size: int = 2000,
    ) -> list[float]:
        """Score multiple query-passage pairs in batched vLLM calls.

        This is more efficient than calling score_pairs repeatedly because
        it batches multiple query-passage pairs per vLLM generate() call.

        Args:
            query_passage_pairs: List of (query, passage) tuples.
            max_batch_size: Maximum pairs per vLLM call to avoid OOM.
                            Default 2000 (~20 queries × 100 passages).

        Returns:
            List of relevance scores (0-1, higher = more relevant).
        """
        if not query_passage_pairs:
            return []

        all_scores = []
        total_pairs = len(query_passage_pairs)

        # Process in chunks to avoid GPU OOM
        for chunk_start in range(0, total_pairs, max_batch_size):
            chunk_end = min(chunk_start + max_batch_size, total_pairs)
            chunk_pairs = query_passage_pairs[chunk_start:chunk_end]

            # Format chunk as messages
            messages = [self._format_message(query, doc) for query, doc in chunk_pairs]

            # Tokenize chunk
            tokenized = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                enable_thinking=False
            )

            # Truncate and add suffix
            max_len = self.max_length - len(self.suffix_tokens)
            tokenized = [tokens[:max_len] + self.suffix_tokens for tokens in tokenized]
            inputs = [TokensPrompt(prompt_token_ids=tokens) for tokens in tokenized]

            # Run generation with logprobs
            outputs = self.model.generate(inputs, self.sampling_params, use_tqdm=False)

            # Compute scores from logprobs
            chunk_scores = self._compute_scores(outputs)
            all_scores.extend(chunk_scores)

        return all_scores


def rerank_passages_batch(
    queries_with_bm25: list[tuple[str, str, list[tuple[str, float]]]],
    passage_texts: dict[str, str],
    reranker: QwenReranker,
) -> dict[str, list[tuple[str, float]]]:
    """Rerank passages for multiple queries in a single batch.

    This is significantly faster than calling rerank_passages repeatedly
    because all query-passage pairs are processed in a single vLLM call.

    Args:
        queries_with_bm25: List of (query_id, query_text, bm25_results) tuples.
            bm25_results is a list of (passage_id, score) tuples.
        passage_texts: Dictionary mapping passage_id to passage text.
        reranker: QwenReranker instance.

    Returns:
        Dictionary mapping query_id to reranked list of (passage_id, score) tuples.
    """
    if not queries_with_bm25:
        return {}

    # Collect all query-passage pairs with tracking info
    all_pairs = []  # (query, passage) tuples
    pair_info = []  # (query_id, passage_id) for each pair

    for query_id, query_text, bm25_results in queries_with_bm25:
        for passage_id, _ in bm25_results:
            text = passage_texts.get(passage_id)
            if text:
                all_pairs.append((query_text, text))
                pair_info.append((query_id, passage_id))

    if not all_pairs:
        return {qid: bm25 for qid, _, bm25 in queries_with_bm25}

    # Score all pairs in a single vLLM call
    all_scores = reranker.score_multi_query(all_pairs)

    # Group scores by query_id
    query_results: dict[str, list[tuple[str, float]]] = {}
    for (query_id, passage_id), score in zip(pair_info, all_scores):
        if query_id not in query_results:
            query_results[query_id] = []
        query_results[query_id].append((passage_id, score))

    # Sort each query's results by score
    for query_id in query_results:
        query_results[query_id].sort(key=lambda x: x[1], reverse=True)

    return query_results
