"""
Baseline Generation Script for SAGEO Experiments

This script generates the baseline data required for all SAGEO experiments.
Run this ONCE before running any experiments to ensure data integrity.

Generates:
    1. Baseline BM25 indices (experiments/baseline/baseline_index/)
    2. Baseline retrieval results (1_retrieval_results.json)
    3. Baseline reranking results (2_rerank_results.json)
    4. Baseline generation results (3_generation_results.json)
    5. Target document selection (4_targets.json, shared across all methods)

Data Integrity:
    - Existing files are NOT overwritten (skipped if exists)
    - Use --force to regenerate specific phases
    - Target selection uses fixed seed for reproducibility

Usage:
    # Run all phases
    python -m src.scripts.run_baseline

    # Run specific phase(s)
    python -m src.scripts.run_baseline --phase index
    python -m src.scripts.run_baseline --phase retrieval
    python -m src.scripts.run_baseline --phase rerank
    python -m src.scripts.run_baseline --phase generation
    python -m src.scripts.run_baseline --phase targets

    # Force regenerate specific phase (deletes existing)
    python -m src.scripts.run_baseline --phase retrieval --force

    # Check existing baseline status
    python -m src.scripts.run_baseline --check

    # Test mode (small subset)
    python -m src.scripts.run_baseline --test --num_per_dataset 5

    # GPU selection for reranking (auto-detects if not provided)
    python -m src.scripts.run_baseline --phase rerank --gpu_ids 0,1,2,3
"""

# =============================================================================
# EARLY GPU DETECTION (before torch imports for cleaner startup)
# =============================================================================
import os
import subprocess
import sys


def _detect_free_gpus(min_free_memory_gb: float = 38.0) -> list[int]:
    """Detect GPUs with sufficient free memory."""
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
        return []


def _early_gpu_setup() -> list[int] | None:
    """Parse --gpu_ids early and auto-detect free GPUs if not provided."""
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--gpu_ids", type=str, default=None)
    args, _ = parser.parse_known_args()

    if args.gpu_ids:
        gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(",")]
        print(f"Using specified GPUs: {gpu_ids}")
        return gpu_ids
    else:
        gpu_ids = _detect_free_gpus(min_free_memory_gb=38.0)
        if gpu_ids:
            print(f"Auto-detected free GPUs (>=38GB free): {gpu_ids}")
            return gpu_ids
        else:
            print("Warning: No GPUs with >=38GB free memory found.")
            return None


# Run early GPU detection
_GPU_IDS = _early_gpu_setup()

# =============================================================================
# IMPORTS
# =============================================================================
import argparse
import asyncio
import json
import random
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv

from src.modules import CORPUS_DIR, DATA_DIR, ROOT_DIR
from src.modules.retriever_bm25 import (
    build_field_jsonl,
    build_indices,
    chunk_text,
    load_searchers,
    search,
)
from src.modules.reranker import QwenReranker, rerank_passages_batch
from src.modules.generator import GenerationModule

load_dotenv(ROOT_DIR / ".env.local")

# =============================================================================
# CONFIGURATION
# =============================================================================

EXPERIMENTS_DIR = ROOT_DIR / "experiments"
BASELINE_DIR = EXPERIMENTS_DIR / "baseline"
QUERIES_DIR = DATA_DIR / "queries"

# Pipeline parameters
RETRIEVAL_K = 100   # BM25 retrieves top 100, all are reranked and saved
GENERATION_K = 10   # Top 10 reranked passages used for generation

# Datasets
ALL_DATASETS = [
    "fiqa", "hotpotqa", "msmarco", "nfcorpus", "nq",
    "quora", "debate", "ecommerce", "researchy"
]

# Random seed for reproducibility
SEED = 42

# Valid phases
VALID_PHASES = ["index", "retrieval", "rerank", "generation", "targets", "all"]


def set_seed(seed: int = SEED):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_corpus() -> tuple[list[dict], dict[str, int]]:
    """Load all documents from corpus.

    Returns:
        (documents, doc_id_to_idx): List of documents and mapping from doc_id to index.
    """
    corpus_path = CORPUS_DIR / "documents_filtered.jsonl"
    if not corpus_path.exists():
        corpus_path = CORPUS_DIR / "documents.json"

    documents = []
    if corpus_path.suffix == ".jsonl":
        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    documents.append(json.loads(line))
    else:
        with open(corpus_path, "r", encoding="utf-8") as f:
            documents = json.load(f)

    # Build doc_id mapping
    doc_id_to_idx = {}
    for idx, doc in enumerate(documents):
        doc_id = doc.get("_id") or f"doc_{idx}"
        doc["_id"] = doc_id
        doc_id_to_idx[doc_id] = idx

    print(f"Loaded {len(documents)} documents from {corpus_path}")
    return documents, doc_id_to_idx


def load_queries(datasets: list[str], num_per_dataset: int = 300) -> list[dict]:
    """Load queries from specified datasets.

    Args:
        datasets: List of dataset names.
        num_per_dataset: Maximum queries per dataset.

    Returns:
        List of query dictionaries with query_id, query, dataset.
    """
    queries = []

    for dataset in datasets:
        query_file = QUERIES_DIR / f"{dataset}_queries_300.jsonl"
        if not query_file.exists():
            print(f"Warning: Query file not found: {query_file}")
            continue

        # Load queries (JSONL)
        dataset_queries = []
        with open(query_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    dataset_queries.append(json.loads(line))

        # Sample if needed
        if len(dataset_queries) > num_per_dataset:
            set_seed(SEED)
            dataset_queries = random.sample(dataset_queries, num_per_dataset)

        # Normalize field names: text -> query, id -> query_id
        for q in dataset_queries:
            if "text" in q and "query" not in q:
                q["query"] = q["text"]
            if "id" in q and "query_id" not in q:
                q["query_id"] = f"{dataset}_{q['id']}"
            if "dataset" not in q:
                q["dataset"] = dataset
            queries.append(q)

    print(f"Loaded {len(queries)} queries from {len(datasets)} datasets")
    return queries


# =============================================================================
# PHASE: INDEX BUILDING
# =============================================================================

def run_index_phase(
    documents: list[dict],
    baseline_dir: Path,
    force: bool = False,
) -> dict[str, str]:
    """Build multi-field BM25 indices for baseline.

    Args:
        documents: List of documents.
        baseline_dir: Baseline directory.
        force: Force rebuild even if exists.

    Returns:
        chunk_to_doc: Mapping from chunk_id to doc_id.
    """
    index_dir = baseline_dir / "baseline_index"
    jsonl_dir = index_dir / "jsonl"
    lucene_dir = index_dir / "lucene"
    metadata_path = jsonl_dir / "metadata.json"

    # Check if exists
    if metadata_path.exists() and not force:
        print(f"Index already exists at {index_dir}, loading metadata...")
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        return metadata["chunk_to_doc"]

    if force and index_dir.exists():
        print(f"Force mode: removing existing index...")
        import shutil
        shutil.rmtree(index_dir)

    print(f"Building baseline index at {index_dir}...")

    # Build JSONL files
    chunk_to_doc = build_field_jsonl(documents, jsonl_dir)
    print(f"Created JSONL files with {len(chunk_to_doc)} chunks")

    # Build Lucene indices
    build_indices(jsonl_dir, lucene_dir)
    print(f"Built Lucene indices at {lucene_dir}")

    return chunk_to_doc


def load_chunk_to_doc(baseline_dir: Path) -> dict[str, str]:
    """Load chunk_to_doc mapping from existing index."""
    metadata_path = baseline_dir / "baseline_index" / "jsonl" / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Index metadata not found: {metadata_path}")
    
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return metadata["chunk_to_doc"]


# =============================================================================
# PHASE: RETRIEVAL
# =============================================================================

def run_retrieval_phase(
    queries: list[dict],
    baseline_dir: Path,
    chunk_to_doc: dict[str, str],
    force: bool = False,
) -> dict[str, list[tuple[str, float]]]:
    """Run BM25 retrieval for all queries.

    Args:
        queries: List of query dictionaries.
        baseline_dir: Baseline directory.
        chunk_to_doc: Mapping from chunk_id to doc_id.
        force: Force regenerate even if exists.

    Returns:
        retrieval_results: Dict mapping query_id to list of (chunk_id, score).
    """
    output_path = baseline_dir / "1_retrieval_results.json"

    # Check if exists
    if output_path.exists() and not force:
        print(f"Loading existing retrieval results from {output_path}")
        with open(output_path, "r", encoding="utf-8") as f:
            return json.load(f)

    if force and output_path.exists():
        print(f"Force mode: removing existing retrieval results...")
        output_path.unlink()

    print(f"Running BM25 retrieval for {len(queries)} queries...")
    
    index_dir = baseline_dir / "baseline_index"
    searchers = load_searchers(index_dir / "lucene")

    retrieval_results = {}
    for i, q in enumerate(queries):
        query_id = q["query_id"]
        query_text = q["query"]

        results = search(query_text, searchers, chunk_to_doc, top_k=RETRIEVAL_K)
        retrieval_results[query_id] = results

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(queries)} queries")

    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(retrieval_results, f)

    print(f"Saved retrieval results to {output_path}")
    return retrieval_results


# =============================================================================
# PHASE: RERANKING
# =============================================================================

def build_passage_texts(
    chunk_ids: list[str],
    documents: list[dict],
    doc_id_to_idx: dict[str, int],
    chunk_to_doc: dict[str, str],
) -> dict[str, str]:
    """Build passage texts for reranking.

    Uses QwenReranker.format_passage() for consistent formatting.
    Format: "Title: ... Description: ... Headings: ... Schema: ... {body_chunk}"

    Args:
        chunk_ids: List of chunk IDs to process.
        documents: List of documents.
        doc_id_to_idx: Mapping from doc_id to document index.
        chunk_to_doc: Mapping from chunk_id to doc_id.

    Returns:
        passage_texts: Dict mapping chunk_id to formatted passage text.
    """
    passage_texts = {}

    for chunk_id in chunk_ids:
        if chunk_id in passage_texts:
            continue

        doc_id = chunk_to_doc.get(chunk_id, chunk_id)
        doc_idx = doc_id_to_idx.get(doc_id)

        if doc_idx is None:
            continue

        doc = documents[doc_idx]

        # Get chunk index from chunk_id
        try:
            chunk_idx = int(chunk_id.split("_chunk_")[-1])
        except (ValueError, IndexError):
            chunk_idx = 0

        # Get body chunk
        body_text = doc.get("body_text", "")
        body_chunks = chunk_text(body_text)
        body_chunk = body_chunks[chunk_idx] if chunk_idx < len(body_chunks) else ""

        # Format using reranker's format_passage
        passage_text = QwenReranker.format_passage(doc, body_chunk)
        passage_texts[chunk_id] = passage_text

    return passage_texts


RERANK_SAVE_INTERVAL = 20  # Save every N queries per GPU


def _rerank_worker(
    gpu_id: int,
    queries_subset: list[tuple[str, str, list]],
    passage_texts: dict[str, str],
    output_path: Path,
    lock_path: Path,
):
    """Worker function for multi-GPU reranking with incremental saving.
    
    All workers write to the same output file using file locking.
    On restart, queries already in the file are skipped.
    """
    import os
    import fcntl
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    from src.modules.reranker import QwenReranker
    
    def load_done_ids() -> set:
        """Load already-completed query IDs from shared output."""
        if not output_path.exists():
            return set()
        try:
            with open(lock_path, "w") as lf:
                fcntl.flock(lf.fileno(), fcntl.LOCK_SH)
                try:
                    with open(output_path, "r") as f:
                        return set(json.load(f).keys())
                finally:
                    fcntl.flock(lf.fileno(), fcntl.LOCK_UN)
        except (json.JSONDecodeError, IOError):
            return set()
    
    def save_batch(batch_results: dict):
        """Merge batch results into shared output with lock."""
        with open(lock_path, "w") as lf:
            fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
            try:
                existing = {}
                if output_path.exists():
                    try:
                        with open(output_path, "r") as f:
                            existing = json.load(f)
                    except (json.JSONDecodeError, IOError):
                        pass
                existing.update(batch_results)
                with open(output_path, "w") as f:
                    json.dump(existing, f)
                return len(existing)
            finally:
                fcntl.flock(lf.fileno(), fcntl.LOCK_UN)
    
    # Filter to remaining queries
    done_ids = load_done_ids()
    remaining = [(qid, qt, bm25) for qid, qt, bm25 in queries_subset if qid not in done_ids]
    
    if not remaining:
        print(f"[GPU {gpu_id}] All {len(queries_subset)} queries already done")
        return
    
    print(f"[GPU {gpu_id}] {len(remaining)}/{len(queries_subset)} queries to process")
    
    # Initialize reranker
    try:
        reranker = QwenReranker(gpu_memory_utilization=0.85)
    except Exception as e:
        print(f"[GPU {gpu_id}] Failed to initialize: {e}")
        return
    
    try:
        for i in range(0, len(remaining), RERANK_SAVE_INTERVAL):
            batch = remaining[i:i + RERANK_SAVE_INTERVAL]
            
            # Collect pairs
            all_pairs = []
            pair_info = []
            for query_id, query_text, bm25_results in batch:
                for passage_id, _ in bm25_results:
                    text = passage_texts.get(passage_id)
                    if text:
                        all_pairs.append((query_text, text))
                        pair_info.append((query_id, passage_id))
            
            if not all_pairs:
                continue
            
            # Score
            scores = reranker.score_multi_query(all_pairs)
            
            # Group by query_id
            batch_results = {}
            for (query_id, passage_id), score in zip(pair_info, scores):
                if query_id not in batch_results:
                    batch_results[query_id] = []
                batch_results[query_id].append((passage_id, score))
            
            # Sort each query's results
            for query_id in batch_results:
                batch_results[query_id].sort(key=lambda x: x[1], reverse=True)
            
            # Save to shared file
            total = save_batch(batch_results)
            print(f"[GPU {gpu_id}] +{len(batch_results)} queries, total {total}")
    
    except Exception as e:
        print(f"[GPU {gpu_id}] Error: {e}")
        raise
    
    finally:
        reranker.cleanup()
    
    print(f"[GPU {gpu_id}] Done")


def run_rerank_phase(
    queries: list[dict],
    retrieval_results: dict[str, list[tuple[str, float]]],
    documents: list[dict],
    doc_id_to_idx: dict[str, int],
    chunk_to_doc: dict[str, str],
    baseline_dir: Path,
    force: bool = False,
    gpu_ids: list[int] = None,
) -> dict[str, list[tuple[str, float]]]:
    """Run neural reranking with incremental saving and recovery support.
    
    All GPUs write to the same output file. On restart, completed queries are skipped.
    """
    output_path = baseline_dir / "2_rerank_results.json"
    lock_path = baseline_dir / ".rerank_lock"

    # Check existing progress
    existing_count = 0
    if output_path.exists() and not force:
        with open(output_path, "r") as f:
            existing = json.load(f)
        existing_count = len(existing)
        if existing_count >= len(queries):
            print(f"Rerank complete ({existing_count}/{len(queries)} queries)")
            return existing
        print(f"Resuming: {existing_count}/{len(queries)} queries done")

    # Force mode: clear progress
    if force and output_path.exists():
        print("Force mode: clearing progress...")
        output_path.unlink()

    print(f"Reranking {len(queries)} queries...")

    # Load or build passage texts (cached)
    passage_cache_path = baseline_dir / "2_rerank_passage_texts.json"
    if passage_cache_path.exists():
        print("Loading cached passage texts...")
        with open(passage_cache_path, "r") as f:
            passage_texts = json.load(f)
        print(f"Loaded {len(passage_texts)} passages from cache")
    else:
        all_chunk_ids = set()
        for results in retrieval_results.values():
            for chunk_id, _ in results:
                all_chunk_ids.add(chunk_id)

        print(f"Building passage texts for {len(all_chunk_ids)} chunks...")
        passage_texts = build_passage_texts(
            list(all_chunk_ids), documents, doc_id_to_idx, chunk_to_doc
        )
        print("Saving passage cache...")
        with open(passage_cache_path, "w") as f:
            json.dump(passage_texts, f)
        print(f"Cached {len(passage_texts)} passages")

    # Prepare queries
    queries_with_bm25 = []
    for q in queries:
        query_id = q["query_id"]
        query_text = q["query"]
        bm25_results = retrieval_results.get(query_id, [])
        queries_with_bm25.append((query_id, query_text, bm25_results))

    # Multi-GPU mode
    if gpu_ids and len(gpu_ids) > 1:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")  # Required for CUDA
        
        num_gpus = len(gpu_ids)
        queries_per_gpu = (len(queries_with_bm25) + num_gpus - 1) // num_gpus
        
        print(f"Multi-GPU: {num_gpus} GPUs, ~{queries_per_gpu} queries each")
        print(f"Save interval: {RERANK_SAVE_INTERVAL} queries")
        
        # Split queries across GPUs
        query_splits = []
        for i, gpu_id in enumerate(gpu_ids):
            start = i * queries_per_gpu
            end = min((i + 1) * queries_per_gpu, len(queries_with_bm25))
            if start < end:
                query_splits.append((gpu_id, queries_with_bm25[start:end]))
        
        # Launch workers (all write to same output_path)
        processes = []
        for gpu_id, queries_subset in query_splits:
            p = ctx.Process(
                target=_rerank_worker,
                args=(gpu_id, queries_subset, passage_texts, output_path, lock_path)
            )
            processes.append((gpu_id, p))
            p.start()
        
        # Wait for all
        failed = []
        for gpu_id, p in processes:
            p.join()
            if p.exitcode != 0:
                print(f"[GPU {gpu_id}] Failed (exit {p.exitcode})")
                failed.append(gpu_id)
        
        if failed:
            print(f"GPUs {failed} failed. Re-run to continue.")
        
    else:
        # Single GPU mode
        print("Single GPU mode...")
        reranker = QwenReranker()
        
        # Load existing
        rerank_results = {}
        if output_path.exists():
            try:
                with open(output_path, "r") as f:
                    rerank_results = json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        
        done_ids = set(rerank_results.keys())
        remaining = [(qid, qt, bm25) for qid, qt, bm25 in queries_with_bm25 if qid not in done_ids]
        
        try:
            for i in range(0, len(remaining), RERANK_SAVE_INTERVAL):
                batch = remaining[i:i + RERANK_SAVE_INTERVAL]
                batch_results = rerank_passages_batch(batch, passage_texts, reranker)
                rerank_results.update(batch_results)
                with open(output_path, "w") as f:
                    json.dump(rerank_results, f)
                print(f"Saved {len(rerank_results)}/{len(queries_with_bm25)} queries")
        finally:
            reranker.cleanup()

    # Load final results
    with open(output_path, "r") as f:
        rerank_results = json.load(f)

    print(f"Done: {len(rerank_results)} rerank results")
    return rerank_results


# =============================================================================
# PHASE: GENERATION
# =============================================================================

def build_generation_context(
    chunk_ids: list[tuple[str, float]],
    documents: list[dict],
    doc_id_to_idx: dict[str, int],
    chunk_to_doc: dict[str, str],
    top_k: int = GENERATION_K,
) -> list[dict]:
    """Build generation context from top-K reranked chunks.

    Uses GenerationModule.format_passage() for HTML-tagged format.

    Args:
        chunk_ids: List of (chunk_id, score) tuples from reranking.
        documents: List of documents.
        doc_id_to_idx: Mapping from doc_id to document index.
        chunk_to_doc: Mapping from chunk_id to doc_id.
        top_k: Number of top chunks to include.

    Returns:
        List of context dicts with passage_id, doc_id, text, score, etc.
    """
    context = []

    for chunk_id, score in chunk_ids[:top_k]:
        doc_id = chunk_to_doc.get(chunk_id, chunk_id)
        doc_idx = doc_id_to_idx.get(doc_id)

        if doc_idx is None:
            continue

        doc = documents[doc_idx]

        # Get chunk index
        try:
            chunk_idx = int(chunk_id.split("_chunk_")[-1])
        except (ValueError, IndexError):
            chunk_idx = 0

        # Get body chunk
        body_text = doc.get("body_text", "")
        body_chunks = chunk_text(body_text)
        body_chunk = body_chunks[chunk_idx] if chunk_idx < len(body_chunks) else ""

        # Format using generator's format_passage (HTML tags)
        passage_text = GenerationModule.format_passage(doc, body_chunk)

        context.append({
            "passage_id": chunk_id,
            "doc_id": doc_id,
            "doc_idx": doc_idx,
            "chunk_idx": chunk_idx,
            "text": passage_text,
            "score": score,
            "title": doc.get("title", ""),
            "link": doc.get("link", ""),
        })

    return context


async def run_generation_phase(
    queries: list[dict],
    rerank_results: dict[str, list[tuple[str, float]]],
    documents: list[dict],
    doc_id_to_idx: dict[str, int],
    chunk_to_doc: dict[str, str],
    baseline_dir: Path,
    force: bool = False,
    max_concurrent: int = 100,
) -> dict[str, dict]:
    """Run LLM generation for all queries.

    Args:
        queries: List of query dictionaries.
        rerank_results: Reranking results.
        documents: List of documents.
        doc_id_to_idx: Mapping from doc_id to document index.
        chunk_to_doc: Mapping from chunk_id to doc_id.
        baseline_dir: Baseline directory.
        force: Force regenerate even if exists.
        max_concurrent: Maximum concurrent API calls.

    Returns:
        generation_results: Dict mapping query_id to generation output.
    """
    output_path = baseline_dir / "3_generation_results.json"

    # Check if exists
    if output_path.exists() and not force:
        print(f"Loading existing generation results from {output_path}")
        with open(output_path, "r", encoding="utf-8") as f:
            return json.load(f)

    if force and output_path.exists():
        print(f"Force mode: removing existing generation results...")
        output_path.unlink()

    print(f"Running LLM generation for {len(queries)} queries...")

    generator = GenerationModule()
    semaphore = asyncio.Semaphore(max_concurrent)
    generation_results = {}

    async def generate_one(q: dict) -> tuple[str, dict]:
        query_id = q["query_id"]
        query_text = q["query"]

        # Get rerank results
        reranked = rerank_results.get(query_id, [])

        # Build context
        context = build_generation_context(
            reranked, documents, doc_id_to_idx, chunk_to_doc
        )

        # Build chunks for generation
        chunks = []
        for ctx in context:
            doc_idx = ctx["doc_idx"]
            doc = documents[doc_idx]
            chunk_idx = ctx["chunk_idx"]
            body_text = doc.get("body_text", "")
            body_chunks = chunk_text(body_text)
            body_chunk = body_chunks[chunk_idx] if chunk_idx < len(body_chunks) else ""
            chunks.append((doc, body_chunk))

        async with semaphore:
            try:
                result = await generator.generate_from_chunks(query_text, chunks)
                return query_id, {
                    "query": query_text,
                    "response": result.response,
                    "reasoning": result.reasoning,
                    "context_passages": context,
                }
            except Exception as e:
                print(f"Error generating for {query_id}: {e}")
                return query_id, {
                    "query": query_text,
                    "response": "",
                    "reasoning": "",
                    "context_passages": context,
                    "error": str(e),
                }

    # Run all in parallel
    tasks = [generate_one(q) for q in queries]
    results = await asyncio.gather(*tasks)

    for query_id, result in results:
        generation_results[query_id] = result

    # Save results
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(generation_results, f, indent=2)

    print(f"Saved generation results to {output_path}")
    return generation_results


# =============================================================================
# PHASE: TARGET SELECTION
# =============================================================================

def run_targets_phase(
    queries: list[dict],
    generation_results: dict[str, dict],
    baseline_dir: Path,
    force: bool = False,
) -> dict[str, dict]:
    """Select target documents for SAGEO optimization.

    For each query, randomly selects one document from the generation context.
    Uses fixed seed for reproducibility across all methods.

    Args:
        queries: List of query dictionaries.
        generation_results: Generation results with context passages.
        baseline_dir: Baseline directory.
        force: Force regenerate even if exists.

    Returns:
        targets: Dict mapping query_id to target info.
    """
    output_path = baseline_dir / "4_targets.json"

    # Check if exists
    if output_path.exists() and not force:
        print(f"Loading existing targets from {output_path}")
        with open(output_path, "r", encoding="utf-8") as f:
            return json.load(f)

    if force and output_path.exists():
        print(f"Force mode: removing existing targets...")
        output_path.unlink()

    print(f"Selecting targets for {len(queries)} queries...")
    set_seed(SEED)  # Fixed seed for reproducibility

    targets = {}

    skipped = 0
    for q in queries:
        query_id = q["query_id"]
        gen_result = generation_results.get(query_id, {})
        
        # Skip failed generations
        if gen_result.get("error") or not gen_result.get("response"):
            skipped += 1
            continue
        
        context = gen_result.get("context_passages", [])
        if not context:
            skipped += 1
            continue

        # Random selection from context
        selected = random.choice(context)

        targets[query_id] = {
            "query_id": query_id,
            "query": q["query"],
            "dataset": q.get("dataset", ""),
            "doc_id": selected["doc_id"],
            "doc_idx": selected["doc_idx"],
            "passage_id": selected["passage_id"],
            "title": selected.get("title", ""),
            "link": selected.get("link", ""),
            "baseline_rank": context.index(selected) + 1,
        }

    # Save targets
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(targets, f, indent=2)

    print(f"Selected {len(targets)} targets (skipped {skipped} failed), saved to {output_path}")
    return targets


# =============================================================================
# CHECK & STATUS
# =============================================================================

def check_baseline(baseline_dir: Path) -> dict:
    """Check existing baseline status."""
    status = {
        "exists": baseline_dir.exists(),
        "index": (baseline_dir / "baseline_index" / "lucene").exists(),
        "retrieval": (baseline_dir / "1_retrieval_results.json").exists(),
        "rerank": (baseline_dir / "2_rerank_results.json").exists(),
        "generation": (baseline_dir / "3_generation_results.json").exists(),
        "targets": (baseline_dir / "4_targets.json").exists(),
    }

    # Count items in each file
    if status["retrieval"]:
        with open(baseline_dir / "1_retrieval_results.json") as f:
            status["retrieval_count"] = len(json.load(f))
    if status["rerank"]:
        with open(baseline_dir / "2_rerank_results.json") as f:
            status["rerank_count"] = len(json.load(f))
    if status["generation"]:
        with open(baseline_dir / "3_generation_results.json") as f:
            status["generation_count"] = len(json.load(f))
    if status["targets"]:
        with open(baseline_dir / "4_targets.json") as f:
            status["targets_count"] = len(json.load(f))

    return status


# =============================================================================
# MAIN
# =============================================================================

async def main():
    parser = argparse.ArgumentParser(description="Generate baseline for SAGEO experiments")
    parser.add_argument("--check", action="store_true", help="Check existing baseline status")
    parser.add_argument("--phase", type=str, default="all", 
                        choices=VALID_PHASES,
                        help="Phase to run: index, retrieval, rerank, generation, targets, all")
    parser.add_argument("--force", action="store_true", help="Force regenerate (overwrites existing)")
    parser.add_argument("--test", action="store_true", help="Test mode (small subset)")
    parser.add_argument("--num_per_dataset", type=int, default=300, help="Queries per dataset")
    parser.add_argument("--datasets", type=str, default=None, help="Comma-separated dataset names")
    parser.add_argument("--gpu_ids", type=str, default=None, 
                        help="Comma-separated GPU IDs for parallel reranking (e.g., '0,1,2,3,4,5'). "
                             "If not provided, auto-detects free GPUs.")
    args = parser.parse_args()
    
    # Use GPU IDs from early detection
    gpu_ids = _GPU_IDS

    # Determine output directory
    baseline_dir = BASELINE_DIR
    if args.test:
        baseline_dir = EXPERIMENTS_DIR / "baseline_test"
        args.num_per_dataset = min(args.num_per_dataset, 10)

    # Check mode
    if args.check:
        status = check_baseline(baseline_dir)
        print("\n=== Baseline Status ===")
        print(f"Directory: {baseline_dir}")
        print(f"Exists: {status['exists']}")
        if status["exists"]:
            print(f"Index: {'✓' if status['index'] else '✗'}")
            print(f"Retrieval: {'✓' if status['retrieval'] else '✗'}" + 
                  (f" ({status.get('retrieval_count', 0)} queries)" if status['retrieval'] else ""))
            print(f"Rerank: {'✓' if status['rerank'] else '✗'}" +
                  (f" ({status.get('rerank_count', 0)} queries)" if status['rerank'] else ""))
            print(f"Generation: {'✓' if status['generation'] else '✗'}" +
                  (f" ({status.get('generation_count', 0)} queries)" if status['generation'] else ""))
            print(f"Targets: {'✓' if status['targets'] else '✗'}" +
                  (f" ({status.get('targets_count', 0)} targets)" if status['targets'] else ""))
        return

    # Create output directory
    baseline_dir.mkdir(parents=True, exist_ok=True)

    # Datasets
    if args.datasets:
        datasets = [d.strip() for d in args.datasets.split(",")]
    else:
        datasets = ALL_DATASETS

    print("\n" + "=" * 60)
    print("SAGEO Baseline Generation")
    print("=" * 60)
    print(f"Output directory: {baseline_dir}")
    print(f"Phase: {args.phase}")
    print(f"Force: {args.force}")
    print(f"Datasets: {datasets}")
    print(f"Queries per dataset: {args.num_per_dataset}")
    print("=" * 60 + "\n")

    # Determine which phases to run
    run_all = args.phase == "all"
    run_index = run_all or args.phase == "index"
    run_retrieval = run_all or args.phase == "retrieval"
    run_rerank = run_all or args.phase == "rerank"
    run_generation = run_all or args.phase == "generation"
    run_targets = run_all or args.phase == "targets"

    # Load data
    print("Loading corpus...")
    documents, doc_id_to_idx = load_corpus()

    print("Loading queries...")
    queries = load_queries(datasets, args.num_per_dataset)

    # Phase: Index
    chunk_to_doc = None
    if run_index:
        print("\n--- Phase: Index Building ---")
        chunk_to_doc = run_index_phase(documents, baseline_dir, force=args.force)
    else:
        # Load existing chunk_to_doc if needed
        try:
            chunk_to_doc = load_chunk_to_doc(baseline_dir)
        except FileNotFoundError as e:
            if run_retrieval or run_rerank or run_generation:
                print(f"Error: {e}")
                print("Run --phase index first")
                sys.exit(1)

    # Phase: Retrieval
    retrieval_results = None
    if run_retrieval:
        print("\n--- Phase: Retrieval ---")
        retrieval_results = run_retrieval_phase(
            queries, baseline_dir, chunk_to_doc, force=args.force
        )
    elif run_rerank or run_generation:
        # Load existing retrieval results
        retrieval_path = baseline_dir / "1_retrieval_results.json"
        if retrieval_path.exists():
            with open(retrieval_path) as f:
                retrieval_results = json.load(f)
        else:
            print("Error: Retrieval results not found. Run --phase retrieval first")
            sys.exit(1)

    # Phase: Reranking
    rerank_results = None
    if run_rerank:
        print("\n--- Phase: Reranking ---")
        rerank_results = run_rerank_phase(
            queries, retrieval_results, documents, doc_id_to_idx, 
            chunk_to_doc, baseline_dir, force=args.force, gpu_ids=gpu_ids
        )
    elif run_generation or run_targets:
        # Load existing rerank results
        rerank_path = baseline_dir / "2_rerank_results.json"
        if rerank_path.exists():
            with open(rerank_path) as f:
                rerank_results = json.load(f)
        else:
            print("Error: Rerank results not found. Run --phase rerank first")
            sys.exit(1)

    # Phase: Generation
    generation_results = None
    if run_generation:
        print("\n--- Phase: Generation ---")
        generation_results = await run_generation_phase(
            queries, rerank_results, documents, doc_id_to_idx,
            chunk_to_doc, baseline_dir, force=args.force
        )
    elif run_targets:
        # Load existing generation results
        generation_path = baseline_dir / "3_generation_results.json"
        if generation_path.exists():
            with open(generation_path) as f:
                generation_results = json.load(f)
        else:
            print("Error: Generation results not found. Run --phase generation first")
            sys.exit(1)

    # Phase: Target Selection
    if run_targets:
        print("\n--- Phase: Target Selection ---")
        targets = run_targets_phase(
            queries, generation_results, baseline_dir, force=args.force
        )

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)

    # Show final status
    status = check_baseline(baseline_dir)
    print(f"\nBaseline status:")
    print(f"  Index: {'✓' if status['index'] else '✗'}")
    print(f"  Retrieval: {'✓' if status['retrieval'] else '✗'}")
    print(f"  Rerank: {'✓' if status['rerank'] else '✗'}")
    print(f"  Generation: {'✓' if status['generation'] else '✗'}")
    print(f"  Targets: {'✓' if status['targets'] else '✗'}")


if __name__ == "__main__":
    asyncio.run(main())
