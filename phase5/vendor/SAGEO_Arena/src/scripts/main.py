"""
SAGEO Main Experiment Runner

Runs GEO optimization experiments with different methods and scopes.
Requires baseline to be generated first via run_baseline.py.

================================================================================
PREREQUISITES
================================================================================

Before running this script, generate baseline:
    python -m src.scripts.run_baseline

================================================================================
USAGE
================================================================================

# Single scope + single method
python -m src.scripts.main --scope body --methods fluency

# Multiple scopes + multiple methods (runs all combinations)
python -m src.scripts.main --scope body,structured --methods fluency,authoritative

# All scopes + all methods
python -m src.scripts.main --scope body,structured,both --methods all

# Run specific phase only
python -m src.scripts.main --scope body --methods fluency --phase optimization

# GPU selection for reranking (auto-detects if not provided)
python -m src.scripts.main --scope body --methods fluency --gpu_ids 0,1,2,3

# Test with sampled subset (N per dataset, saved under experiments/test_N/)
python -m src.scripts.main --scope both --methods stage_aware --num_per_dataset 100

================================================================================
SCOPES
================================================================================

- body: Optimize body_text only
- structured: Optimize title, meta_description, headings, jsonld only
- both: Optimize all fields

================================================================================
METHODS
================================================================================

Standard methods work with all scopes:
- autogeo, fluency, authoritative, citations, statistics
- unique_words, quotes, simple_language, technical_terms, content_improvement

Combo/stage-aware methods are intended for 'both' scope (not enforced):
- combo_easy_quote, combo_easy_stats, combo_fluency_quote, combo_fluency_stats
- stage_aware

================================================================================
OUTPUT STRUCTURE
================================================================================

experiments/
├── baseline/                      # From run_baseline.py
├── body/
│   └── fluency/
│       ├── 1_optimized_documents.json
│       ├── failed_docs.json
│       ├── index/
│       ├── 2_retrieval_results.json
│       ├── 3_rerank_results.json
│       ├── 4_generation_results.json
│       └── 5_evaluation_results.json
├── structured/
│   └── ...
└── both/
    └── ...
"""

# =============================================================================
# EARLY GPU DETECTION (before torch imports for cleaner startup)
# =============================================================================
import os
import subprocess
import sys

def _detect_free_gpus(min_free_memory_gb: float = 38.0) -> list[int]:
    """Detect GPUs with sufficient free memory.
    
    Args:
        min_free_memory_gb: Minimum free memory in GB required.
    
    Returns:
        List of GPU indices with sufficient free memory.
    """
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
    """Parse --gpu_ids early and auto-detect free GPUs if not provided.
    
    Returns:
        List of GPU IDs to use for data parallelism, or None to use default.
    """
    import argparse
    
    # Quick parse just for --gpu_ids
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
import shutil
from pathlib import Path

from src.modules import ROOT_DIR
from src.modules.optimizer import DocumentOptimizer
from src.modules.generator import GenerationModule
from src.modules.reranker import QwenReranker
from src.modules.retriever_bm25 import (
    build_field_jsonl,
    build_indices,
    load_searchers,
    search,
    chunk_text,
)
from src.modules.utils.sageo_methods import (
    get_method,
    list_methods,
    TARGET_FIELDS_BODY,
    TARGET_FIELDS_STRUCTURED,
    TARGET_FIELDS_ALL,
)
from src.evaluation import run_evaluation_phase

# =============================================================================
# CONSTANTS
# =============================================================================

EXPERIMENTS_DIR = ROOT_DIR / "experiments"
BASELINE_DIR = EXPERIMENTS_DIR / "baseline"

RETRIEVAL_K = 100  # Top-K for BM25 retrieval
GENERATION_K = 10  # Top-K chunks for generation context
SEED = 42

# Scope name to target_fields mapping
SCOPE_TO_TARGET_FIELDS = {
    "body": TARGET_FIELDS_BODY,
    "structured": TARGET_FIELDS_STRUCTURED,
    "both": TARGET_FIELDS_ALL,
}

VALID_SCOPES = list(SCOPE_TO_TARGET_FIELDS.keys())
VALID_PHASES = ["optimization", "index", "retrieval", "rerank", "generation", "evaluation", "all"]

# =============================================================================
# BASELINE LOADING
# =============================================================================


def load_baseline(num_per_dataset: int = None) -> dict:
    """Load baseline data (read-only).

    Args:
        num_per_dataset: If set, sample N targets per dataset.

    Returns:
        Dict with: targets, retrieval_results, rerank_results, generation_results,
                   documents, doc_id_to_idx, chunk_to_doc, queries
    """
    # Check baseline exists
    required_files = [
        "4_targets.json",
        "1_retrieval_results.json",
        "2_rerank_results.json", 
        "3_generation_results.json",
        "baseline_index/jsonl/metadata.json",
    ]
    
    for f in required_files:
        if not (BASELINE_DIR / f).exists():
            print(f"Error: Baseline file not found: {BASELINE_DIR / f}")
            print("Run 'python -m src.scripts.run_baseline' first")
            sys.exit(1)
    
    print("Loading baseline data...")
    
    # Load targets
    with open(BASELINE_DIR / "4_targets.json") as f:
        targets = json.load(f)

    # Sample N per dataset if requested
    if num_per_dataset is not None:
        from collections import defaultdict
        by_dataset = defaultdict(list)
        for qid, t in targets.items():
            by_dataset[t.get("dataset", "")].append(qid)

        random.seed(SEED)
        sampled = {}
        for dataset, qids in sorted(by_dataset.items()):
            selected = random.sample(sorted(qids), min(num_per_dataset, len(qids)))
            for qid in selected:
                sampled[qid] = targets[qid]
        print(f"  Targets: {len(sampled)}/{len(targets)} (sampled {num_per_dataset}/dataset)")
        targets = sampled
    else:
        print(f"  Targets: {len(targets)} queries")
    
    # Load baseline results
    with open(BASELINE_DIR / "1_retrieval_results.json") as f:
        retrieval_results = json.load(f)
    with open(BASELINE_DIR / "2_rerank_results.json") as f:
        rerank_results = json.load(f)
    with open(BASELINE_DIR / "3_generation_results.json") as f:
        generation_results = json.load(f)
    print(f"  Baseline results: {len(retrieval_results)} retrieval, {len(rerank_results)} rerank, {len(generation_results)} generation")
    
    # Load chunk_to_doc mapping
    with open(BASELINE_DIR / "baseline_index/jsonl/metadata.json") as f:
        metadata = json.load(f)
    chunk_to_doc = metadata["chunk_to_doc"]
    print(f"  Chunk mappings: {len(chunk_to_doc)}")
    
    # Load corpus
    corpus_path = ROOT_DIR / "data" / "corpus" / "documents_filtered.jsonl"
    documents = []
    with open(corpus_path) as f:
        for line in f:
            documents.append(json.loads(line))
    
    # Build doc_id mapping
    doc_id_to_idx = {}
    for idx, doc in enumerate(documents):
        doc_id = doc.get("_id") or f"doc_{idx}"
        doc["_id"] = doc_id
        doc_id_to_idx[doc_id] = idx
    print(f"  Corpus: {len(documents)} documents")
    
    # Build queries list from targets
    queries = []
    for query_id, target in targets.items():
        queries.append({
            "query_id": query_id,
            "query": target["query"],
        })
    
    return {
        "targets": targets,
        "retrieval_results": retrieval_results,
        "rerank_results": rerank_results,
        "generation_results": generation_results,
        "documents": documents,
        "doc_id_to_idx": doc_id_to_idx,
        "chunk_to_doc": chunk_to_doc,
        "queries": queries,
    }


# =============================================================================
# PHASE 1: OPTIMIZATION
# =============================================================================


async def run_optimization_phase(
    targets: dict,
    documents: list[dict],
    doc_id_to_idx: dict[str, int],
    method: str,
    scope: str,
    exp_dir: Path,
    force: bool = False,
) -> tuple[dict, set]:
    """Optimize target documents.
    
    Args:
        targets: Target info dict (query_id -> target).
        documents: Corpus documents.
        doc_id_to_idx: Doc ID to index mapping.
        method: GEO method name.
        scope: Optimization scope (body, structured, both).
        exp_dir: Experiment directory.
        force: Force regenerate.
    
    Returns:
        (optimized_docs, failed_docs): Optimized documents and failed doc IDs.
    """
    output_path = exp_dir / "1_optimized_documents.json"
    failed_path = exp_dir / "failed_docs.json"
    target_fields = SCOPE_TO_TARGET_FIELDS[scope]
    
    # Load existing results
    optimized_docs = {}
    if output_path.exists() and not force:
        with open(output_path) as f:
            optimized_docs = json.load(f)
    
    # Load failed docs (never retry)
    failed_docs = set()
    if failed_path.exists():
        with open(failed_path) as f:
            failed_docs = set(json.load(f))
    
    if force:
        optimized_docs = {}
        failed_docs = set()
    
    # Collect unique target doc_ids
    target_doc_ids = set(t["doc_id"] for t in targets.values())
    
    # Filter to docs needing optimization
    docs_to_optimize = {
        doc_id for doc_id in target_doc_ids
        if doc_id not in optimized_docs and doc_id not in failed_docs
    }
    
    if not docs_to_optimize:
        print(f"  All {len(target_doc_ids)} documents already processed")
        return optimized_docs, failed_docs
    
    print(f"  Optimizing {len(docs_to_optimize)} documents ({len(optimized_docs)} done, {len(failed_docs)} failed)...")
    
    optimizer = DocumentOptimizer()
    semaphore = asyncio.Semaphore(50)  # Max concurrent
    
    async def optimize_one(doc_id: str):
        doc_idx = doc_id_to_idx.get(doc_id)
        if doc_idx is None:
            return
        
        doc = documents[doc_idx]
        
        async with semaphore:
            try:
                result = await optimizer.optimize_document(doc, method, target_fields)
                optimized_docs[doc_id] = result
                
                # Incremental save
                with open(output_path, "w") as f:
                    json.dump(optimized_docs, f)
                
                print(f"    Optimized {doc_id} ({len(optimized_docs)}/{len(target_doc_ids)})")
                
            except Exception as e:
                print(f"    DROPPED {doc_id}: {str(e)[:100]}")
                failed_docs.add(doc_id)
                with open(failed_path, "w") as f:
                    json.dump(list(failed_docs), f)
    
    tasks = [optimize_one(doc_id) for doc_id in docs_to_optimize]
    await asyncio.gather(*tasks)
    
    print(f"  Optimization complete: {len(optimized_docs)} success, {len(failed_docs)} failed")
    return optimized_docs, failed_docs


# =============================================================================
# PHASE 2: INDEX BUILDING
# =============================================================================


def run_index_phase(
    optimized_docs: dict,
    failed_docs: set,
    documents: list[dict],
    doc_id_to_idx: dict[str, int],
    exp_dir: Path,
    force: bool = False,
) -> dict[str, str]:
    """Build BM25 index with optimized documents.
    
    Returns:
        chunk_to_doc: Mapping from chunk_id to doc_id.
    """
    index_dir = exp_dir / "index"
    jsonl_dir = index_dir / "jsonl"
    lucene_dir = index_dir / "lucene"
    metadata_path = jsonl_dir / "metadata.json"
    
    # Check if exists
    if metadata_path.exists() and not force:
        print(f"  Loading existing index from {index_dir}")
        with open(metadata_path) as f:
            return json.load(f)["chunk_to_doc"]
    
    # Clear and rebuild
    if index_dir.exists():
        shutil.rmtree(index_dir)
    jsonl_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"  Building index with {len(optimized_docs)} optimized documents...")
    
    # Create modified document list
    modified_docs = []
    for idx, doc in enumerate(documents):
        doc_id = doc.get("_id") or f"doc_{idx}"
        
        if doc_id in failed_docs:
            continue  # Skip failed docs entirely
        
        if doc_id in optimized_docs:
            modified_docs.append(optimized_docs[doc_id])
        else:
            modified_docs.append(doc)
    
    print(f"  Total documents in index: {len(modified_docs)}")
    
    # Build JSONL and indices
    chunk_to_doc = build_field_jsonl(modified_docs, jsonl_dir)
    print(f"  Created {len(chunk_to_doc)} chunks")
    
    build_indices(jsonl_dir, lucene_dir)
    print(f"  Built Lucene indices")
    
    return chunk_to_doc


# =============================================================================
# PHASE 3: RETRIEVAL
# =============================================================================


def run_retrieval_phase(
    queries: list[dict],
    targets: dict,
    failed_docs: set,
    chunk_to_doc: dict[str, str],
    exp_dir: Path,
    force: bool = False,
) -> dict:
    """Run BM25 retrieval.
    
    Returns:
        retrieval_results: query_id -> [(chunk_id, score), ...]
    """
    output_path = exp_dir / "2_retrieval_results.json"
    index_dir = exp_dir / "index"
    lucene_dir = index_dir / "lucene"
    
    # Load existing
    retrieval_results = {}
    if output_path.exists() and not force:
        with open(output_path) as f:
            retrieval_results = json.load(f)
    
    # Filter queries (skip failed target docs)
    queries_to_run = []
    for q in queries:
        query_id = q["query_id"]
        if query_id in retrieval_results:
            continue
        target = targets.get(query_id, {})
        if target.get("doc_id") in failed_docs:
            continue
        queries_to_run.append(q)
    
    if not queries_to_run:
        print(f"  All {len(queries)} queries already retrieved")
        return retrieval_results
    
    print(f"  Running retrieval for {len(queries_to_run)} queries...")
    
    searchers = load_searchers(lucene_dir)
    
    for i, q in enumerate(queries_to_run):
        query_id = q["query_id"]
        query_text = q["query"]
        
        results = search(query_text, searchers, chunk_to_doc, top_k=RETRIEVAL_K)
        retrieval_results[query_id] = results
        
        if (i + 1) % 100 == 0:
            print(f"    Processed {i + 1}/{len(queries_to_run)} queries")
    
    # Save
    with open(output_path, "w") as f:
        json.dump(retrieval_results, f)
    
    print(f"  Saved {len(retrieval_results)} retrieval results")
    return retrieval_results


# =============================================================================
# PHASE 4: RERANKING
# =============================================================================


def build_passage_texts(
    chunk_ids: list[str],
    documents: list[dict],
    doc_id_to_idx: dict[str, int],
    chunk_to_doc: dict[str, str],
    optimized_docs: dict,
) -> dict[str, str]:
    """Build passage texts for reranking."""
    from src.modules.reranker import QwenReranker
    
    passage_texts = {}
    
    for chunk_id in chunk_ids:
        if chunk_id in passage_texts:
            continue
        
        doc_id = chunk_to_doc.get(chunk_id, chunk_id)
        
        # Use optimized doc if available
        if doc_id in optimized_docs:
            doc = optimized_docs[doc_id]
        else:
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
        
        # Format passage
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
    
    Each worker uses 1 GPU (data parallelism). All workers write to the same 
    output file using file locking. On restart, queries already in the file are skipped.
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
    
    # Initialize reranker (single GPU mode)
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
    targets: dict,
    failed_docs: set,
    retrieval_results: dict,
    documents: list[dict],
    doc_id_to_idx: dict[str, int],
    chunk_to_doc: dict[str, str],
    optimized_docs: dict,
    exp_dir: Path,
    force: bool = False,
    gpu_ids: list[int] = None,
) -> dict:
    """Run neural reranking with multi-GPU data parallelism.
    
    Uses the same approach as run_baseline.py: each GPU runs in a separate process.
    All GPUs write to the same output file using file locking.
    
    Returns:
        rerank_results: query_id -> [(chunk_id, score), ...]
    """
    output_path = exp_dir / "3_rerank_results.json"
    lock_path = exp_dir / ".rerank_lock"
    
    # Check existing progress
    existing_count = 0
    if output_path.exists() and not force:
        with open(output_path) as f:
            existing = json.load(f)
        existing_count = len(existing)
    
    # Filter queries (exclude already done, failed docs, missing retrieval)
    valid_query_ids = set()
    for q in queries:
        query_id = q["query_id"]
        target = targets.get(query_id, {})
        if target.get("doc_id") in failed_docs:
            continue
        if query_id not in retrieval_results:
            continue
        valid_query_ids.add(query_id)
    
    if existing_count >= len(valid_query_ids):
        print(f"  Rerank complete ({existing_count}/{len(valid_query_ids)} queries)")
        with open(output_path) as f:
            return json.load(f)
    
    if existing_count > 0:
        print(f"  Resuming: {existing_count}/{len(valid_query_ids)} queries done")
    
    # Force mode: clear progress
    if force and output_path.exists():
        print("  Force mode: clearing progress...")
        output_path.unlink()
    
    print(f"  Reranking {len(valid_query_ids)} queries...")
    
    # Collect all chunk IDs needed
    all_chunk_ids = set()
    for query_id in valid_query_ids:
        results = retrieval_results.get(query_id, [])
        for chunk_id, _ in results:
            all_chunk_ids.add(chunk_id)
    
    # Build or load cached passage texts
    passage_cache_path = exp_dir / "3_rerank_passage_texts.json"
    
    if passage_cache_path.exists() and not force:
        print(f"  Loading cached passage texts...")
        with open(passage_cache_path) as f:
            passage_texts = json.load(f)
        # Check if we need to build any missing chunks
        missing_chunks = all_chunk_ids - set(passage_texts.keys())
        if missing_chunks:
            print(f"  Building {len(missing_chunks)} additional passage texts...")
            new_texts = build_passage_texts(
                list(missing_chunks), documents, doc_id_to_idx, chunk_to_doc, optimized_docs
            )
            passage_texts.update(new_texts)
            with open(passage_cache_path, "w") as f:
                json.dump(passage_texts, f)
    else:
        print(f"  Building passage texts for {len(all_chunk_ids)} chunks...")
        passage_texts = build_passage_texts(
            list(all_chunk_ids), documents, doc_id_to_idx, chunk_to_doc, optimized_docs
        )
        print(f"  Saving passage texts cache...")
        with open(passage_cache_path, "w") as f:
            json.dump(passage_texts, f)
    
    # Prepare queries with their retrieval results
    queries_with_bm25 = []
    for q in queries:
        query_id = q["query_id"]
        if query_id not in valid_query_ids:
            continue
        query_text = q["query"]
        bm25_results = retrieval_results.get(query_id, [])
        queries_with_bm25.append((query_id, query_text, bm25_results))
    
    # Multi-GPU mode (data parallelism - each worker uses 1 GPU)
    if gpu_ids and len(gpu_ids) > 1:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")  # Required for CUDA
        
        num_gpus = len(gpu_ids)
        queries_per_gpu = (len(queries_with_bm25) + num_gpus - 1) // num_gpus
        
        print(f"  Multi-GPU: {num_gpus} GPUs, ~{queries_per_gpu} queries each")
        print(f"  Save interval: {RERANK_SAVE_INTERVAL} queries")
        
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
                print(f"  [GPU {gpu_id}] Failed (exit {p.exitcode})")
                failed.append(gpu_id)
        
        if failed:
            print(f"  GPUs {failed} failed. Re-run to continue.")
    
    else:
        # Single GPU mode
        print("  Single GPU mode...")
        reranker = QwenReranker()
        
        # Load existing
        rerank_results = {}
        if output_path.exists():
            try:
                with open(output_path) as f:
                    rerank_results = json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        
        try:
            BATCH_SIZE = 20
            remaining = [(qid, qt, bm25) for qid, qt, bm25 in queries_with_bm25 
                         if qid not in rerank_results]
            
            for i in range(0, len(remaining), BATCH_SIZE):
                batch = remaining[i:i + BATCH_SIZE]
                
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
                
                # Group by query
                batch_results = {}
                for (query_id, chunk_id), score in zip(pair_info, scores):
                    if query_id not in batch_results:
                        batch_results[query_id] = []
                    batch_results[query_id].append((chunk_id, score))
                
                # Sort
                for query_id in batch_results:
                    batch_results[query_id].sort(key=lambda x: x[1], reverse=True)
                
                rerank_results.update(batch_results)
                
                # Save checkpoint
                with open(output_path, "w") as f:
                    json.dump(rerank_results, f)
                
                print(f"    Progress: {min(i + BATCH_SIZE, len(remaining))}/{len(remaining)}")
        
        finally:
            reranker.cleanup()
    
    # Load final results
    with open(output_path) as f:
        rerank_results = json.load(f)
    
    print(f"  Saved {len(rerank_results)} rerank results")
    return rerank_results


# =============================================================================
# PHASE 5: GENERATION
# =============================================================================


async def run_generation_phase(
    queries: list[dict],
    targets: dict,
    failed_docs: set,
    rerank_results: dict,
    documents: list[dict],
    doc_id_to_idx: dict[str, int],
    chunk_to_doc: dict[str, str],
    optimized_docs: dict,
    exp_dir: Path,
    force: bool = False,
) -> dict:
    """Run LLM generation.
    
    Returns:
        generation_results: query_id -> generation output
    """
    output_path = exp_dir / "4_generation_results.json"
    
    # Load existing
    generation_results = {}
    if output_path.exists() and not force:
        with open(output_path) as f:
            generation_results = json.load(f)
    
    # Filter queries
    queries_to_run = []
    for q in queries:
        query_id = q["query_id"]
        if query_id in generation_results and generation_results[query_id].get("response"):
            continue
        target = targets.get(query_id, {})
        if target.get("doc_id") in failed_docs:
            continue
        if query_id not in rerank_results:
            continue
        queries_to_run.append(q)
    
    if not queries_to_run:
        print(f"  All queries already generated")
        return generation_results
    
    print(f"  Generating for {len(queries_to_run)} queries...")
    
    generator = GenerationModule()
    semaphore = asyncio.Semaphore(100)
    
    async def generate_one(q: dict):
        query_id = q["query_id"]
        query_text = q["query"]
        
        # Build context from top-K reranked
        reranked = rerank_results.get(query_id, [])[:GENERATION_K]
        
        chunks = []
        context_passages = []
        
        for chunk_id, score in reranked:
            doc_id = chunk_to_doc.get(chunk_id)
            
            # Use optimized doc if available
            if doc_id in optimized_docs:
                doc = optimized_docs[doc_id]
            else:
                doc_idx = doc_id_to_idx.get(doc_id)
                if doc_idx is None:
                    continue
                doc = documents[doc_idx]
            
            # Get chunk
            try:
                chunk_idx = int(chunk_id.split("_chunk_")[-1])
            except (ValueError, IndexError):
                chunk_idx = 0
            
            body_text = doc.get("body_text", "")
            body_chunks = chunk_text(body_text)
            body_chunk = body_chunks[chunk_idx] if chunk_idx < len(body_chunks) else ""
            
            chunks.append((doc, body_chunk))
            context_passages.append({
                "passage_id": chunk_id,
                "doc_id": doc_id,
                "score": score,
            })
        
        async with semaphore:
            try:
                result = await generator.generate_from_chunks(query_text, chunks)
                return query_id, {
                    "query": query_text,
                    "response": result.response,
                    "reasoning": result.reasoning,
                    "context_passages": context_passages,
                }
            except Exception as e:
                print(f"    Error for {query_id}: {str(e)[:50]}")
                return query_id, {
                    "query": query_text,
                    "response": "",
                    "reasoning": "",
                    "context_passages": context_passages,
                    "error": str(e),
                }
    
    tasks = [generate_one(q) for q in queries_to_run]
    results = await asyncio.gather(*tasks)
    
    for query_id, result in results:
        generation_results[query_id] = result
    
    # Save
    with open(output_path, "w") as f:
        json.dump(generation_results, f, indent=2)
    
    print(f"  Saved {len(generation_results)} generation results")
    return generation_results


# =============================================================================
# PHASE 6: EVALUATION
# =============================================================================


# =============================================================================
# MAIN EXPERIMENT RUNNER
# =============================================================================


async def run_experiment(
    scope: str,
    method: str,
    baseline: dict,
    phases: list[str] = None,
    force: bool = False,
    gpu_ids: list[int] = None,
    experiments_dir: Path = None,
):
    """Run a single experiment (scope + method combination)."""
    if phases is None:
        phases = ["all"]
    if experiments_dir is None:
        experiments_dir = EXPERIMENTS_DIR

    exp_dir = experiments_dir / scope / method
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Experiment: {scope}/{method}")
    print(f"{'='*60}")
    
    # Determine which phases to run
    run_all = "all" in phases
    run_optimization = run_all or "optimization" in phases
    run_index = run_all or "index" in phases
    run_retrieval = run_all or "retrieval" in phases
    run_rerank = run_all or "rerank" in phases
    run_generation = run_all or "generation" in phases
    run_evaluation = run_all or "evaluation" in phases
    
    # Define file paths for dependency checking
    opt_path = exp_dir / "1_optimized_documents.json"
    failed_path = exp_dir / "failed_docs.json"
    index_path = exp_dir / "index" / "lucene"
    meta_path = exp_dir / "index" / "jsonl" / "metadata.json"
    ret_path = exp_dir / "2_retrieval_results.json"
    rerank_path = exp_dir / "3_rerank_results.json"
    gen_path = exp_dir / "4_generation_results.json"
    
    # Validate dependencies before running
    def check_dependency(phase_name: str, required_file: Path, required_phase: str):
        if not required_file.exists():
            print(f"Error: {phase_name} requires {required_phase} to complete first.")
            print(f"  Missing: {required_file}")
            print(f"  Run: --phase {required_phase}")
            sys.exit(1)
    
    if run_index and not run_optimization:
        check_dependency("Index", opt_path, "optimization")
    if run_retrieval and not run_index:
        check_dependency("Retrieval", index_path, "index")
    if run_rerank and not run_retrieval:
        check_dependency("Rerank", ret_path, "retrieval")
    if run_generation and not run_rerank:
        check_dependency("Generation", rerank_path, "rerank")
    if run_evaluation and not run_generation:
        check_dependency("Evaluation", gen_path, "generation")
    
    # Track state
    optimized_docs = {}
    failed_docs = set()
    chunk_to_doc = baseline["chunk_to_doc"]
    retrieval_results = {}
    rerank_results = {}
    generation_results = {}
    
    # Load existing state
    if failed_path.exists():
        with open(failed_path) as f:
            failed_docs = set(json.load(f))
    
    if opt_path.exists():
        with open(opt_path) as f:
            optimized_docs = json.load(f)
    
    # Phase 1: Optimization
    if run_optimization:
        print("\n--- Phase 1: Optimization ---")
        optimized_docs, failed_docs = await run_optimization_phase(
            baseline["targets"],
            baseline["documents"],
            baseline["doc_id_to_idx"],
            method,
            scope,
            exp_dir,
            force=force,
        )
    
    # Phase 2: Index Building
    if run_index:
        print("\n--- Phase 2: Index Building ---")
        chunk_to_doc = run_index_phase(
            optimized_docs,
            failed_docs,
            baseline["documents"],
            baseline["doc_id_to_idx"],
            exp_dir,
            force=force,
        )
    else:
        # Load existing
        if meta_path.exists():
            with open(meta_path) as f:
                chunk_to_doc = json.load(f)["chunk_to_doc"]
    
    # Phase 3: Retrieval
    if run_retrieval:
        print("\n--- Phase 3: Retrieval ---")
        retrieval_results = run_retrieval_phase(
            baseline["queries"],
            baseline["targets"],
            failed_docs,
            chunk_to_doc,
            exp_dir,
            force=force,
        )
    else:
        if ret_path.exists():
            with open(ret_path) as f:
                retrieval_results = json.load(f)
    
    # Phase 4: Reranking
    if run_rerank:
        print("\n--- Phase 4: Reranking ---")
        rerank_results = run_rerank_phase(
            baseline["queries"],
            baseline["targets"],
            failed_docs,
            retrieval_results,
            baseline["documents"],
            baseline["doc_id_to_idx"],
            chunk_to_doc,
            optimized_docs,
            exp_dir,
            force=force,
            gpu_ids=gpu_ids,
        )
    else:
        if rerank_path.exists():
            with open(rerank_path) as f:
                rerank_results = json.load(f)
    
    # Phase 5: Generation
    if run_generation:
        print("\n--- Phase 5: Generation ---")
        generation_results = await run_generation_phase(
            baseline["queries"],
            baseline["targets"],
            failed_docs,
            rerank_results,
            baseline["documents"],
            baseline["doc_id_to_idx"],
            chunk_to_doc,
            optimized_docs,
            exp_dir,
            force=force,
        )
    else:
        if gen_path.exists():
            with open(gen_path) as f:
                generation_results = json.load(f)
    
    # Phase 6: Evaluation
    if run_evaluation:
        print("\n--- Phase 6: Evaluation ---")
        run_evaluation_phase(
            baseline["targets"],
            failed_docs,
            baseline["retrieval_results"],
            baseline["rerank_results"],
            baseline["generation_results"],
            retrieval_results,
            rerank_results,
            generation_results,
            chunk_to_doc,
            exp_dir,
            force=force,
        )
    
    print(f"\nExperiment {scope}/{method} complete!")


async def main():
    parser = argparse.ArgumentParser(description="Run SAGEO experiments")
    parser.add_argument("--scope", type=str, required=True,
                        help="Optimization scope(s): body, structured, both (comma-separated)")
    parser.add_argument("--methods", type=str, required=True,
                        help="GEO method(s): method names or 'all' (comma-separated)")
    parser.add_argument("--phase", type=str, default="all",
                        help="Phase(s) to run: optimization, index, retrieval, rerank, generation, evaluation, all (comma-separated)")
    parser.add_argument("--force", action="store_true",
                        help="Force regenerate existing results")
    parser.add_argument("--gpu_ids", type=str, default=None,
                        help="GPU IDs for reranking (comma-separated). If not provided, auto-detects free GPUs.")
    parser.add_argument("--check", action="store_true",
                        help="Check status only, don't run")
    parser.add_argument("--num_per_dataset", type=int, default=None,
                        help="Sample N targets per dataset. Results saved under experiments/test_{N}/")

    args = parser.parse_args()
    
    # Parse scopes
    scopes = [s.strip() for s in args.scope.split(",")]
    for s in scopes:
        if s not in VALID_SCOPES:
            print(f"Error: Invalid scope '{s}'. Valid: {VALID_SCOPES}")
            sys.exit(1)
    
    # Parse methods
    if args.methods == "all":
        methods = list_methods()
    else:
        methods = [m.strip() for m in args.methods.split(",")]
        all_methods = list_methods()
        for m in methods:
            if m not in all_methods:
                print(f"Error: Invalid method '{m}'. Valid: {all_methods}")
                sys.exit(1)
    
    # Use GPU IDs from early detection
    gpu_ids = _GPU_IDS
    
    # Parse phases
    phases = [p.strip() for p in args.phase.split(",")]
    for p in phases:
        if p not in VALID_PHASES:
            print(f"Error: Invalid phase '{p}'. Valid: {VALID_PHASES}")
            sys.exit(1)
    
    # Determine experiments directory
    if args.num_per_dataset is not None:
        experiments_dir = EXPERIMENTS_DIR / f"test_{args.num_per_dataset}"
    else:
        experiments_dir = EXPERIMENTS_DIR

    # Check mode
    if args.check:
        print("Status check:")
        for scope in scopes:
            for method in methods:
                exp_dir = experiments_dir / scope / method
                status = []
                if (exp_dir / "1_optimized_documents.json").exists():
                    status.append("opt")
                if (exp_dir / "index" / "lucene").exists():
                    status.append("idx")
                if (exp_dir / "2_retrieval_results.json").exists():
                    status.append("ret")
                if (exp_dir / "3_rerank_results.json").exists():
                    status.append("rer")
                if (exp_dir / "4_generation_results.json").exists():
                    status.append("gen")
                if (exp_dir / "5_evaluation_results.json").exists():
                    status.append("eval")
                
                status_str = ",".join(status) if status else "none"
                print(f"  {scope}/{method}: {status_str}")
        return
    
    # Load baseline once
    baseline = load_baseline(num_per_dataset=args.num_per_dataset)
    
    # Run experiments
    total = len(scopes) * len(methods)
    print(f"\nRunning {total} experiment(s): {scopes} x {methods}")
    
    for scope in scopes:
        for method in methods:
            await run_experiment(
                scope,
                method,
                baseline,
                phases=phases,
                force=args.force,
                gpu_ids=gpu_ids,
                experiments_dir=experiments_dir,
            )
    
    print(f"\n{'='*60}")
    print("All experiments complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
