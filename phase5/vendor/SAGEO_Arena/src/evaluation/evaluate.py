"""
Evaluation logic for SAGEO experiments.

Compares baseline vs GEO results across all phases:
- Retrieval: Hit@20, delta_rank (k=100 pool)
- Reranking: Hit@10, delta_rank (k=10 pool, clamped to max_rank=11)  
- Generation: citation rate, inline citation rank change(delta_rank) (clamped to max_rank=11)
"""

import json
from pathlib import Path

from src.evaluation.metrics import find_doc_rank, hit_at_k
# from src.evaluation.pawc_metrics import find_target_citation_index
from src.evaluation.citation_parser import extract_citation_order


# Constants matching legacy evaluation
HIT_K_RETRIEVAL = 20
RETRIEVAL_K = 100  # Pool size for retrieval
HIT_K_RERANK = 10
GENERATION_K = 10  # Top-K chunks used for generation context
MAX_RANK_RETRIEVAL = RETRIEVAL_K + 1  # For clamping delta_rank calculation
MAX_RANK_RERANK = HIT_K_RERANK + 1  # For clamping delta_rank calculation
MAX_RANK_GENERATION = GENERATION_K + 1  # For clamping delta_rank calculation


def run_evaluation_phase(
    targets: dict,
    failed_docs: set,
    baseline_retrieval: dict,
    baseline_rerank: dict,
    baseline_generation: dict,
    geo_retrieval: dict,
    geo_rerank: dict,
    geo_generation: dict,
    chunk_to_doc: dict[str, str],
    exp_dir: Path,
    force: bool = False,
) -> dict:
    """Evaluate SAGEO vs baseline across all phases.
    
    Args:
        targets: Dict mapping query_id to target info.
        failed_docs: Set of failed document IDs to exclude.
        baseline_retrieval: Baseline retrieval results (query_id -> [(chunk_id, score)]).
        baseline_rerank: Baseline reranking results.
        baseline_generation: Baseline generation results.
        geo_retrieval: GEO retrieval results.
        geo_rerank: GEO reranking results.
        geo_generation: GEO generation results.
        chunk_to_doc: Mapping from chunk_id to doc_id.
        exp_dir: Experiment directory for output.
        force: Force regenerate if exists.
    
    Returns:
        Evaluation results dict with metrics and per-query breakdown.
    """
    output_path = exp_dir / "5_evaluation_results.json"
    
    if output_path.exists() and not force:
        print(f"  Loading existing evaluation")
        with open(output_path) as f:
            return json.load(f)
    
    print(f"  Evaluating {len(targets)} targets...")
    
    per_query = {}
    excluded_queries = []
    
    for query_id, target in targets.items():
        target_doc_id = target["doc_id"]
        
        # Skip failed docs
        if target_doc_id in failed_docs:
            excluded_queries.append(query_id)
            continue
        
        # Skip if missing results
        if query_id not in geo_rerank or query_id not in geo_generation:
            continue
        
        # === RETRIEVAL METRICS ===
        baseline_ret_rank = find_doc_rank(
            baseline_retrieval.get(query_id, []),
            target_doc_id,
            chunk_to_doc
        )
        geo_ret_rank = find_doc_rank(
            geo_retrieval.get(query_id, []),
            target_doc_id,
            chunk_to_doc
        )
        
        # === RERANKING METRICS ===
        baseline_rerank_rank = find_doc_rank(
            baseline_rerank.get(query_id, []),
            target_doc_id,
            chunk_to_doc
        )
        geo_rerank_rank = find_doc_rank(
            geo_rerank.get(query_id, []),
            target_doc_id,
            chunk_to_doc
        )
        
        # === GENERATION METRICS ===
        baseline_gen = baseline_generation.get(query_id, {})
        geo_gen = geo_generation.get(query_id, {})
        
        # Find citation rank in response (1-indexed position in citation order)
        baseline_gen_rank = _get_citation_rank(
            baseline_gen.get("response", ""),
            baseline_gen.get("context_passages", []),
            target_doc_id,
        )
        geo_gen_rank = _get_citation_rank(
            geo_gen.get("response", ""),
            geo_gen.get("context_passages", []),
            target_doc_id,
        )
        
        per_query[query_id] = {
            "target_doc_id": target_doc_id,
            "retrieval": {
                "baseline_rank": baseline_ret_rank,
                "geo_rank": geo_ret_rank,
                "delta_rank": _delta_rank(baseline_ret_rank, geo_ret_rank),
                "baseline_hit": hit_at_k(baseline_ret_rank, HIT_K_RETRIEVAL),
                "geo_hit": hit_at_k(geo_ret_rank, HIT_K_RETRIEVAL),
            },
            "reranking": {
                "baseline_rank": baseline_rerank_rank,
                "geo_rank": geo_rerank_rank,
                "delta_rank": _delta_rank(baseline_rerank_rank, geo_rerank_rank),
                "baseline_hit": hit_at_k(baseline_rerank_rank, HIT_K_RERANK),
                "geo_hit": hit_at_k(geo_rerank_rank, HIT_K_RERANK),
            },
            "generation": {
                "baseline_rank": baseline_gen_rank,
                "geo_rank": geo_gen_rank,
                "delta_rank": _delta_rank(baseline_gen_rank, geo_gen_rank),
                "baseline_cited": baseline_gen_rank is not None,
                "geo_cited": geo_gen_rank is not None,
            },
        }
    
    # Aggregate metrics
    valid = list(per_query.values())
    n = len(valid)
    
    if n > 0:
        metrics = {
            "num_queries": n,
            "retrieval": {
                "baseline_hit_rate": sum(1 for v in valid if v["retrieval"]["baseline_hit"]) / n,
                "geo_hit_rate": sum(1 for v in valid if v["retrieval"]["geo_hit"]) / n,
                "avg_delta_rank": _compute_retrieval_delta_avg(valid),
            },
            "reranking": {
                "baseline_hit_rate": sum(1 for v in valid if v["reranking"]["baseline_hit"]) / n,
                "geo_hit_rate": sum(1 for v in valid if v["reranking"]["geo_hit"]) / n,
                "avg_delta_rank": _compute_rerank_delta_avg(valid),
            },
            "generation": {
                "baseline_citation_rate": sum(1 for v in valid if v["generation"]["baseline_cited"]) / n,
                "geo_citation_rate": sum(1 for v in valid if v["generation"]["geo_cited"]) / n,
                "avg_delta_rank": _compute_generation_delta_avg(valid),
            },
        }
    else:
        metrics = {"num_queries": 0}
    
    results = {
        "metrics": metrics,
        "per_query": per_query,
        "excluded_docs": list(failed_docs),
        "excluded_queries": excluded_queries,
    }
    
    # Save
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    if n > 0:
        m = metrics
        print(f"  Evaluation complete ({n} queries, {len(excluded_queries)} excluded):")
        print(f"    Retrieval:  Hit@{HIT_K_RETRIEVAL}={m['retrieval']['geo_hit_rate']:.3f} (baseline={m['retrieval']['baseline_hit_rate']:.3f}), ΔRank={m['retrieval']['avg_delta_rank']:+.2f}")
        print(f"    Reranking:  Hit@{HIT_K_RERANK}={m['reranking']['geo_hit_rate']:.3f} (baseline={m['reranking']['baseline_hit_rate']:.3f}), ΔRank={m['reranking']['avg_delta_rank']:+.2f}")
        print(f"    Generation: Cited={m['generation']['geo_citation_rate']:.3f} (baseline={m['generation']['baseline_citation_rate']:.3f}), ΔRank={m['generation']['avg_delta_rank']:+.2f}")
    
    return results


def _delta_rank(baseline_rank: int | None, geo_rank: int | None) -> int | None:
    """Calculate rank improvement (positive = improved)."""
    if baseline_rank is None or geo_rank is None:
        return None
    return baseline_rank - geo_rank


def _compute_retrieval_delta_avg(per_query_values: list, max_rank: int = MAX_RANK_RETRIEVAL) -> float:
    """Compute retrieval avg_delta_rank with clamped max_rank (101)."""
    if not per_query_values:
        return 0.0
    
    deltas = []
    for q in per_query_values:
        bl = q["retrieval"]["baseline_rank"]
        geo = q["retrieval"]["geo_rank"]
        bl_clamped = min(bl, max_rank) if bl is not None else max_rank
        geo_clamped = min(geo, max_rank) if geo is not None else max_rank
        deltas.append(bl_clamped - geo_clamped)
    
    return sum(deltas) / len(deltas)


def _compute_rerank_delta_avg(per_query_values: list, max_rank: int = MAX_RANK_RERANK) -> float:
    """Compute reranking avg_delta_rank with clamped max_rank.
    
    Matches legacy behavior: ranks beyond top-K are clamped to max_rank.
    """
    if not per_query_values:
        return 0.0
    
    deltas = []
    for q in per_query_values:
        bl = q["reranking"]["baseline_rank"]
        geo = q["reranking"]["geo_rank"]
        # Clamp to max_rank if rank is None (not found) or beyond
        bl_clamped = min(bl, max_rank) if bl is not None else max_rank
        geo_clamped = min(geo, max_rank) if geo is not None else max_rank
        deltas.append(bl_clamped - geo_clamped)
    
    return sum(deltas) / len(deltas)


def _get_citation_rank(
    response: str,
    context_passages: list[dict],
    target_doc_id: str,
) -> int | None:
    """Get the rank of target doc in inline citation order.

    Returns the position of the cited chunk in citation order.
    Returns None if target is not cited.
    """
    if not response or not context_passages:
        return None

    # Find ALL citation indices for target doc (1-indexed)
    target_indices = {i + 1 for i, p in enumerate(context_passages) if p.get('doc_id') == target_doc_id}
    if not target_indices:
        return None

    # Extract citation order from response
    citation_order = extract_citation_order(response, max_citations=len(context_passages))

    # Find first-cited chunk of target doc in citation order
    for rank, cite_idx in enumerate(citation_order, 1):
        if cite_idx in target_indices:
            return rank

    return None  # Not cited


def _compute_generation_delta_avg(per_query_values: list, max_rank: int = MAX_RANK_GENERATION) -> float:
    """Compute generation avg_delta_rank with clamped max_rank (11).

    """
    if not per_query_values:
        return 0.0

    deltas = []
    for q in per_query_values:
        bl = q["generation"]["baseline_rank"]
        geo = q["generation"]["geo_rank"]
        bl_clamped = min(bl, max_rank) if bl is not None else max_rank
        geo_clamped = min(geo, max_rank) if geo is not None else max_rank
        deltas.append(bl_clamped - geo_clamped)

    return sum(deltas) / len(deltas)
