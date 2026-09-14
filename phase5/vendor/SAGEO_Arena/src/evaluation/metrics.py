"""
Core evaluation metrics for SAGEO experiments.

Metrics:
- Ranking: Hit@K, MRR, Delta Rank
- Citation: Citation rate of target documents
"""
# from src.evaluation.pawc_metrics import find_target_citation_index
import re


def find_doc_rank(
    results: list[tuple[str, float]],
    target_doc_id: str,
    chunk_to_doc: dict[str, str],
) -> int | None:
    """Find rank of target document in ranked results.

    Document rank = position of its highest-ranked chunk (1-indexed).
    The best chunk serves as the visibility indicator for the document.

    Args:
        results: List of (chunk_id, score) tuples, sorted by score descending.
        target_doc_id: Target document ID to find.
        chunk_to_doc: Mapping from chunk_id to doc_id.

    Returns:
        1-indexed chunk position of target document, or None if not found.
    """
    for rank, (chunk_id, _) in enumerate(results, start=1):
        doc_id = chunk_to_doc.get(chunk_id)
        if doc_id is None:
            # Fallback: extract from chunk_id
            doc_id = chunk_id.rsplit("_chunk_", 1)[0]

        if doc_id == target_doc_id:
            return rank

    return None


def parse_citations(text: str) -> set[int]:
    """Extract citation numbers from generated text.
    
    Looks for patterns like [1], [2], [1][2][3].
    
    Args:
        text: Generated response text.
    
    Returns:
        Set of citation numbers (1-indexed).
    """
    if not text:
        return set()
    return set(int(m) for m in re.findall(r'\[(\d+)\]', text))


def hit_at_k(rank: int | None, k: int = 10) -> bool:
    """Check if target document is in top-K results.
    
    Args:
        rank: 1-indexed rank of target, or None if not found.
        k: Top-K threshold.
    
    Returns:
        True if rank <= k, False otherwise.
    """
    return rank is not None and rank <= k


def mrr(rank: int | None) -> float:
    """Calculate Mean Reciprocal Rank contribution.
    
    Args:
        rank: 1-indexed rank of target, or None if not found.
    
    Returns:
        1/rank if found, 0 otherwise.
    """
    if rank is None or rank <= 0:
        return 0.0
    return 1.0 / rank


def delta_rank(baseline_rank: int | None, geo_rank: int | None) -> int | None:
    """Calculate rank improvement (positive = improved).
    
    Args:
        baseline_rank: Rank in baseline results.
        geo_rank: Rank in GEO results.
    
    Returns:
        baseline_rank - geo_rank (positive means improvement), or None if either is missing.
    """
    if baseline_rank is None or geo_rank is None:
        return None
    return baseline_rank - geo_rank


# def is_target_cited(
#     response: str,
#     context_passages: list[dict],
#     target_doc_id: str,
# ) -> bool:
#     """Check if target document is cited in generated response."""
#     target_cite_num = find_target_citation_index(context_passages, target_doc_id)
#     if target_cite_num is None:
#         return False
#     citations = parse_citations(response)
#     return target_cite_num in citations
