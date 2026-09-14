"""
Evaluation module for SAGEO experiments.

Core metrics (metrics.py):
- find_doc_rank: Find document rank in reranked results
- hit_at_k, mrr, delta_rank: Ranking metrics
- parse_citations, is_target_cited: Citation metrics

Citation parsing (citation_parser.py):
- extract_citation_order: Order of citations in text
- extract_sentence_citations: Per-sentence citation info

PAWC metrics (pawc_metrics.py):
- compute_pawc_scores: Position-Adjusted Word Count
- compute_pawc_improvement: PAWC improvement percentage
- find_target_citation_index: Find target doc's citation index

Evaluation (evaluate.py):
- run_evaluation_phase: Full baseline vs GEO evaluation
"""

from src.evaluation.metrics import (
    find_doc_rank,
    parse_citations,
    hit_at_k,
    mrr,
    delta_rank,
    # is_target_cited,
)
from src.evaluation.evaluate import run_evaluation_phase
from src.evaluation.citation_parser import (
    extract_citation_order,
    extract_sentence_citations,
)
# from src.evaluation.pawc_metrics import (
#     compute_pawc_scores,
#     compute_pawc_improvement,
#     find_target_citation_index,
# )