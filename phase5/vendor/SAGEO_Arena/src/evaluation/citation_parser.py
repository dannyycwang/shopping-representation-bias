"""
Purpose:
    Parse inline citations from LLM-generated responses.
    Extract citation order and per-sentence citation information.

Usage:
    from src.evaluation.citation_parser import extract_citation_order, extract_sentence_citations
"""

import re
from typing import List, Dict


def extract_citation_order(text: str, max_citations: int = None) -> List[int]:
    """Extract the order in which citations first appear in the text.

    Args:
        text (str): LLM response text containing citations like [1], [2], etc.
        max_citations (int): If provided, filter out citations outside [1, max_citations].

    Returns:
        List[int]: List of citation indices (1-based) in order of first appearance.
    """
    citation_pattern = r'\[(\d+)\]'
    seen = set()
    order = []
    
    for match in re.finditer(citation_pattern, text):
        citation_idx = int(match.group(1))
        if max_citations is not None and (citation_idx < 1 or citation_idx > max_citations):
            continue
        if citation_idx not in seen:
            seen.add(citation_idx)
            order.append(citation_idx)
    
    return order


def extract_sentence_citations(text: str, max_citations: int = None) -> List[Dict]:
    """Extract citations and word count for each sentence.

    Args:
        text (str): LLM response text.
        max_citations (int): If provided, filter out citations outside [1, max_citations].

    Returns:
        List[Dict]: List of dicts with 'word_count' and 'citations' for each sentence.
    """
    sentences = _split_into_sentences(text)
    results = []
    
    for sentence in sentences:
        citations = _extract_citations_from_sentence(sentence, max_citations)
        word_count = _count_words(sentence)
        
        if word_count > 0:
            results.append({
                'sentence': sentence,
                'word_count': word_count,
                'citations': citations
            })
    
    return results


def _split_into_sentences(text: str) -> List[str]:
    """Split text into sentences.

    Args:
        text (str): Input text.

    Returns:
        List[str]: List of sentences.
    """
    sentence_pattern = r'[^.!?]*[.!?]+'
    sentences = re.findall(sentence_pattern, text)
    
    if not sentences:
        return [text] if text.strip() else []
    
    return [s.strip() for s in sentences if s.strip()]


def _extract_citations_from_sentence(sentence: str, max_citations: int = None) -> List[int]:
    """Extract all citation indices from a sentence.

    Args:
        sentence (str): Single sentence.
        max_citations (int): If provided, filter out citations outside [1, max_citations].

    Returns:
        List[int]: List of citation indices (1-based), may contain duplicates.
    """
    citation_pattern = r'\[(\d+)\]'
    citations = [int(m.group(1)) for m in re.finditer(citation_pattern, sentence)]
    if max_citations is not None:
        citations = [c for c in citations if 1 <= c <= max_citations]
    return citations


def _count_words(text: str) -> int:
    """Count words with more than 2 characters.

    Args:
        text (str): Input text.

    Returns:
        int: Number of words with length > 2.
    """
    text_without_citations = re.sub(r'\[\d+\]', '', text)
    words = re.findall(r'\b\w+\b', text_without_citations)
    return sum(1 for w in words if len(w) > 2)
