"""
Purpose:
    BM25 Multi-Field Retriever using Pyserini.
    Builds separate indices per field and fuses results with weighted RRF.

Usage:
    from src.modules.retriever_bm25 import build_field_jsonl, build_indices, load_searchers, search
"""
import json
import re
import subprocess
from collections import defaultdict
from urllib.parse import urlparse

import tiktoken
from pyserini.search.lucene import LuceneSearcher

from src.modules import CORPUS_DIR, DATA_DIR
from src.modules.utils.jsonld_extractor import extract_jsonld_text

# Directory paths
INDEX_DIR = DATA_DIR / "bm25_index"
JSONL_DIR = DATA_DIR / "bm25_jsonl"

# Tokenizer for chunking
enc = tiktoken.get_encoding("cl100k_base")


# Field names for indexing
FIELDS = ["title", "meta_description", "jsonld_text", "headings", "body_text"]  # "url_keywords"

# Chunking parameters for body_text (token-based)
CHUNK_SIZE = 256  # tokens
CHUNK_OVERLAP = 64  # tokens

# Field weights for weighted RRF fusion (equal weights)
FIELD_WEIGHTS = {
    "title": 1.0,
    "meta_description": 1.0,
    "headings": 1.0,
    "body_text": 1.0,
    "jsonld_text": 1.0,
    # "url_keywords": 1.0,
}


def extract_url_keywords(url: str) -> str:
    """Extract keywords from URL path.

    Args:
        url: Full URL string.

    Returns:
        Space-separated keywords from URL path.
    """
    if not url:
        return ""
    
    parsed = urlparse(url)
    path = parsed.path
    
    tokens = re.split(r"[/\-_]", path)
    keywords = [t for t in tokens if t and not t.startswith(".") and len(t) > 1]
    
    return " ".join(keywords)


def extract_headings_text(headings: list[str]) -> str:
    """Extract text from headings list, removing HTML tags.

    Args:
        headings: List of heading strings with HTML tags.

    Returns:
        Space-separated heading texts.
    """
    if not headings:
        return ""
    
    texts = []
    for h in headings:
        text = re.sub(r"<[^>]+>", "", h).strip()
        if text:
            texts.append(text)
    
    return " ".join(texts)


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into token-based chunks with overlap, respecting sentence boundaries.

    Args:
        text: Input text to chunk.
        size: Number of tokens per chunk (default: 256).
        overlap: Number of overlapping tokens between chunks (default: 64).

    Returns:
        List of text chunks.
    """
    if not text or not text.strip():
        return [""]

    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return [text] if text.strip() else [""]

    chunks = []
    current_sentences = []
    current_tokens = 0

    for sentence in sentences:
        tokens = enc.encode(sentence)

        # Handle sentences longer than chunk size
        if len(tokens) > size:
            if current_sentences:
                chunks.append(" ".join(current_sentences))
            current_sentences = []
            current_tokens = 0

            # Split long sentence into chunks
            for i in range(0, len(tokens), size - overlap):
                chunk_tokens = tokens[i : i + size]
                chunks.append(enc.decode(chunk_tokens))
            continue

        # Start new chunk if current would exceed size
        if current_tokens + len(tokens) > size:
            if current_sentences:
                chunks.append(" ".join(current_sentences))

            # Keep overlap sentences for next chunk
            overlap_sentences = []
            overlap_tokens = 0
            for s in reversed(current_sentences):
                t = len(enc.encode(s))
                if overlap_tokens + t <= overlap:
                    overlap_sentences.insert(0, s)
                    overlap_tokens += t
                else:
                    break

            current_sentences = overlap_sentences
            current_tokens = overlap_tokens

        current_sentences.append(sentence)
        current_tokens += len(tokens)

    # Add remaining sentences as final chunk
    if current_sentences:
        chunks.append(" ".join(current_sentences))

    return chunks if chunks else [""]


def preprocess_document(doc: dict, doc_idx: int) -> dict[str, str | list[tuple[str, str]]]:
    """Extract and preprocess all fields from a document.

    Args:
        doc: Document dictionary from documents.json.
        doc_idx: Index of the document for ID generation.

    Returns:
        Dictionary with field names as keys and processed text/chunks as values.
    """
    # Use _id field if provided (for variant documents), otherwise generate from index
    doc_id = doc.get("_id") or f"doc_{doc_idx}"

    title = doc.get("title") or ""
    meta_description = doc.get("meta_description") or ""
    # Use optimized jsonld_text if exists, otherwise extract from jsonld
    jsonld_text = doc.get("jsonld_text") or extract_jsonld_text(doc.get("jsonld", []))
    headings = extract_headings_text(doc.get("headings", []))
    body_text = doc.get("body_text") or ""
    url_keywords = extract_url_keywords(doc.get("link", ""))
    
    body_chunks = chunk_text(body_text)
    body_chunk_entries = [(f"{doc_id}_chunk_{i}", chunk) for i, chunk in enumerate(body_chunks)]
    
    return {
        "doc_id": doc_id,
        "title": (doc_id, title),
        "meta_description": (doc_id, meta_description),
        "jsonld_text": (doc_id, jsonld_text),
        "headings": (doc_id, headings),
        "body_text": body_chunk_entries,
        "url_keywords": (doc_id, url_keywords),
        "original": doc,
    }


def build_field_jsonl(docs: list[dict], output_dir) -> dict[str, list[str]]:
    """Build JSONL files for each field for Pyserini indexing.

    Args:
        docs: List of documents from documents.json.
        output_dir: Directory to save JSONL files.

    Returns:
        Dictionary mapping chunk_id to doc_id for body_text field.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    field_data = {field: [] for field in FIELDS}
    chunk_to_doc = {}
    doc_metadata = []
    
    for idx, doc in enumerate(docs):
        processed = preprocess_document(doc, idx)
        doc_id = processed["doc_id"]
        
        doc_metadata.append({
            "doc_id": doc_id,
            "link": doc.get("link", ""),
            "query": doc.get("query", ""),
            "query_id": doc.get("query_id", ""),
        })
        
        for field in FIELDS:
            if field == "body_text":
                for chunk_id, chunk_text in processed[field]:
                    field_data[field].append({"id": chunk_id, "contents": chunk_text})
                    chunk_to_doc[chunk_id] = doc_id
            else:
                entry_id, text = processed[field]
                field_data[field].append({"id": entry_id, "contents": text})
    
    for field in FIELDS:
        field_dir = output_dir / field
        field_dir.mkdir(parents=True, exist_ok=True)
        
        jsonl_path = field_dir / "docs.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for entry in field_data[field]:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump({
            "chunk_to_doc": chunk_to_doc,
            "documents": doc_metadata,
        }, f, ensure_ascii=False, indent=2)
    
    return chunk_to_doc


def build_indices(jsonl_dir, index_dir):
    """Build Pyserini/Lucene indices for each field.

    Args:
        jsonl_dir: Directory containing JSONL files for each field.
        index_dir: Directory to save Lucene indices.
    """
    index_dir.mkdir(parents=True, exist_ok=True)
    
    for field in FIELDS:
        field_jsonl_dir = jsonl_dir / field
        field_index_dir = index_dir / field
        
        if not field_jsonl_dir.exists():
            print(f"Skipping {field}: JSONL directory not found")
            continue
        
        print(f"Building index for {field}...")
        
        cmd = [
            "python", "-m", "pyserini.index.lucene",
            "--collection", "JsonCollection",
            "--input", str(field_jsonl_dir),
            "--index", str(field_index_dir),
            "--generator", "DefaultLuceneDocumentGenerator",
            "--threads", "4",
            "--storeRaw",
        ]
        
        subprocess.run(cmd, check=True)
        print(f"Index built for {field}")


def load_searchers(index_dir) -> dict[str, LuceneSearcher]:
    """Load Pyserini searchers for each field.

    Args:
        index_dir: Directory containing Lucene indices.

    Returns:
        Dictionary mapping field names to LuceneSearcher instances.
    """
    searchers = {}
    
    for field in FIELDS:
        field_index_dir = index_dir / field
        if field_index_dir.exists():
            searchers[field] = LuceneSearcher(str(field_index_dir))
    
    return searchers


def weighted_rrf_fusion(
    field_results: dict[str, list[str]],
    weights: dict[str, float],
    chunk_to_doc: dict[str, str],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Fuse rankings from multiple fields using weighted RRF (chunk-level output).

    Returns chunk-level results where chunks compete independently.
    Each chunk's score = body_text contribution + parent doc's contribution from other fields.

    Args:
        field_results: Dictionary mapping field names to ranked lists of doc/chunk IDs.
        weights: Dictionary mapping field names to weight values.
        chunk_to_doc: Mapping from chunk_id to doc_id for body_text field.
        k: RRF parameter (default 60).

    Returns:
        List of (chunk_id, score) tuples sorted by score descending.
    """
    # Step 1: Compute document-level scores from non-body fields
    doc_scores = defaultdict(float)
    for field, ranking in field_results.items():
        if field == "body_text":
            continue  # Handle separately
        weight = weights.get(field, 1.0)
        for rank, doc_id in enumerate(ranking, start=1):
            doc_scores[doc_id] += weight / (k + rank)

    # Step 2: Compute chunk-level scores
    # Each chunk gets: body_text contribution + parent doc's non-body contribution
    chunk_scores = defaultdict(float)

    body_ranking = field_results.get("body_text", [])
    body_weight = weights.get("body_text", 1.0)

    for rank, chunk_id in enumerate(body_ranking, start=1):
        # Body text contribution for this specific chunk
        chunk_scores[chunk_id] += body_weight / (k + rank)

        # Add parent doc's contribution from other fields
        doc_id = chunk_to_doc.get(chunk_id, chunk_id)
        chunk_scores[chunk_id] += doc_scores.get(doc_id, 0.0)

    # Also include chunks that weren't in body_text results but their doc appeared in other fields
    # (This handles edge cases where body_text had no results for a doc)
    doc_to_chunks = defaultdict(list)
    for chunk_id, doc_id in chunk_to_doc.items():
        doc_to_chunks[doc_id].append(chunk_id)

    for doc_id, doc_score in doc_scores.items():
        if doc_score > 0:
            # If doc has chunks, ensure at least one chunk is represented
            chunks = doc_to_chunks.get(doc_id, [])
            for chunk_id in chunks:
                if chunk_id not in chunk_scores:
                    chunk_scores[chunk_id] = doc_score

    sorted_results = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_results


def search(
    query: str,
    searchers: dict[str, LuceneSearcher],
    chunk_to_doc: dict[str, str],
    top_k: int = 100,
    per_field_k: int = 200,
) -> list[tuple[str, float]]:
    """Search across all fields and fuse with weighted RRF.

    Returns chunk-level results where chunks compete independently.
    Each chunk's score combines body_text contribution + parent doc's contribution
    from other fields (title, meta_description, etc.).

    Args:
        query: Search query string.
        searchers: Dictionary of LuceneSearcher instances per field.
        chunk_to_doc: Mapping from chunk_id to doc_id.
        top_k: Number of final results to return.
        per_field_k: Number of results to retrieve per field before fusion.

    Returns:
        List of (chunk_id, score) tuples sorted by score descending.
    """
    field_results = {}

    for field, searcher in searchers.items():
        hits = searcher.search(query, k=per_field_k)
        field_results[field] = [hit.docid for hit in hits]

    fused = weighted_rrf_fusion(field_results, FIELD_WEIGHTS, chunk_to_doc)
    return fused[:top_k]
