"""
Modules package for SAGEO_Arena.

Submodules:
    - retriever_bm25: BM25 multi-field retrieval using Pyserini
    - reranker: Neural reranking with Qwen3-Reranker
    - generator: LLM-based answer generation
    - optimizer: Document-level SAGEO optimization
    - utils/
        - sageo_methods: SAGEO optimization method definitions
        - jsonld_extractor: JSON-LD structured data extraction
"""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
CORPUS_DIR = DATA_DIR / "corpus"

