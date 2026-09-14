"""
Web crawling module.

Modules:
    - crawler: Crawl URLs and build clean document corpus
"""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
CORPUS_DIR = DATA_DIR / "corpus"
QUERIES_DIR = DATA_DIR / "queries"

