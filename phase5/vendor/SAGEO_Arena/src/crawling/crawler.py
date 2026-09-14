"""
Crawl URLs from Google search results and build a clean document corpus.

Purpose: Fetch web pages, extract structured fields + body text, filter garbage,
         and produce a clean JSONL corpus ready for the SAGEO Arena pipeline.
Usage:
    python -m src.crawling.crawler
    python -m src.crawling.crawler --input google_search_responses.jsonl
    python -m src.crawling.crawler --num_browsers 10 --token_limit 40000
"""
import asyncio
import json
import re
import warnings
from collections import defaultdict
from typing import Dict, Any, List, Tuple

import httpx
import tiktoken
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
import extruct
from playwright.async_api import async_playwright, Page
from playwright_stealth import Stealth
import trafilatura

from src.crawling import CORPUS_DIR


warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MIN_WORDS = 50
TOKEN_LIMIT = 40000
TIMEOUT = 60
TOKENIZER = tiktoken.get_encoding("cl100k_base")

# ===========================================================================
# Anti-bot detection (runs on raw HTML before extraction)
# Adapted from crawl4ai's antibot_detector.py — 3-tier layered detection.
# ===========================================================================

# Tier 1: High-confidence structural markers (any page size)
_TIER1_PATTERNS = [
    (re.compile(r"Reference\s*#\s*[\d]+\.[0-9a-f]+\.\d+\.[0-9a-f]+", re.I), "Akamai block"),
    (re.compile(r"Pardon\s+Our\s+Interruption", re.I), "Akamai challenge"),
    (re.compile(r'challenge-form.*?__cf_chl_f_tk=', re.I | re.DOTALL), "Cloudflare challenge form"),
    (re.compile(r'<span\s+class="cf-error-code">\d{4}</span>', re.I), "Cloudflare firewall"),
    (re.compile(r'/cdn-cgi/challenge-platform/\S+orchestrate', re.I), "Cloudflare JS challenge"),
    (re.compile(r"window\._pxAppId\s*=", re.I), "PerimeterX block"),
    (re.compile(r"captcha\.px-cdn\.net", re.I), "PerimeterX captcha"),
    (re.compile(r"captcha-delivery\.com", re.I), "DataDome captcha"),
    (re.compile(r"_Incapsula_Resource", re.I), "Imperva block"),
    (re.compile(r"Incapsula\s+incident\s+ID", re.I), "Imperva incident"),
    (re.compile(r"Sucuri\s+WebSite\s+Firewall", re.I), "Sucuri firewall"),
    (re.compile(r"KPSDK\.scriptStart\s*=\s*KPSDK\.now\(\)", re.I), "Kasada challenge"),
    (re.compile(r"blocked\s+by\s+network\s+security", re.I), "Network security block"),
]

# Tier 2: Medium-confidence — only on short pages (<10KB) or error status
_TIER2_PATTERNS = [
    (re.compile(r"Access\s+Denied", re.I), "Access Denied"),
    (re.compile(r"Checking\s+your\s+browser", re.I), "Cloudflare browser check"),
    (re.compile(r"<title>\s*Just\s+a\s+moment", re.I), "Cloudflare interstitial"),
    (re.compile(r'class=["\']g-recaptcha["\']', re.I), "reCAPTCHA"),
    (re.compile(r'class=["\']h-captcha["\']', re.I), "hCaptcha"),
    (re.compile(r"Access\s+to\s+This\s+Page\s+Has\s+Been\s+Blocked", re.I), "PerimeterX block page"),
    (re.compile(r"blocked\s+by\s+security", re.I), "Blocked by security"),
    (re.compile(r"Request\s+unsuccessful", re.I), "Request unsuccessful"),
]
_TIER2_MAX_SIZE = 10000

# Tier 3: Structural integrity helpers
_CONTENT_ELEMENTS_RE = re.compile(r'<(?:p|h[1-6]|article|section|li|td|a|pre)\b', re.I)
_SCRIPT_BLOCK_RE = re.compile(r'<script\b[\s\S]*?</script>', re.I)
_STYLE_TAG_RE = re.compile(r'<style\b[\s\S]*?</style>', re.I)
_TAG_RE = re.compile(r'<[^>]+>')
_BODY_RE = re.compile(r'<body\b', re.I)
_STRUCTURAL_MAX_SIZE = 50000


def _looks_like_data(html: str) -> bool:
    """Check if content is a JSON/XML API response, not an HTML page."""
    stripped = html.strip()
    if not stripped:
        return False
    if stripped[0] in ('{', '['):
        return True
    if stripped[:10].lower().startswith(('<html', '<!')):
        if re.search(r'<body[^>]*>\s*<pre[^>]*>\s*[{\[]', stripped[:500], re.I):
            return True
    return False


def _structural_integrity_check(html: str) -> Tuple[bool, str]:
    """Tier 3: catch silent blocks — empty shells, incomplete renders."""
    html_len = len(html)
    if html_len > _STRUCTURAL_MAX_SIZE or _looks_like_data(html):
        return False, ""

    signals = []

    if not _BODY_RE.search(html):
        return True, f"Structural: no <body> tag ({html_len} bytes)"

    body_match = re.search(r'<body\b[^>]*>([\s\S]*)</body>', html, re.I)
    body_content = body_match.group(1) if body_match else html
    stripped = _SCRIPT_BLOCK_RE.sub('', body_content)
    stripped = _STYLE_TAG_RE.sub('', stripped)
    visible_text = _TAG_RE.sub('', stripped).strip()
    visible_len = len(visible_text)

    if visible_len < 50:
        signals.append("minimal_text")
    if len(_CONTENT_ELEMENTS_RE.findall(html)) == 0:
        signals.append("no_content_elements")

    script_count = len(re.findall(r'<script\b', html, re.I))
    if script_count > 0 and len(_CONTENT_ELEMENTS_RE.findall(html)) == 0 and visible_len < 100:
        signals.append("script_heavy_shell")

    if len(signals) >= 2:
        return True, f"Structural: {', '.join(signals)} ({html_len} bytes)"
    if len(signals) == 1 and html_len < 5000:
        return True, f"Structural: {signals[0]} ({html_len} bytes)"

    return False, ""


def is_blocked(status_code: int, html: str) -> Tuple[bool, str]:
    """Detect anti-bot blocking on raw HTML before extraction.

    3-tier detection: WAF structural markers → size-gated generic patterns
    → structural integrity. Runs before trafilatura extraction.
    """
    html = html or ""
    html_len = len(html)

    if status_code == 429:
        return True, "HTTP 429 Too Many Requests"

    # Tier 1: high-confidence patterns (any page size)
    snippet = html[:15000]
    for pattern, reason in _TIER1_PATTERNS:
        if pattern.search(snippet):
            return True, reason

    # Tier 1 deep scan for large pages (scripts/styles stripped)
    if html_len > 15000:
        stripped = _SCRIPT_BLOCK_RE.sub('', html[:500000])
        stripped = _STYLE_TAG_RE.sub('', stripped)
        deep = stripped[:30000]
        for pattern, reason in _TIER1_PATTERNS:
            if pattern.search(deep):
                return True, reason

    # HTTP 403/503 with HTML content → always blocked
    if status_code in (403, 503) and not _looks_like_data(html):
        if html_len < 100:
            return True, f"HTTP {status_code} near-empty ({html_len} bytes)"
        check = snippet
        if html_len > _TIER2_MAX_SIZE:
            s = _SCRIPT_BLOCK_RE.sub('', html[:500000])
            s = _STYLE_TAG_RE.sub('', s)
            check = s[:30000]
        for pattern, reason in _TIER2_PATTERNS:
            if pattern.search(check):
                return True, f"{reason} (HTTP {status_code})"
        return True, f"HTTP {status_code} HTML content ({html_len} bytes)"

    # Tier 2: medium-confidence on other 4xx/5xx + short page
    if status_code and status_code >= 400 and html_len < _TIER2_MAX_SIZE:
        for pattern, reason in _TIER2_PATTERNS:
            if pattern.search(snippet):
                return True, f"{reason} (HTTP {status_code})"

    # HTTP 200 + near-empty
    if status_code == 200 and len(html.strip()) < 100 and not _looks_like_data(html):
        return True, f"Near-empty HTTP 200 ({len(html.strip())} bytes)"

    # Tier 3: structural integrity
    blocked, reason = _structural_integrity_check(html)
    if blocked:
        return True, reason

    return False, ""


# ===========================================================================
# Post-extraction body text filters
# ===========================================================================

# Body-text bot patterns (applied after trafilatura extraction)
BOT_PATTERNS = [
    # JavaScript required
    r"enable javascript", r"javascript is disabled", r"javascript is required",
    r"please enable javascript", r"you need to enable javascript", r"turn on javascript",
    r"requires javascript", r"enable javascript to see", r"does not support javascript",
    r"without javascript", r"if this page doesn't load", r"javascript is not available",
    r"javascript is turned off", r"this app works best with javascript enabled",
    # Bot verification / CAPTCHA
    r"verify.*not a robot", r"making sure you'?re not a bot", r"checking your browser",
    r"confirm you are a human", r"prove that you'?re human", r"verify that you are human",
    r"are you a human", r"captcha", r"security check",
    r"please wait while.*verif", r"loading.*please wait",
    # WAF / CDN blocks
    r"cloudflare", r"access denied", r"request rejected", r"403 forbidden",
    r"sorry,? you have been blocked", r"automated access", r"too many requests",
    r"anubis.*protect", r"your activity looks suspicious",
    # Browser / cookie blocks
    r"browser.*not supported", r"browser is not supported",
    r"cookies.*disabled", r"cookies.*enabled", r"please enable cookies",
    r"press the continue button",
    # Login / paywall gates
    r"please log in", r"please sign in",
    # Ad-blocker gates
    r"ad.?blocker", r"blocks? ads hinders",
    # E-commerce shells
    r"continue shopping",
    # YouTube / Google boilerplate
    r"© 20\d{2} Google LLC",
    # Service errors
    r"temporarily unavailable", r"service unavailable",
    r"under maintenance", r"page not found",
]
_BOT_RE = re.compile("|".join(BOT_PATTERNS), re.IGNORECASE)


def is_bot_body(body: str, title: str) -> bool:
    """Post-extraction check on body text + title (52 patterns)."""
    if _BOT_RE.search(body[:500].lower()):
        return True
    if title and _BOT_RE.search(title.lower()):
        return True
    return False


def is_short(body: str) -> bool:
    return len(body.split()) < MIN_WORDS


def is_too_long(body: str) -> bool:
    return len(TOKENIZER.encode(body)) > TOKEN_LIMIT


def is_garbled_text(text: str) -> bool:
    if not text or len(text) < 50:
        return False
    bad_chars = sum(1 for c in text if ord(c) < 32 and c not in '\t\n\r')
    bad_chars += text.count('�')
    return bad_chars > len(text) * 0.05


# ---------------------------------------------------------------------------
# HTTP fetching
# ---------------------------------------------------------------------------
def fetch_html(url: str) -> Tuple[str | None, int]:
    """Fetch HTML content from URL using httpx."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }
    response = httpx.get(url, headers=headers, timeout=TIMEOUT, follow_redirects=True)
    return response.text, response.status_code


async def fetch_html_js(page: Page, url: str) -> Tuple[str | None, int]:
    """Fetch HTML content using Playwright (headless browser)."""
    response = await page.goto(url, timeout=TIMEOUT * 1000, wait_until="load")
    await page.wait_for_timeout(3000)
    html = await page.content()
    status_code = response.status if response else 0
    return html, status_code


# ---------------------------------------------------------------------------
# Content extraction
# ---------------------------------------------------------------------------
def extract_title(soup: BeautifulSoup) -> str | None:
    title_tag = soup.find("title")
    if not title_tag:
        return None
    text = title_tag.get_text(strip=True)
    return " ".join(text.split())


def extract_meta_description(soup: BeautifulSoup) -> str | None:
    meta = soup.find("meta", attrs={"name": "description"})
    return meta.get("content", "").strip() if meta else None


def extract_headings(soup: BeautifulSoup) -> List[str]:
    headings = []
    for tag in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
        text = tag.get_text(strip=True)
        if text:
            headings.append(f"<{tag.name}>{text}</{tag.name}>")
    return headings


def extract_structured_data(html: str, url: str) -> List[Dict[str, Any]]:
    """Extract JSON-LD using extruct, with BeautifulSoup fallback."""
    try:
        data = extruct.extract(html, base_url=url, syntaxes=["json-ld"])
        return data.get("json-ld", [])
    except Exception:
        pass

    try:
        soup = BeautifulSoup(html, "lxml")
        results = []
        for script in soup.find_all("script", type="application/ld+json"):
            if script.string:
                parsed = json.loads(script.string)
                if isinstance(parsed, list):
                    results.extend(parsed)
                else:
                    results.append(parsed)
        return results
    except Exception:
        return []


def extract_body_text(html: str, soup: BeautifulSoup) -> str:
    """Extract main body text using trafilatura, with <p> tag fallback."""
    text = trafilatura.extract(html)
    if text:
        return text.replace("\n", " ")

    for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
        tag.decompose()

    paragraphs = soup.find_all("p")
    fallback_text = " ".join(p.get_text(strip=True) for p in paragraphs)
    return fallback_text.replace("\n", " ")


def extract_page_data(html: str, url: str, fallback_title: str) -> Tuple[Dict[str, Any] | None, str | None]:
    """Extract all page data from HTML. Returns (page_data, error_reason)."""
    soup = BeautifulSoup(html, "lxml")

    title = extract_title(soup) or fallback_title
    meta_description = extract_meta_description(soup)
    headings = extract_headings(soup)
    structured = extract_structured_data(html, url)
    body_text = extract_body_text(html, soup)

    if not body_text:
        return None, "no_body_text"

    if is_garbled_text(body_text):
        return None, "garbled_text"

    page_data = {
        "url": url,
        "title": title,
        "meta_description": meta_description or "",
        "jsonld": structured,
        "headings": headings,
        "body_text": body_text,
    }
    return page_data, None


# ---------------------------------------------------------------------------
# Metatag filtering
# ---------------------------------------------------------------------------
_METATAG_EXCLUDE = {
    "csrf-token", "csrf-param", "viewport", "format-detection",
    "og:image:width", "og:image:height", "theme-color",
    "og:url", "twitter:url",
}


def filter_metatags(metatags: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    filtered = []
    for tag_dict in metatags:
        filtered_tag = {k: v for k, v in tag_dict.items() if k not in _METATAG_EXCLUDE}
        if filtered_tag:
            filtered.append(filtered_tag)
    return filtered


def build_document(record: Dict[str, Any]) -> Dict[str, Any]:
    """Build final document record from crawl record + page data."""
    page_data = record.get("page_data", {})
    pagemap = record.get("pagemap", {})
    metatags = pagemap.get("metatags", [])

    return {
        "query": record.get("query"),
        "query_id": record.get("query_id"),
        "dataset": record.get("dataset"),
        "domain": record.get("domain"),
        "link": record.get("link"),
        "title": page_data.get("title") or record.get("title"),
        "meta_description": page_data.get("meta_description", ""),
        "jsonld": page_data.get("jsonld", []),
        "headings": page_data.get("headings", []),
        "body_text": page_data.get("body_text", ""),
        "metatags": filter_metatags(metatags),
    }


# ---------------------------------------------------------------------------
# Single-URL crawl + filter
# ---------------------------------------------------------------------------
async def _fetch_and_check(
    url: str, page: Page | None, use_playwright: bool = False,
) -> Tuple[str | None, int, str | None]:
    """Fetch HTML and run anti-bot check. Returns (html, status, block_reason)."""
    html, status_code = None, 0

    if use_playwright and page:
        try:
            html, status_code = await fetch_html_js(page, url)
        except Exception as e:
            return None, 0, f"playwright_error ({type(e).__name__})"
    else:
        try:
            html, status_code = fetch_html(url)
        except httpx.TimeoutException:
            return None, 0, "httpx_timeout"
        except Exception as e:
            return None, 0, f"httpx_error ({type(e).__name__})"

    if not html:
        return None, status_code, "empty_response"

    blocked, reason = is_blocked(status_code, html)
    if blocked:
        return html, status_code, reason

    return html, status_code, None


async def crawl_and_filter(
    url: str, fallback_title: str, page: Page | None = None
) -> Tuple[Dict[str, Any] | None, str | None]:
    """Crawl a URL, extract content, and apply all quality filters.

    Flow: httpx → anti-bot check → if blocked, Playwright retry →
    anti-bot check → extract → body quality filters.
    """
    # --- Layer 1: Fetch with anti-bot detection on raw HTML ---
    html, status_code, block_reason = await _fetch_and_check(url, page)

    if block_reason and page:
        # Retry with Playwright
        html, status_code, block_reason = await _fetch_and_check(url, page, use_playwright=True)

    if block_reason:
        return None, block_reason

    # --- Layer 2: Extract content ---
    page_data, error = extract_page_data(html, url, fallback_title)

    if not page_data and page:
        # Extraction failed — retry with Playwright (may need JS rendering)
        html, status_code, block_reason = await _fetch_and_check(url, page, use_playwright=True)
        if block_reason:
            return None, block_reason
        page_data, error = extract_page_data(html, url, fallback_title)

    if not page_data:
        return None, error

    body = page_data.get("body_text", "")
    title = page_data.get("title", "")

    # --- Layer 3: Post-extraction quality filters ---

    # Bot patterns on extracted body text
    if is_bot_body(body, title):
        if page:
            html, status_code, block_reason = await _fetch_and_check(url, page, use_playwright=True)
            if block_reason:
                return None, block_reason
            page_data, error = extract_page_data(html, url, fallback_title)
            if not page_data:
                return None, error
            body = page_data.get("body_text", "")
            title = page_data.get("title", "")
            if is_bot_body(body, title):
                return None, "bot_pattern"
        else:
            return None, "bot_pattern"

    # Minimum word count
    if is_short(body):
        if page:
            html, status_code, block_reason = await _fetch_and_check(url, page, use_playwright=True)
            if block_reason:
                return None, block_reason
            page_data, error = extract_page_data(html, url, fallback_title)
            if not page_data:
                return None, error
            body = page_data.get("body_text", "")
            if is_short(body):
                return None, f"too_short ({len(body.split())} words)"
        else:
            return None, f"too_short ({len(body.split())} words)"

    return page_data, None


# ---------------------------------------------------------------------------
# Browser-based parallel crawling
# ---------------------------------------------------------------------------
async def crawl_with_browser(
    records: List[Dict[str, Any]],
    browser_idx: int,
    num_browsers: int,
    total_records: int,
    output_file,
    error_file,
    file_lock: asyncio.Lock,
    seen_urls: set,
    counter: List[int],
) -> Tuple[int, int, int]:
    """Crawl URLs assigned to this browser instance.

    Returns (success_count, error_count, dedup_count).
    """
    stealth = Stealth()
    success_count = 0
    error_count = 0
    dedup_count = 0

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            viewport={"width": 1920, "height": 1080},
        )
        await stealth.apply_stealth_async(context)
        page = await context.new_page()

        for i, record in enumerate(records):
            if i % num_browsers != browser_idx:
                continue

            url = record.get("link")
            query_id = record.get("query_id")
            fallback_title = record.get("title", "")

            # Dedup: skip if (query_id, url) already seen
            dedup_key = (query_id, url)
            async with file_lock:
                if dedup_key in seen_urls:
                    dedup_count += 1
                    continue
                seen_urls.add(dedup_key)

            page_data, error_reason = await crawl_and_filter(url, fallback_title, page)

            async with file_lock:
                counter[0] += 1
                n = counter[0]
                if page_data is None:
                    error_record = {
                        "query_id": query_id,
                        "query": record.get("query"),
                        "link": url,
                        "reason": error_reason,
                    }
                    error_file.write(json.dumps(error_record, ensure_ascii=False) + "\n")
                    error_file.flush()
                    error_count += 1
                    print(f"[{n}/{total_records}] FAIL | {error_reason} | {url[:70]}")
                else:
                    record["page_data"] = page_data
                    doc = build_document(record)
                    output_file.write(json.dumps(doc, ensure_ascii=False) + "\n")
                    output_file.flush()
                    success_count += 1
                    print(f"[{n}/{total_records}] OK   | {len(doc['body_text'].split())}w | {url[:70]}")

            await asyncio.sleep(0.5)

        await browser.close()

    return success_count, error_count, dedup_count


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main(
    input_file: str = "",
    suffix: str = "",
    num_browsers: int = 10,
    token_limit: int = 40000,
    timeout: int = 60,
) -> None:
    global TOKEN_LIMIT, TIMEOUT
    TOKEN_LIMIT = token_limit
    TIMEOUT = timeout

    if input_file:
        input_path = CORPUS_DIR.parent / input_file
    else:
        input_path = CORPUS_DIR.parent / "google_search_results" / "google_search_responses.jsonl"

    CORPUS_DIR.mkdir(parents=True, exist_ok=True)

    suffix_str = f"_{suffix}" if suffix else ""
    output_path = CORPUS_DIR / f"documents_filtered{suffix_str}.jsonl"
    error_path = CORPUS_DIR / f"crawl_errors{suffix_str}.jsonl"

    # Load already-crawled URLs for resume
    seen_urls = set()
    if output_path.exists():
        print(f"Resuming: loading existing {output_path}")
        with output_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        r = json.loads(line)
                        seen_urls.add((r.get("query_id"), r.get("link")))
                    except Exception:
                        pass
        print(f"  Already crawled: {len(seen_urls)} (query_id, url) pairs")

    # Also skip previously failed URLs
    failed_urls = set()
    if error_path.exists():
        with error_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        r = json.loads(line)
                        failed_urls.add((r.get("query_id"), r.get("link")))
                    except Exception:
                        pass
        print(f"  Previously failed: {len(failed_urls)} URLs")
        seen_urls.update(failed_urls)

    # Load input records
    records = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                url = record.get("link")
                query_id = record.get("query_id")
                if (query_id, url) not in seen_urls:
                    records.append(record)

    print(f"Records to crawl: {len(records)}")
    print(f"Output: {output_path}")
    print(f"Errors: {error_path}")
    print(f"Filters: min {MIN_WORDS} words, bot patterns, dedup by (query_id, url)")

    file_lock = asyncio.Lock()
    counter = [0]

    with output_path.open("a", encoding="utf-8") as output_file, \
         error_path.open("a", encoding="utf-8") as error_file:

        tasks = [
            crawl_with_browser(
                records, i, num_browsers, len(records),
                output_file, error_file, file_lock, seen_urls, counter,
            )
            for i in range(num_browsers)
        ]

        results = await asyncio.gather(*tasks)

    total_success = sum(s for s, e, d in results)
    total_errors = sum(e for s, e, d in results)
    total_dedup = sum(d for s, e, d in results)

    print(f"\nDone!")
    print(f"  Crawled: {total_success}")
    print(f"  Errors:  {total_errors}")
    print(f"  Deduped: {total_dedup}")
    print(f"  Total:   {total_success + total_errors + total_dedup}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Crawl URLs and build clean corpus")
    parser.add_argument("--input", type=str, default="", help="Input JSONL file in data/ directory")
    parser.add_argument("--suffix", type=str, default="", help="Output file suffix")
    parser.add_argument("--num_browsers", type=int, default=10, help="Parallel browser instances")
    parser.add_argument("--token_limit", type=int, default=40000, help="Max body tokens (cl100k_base)")
    parser.add_argument("--timeout", type=int, default=60, help="Timeout per request in seconds")
    args = parser.parse_args()
    asyncio.run(main(
        input_file=args.input,
        suffix=args.suffix,
        num_browsers=args.num_browsers,
        token_limit=args.token_limit,
        timeout=args.timeout,
    ))
