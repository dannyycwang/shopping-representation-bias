"""
JSON-LD Text Extraction for BM25 Indexing.

Extracts meaningful text from schema.org JSON-LD structured data using a
whitelist of common types and fields.

Usage:
    from src.modules.utils.jsonld_extractor import extract_jsonld_text
"""
from typing import Any


# shared, common 
JSONLD_WHITELIST = {
    "Organization": ["name", "description", "alternateName"],
    "WebPage": ["name", "description", "headline"],
    "WebSite": ["name", "description", "alternateName"],
    "Person": ["name", "description", "jobTitle"],
    "Article": ["headline", "description", "author", "keywords", "articleSection"],
    "NewsArticle": ["headline", "description", "author", "keywords", "articleSection"],
    "BlogPosting": ["headline", "description", "author", "keywords", "articleSection"],
    "ScholarlyArticle": ["headline", "description", "keywords"],
    "Product": ["name", "description", "brand"],
    "Review": ["reviewBody", "author"],
    "FAQPage": ["mainEntity"],
    "QAPage": ["mainEntity", "name"],
    "Question": ["name", "text", "acceptedAnswer"],
    "Answer": ["text"],
    "HowTo": ["name", "description", "step"],
    "HowToStep": ["name", "text"],
    "SoftwareApplication": ["name", "description", "applicationCategory"],
    "LocalBusiness": ["name", "description"],
    "Event": ["name", "description", "location"],
    "VideoObject": ["name", "description", "transcript"],
    "Movie": ["name", "description", "genre"],
    "TVSeries": ["name", "description", "genre"],
    "Recipe": ["name", "description", "recipeIngredient"],
    "Service": ["name", "description"],
    "Course": ["name", "description"],
    "CreativeWork": ["name", "description", "headline"],
    "Dataset": ["name", "description"],
    "Comment": ["text"],
    "DiscussionForumPosting": ["headline", "text", "articleBody"],
    "MedicalWebPage": ["name", "description", "headline", "about"],
}
# Fallback for unlisted types: check if the object has any of these keys
# and extract whichever are present (skips noisy fields like ID etc.)
DEFAULT_KEYS = ["name", "description", "headline", "text"]


def extract_jsonld_text(jsonld: list) -> str:
    """Extract meaningful text from JSON-LD using whitelist keys in key: value format.
    
    Used by BM25 retriever and reranker for text-based matching/scoring.

    Args:
        jsonld: List of JSON-LD objects from the document.

    Returns:
        Key-value formatted string (e.g., "type: Product. name: iPhone. description: A smartphone.")
    """
    if not jsonld:
        return ""

    key_values = []

    def extract_value(obj: Any) -> str:
        """Recursively extract string value from an object."""
        if isinstance(obj, str):
            return obj.strip()
        if isinstance(obj, list):
            parts = [extract_value(item) for item in obj]
            return ", ".join(p for p in parts if p)
        if isinstance(obj, dict):
            if "name" in obj:
                return extract_value(obj["name"])
            if "text" in obj:
                return extract_value(obj["text"])
            parts = [extract_value(v) for v in obj.values()]
            return " ".join(p for p in parts if p)
        return ""

    def process_jsonld_obj(obj: dict) -> list[str]:
        """Process a single JSON-LD object to key: value pairs."""
        if not isinstance(obj, dict):
            return []

        result = []
        obj_type = obj.get("@type", "")

        if isinstance(obj_type, list):
            obj_type = obj_type[0] if obj_type else ""

        if not isinstance(obj_type, str):
            obj_type = ""

        if obj_type:
            result.append(f"type: {obj_type}")

        # Use whitelist if type is known, otherwise fall back to default keys
        whitelist_keys = JSONLD_WHITELIST.get(obj_type, DEFAULT_KEYS)

        for key in whitelist_keys:
            if key in obj:
                value = extract_value(obj[key])
                if value:
                    result.append(f"{key}: {value}")

        return result

    def process_item(item: Any) -> None:
        """Process a JSON-LD item, handling @graph arrays recursively."""
        if not isinstance(item, dict):
            return
        
        key_values.extend(process_jsonld_obj(item))
        
        if "@graph" in item and isinstance(item["@graph"], list):
            for graph_item in item["@graph"]:
                process_item(graph_item)

    for item in jsonld:
        process_item(item)

    return ". ".join(key_values)


def filter_jsonld_for_generator(jsonld: list) -> list:
    """Filter JSON-LD objects using whitelist, returning filtered JSON structure.

    Used by generator for LLM input where we want standard JSON format
    (matching real HTML) but with non-semantic fields removed.

    Args:
        jsonld: List of JSON-LD objects from the document.

    Returns:
        Filtered list of JSON-LD objects with only whitelisted fields.
    """
    if not jsonld:
        return []

    def filter_obj(obj: Any) -> Any:
        """Recursively filter a JSON-LD object."""
        if not isinstance(obj, dict):
            return obj

        obj_type = obj.get("@type", "")
        if isinstance(obj_type, list):
            obj_type = obj_type[0] if obj_type else ""
        if not isinstance(obj_type, str):
            obj_type = ""

        whitelist_keys = JSONLD_WHITELIST.get(obj_type, DEFAULT_KEYS)
        if not whitelist_keys:
            return None

        filtered = {}
        
        # Always keep @type and @context
        if "@type" in obj:
            filtered["@type"] = obj["@type"]
        if "@context" in obj:
            filtered["@context"] = obj["@context"]

        # Keep whitelisted keys
        for key in whitelist_keys:
            if key in obj:
                value = obj[key]
                # Recursively filter nested objects
                if isinstance(value, dict):
                    filtered_value = filter_obj(value)
                    if filtered_value:
                        filtered[key] = filtered_value
                elif isinstance(value, list):
                    filtered_list = [filter_obj(v) if isinstance(v, dict) else v for v in value]
                    filtered_list = [v for v in filtered_list if v is not None]
                    if filtered_list:
                        filtered[key] = filtered_list
                else:
                    filtered[key] = value

        # Handle @graph recursively
        if "@graph" in obj and isinstance(obj["@graph"], list):
            filtered_graph = [filter_obj(item) for item in obj["@graph"]]
            filtered_graph = [item for item in filtered_graph if item]
            if filtered_graph:
                filtered["@graph"] = filtered_graph

        return filtered if len(filtered) > 1 or (len(filtered) == 1 and "@type" not in filtered) else None

    result = []
    for item in jsonld:
        filtered = filter_obj(item)
        if filtered:
            result.append(filtered)

    return result


def jsonld_text_to_jsonld(jsonld_text: str) -> list:
    """Convert key: value formatted text back to JSON-LD structure.

    Used by optimizer to convert its output back to JSON-LD format,
    so downstream components (reranker, generator) can process it normally.

    Args:
        jsonld_text: Text in "type: X. name: Y. description: Z" format.

    Returns:
        List of JSON-LD objects, e.g., [{"@type": "X", "name": "Y", "description": "Z"}]
    """
    if not jsonld_text or not jsonld_text.strip():
        return []

    # Split by ". " to get key-value pairs, but be careful with periods in values
    # The format is: "key1: value1. key2: value2. key3: value3"
    
    result = []
    current_obj = {}
    
    # Split on ". " but handle edge cases
    # We need to identify "key: value" pairs
    parts = jsonld_text.split(". ")
    
    for part in parts:
        part = part.strip()
        if not part:
            continue
        
        # Remove trailing period if present
        if part.endswith("."):
            part = part[:-1]
        
        # Find the first colon to split key and value
        colon_idx = part.find(": ")
        if colon_idx == -1:
            # No colon found, might be continuation of previous value
            continue
        
        key = part[:colon_idx].strip()
        value = part[colon_idx + 2:].strip()
        
        if not key or not value:
            continue
        
        # Handle special keys
        if key == "type":
            # Start a new object if we already have one with a type
            if "@type" in current_obj:
                if current_obj:
                    result.append(current_obj)
                current_obj = {}
            current_obj["@type"] = value
        else:
            # Map common keys
            current_obj[key] = value
    
    # Don't forget the last object
    if current_obj:
        result.append(current_obj)
    
    return result
