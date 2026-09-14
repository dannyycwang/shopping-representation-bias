"""
Purpose:
    Generator module using LangChain + GPT for answer generation.
    Builds prompts from search results with adaptive truncation for long documents.

Usage:
    python -m src.modules.generator
"""
import asyncio
import json
import os
import re
from typing import Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.modules import ROOT_DIR
from src.modules.utils.jsonld_extractor import filter_jsonld_for_generator


load_dotenv(ROOT_DIR / ".env.local")

# Heading limits (some documents have extremely long mis-tagged "headings")
MAX_CHARS_PER_HEADING = 500
MAX_TOTAL_HEADINGS_CHARS = 3500

# Default model
DEFAULT_MODEL = "gpt-5-mini"

# Generation prompt template
# Note: Documents are labeled [Document 1], [Document 2], etc. but citations in the answer use [1], [2]
GENERATION_PROMPT = """
    Write an accurate and concise answer for the given user question, using _only_ the provided summarized web search results. The answer should be correct, high-quality, and written by an expert using an unbiased and journalistic tone. The user's language of choice such as English, Français, Español, Deutsch, or others should be used. The answer should be informative, interesting, and engaging. The answer's logic and reasoning should be rigorous and defensible. Every sentence in the answer should be _immediately followed_ by an in-line citation to the search result(s). The cited search result(s) should fully support _all_ the information in the sentence. Search results need to be cited using [index]. When citing several search results, use [1][2][3] format rather than [1, 2, 3]. You can use multiple search results to respond comprehensively while avoiding irrelevant search results.

    Question: {query}

    Search Results:
    {search_results}
"""



class GenerationResponse(BaseModel):
    """Structured output for generation responses."""
    response: str = Field(description="The main response content")
    reasoning: str = Field(description="Explanation of the reasoning behind the response")


class GenerationModule:
    """LLM-based answer generation module with adaptive truncation."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        service_tier: str = "flex",
        max_retries: int = 5,
    ):
        """Initialize the generation module.

        Args:
            model: OpenAI model name.
            api_key: OpenAI API key. If None, uses OPENAI_API_KEY env var.
            service_tier: OpenAI service tier ("flex" for 50% discount, "auto" for standard).
            max_retries: Max retries for Flex tier before fallback.
        """
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")

        self.model = model
        self.api_key = api_key
        self.service_tier = service_tier
        self.max_retries = max_retries

        self.llm = ChatOpenAI(
            model=model,
            output_version="responses/v1",
            api_key=api_key,
            reasoning={"effort": "minimal"},
            verbosity="low",
            service_tier=service_tier,
        )

        # Fallback LLM with standard tier
        self.llm_fallback = ChatOpenAI(
            model=model,
            output_version="responses/v1",
            api_key=api_key,
            reasoning={"effort": "minimal"},
            verbosity="low",
            service_tier="auto",
        )

        self.prompt = ChatPromptTemplate.from_messages([
            ("human", "{input}"),
        ])

        self.runnable = self.prompt | self.llm.with_structured_output(GenerationResponse)
        self.runnable_fallback = self.prompt | self.llm_fallback.with_structured_output(GenerationResponse)

    @staticmethod
    def format_passage(doc: dict, body_chunk: str) -> str:
        """Format a document chunk as HTML-tagged passage for generation.

        Similar to reranker's format_passage but outputs HTML-like structure
        for LLM understanding. Uses the same passage_id/chunk as reranker.

        Output format:
            <title>{title}</title>
            <meta name="description" content="{meta_desc}">
            <script type="application/ld+json">{jsonld}</script>
            {headings as <h1>, <h2>, etc.}
            <body>{body_chunk}</body>

        Args:
            doc: Document dict with keys: title, meta_description, headings, jsonld.
            body_chunk: The body text chunk to include.

        Returns:
            HTML-formatted passage string.
        """
        title = doc.get("title", "").strip()
        meta_desc = doc.get("meta_description", "").strip()
        headings = doc.get("headings", [])
        jsonld = doc.get("jsonld", [])

        parts = []

        # Title
        if title:
            parts.append(f"<title>{title}</title>")

        # Meta description
        if meta_desc:
            parts.append(f'<meta name="description" content="{meta_desc}">')

        # JSON-LD (filtered)
        if jsonld:
            filtered = filter_jsonld_for_generator(jsonld)
            if filtered:
                jsonld_str = json.dumps(filtered, ensure_ascii=False)
                parts.append(f'<script type="application/ld+json">{jsonld_str}</script>')

        # Headings (with HTML tags, truncated if needed)
        if headings:
            total_chars = 0
            for h in headings:
                if not h:
                    continue
                text = h.strip()

                # Truncate if needed
                if len(text) > MAX_CHARS_PER_HEADING:
                    # Extract heading level (h1-h6) if present
                    tag_match = re.match(r'^<(h[1-6])>', text, re.IGNORECASE)
                    tag_name = tag_match.group(1) if tag_match else None

                    text = text[:MAX_CHARS_PER_HEADING]
                    # If we have a tag and the closing tag was cut off, add it back
                    if tag_name and not text.endswith(f'</{tag_name}>'):
                        text = re.sub(r'</?[^>]*$', '', text)
                        text = text + f'</{tag_name}>'

                if text:
                    if total_chars + len(text) > MAX_TOTAL_HEADINGS_CHARS:
                        break
                    parts.append(text)
                    total_chars += len(text)

        # Body chunk
        parts.append(f"<body>{body_chunk}</body>")

        return "\n".join(parts)

    async def _invoke_with_flex_retry(self, prompt_text: str) -> GenerationResponse:
        """Invoke LLM with Flex tier retry and fallback to standard."""
        last_error = None

        for attempt in range(self.max_retries):
            try:
                result: GenerationResponse = await self.runnable.ainvoke({"input": prompt_text})
                return result
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "resource" in error_str.lower() or "500" in error_str or "server_error" in error_str.lower():
                    wait_time = 2 ** attempt
                    print(f"  Flex error (attempt {attempt + 1}/{self.max_retries}), retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                    last_error = e
                else:
                    raise

        # Fallback to standard tier
        print(f"  Flex failed after {self.max_retries} attempts, falling back to standard tier")
        try:
            result: GenerationResponse = await self.runnable_fallback.ainvoke({"input": prompt_text})
            return result
        except Exception as e:
            raise last_error or e

    def build_prompt_from_chunks(
        self,
        query: str,
        chunks: list[tuple[dict, str]]
    ) -> str:
        """Build prompt from query and document chunks with HTML-tagged format.

        This is the primary method for generation - takes raw doc fields and
        body_chunk, formats each with HTML tags using format_passage().

        Args:
            query: User query string.
            chunks: List of (doc_dict, body_chunk) tuples where doc_dict contains
                title, meta_description, headings, jsonld fields.

        Returns:
            Formatted prompt string.
        """
        passage_parts = []

        for i, (doc, body_chunk) in enumerate(chunks, 1):
            formatted = self.format_passage(doc, body_chunk)
            passage_section = f"[Document {i}]\n{formatted}"
            passage_parts.append(passage_section)

        search_results_str = "\n\n".join(passage_parts)

        prompt = GENERATION_PROMPT.format(
            query=query,
            search_results=search_results_str,
        )
        return prompt

    async def generate_from_chunks(
        self,
        query: str,
        chunks: list[tuple[dict, str]],
    ) -> GenerationResponse:
        """Generate answer from query and document chunks.

        Convenience method that builds prompt from chunks and invokes LLM.

        Args:
            query: User query string.
            chunks: List of (doc_dict, body_chunk) tuples.

        Returns:
            GenerationResponse with response and reasoning.
        """
        prompt_text = self.build_prompt_from_chunks(query, chunks)
        return await self._invoke_with_flex_retry(prompt_text)
