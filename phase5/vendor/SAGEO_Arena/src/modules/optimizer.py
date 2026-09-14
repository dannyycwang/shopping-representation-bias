"""
GEO Optimizer for document-level optimization.

Supports three settings:
- body: Optimize body_text only
- structured: Optimize title, meta_description, headings only
- all: Optimize all fields

Uses OpenAI Responses API with structured output (Pydantic).
"""

import asyncio
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from pathlib import Path

from src.modules.utils.sageo_methods import (
    get_method,
    TARGET_FIELDS_BODY,
    TARGET_FIELDS_STRUCTURED,
    TARGET_FIELDS_ALL,
)
from src.modules.utils.jsonld_extractor import extract_jsonld_text, jsonld_text_to_jsonld
from src.modules import ROOT_DIR
load_dotenv(ROOT_DIR / ".env.local")


class BodyOutput(BaseModel):
    """Output for body-only optimization."""
    body_text: str = Field(description="Optimized body text")


class StructuredOutput(BaseModel):
    """Output for structured-only optimization."""
    title: str = Field(description="Optimized title")
    meta_description: str = Field(description="Optimized meta description")
    headings: list[str] = Field(description="Optimized headings list")
    jsonld_text: str = Field(description="Structured data in 'key: value' format with relevant fields based on content type - NOT JSON markup")


class AllFieldsOutput(BaseModel):
    """Output for all-fields optimization."""
    title: str = Field(description="Optimized title")
    meta_description: str = Field(description="Optimized meta description")
    headings: list[str] = Field(description="Optimized headings list")
    jsonld_text: str = Field(description="Structured data in 'key: value' format with relevant fields based on content type - NOT JSON markup")
    body_text: str = Field(description="Optimized body text")


class DocumentOptimizer:
    """Document-level GEO optimizer using OpenAI Responses API with structured output."""

    def __init__(self, model: str = "gpt-5-mini", max_retries: int = 5):
        self.client = AsyncOpenAI()
        self.model = model
        self.max_retries = max_retries

    async def _call_api_with_flex(
        self,
        prompt: str,
        output_format: type[BaseModel],
    ):
        """Call API with Flex tier, retry with exponential backoff, fallback to standard.

        Args:
            prompt: The prompt to send.
            output_format: Pydantic model for structured output.

        Returns:
            API response.
        """
        last_error = None

        # Try Flex tier with exponential backoff
        for attempt in range(self.max_retries):
            try:
                response = await self.client.responses.parse(
                    model=self.model,
                    input=[{"role": "user", "content": prompt}],
                    text_format=output_format,
                    service_tier="flex",
                )
                return response
            except Exception as e:
                error_str = str(e)
                # Check for 429 Resource Unavailable (Flex capacity issue)
                if "429" in error_str or "resource" in error_str.lower():
                    wait_time = 2 ** attempt
                    print(f"  Flex unavailable (attempt {attempt + 1}/{self.max_retries}), retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                    last_error = e
                else:
                    # Other errors, raise immediately
                    raise

        # Fallback to standard tier
        print(f"  Flex failed after {self.max_retries} attempts, falling back to standard tier")
        try:
            response = await self.client.responses.parse(
                model=self.model,
                input=[{"role": "user", "content": prompt}],
                text_format=output_format,
                service_tier="auto",
            )
            return response
        except Exception as e:
            # If standard also fails, raise the original flex error
            raise last_error or e

    async def optimize_document(
        self,
        document: dict,
        method: str,
        target_fields: str,
    ) -> dict:
        """Optimize a document with specified method and target fields.

        Uses OpenAI Responses API with structured output (Pydantic) for reliable JSON.
        Uses Flex tier with fallback to standard tier.

        Args:
            document: Document dict with fields (body_text, title, meta_description, headings, etc.)
            method: GEO method name (fluency, authoritative, etc.)
            target_fields: One of "body", "structured", "all"

        Returns:
            Updated document dict with optimized fields.
        """
        geo_method = get_method(method)
        template = geo_method.get_prompt_template(target_fields)

        # Prepare input based on target_fields
        if target_fields == TARGET_FIELDS_BODY:
            prompt = template.format(
                body_text=document.get("body_text", "")
            )
            output_format = BodyOutput

        elif target_fields == TARGET_FIELDS_STRUCTURED:
            headings = document.get("headings", [])
            if isinstance(headings, list):
                headings_str = "\n".join(f"- {h}" for h in headings)
            else:
                headings_str = str(headings)

            # Extract jsonld text (empty string triggers LLM to generate)
            jsonld_text = extract_jsonld_text(document.get("jsonld"))

            prompt = template.format(
                title=document.get("title", ""),
                meta_description=document.get("meta_description", ""),
                headings=headings_str,
                jsonld_text=jsonld_text if jsonld_text else "(No existing JSON-LD - generate appropriate structured data text)",
                body_text=document.get("body_text", ""),  # Include body as read-only context
            )
            output_format = StructuredOutput

        elif target_fields == TARGET_FIELDS_ALL:
            headings = document.get("headings", [])
            if isinstance(headings, list):
                headings_str = "\n".join(f"- {h}" for h in headings)
            else:
                headings_str = str(headings)

            # Extract jsonld text (empty string triggers LLM to generate)
            jsonld_text = extract_jsonld_text(document.get("jsonld"))

            prompt = template.format(
                title=document.get("title", ""),
                meta_description=document.get("meta_description", ""),
                headings=headings_str,
                jsonld_text=jsonld_text if jsonld_text else "(No existing JSON-LD - generate appropriate structured data text)",
                body_text=document.get("body_text", ""),
            )
            output_format = AllFieldsOutput

        else:
            raise ValueError(f"Unknown target_fields: {target_fields}")

        # Call OpenAI Responses API with Flex tier (fallback to standard)
        response = await self._call_api_with_flex(prompt, output_format)

        # Handle incomplete response (max_output_tokens reached)
        if response.status == "incomplete":
            reason = response.incomplete_details.reason if response.incomplete_details else "unknown"
            raise ValueError(f"Incomplete response: {reason}")

        # Handle refusal
        if response.output and response.output[0].content:
            content = response.output[0].content[0]
            if hasattr(content, "refusal") and content.refusal:
                raise ValueError(f"Model refused: {content.refusal}")

        # Get parsed output
        parsed = response.output_parsed

        # Update document with optimized fields
        result = document.copy()

        if target_fields == TARGET_FIELDS_BODY:
            result["body_text"] = parsed.body_text

        elif target_fields == TARGET_FIELDS_STRUCTURED:
            result["title"] = parsed.title
            result["meta_description"] = parsed.meta_description
            result["headings"] = parsed.headings
            result["jsonld_text"] = parsed.jsonld_text
            # Convert jsonld_text back to jsonld structure for downstream components
            result["jsonld"] = jsonld_text_to_jsonld(parsed.jsonld_text)

        elif target_fields == TARGET_FIELDS_ALL:
            result["title"] = parsed.title
            result["meta_description"] = parsed.meta_description
            result["headings"] = parsed.headings
            result["jsonld_text"] = parsed.jsonld_text
            # Convert jsonld_text back to jsonld structure for downstream components
            result["jsonld"] = jsonld_text_to_jsonld(parsed.jsonld_text)
            result["body_text"] = parsed.body_text

        return result
