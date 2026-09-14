"""
SAGEO (Structured and Augmented GEO) optimization method classes.

Usage:
    from src.modules.utils.sageo_methods import get_method
    method = get_method("fluency")
    prompt = method.get_prompt_template(target_fields="body")
"""

# Target field options
TARGET_FIELDS_BODY = "body"           # Body text only
TARGET_FIELDS_STRUCTURED = "structured"  # Title, meta_description, headings, jsonld_text (no body)
TARGET_FIELDS_ALL = "all"             # All fields
TARGET_FIELDS_PASSAGE = "passage"     # Single passage optimization


class GEOMethod:
    """Base class for GEO methods."""

    name: str = ""
    description: str = ""

    # Templates for different target fields
    prompt_template_body: str = ""
    prompt_template_structured: str = ""
    prompt_template_all: str = ""
    prompt_template_passage: str = ""  # For passage-level optimization

    def get_prompt_template(self, target_fields: str = TARGET_FIELDS_BODY) -> str:
        """Get prompt template for specified target fields.

        Args:
            target_fields: One of "body", "structured", "all", or "passage".

        Returns:
            Prompt template string.
        """
        if target_fields == TARGET_FIELDS_BODY:
            return self.prompt_template_body
        elif target_fields == TARGET_FIELDS_STRUCTURED:
            return self.prompt_template_structured
        elif target_fields == TARGET_FIELDS_ALL:
            return self.prompt_template_all
        elif target_fields == TARGET_FIELDS_PASSAGE:
            return self.prompt_template_passage
        else:
            raise ValueError(f"Unknown target_fields: {target_fields}")

    @property
    def prompt_template(self) -> str:
        """Backward compatibility: return body template."""
        return self.prompt_template_body

    def post_processing(self, text: str) -> str:
        """Post-process the LLM response.

        Args:
            text: Raw LLM response text.

        Returns:
            Processed text.
        """
        return text


class AutoGEO(GEOMethod):
    """Comprehensive content quality optimization based on GEO best practices."""

    name = "autogeo"
    description = "Comprehensive content quality optimization (accuracy, structure, clarity, depth)"

    prompt_template_passage = """Optimize the following passage by applying these content quality principles:

1. Ensure all information is factually accurate and verifiable, citing credible sources.
2. Ensure the passage is self-contained and comprehensive, providing all necessary context.
3. Organize content with a clear, logical hierarchy, using elements like headings, lists, and tables where appropriate.
4. Use clear and unambiguous language, defining technical terms, acronyms, and jargon upon first use.
5. Ensure information is current and up-to-date, especially for time-sensitive topics.
6. Write concisely, eliminating verbose language, redundancy, and filler content.
7. Explain the underlying mechanisms and principles (the 'why' and 'how'), not just surface-level facts.
8. State the primary conclusion directly at the beginning.
9. Maintain a singular focus on the core topic, excluding tangential information and noise.
10. Use specific, concrete details and examples instead of abstract generalizations.
11. Present a balanced and objective view on debatable topics, including multiple significant perspectives.
12. Provide specific, actionable guidance, such as step-by-step instructions, for procedural topics.

## Passage
{passage}

Important:
- Maintain the same language as the original
- Preserve ALL information - do not truncate or summarize
- Keep similar length to original"""

    prompt_template_body = """Optimize the following text by applying these content quality principles:

1. Ensure all information is factually accurate and verifiable, citing credible sources.
2. Ensure the document is self-contained and comprehensive, providing all necessary context and sub-topic information.
3. Organize content with a clear, logical hierarchy, using elements like headings, lists, and tables.
4. Use clear and unambiguous language, defining technical terms, acronyms, and jargon upon first use.
5. Ensure information is current and up-to-date, especially for time-sensitive topics.
6. Write concisely, eliminating verbose language, redundancy, and filler content.
7. Explain the underlying mechanisms and principles (the 'why' and 'how'), not just surface-level facts.
8. State the primary conclusion directly at the beginning of the document.
9. Maintain a singular focus on the core topic, excluding tangential information, promotional content, and document 'noise' (e.g., navigation, ads).
10. Use specific, concrete details and examples instead of abstract generalizations.
11. Present a balanced and objective view on debatable topics, including multiple significant perspectives.
12. Provide specific, actionable guidance, such as step-by-step instructions, for procedural topics.

## Text
{body_text}

Important:
- Maintain the same language as the original
- Preserve ALL information - do not truncate or summarize
- Keep similar length to original"""

    prompt_template_structured = """Optimize the following structured metadata fields by applying these content quality principles:

1. Ensure all information is factually accurate and verifiable, citing credible sources.
2. Ensure the document is self-contained and comprehensive, providing all necessary context and sub-topic information.
3. Organize content with a clear, logical hierarchy, using elements like headings, lists, and tables.
4. Use clear and unambiguous language, defining technical terms, acronyms, and jargon upon first use.
5. Ensure information is current and up-to-date, especially for time-sensitive topics.
6. Write concisely, eliminating verbose language, redundancy, and filler content.
7. Explain the underlying mechanisms and principles (the 'why' and 'how'), not just surface-level facts.
8. State the primary conclusion directly at the beginning.
9. Maintain a singular focus on the core topic, excluding tangential information, promotional content, and document 'noise'.
10. Use specific, concrete details and examples instead of abstract generalizations.
11. Present a balanced and objective view on debatable topics, including multiple significant perspectives.
12. Provide specific, actionable guidance, such as step-by-step instructions, for procedural topics.

## Document Body (READ-ONLY CONTEXT - use this to understand the document content, but do NOT include in output)
{body_text}

## Fields to Optimize

**Title**: {title}

**Meta Description**: {meta_description}

**Headings**: {headings}

**JSON-LD Text**: {jsonld_text}

## Output Format

Return a JSON object with the optimized fields:

```json
{{
  "title": "optimized title",
  "meta_description": "optimized meta description",
  "headings": ["heading 1", "heading 2", ...],
  "jsonld_text": "type: Article. name: X. headline: Y. description: Z."
}}
```

Important:
- Return ONLY the JSON object, no additional text
- Maintain the same language as the original
- Keep headings as a list of strings
- For jsonld_text: use "key: value" format with relevant structured data fields based on the content (e.g., "type: Product. name: iPhone. brand: Apple. description: A smartphone."). Include type and any relevant fields that describe the content. NOT JSON markup
- Apply enhancements without damaging the original content's completeness or detail - avoid truncation
- Follow real-world SEO conventions for field lengths:
  - Title: Keep under 60 characters, similar length to original
  - Meta description: Keep between 120-160 characters, similar length to original
  - Headings: Keep concise, similar length to originals"""

    prompt_template_all = """Optimize the following document by applying these content quality principles:

1. Ensure all information is factually accurate and verifiable, citing credible sources.
2. Ensure the document is self-contained and comprehensive, providing all necessary context and sub-topic information.
3. Organize content with a clear, logical hierarchy, using elements like headings, lists, and tables.
4. Use clear and unambiguous language, defining technical terms, acronyms, and jargon upon first use.
5. Ensure information is current and up-to-date, especially for time-sensitive topics.
6. Write concisely, eliminating verbose language, redundancy, and filler content.
7. Explain the underlying mechanisms and principles (the 'why' and 'how'), not just surface-level facts.
8. State the primary conclusion directly at the beginning of the document.
9. Maintain a singular focus on the core topic, excluding tangential information, promotional content, and document 'noise' (e.g., navigation, ads).
10. Use specific, concrete details and examples instead of abstract generalizations.
11. Present a balanced and objective view on debatable topics, including multiple significant perspectives.
12. Provide specific, actionable guidance, such as step-by-step instructions, for procedural topics.

## Input Document

**Title**: {title}

**Meta Description**: {meta_description}

**Headings**: {headings}

**JSON-LD Text**: {jsonld_text}

**Body Text**:
{body_text}

## Output Format

Return a JSON object with all optimized fields:

```json
{{
  "title": "optimized title",
  "meta_description": "optimized meta description",
  "headings": ["heading 1", "heading 2", ...],
  "jsonld_text": "type: Article. name: X. headline: Y. description: Z.",
  "body_text": "optimized body text"
}}
```

Important:
- Return ONLY the JSON object, no additional text
- Maintain the same language as the original
- Keep headings as a list of strings
- For jsonld_text: use "key: value" format with relevant structured data fields based on the content (e.g., "type: Product. name: iPhone. brand: Apple. description: A smartphone."). Include type and any relevant fields that describe the content. NOT JSON markup
- Apply enhancements without damaging the original content's completeness or detail - avoid truncation"""


class Fluency(GEOMethod):
    """Improves the fluency of the text."""

    name = "fluency"
    description = "Improves fluency without altering core content"

    prompt_template_passage = """Rewrite the following passage to make it more fluent. The sentences should flow smoothly and the language should be clear and engaging while preserving the original information.

## Passage
{passage}

Important:
- Maintain the same language as the original
- Preserve ALL information - do not truncate or summarize
- Keep similar length to original"""

    prompt_template_body = """Rewrite the following text to make it more fluent. The sentences should flow smoothly and the language should be clear and engaging while preserving the original information.

## Text
{body_text}

Important:
- Maintain the same language as the original
- Preserve ALL information - do not truncate or summarize
- Keep similar length to original"""

    prompt_template_structured = """Optimize the following structured metadata fields to make them more fluent and engaging without altering the core content. The language should be clear and compelling while preserving the original information.

## Document Body (READ-ONLY CONTEXT - use this to understand the document content, but do NOT include in output)
{body_text}

## Fields to Optimize

**Title**: {title}

**Meta Description**: {meta_description}

**Headings**: {headings}

**JSON-LD Text**: {jsonld_text}

## Output Format

Return a JSON object with the optimized fields:

```json
{{
  "title": "optimized title",
  "meta_description": "optimized meta description",
  "headings": ["heading 1", "heading 2", ...],
  "jsonld_text": "type: Article. name: X. headline: Y. description: Z."
}}
```

Important:
- Return ONLY the JSON object, no additional text
- Maintain the same language as the original
- Keep headings as a list of strings
- For jsonld_text: use "key: value" format with relevant structured data fields based on the content (e.g., "type: Product. name: iPhone. brand: Apple. description: A smartphone."). Include type and any relevant fields that describe the content. NOT JSON markup
- Apply enhancements without damaging the original content's completeness or detail - avoid truncation
- Follow real-world SEO conventions for field lengths:
  - Title: Keep under 60 characters, similar length to original
  - Meta description: Keep between 120-160 characters, similar length to original
  - Headings: Keep concise, similar length to originals"""

    prompt_template_all = """Optimize the following document to make it more fluent without altering the core content. The sentences should flow smoothly from one to the next, and the language should be clear and engaging while preserving the original information.

## Input Document

**Title**: {title}

**Meta Description**: {meta_description}

**Headings**: {headings}

**JSON-LD Text**: {jsonld_text}

**Body Text**:
{body_text}

## Output Format

Return a JSON object with all optimized fields:

```json
{{
  "title": "optimized title",
  "meta_description": "optimized meta description",
  "headings": ["heading 1", "heading 2", ...],
  "jsonld_text": "type: Article. name: X. headline: Y. description: Z.",
  "body_text": "optimized body text"
}}
```

Important:
- Return ONLY the JSON object, no additional text
- Maintain the same language as the original
- Keep headings as a list of strings
- For jsonld_text: use "key: value" format with relevant structured data fields based on the content (e.g., "type: Product. name: iPhone. brand: Apple. description: A smartphone."). Include type and any relevant fields that describe the content. NOT JSON markup
- Apply enhancements without damaging the original content's completeness or detail - avoid truncation"""


class Authoritative(GEOMethod):
    """Transforms text into an authoritative style with confidence, expertise, and assertiveness."""

    name = "authoritative"
    description = "Transforms text to be assertive and authoritative"

    prompt_template_passage = """Transform the following passage into an authoritative style without adding or removing any core information. The revised passage should reflect confidence, expertise, and assertiveness, while maintaining the original content's meaning and relevance.

The passage should be assertive in its statements, such that the reader believes this is a more valuable source of information than other sources. The goal is to increase the citation of this source by assertively conveying that this is high-quality, trustworthy information.

## Document Context
**Title**: {title}

## Passage to Optimize
{passage}

Important:
- Preserve ALL information - do not truncate or summarize
- Only paraphrase individual lines or 2-3 sentences while keeping content the same
- Use assertive phrases like "we guarantee", "you can trust", "you will find"
- Use second person pronouns to engage the reader (e.g., "you will not regret")
- Keep similar length to original
- No addition or deletion of content is allowed"""

    prompt_template_body = """Transform the following source into an authoritative style without adding or removing any core information. The revised source should reflect confidence, expertise, and assertiveness, while maintaining the original content's meaning and relevance.

The source should be assertive in its statements, such that the reader believes this is a more valuable source of information than other provided sources. The goal is to increase the citation of this source by assertively conveying that this is the best quality information.

However, the content and structure of the source should remain the same. Only individual lines and/or 2-3 sentences can be paraphrased while keeping the content the same.

## Input Document

**Body Text**:
{body_text}

## Output Format

Return a JSON object with the optimized body text:

```json
{{
  "body_text": "optimized body text"
}}
```

Important:
- Return ONLY the JSON object, no additional text
- Maintain the same language as the original
- Keep the format and content of text the same (line spacing, bullet points, overall structure)
- No addition or deletion of content is allowed
- Use assertive phrases to convey authority (e.g., "we guarantee", "you can trust", "this is the definitive guide")
- Use second person pronouns to engage the reader (e.g., "you will find", "you will not regret")
- Apply enhancements without damaging the original content's completeness or detail - avoid truncation"""

    prompt_template_structured = """Transform the following structured metadata into an authoritative style. Make the title and description sound confident, expert, and assertive.

## Document Body (READ-ONLY CONTEXT - use this to understand the document content, but do NOT include in output)
{body_text}

## Fields to Optimize

**Title**: {title}

**Meta Description**: {meta_description}

**Headings**: {headings}

**JSON-LD Text**: {jsonld_text}

## Output Format

Return a JSON object with the optimized fields:

```json
{{
  "title": "optimized title",
  "meta_description": "optimized meta description",
  "headings": ["heading 1", "heading 2", ...],
  "jsonld_text": "type: Article. name: X. headline: Y. description: Z."
}}
```

Important:
- Return ONLY the JSON object, no additional text
- Maintain the same language as the original
- Use assertive language that conveys confidence and expertise
- For jsonld_text: use "key: value" format with relevant structured data fields based on the content (e.g., "type: Product. name: iPhone. brand: Apple. description: A smartphone."). Include type and any relevant fields that describe the content. NOT JSON markup
- Apply enhancements without damaging the original content's completeness or detail - avoid truncation
- Follow real-world SEO conventions for field lengths:
  - Title: Keep under 60 characters, similar length to original
  - Meta description: Keep between 120-160 characters, similar length to original
  - Headings: Keep concise, similar length to originals"""

    prompt_template_all = """Transform the following document into an authoritative style without adding or removing any core information. The revised document should reflect confidence, expertise, and assertiveness, while maintaining the original content's meaning and relevance.

The document should be assertive in its statements, such that the reader believes this is a more valuable source of information than other sources. The goal is to increase the citation of this source by assertively conveying that this is the best quality information.

However, the content and structure should remain the same. Only individual lines and/or 2-3 sentences can be paraphrased while keeping the content the same.

## Input Document

**Title**: {title}

**Meta Description**: {meta_description}

**Headings**: {headings}

**JSON-LD Text**: {jsonld_text}

**Body Text**:
{body_text}

## Output Format

Return a JSON object with all optimized fields:

```json
{{
  "title": "optimized title",
  "meta_description": "optimized meta description",
  "headings": ["heading 1", "heading 2", ...],
  "jsonld_text": "type: Article. name: X. headline: Y. description: Z.",
  "body_text": "optimized body text"
}}
```

Important:
- Return ONLY the JSON object, no additional text
- Maintain the same language as the original
- Keep the format and content of text the same (line spacing, bullet points, overall structure)
- No addition or deletion of content is allowed
- Use assertive phrases to convey authority (e.g., "we guarantee", "you can trust", "this is the definitive guide")
- Use second person pronouns to engage the reader (e.g., "you will find", "you will not regret")
- For jsonld_text: use "key: value" format with relevant structured data fields based on the content (e.g., "type: Product. name: iPhone. brand: Apple. description: A smartphone."). Include type and any relevant fields that describe the content. NOT JSON markup
- Apply enhancements without damaging the original content's completeness or detail - avoid truncation"""


class Citations(GEOMethod):
    """Adds relevant citations from credible sources."""

    name = "citations"
    description = "Adds citations from credible sources"

    prompt_template_passage = """Revise the following passage to include citations from credible sources. You may invent sources but ensure they sound plausible. Citations should be in rephrased words, e.g., "According to Google's latest report..."

## Document Context
**Title**: {title}

## Passage to Optimize
{passage}

Important:
- Preserve ALL information - do not truncate or summarize
- Add 1-2 citations naturally within the text
- Keep similar length to original"""

    prompt_template_body = """Revise the following source to include citations from credible sources. You may invent these sources but ensure they sound plausible. Citations should be in rephrased words. For example: "According to Google's latest report..."

## Input Document

**Body Text**:
{body_text}

## Output Format

Return a JSON object with the optimized body text:

```json
{{
  "body_text": "optimized body text"
}}
```

Important:
- Return ONLY the JSON object, no additional text
- Don't overdo citing, 5-6 citations are enough
- Apply enhancements without damaging the original content's completeness or detail - avoid truncation"""

    prompt_template_structured = """Revise the following structured metadata to include references to credible sources where appropriate.

## Document Body (READ-ONLY CONTEXT - use this to understand the document content, but do NOT include in output)
{body_text}

## Fields to Optimize

**Title**: {title}

**Meta Description**: {meta_description}

**Headings**: {headings}

**JSON-LD Text**: {jsonld_text}

## Output Format

Return a JSON object with the optimized fields:

```json
{{
  "title": "optimized title",
  "meta_description": "optimized meta description",
  "headings": ["heading 1", "heading 2", ...],
  "jsonld_text": "type: Article. name: X. headline: Y. description: Z."
}}
```

Important:
- Return ONLY the JSON object, no additional text
- For jsonld_text: use "key: value" format with relevant structured data fields based on the content (e.g., "type: Product. name: iPhone. brand: Apple. description: A smartphone."). Include type and any relevant fields that describe the content. NOT JSON markup
- Apply enhancements without damaging the original content's completeness or detail - avoid truncation
- Follow real-world SEO conventions for field lengths:
  - Title: Keep under 60 characters, similar length to original
  - Meta description: Keep between 120-160 characters, similar length to original
  - Headings: Keep concise, similar length to originals"""

    prompt_template_all = """Revise the following document to include citations from credible sources. Citations should be in rephrased words.

## Input Document

**Title**: {title}

**Meta Description**: {meta_description}

**Headings**: {headings}

**JSON-LD Text**: {jsonld_text}

**Body Text**:
{body_text}

## Output Format

Return a JSON object with all optimized fields:

```json
{{
  "title": "optimized title",
  "meta_description": "optimized meta description",
  "headings": ["heading 1", "heading 2", ...],
  "jsonld_text": "type: Article. name: X. headline: Y. description: Z.",
  "body_text": "optimized body text"
}}
```

Important:
- Return ONLY the JSON object, no additional text
- Don't overdo citing, 5-6 citations are enough
- For jsonld_text: use "key: value" format with relevant structured data fields based on the content (e.g., "type: Product. name: iPhone. brand: Apple. description: A smartphone."). Include type and any relevant fields that describe the content. NOT JSON markup
- Apply enhancements without damaging the original content's completeness or detail - avoid truncation"""


class Statistics(GEOMethod):
    """Adds positive, compelling statistics to the text."""

    name = "statistics"
    description = "Adds compelling statistics and numbers"

    prompt_template_passage = """Add positive, compelling statistics (even if hypothetical) at relevant places in this passage. Statistics means objective facts such as x% growth, numbers, interesting numerical facts.

## Document Context
**Title**: {title}

## Passage to Optimize
{passage}

Important:
- Preserve ALL information - do not truncate or summarize
- Add statistics subtly inline within sentences
- Keep similar length to original"""

    prompt_template_body = """Add positive, compelling statistics (even if hypothetical) at relevant places in the text. Statistics means objective facts such as x% growth, numbers, interesting numerical facts.

## Input Document

**Body Text**:
{body_text}

## Guidelines:
1. Statistics should be subtly added inline within sentences
2. Do not alter content except where adding statistics
3. Just output the optimized source text

## Output Format

Return a JSON object with the optimized body text:

```json
{{
  "body_text": "optimized body text"
}}
```

Important:
- Return ONLY the JSON object, no additional text
- Apply enhancements without damaging the original content's completeness or detail - avoid truncation"""

    prompt_template_structured = """Add compelling statistics to the structured metadata where appropriate.

## Document Body (READ-ONLY CONTEXT - use this to understand the document content, but do NOT include in output)
{body_text}

## Fields to Optimize

**Title**: {title}

**Meta Description**: {meta_description}

**Headings**: {headings}

**JSON-LD Text**: {jsonld_text}

## Output Format

Return a JSON object with the optimized fields:

```json
{{
  "title": "optimized title",
  "meta_description": "optimized meta description",
  "headings": ["heading 1", "heading 2", ...],
  "jsonld_text": "type: Article. name: X. headline: Y. description: Z."
}}
```

Important:
- Return ONLY the JSON object, no additional text
- For jsonld_text: use "key: value" format with relevant structured data fields based on the content (e.g., "type: Product. name: iPhone. brand: Apple. description: A smartphone."). Include type and any relevant fields that describe the content. NOT JSON markup
- Apply enhancements without damaging the original content's completeness or detail - avoid truncation
- Follow real-world SEO conventions for field lengths:
  - Title: Keep under 60 characters, similar length to original
  - Meta description: Keep between 120-160 characters, similar length to original
  - Headings: Keep concise, similar length to originals"""

    prompt_template_all = """Add positive, compelling statistics at relevant places in the document.

## Input Document

**Title**: {title}

**Meta Description**: {meta_description}

**Headings**: {headings}

**JSON-LD Text**: {jsonld_text}

**Body Text**:
{body_text}

## Output Format

Return a JSON object with all optimized fields:

```json
{{
  "title": "optimized title",
  "meta_description": "optimized meta description",
  "headings": ["heading 1", "heading 2", ...],
  "jsonld_text": "type: Article. name: X. headline: Y. description: Z.",
  "body_text": "optimized body text"
}}
```

Important:
- Return ONLY the JSON object, no additional text
- For jsonld_text: use "key: value" format with relevant structured data fields based on the content (e.g., "type: Product. name: iPhone. brand: Apple. description: A smartphone."). Include type and any relevant fields that describe the content. NOT JSON markup
- Apply enhancements without damaging the original content's completeness or detail - avoid truncation"""

    def post_processing(self, text: str) -> str:
        """Post-process to extract text after 'Updated Output:' if present."""
        if "Updated Output:" in text:
            text = text.split("Updated Output:")[1].strip()
        return text


class UniqueWords(GEOMethod):
    """Adds unique and rare words to the text."""

    name = "unique_words"
    description = "Incorporates unique and rare words"

    prompt_template_passage = """Revise the following passage by incorporating more unique and rare words, without altering the core information.

## Document Context
**Title**: {title}

## Passage to Optimize
{passage}

Important:
- Preserve ALL information - do not truncate or summarize
- Replace common words with more unique alternatives where appropriate
- Keep similar length to original"""

    prompt_template_body = """Revise the following source by incorporating more unique and rare words, without altering the core information.

## Input Document

**Body Text**:
{body_text}

## Output Format

Return a JSON object with the optimized body text:

```json
{{
  "body_text": "optimized body text"
}}
```

Important:
- Return ONLY the JSON object, no additional text
- Apply enhancements without damaging the original content's completeness or detail - avoid truncation"""

    prompt_template_structured = """Revise the structured metadata by incorporating more unique and engaging words.

## Document Body (READ-ONLY CONTEXT - use this to understand the document content, but do NOT include in output)
{body_text}

## Fields to Optimize

**Title**: {title}

**Meta Description**: {meta_description}

**Headings**: {headings}

**JSON-LD Text**: {jsonld_text}

## Output Format

Return a JSON object with the optimized fields:

```json
{{
  "title": "optimized title",
  "meta_description": "optimized meta description",
  "headings": ["heading 1", "heading 2", ...],
  "jsonld_text": "type: Article. name: X. headline: Y. description: Z."
}}
```

Important:
- Return ONLY the JSON object, no additional text
- For jsonld_text: use "key: value" format with relevant structured data fields based on the content (e.g., "type: Product. name: iPhone. brand: Apple. description: A smartphone."). Include type and any relevant fields that describe the content. NOT JSON markup
- Apply enhancements without damaging the original content's completeness or detail - avoid truncation
- Follow real-world SEO conventions for field lengths:
  - Title: Keep under 60 characters, similar length to original
  - Meta description: Keep between 120-160 characters, similar length to original
  - Headings: Keep concise, similar length to originals"""

    prompt_template_all = """Revise the document by incorporating more unique and rare words, without altering the core information.

## Input Document

**Title**: {title}

**Meta Description**: {meta_description}

**Headings**: {headings}

**JSON-LD Text**: {jsonld_text}

**Body Text**:
{body_text}

## Output Format

Return a JSON object with all optimized fields:

```json
{{
  "title": "optimized title",
  "meta_description": "optimized meta description",
  "headings": ["heading 1", "heading 2", ...],
  "jsonld_text": "type: Article. name: X. headline: Y. description: Z.",
  "body_text": "optimized body text"
}}
```

Important:
- Return ONLY the JSON object, no additional text
- For jsonld_text: use "key: value" format with relevant structured data fields based on the content (e.g., "type: Product. name: iPhone. brand: Apple. description: A smartphone."). Include type and any relevant fields that describe the content. NOT JSON markup
- Apply enhancements without damaging the original content's completeness or detail - avoid truncation"""


class Quotes(GEOMethod):
    """Adds quotes from authoritative figures to the text."""

    name = "quotes"
    description = "Adds quotes from authoritative figures"

    prompt_template_passage = """Modify the following passage by including quotes from authoritative figures. The quotes should be relevant and increase credibility.

## Document Context
**Title**: {title}

## Passage to Optimize
{passage}

Important:
- Preserve ALL information - do not truncate or summarize
- Add 1-2 relevant quotes naturally
- Keep similar length to original"""

    prompt_template_body = """Modify the following source by including quotes from authoritative figures. The quotes should be relevant and increase credibility.

## Input Document

**Body Text**:
{body_text}

## Output Format

Return a JSON object with the optimized body text:

```json
{{
  "body_text": "optimized body text"
}}
```

Important:
- Return ONLY the JSON object, no additional text
- Apply enhancements without damaging the original content's completeness or detail - avoid truncation"""

    prompt_template_structured = """Modify the structured metadata to include references to authoritative sources or expert opinions.

## Document Body (READ-ONLY CONTEXT - use this to understand the document content, but do NOT include in output)
{body_text}

## Fields to Optimize

**Title**: {title}

**Meta Description**: {meta_description}

**Headings**: {headings}

**JSON-LD Text**: {jsonld_text}

## Output Format

Return a JSON object with the optimized fields:

```json
{{
  "title": "optimized title",
  "meta_description": "optimized meta description",
  "headings": ["heading 1", "heading 2", ...],
  "jsonld_text": "type: Article. name: X. headline: Y. description: Z."
}}
```

Important:
- Return ONLY the JSON object, no additional text
- For jsonld_text: use "key: value" format with relevant structured data fields based on the content (e.g., "type: Product. name: iPhone. brand: Apple. description: A smartphone."). Include type and any relevant fields that describe the content. NOT JSON markup
- Apply enhancements without damaging the original content's completeness or detail - avoid truncation
- Follow real-world SEO conventions for field lengths:
  - Title: Keep under 60 characters, similar length to original
  - Meta description: Keep between 120-160 characters, similar length to original
  - Headings: Keep concise, similar length to originals"""

    prompt_template_all = """Modify the document by including quotes from authoritative figures. The quotes should increase credibility.

## Input Document

**Title**: {title}

**Meta Description**: {meta_description}

**Headings**: {headings}

**JSON-LD Text**: {jsonld_text}

**Body Text**:
{body_text}

## Output Format

Return a JSON object with all optimized fields:

```json
{{
  "title": "optimized title",
  "meta_description": "optimized meta description",
  "headings": ["heading 1", "heading 2", ...],
  "jsonld_text": "type: Article. name: X. headline: Y. description: Z.",
  "body_text": "optimized body text"
}}
```

Important:
- Return ONLY the JSON object, no additional text
- For jsonld_text: use "key: value" format with relevant structured data fields based on the content (e.g., "type: Product. name: iPhone. brand: Apple. description: A smartphone."). Include type and any relevant fields that describe the content. NOT JSON markup
- Apply enhancements without damaging the original content's completeness or detail - avoid truncation"""


class SimpleLanguage(GEOMethod):
    """Simplifies the language of the text."""

    name = "simple_language"
    description = "Simplifies language for easier understanding"

    prompt_template_passage = """Simplify the following passage using easy-to-understand language while preserving the key information.

## Document Context
**Title**: {title}

## Passage to Optimize
{passage}

Important:
- Preserve ALL information - do not truncate or summarize
- Use simpler words and shorter sentences
- Keep similar length to original"""

    prompt_template_body = """Simplify the following source using easy-to-understand language while preserving the key information.

## Input Document

**Body Text**:
{body_text}

## Output Format

Return a JSON object with the optimized body text:

```json
{{
  "body_text": "optimized body text"
}}
```

Important:
- Return ONLY the JSON object, no additional text
- Apply enhancements without damaging the original content's completeness or detail - avoid truncation"""

    prompt_template_structured = """Simplify the structured metadata using clear, easy-to-understand language.

## Document Body (READ-ONLY CONTEXT - use this to understand the document content, but do NOT include in output)
{body_text}

## Fields to Optimize

**Title**: {title}

**Meta Description**: {meta_description}

**Headings**: {headings}

**JSON-LD Text**: {jsonld_text}

## Output Format

Return a JSON object with the optimized fields:

```json
{{
  "title": "optimized title",
  "meta_description": "optimized meta description",
  "headings": ["heading 1", "heading 2", ...],
  "jsonld_text": "type: Article. name: X. headline: Y. description: Z."
}}
```

Important:
- Return ONLY the JSON object, no additional text
- For jsonld_text: use "key: value" format with relevant structured data fields based on the content (e.g., "type: Product. name: iPhone. brand: Apple. description: A smartphone."). Include type and any relevant fields that describe the content. NOT JSON markup
- Apply enhancements without damaging the original content's completeness or detail - avoid truncation
- Follow real-world SEO conventions for field lengths:
  - Title: Keep under 60 characters, similar length to original
  - Meta description: Keep between 120-160 characters, similar length to original
  - Headings: Keep concise, similar length to originals"""

    prompt_template_all = """Simplify the document using easy-to-understand language while preserving key information.

## Input Document

**Title**: {title}

**Meta Description**: {meta_description}

**Headings**: {headings}

**JSON-LD Text**: {jsonld_text}

**Body Text**:
{body_text}

## Output Format

Return a JSON object with all optimized fields:

```json
{{
  "title": "optimized title",
  "meta_description": "optimized meta description",
  "headings": ["heading 1", "heading 2", ...],
  "jsonld_text": "type: Article. name: X. headline: Y. description: Z.",
  "body_text": "optimized body text"
}}
```

Important:
- Return ONLY the JSON object, no additional text
- For jsonld_text: use "key: value" format with relevant structured data fields based on the content (e.g., "type: Product. name: iPhone. brand: Apple. description: A smartphone."). Include type and any relevant fields that describe the content. NOT JSON markup
- Apply enhancements without damaging the original content's completeness or detail - avoid truncation"""


class TechnicalTerms(GEOMethod):
    """Adds technical terms to the text."""

    name = "technical_terms"
    description = "Adds technical terms and precise language"

    prompt_template_passage = """Make the following passage more technical by adding technical terms and facts while preserving key information.

## Document Context
**Title**: {title}

## Passage to Optimize
{passage}

Important:
- Preserve ALL information - do not truncate or summarize
- Add technical terminology where appropriate
- Keep similar length to original"""

    prompt_template_body = """Make the following source more technical by adding technical terms and facts while preserving key information.

## Input Document

**Body Text**:
{body_text}

## Output Format

Return a JSON object with the optimized body text:

```json
{{
  "body_text": "optimized body text"
}}
```

Important:
- Return ONLY the JSON object, no additional text
- Apply enhancements without damaging the original content's completeness or detail - avoid truncation"""

    prompt_template_structured = """Make the structured metadata more technical and precise.

## Document Body (READ-ONLY CONTEXT - use this to understand the document content, but do NOT include in output)
{body_text}

## Fields to Optimize

**Title**: {title}

**Meta Description**: {meta_description}

**Headings**: {headings}

**JSON-LD Text**: {jsonld_text}

## Output Format

Return a JSON object with the optimized fields:

```json
{{
  "title": "optimized title",
  "meta_description": "optimized meta description",
  "headings": ["heading 1", "heading 2", ...],
  "jsonld_text": "type: Article. name: X. headline: Y. description: Z."
}}
```

Important:
- Return ONLY the JSON object, no additional text
- For jsonld_text: use "key: value" format with relevant structured data fields based on the content (e.g., "type: Product. name: iPhone. brand: Apple. description: A smartphone."). Include type and any relevant fields that describe the content. NOT JSON markup
- Apply enhancements without damaging the original content's completeness or detail - avoid truncation
- Follow real-world SEO conventions for field lengths:
  - Title: Keep under 60 characters, similar length to original
  - Meta description: Keep between 120-160 characters, similar length to original
  - Headings: Keep concise, similar length to originals"""

    prompt_template_all = """Make the document more technical by adding technical terms and facts.

## Input Document

**Title**: {title}

**Meta Description**: {meta_description}

**Headings**: {headings}

**JSON-LD Text**: {jsonld_text}

**Body Text**:
{body_text}

## Output Format

Return a JSON object with all optimized fields:

```json
{{
  "title": "optimized title",
  "meta_description": "optimized meta description",
  "headings": ["heading 1", "heading 2", ...],
  "jsonld_text": "type: Article. name: X. headline: Y. description: Z.",
  "body_text": "optimized body text"
}}
```

Important:
- Return ONLY the JSON object, no additional text
- For jsonld_text: use "key: value" format with relevant structured data fields based on the content (e.g., "type: Product. name: iPhone. brand: Apple. description: A smartphone."). Include type and any relevant fields that describe the content. NOT JSON markup
- Apply enhancements without damaging the original content's completeness or detail - avoid truncation"""


class ContentImprovement(GEOMethod):
    """Improves the fluency, authority, and persuasiveness of texts."""

    name = "content_improvement"
    description = "Improves fluency, authority, and persuasiveness"

    prompt_template_passage = """Rewrite the following text to make it more fluent, authoritative, and persuasive without altering the core content. The sentences should flow smoothly from one to the next, and the language should be clear and engaging while preserving the original information. The revised text should reflect confidence, expertise, and assertiveness while maintaining the original content's meaning and relevance. The text should be assertive in its statements, such that the reader believes that this is a more valuable source of information than other texts. Lastly, give structure to the text.

## Document Context
**Title**: {title}

## Passage to Optimize
{passage}

Important:
- Preserve ALL information - do not truncate or summarize
- Keep similar length to original"""

    prompt_template_body = """Rewrite the following text to make it more fluent, authoritative, and persuasive without altering the core content. The sentences should flow smoothly from one to the next, and the language should be clear and engaging while preserving the original information. The revised text should reflect confidence, expertise, and assertiveness while maintaining the original content's meaning and relevance. The text should be assertive in its statements, such that the reader believes that this is a more valuable source of information than other texts. Lastly, give structure to the text.

## Input Document

**Body Text**:
{body_text}

## Output Format

Return a JSON object with the optimized body text:

```json
{{
  "body_text": "optimized body text"
}}
```

Important:
- Return ONLY the JSON object, no additional text"""

    prompt_template_structured = """Rewrite the following text to make it more fluent, authoritative, and persuasive without altering the core content. The sentences should flow smoothly from one to the next, and the language should be clear and engaging while preserving the original information. The revised text should reflect confidence, expertise, and assertiveness while maintaining the original content's meaning and relevance. The text should be assertive in its statements, such that the reader believes that this is a more valuable source of information than other texts. Lastly, give structure to the text.

## Document Body (READ-ONLY CONTEXT - use this to understand the document content, but do NOT include in output)
{body_text}

## Fields to Optimize

**Title**: {title}

**Meta Description**: {meta_description}

**Headings**: {headings}

**JSON-LD Text**: {jsonld_text}

## Output Format

Return a JSON object with the optimized fields:

```json
{{
  "title": "optimized title",
  "meta_description": "optimized meta description",
  "headings": ["heading 1", "heading 2", ...],
  "jsonld_text": "type: Article. name: X. headline: Y. description: Z."
}}
```

Important:
- Return ONLY the JSON object, no additional text
- For jsonld_text: use "key: value" format with relevant structured data fields based on the content (e.g., "type: Product. name: iPhone. brand: Apple. description: A smartphone."). Include type and any relevant fields that describe the content. NOT JSON markup
- Follow real-world SEO conventions for field lengths:
  - Title: Keep under 60 characters, similar length to original
  - Meta description: Keep between 120-160 characters, similar length to original
  - Headings: Keep concise, similar length to originals"""

    prompt_template_all = """Rewrite the following text to make it more fluent, authoritative, and persuasive without altering the core content. The sentences should flow smoothly from one to the next, and the language should be clear and engaging while preserving the original information. The revised text should reflect confidence, expertise, and assertiveness while maintaining the original content's meaning and relevance. The text should be assertive in its statements, such that the reader believes that this is a more valuable source of information than other texts. Lastly, give structure to the text.

## Input Document

**Title**: {title}

**Meta Description**: {meta_description}

**Headings**: {headings}

**JSON-LD Text**: {jsonld_text}

**Body Text**:
{body_text}

## Output Format

Return a JSON object with all optimized fields:

```json
{{
  "title": "optimized title",
  "meta_description": "optimized meta description",
  "headings": ["heading 1", "heading 2", ...],
  "jsonld_text": "type: Article. name: X. headline: Y. description: Z.",
  "body_text": "optimized body text"
}}
```

Important:
- Return ONLY the JSON object, no additional text
- For jsonld_text: use "key: value" format with relevant structured data fields based on the content (e.g., "type: Product. name: iPhone. brand: Apple. description: A smartphone."). Include type and any relevant fields that describe the content. NOT JSON markup"""

class ComboEasyQuote(GEOMethod):
    """Combine simple language + authoritative quotes."""

    name = "combo_easy_quote"
    description = "Simple language + authoritative quotes — accessibility meets credibility"

    prompt_template_all = """Optimize the following document by applying two strategies together:

1. **Simplify the language**: Use easy-to-understand words and shorter sentences throughout. Replace jargon or complex phrasing with plain, accessible alternatives while preserving all information.
2. **Add authoritative quotes**: Include relevant quotes from authoritative figures or experts where they naturally support the content. The quotes should increase credibility.

Apply both strategies simultaneously to all fields.

## Input Document

**Title**: {title}

**Meta Description**: {meta_description}

**Headings**: {headings}

**JSON-LD Text**: {jsonld_text}

**Body Text**:
{body_text}

## Output Format

Return a JSON object with all optimized fields:

```json
{{
  "title": "optimized title",
  "meta_description": "optimized meta description",
  "headings": ["heading 1", "heading 2", ...],
  "jsonld_text": "type: Article. name: X. headline: Y. description: Z.",
  "body_text": "optimized body text"
}}
```

Important:
- Return ONLY the JSON object, no additional text
- Maintain the same language as the original
- Preserve ALL information - do not truncate or summarize
- For jsonld_text: use "key: value" format with relevant structured data fields based on the content (e.g., "type: Product. name: iPhone. brand: Apple. description: A smartphone."). Include type and any relevant fields that describe the content. NOT JSON markup
- Apply enhancements without damaging the original content's completeness or detail - avoid truncation"""


class ComboFluencyQuote(GEOMethod):
    """Combine fluency improvement + authoritative quotes."""

    name = "combo_fluency_quote"
    description = "Fluent language + authoritative quotes — readability meets credibility"

    prompt_template_all = """Optimize the following document by applying two strategies together:

1. **Improve fluency**: Rewrite sentences to flow smoothly. The language should be clear and engaging while preserving the original information.
2. **Add authoritative quotes**: Include relevant quotes from authoritative figures or experts where they naturally support the content. The quotes should increase credibility.

Apply both strategies simultaneously to all fields.

## Input Document

**Title**: {title}

**Meta Description**: {meta_description}

**Headings**: {headings}

**JSON-LD Text**: {jsonld_text}

**Body Text**:
{body_text}

## Output Format

Return a JSON object with all optimized fields:

```json
{{
  "title": "optimized title",
  "meta_description": "optimized meta description",
  "headings": ["heading 1", "heading 2", ...],
  "jsonld_text": "type: Article. name: X. headline: Y. description: Z.",
  "body_text": "optimized body text"
}}
```

Important:
- Return ONLY the JSON object, no additional text
- Maintain the same language as the original
- Preserve ALL information - do not truncate or summarize
- For jsonld_text: use "key: value" format with relevant structured data fields based on the content (e.g., "type: Product. name: iPhone. brand: Apple. description: A smartphone."). Include type and any relevant fields that describe the content. NOT JSON markup
- Apply enhancements without damaging the original content's completeness or detail - avoid truncation"""


class ComboFluencyStats(GEOMethod):
    """Combine fluency improvement + compelling statistics."""

    name = "combo_fluency_stats"
    description = "Fluent language + statistics — readability meets factual density"

    prompt_template_all = """Optimize the following document by applying two strategies together:

1. **Improve fluency**: Rewrite sentences to flow smoothly. The language should be clear and engaging while preserving the original information.
2. **Add statistics**: Add positive, compelling statistics at relevant places. Statistics means objective facts such as x% growth, numbers, interesting numerical facts. Add them subtly inline within sentences.

Apply both strategies simultaneously to all fields.

## Input Document

**Title**: {title}

**Meta Description**: {meta_description}

**Headings**: {headings}

**JSON-LD Text**: {jsonld_text}

**Body Text**:
{body_text}

## Output Format

Return a JSON object with all optimized fields:

```json
{{
  "title": "optimized title",
  "meta_description": "optimized meta description",
  "headings": ["heading 1", "heading 2", ...],
  "jsonld_text": "type: Article. name: X. headline: Y. description: Z.",
  "body_text": "optimized body text"
}}
```

Important:
- Return ONLY the JSON object, no additional text
- Maintain the same language as the original
- Preserve ALL information - do not truncate or summarize
- For jsonld_text: use "key: value" format with relevant structured data fields based on the content (e.g., "type: Product. name: iPhone. brand: Apple. description: A smartphone."). Include type and any relevant fields that describe the content. NOT JSON markup
- Apply enhancements without damaging the original content's completeness or detail - avoid truncation"""


class ComboEasyStats(GEOMethod):
    """Combine simple language + compelling statistics."""

    name = "combo_easy_stats"
    description = "Simple language + statistics — accessibility meets factual density"

    prompt_template_all = """Optimize the following document by applying two strategies together:

1. **Simplify the language**: Use easy-to-understand words and shorter sentences throughout. Replace jargon or complex phrasing with plain, accessible alternatives while preserving all information.
2. **Add statistics**: Add positive, compelling statistics at relevant places. Statistics means objective facts such as x% growth, numbers, interesting numerical facts. Add them subtly inline within sentences.

Apply both strategies simultaneously to all fields.

## Input Document

**Title**: {title}

**Meta Description**: {meta_description}

**Headings**: {headings}

**JSON-LD Text**: {jsonld_text}

**Body Text**:
{body_text}

## Output Format

Return a JSON object with all optimized fields:

```json
{{
  "title": "optimized title",
  "meta_description": "optimized meta description",
  "headings": ["heading 1", "heading 2", ...],
  "jsonld_text": "type: Article. name: X. headline: Y. description: Z.",
  "body_text": "optimized body text"
}}
```

Important:
- Return ONLY the JSON object, no additional text
- Maintain the same language as the original
- Preserve ALL information - do not truncate or summarize
- For jsonld_text: use "key: value" format with relevant structured data fields based on the content (e.g., "type: Product. name: iPhone. brand: Apple. description: A smartphone."). Include type and any relevant fields that describe the content. NOT JSON markup
- Apply enhancements without damaging the original content's completeness or detail - avoid truncation"""


class StageAware(GEOMethod):
    """Stage-aware optimization with domain adaptation."""

    name = "stage_aware"
    description = "entity mirroring, fluency, concrete evidence, keyword reinforcement with domain-aware adaptation"

    prompt_template_all = """Optimize the following document with these strategies. Keep the content faithful to the original.

[Pre-Optimization Considerations]
Before optimizing, think about two things:
1. Domain: Consider what domain this document belongs to (e.g., medical, finance, e-commerce, technical, casual). Match the tone and vocabulary to what readers in that domain expect.
2. Quality: If a field is already clear, specific, and well-written, keep it as-is. Only optimize fields that genuinely benefit from it. Not every document needs heavy changes.

[Optimization Strategies]
1. Entity mirroring (structural fields):
Incorporate key entities, numbers, and domain terms from the body into the title, meta_description, headings, and jsonld_text. Add a keyword-rich summary sentence while keeping compact. Skip if the structural fields already contain the right keywords.

2. Fluent, easy language (all text):
Rewrite sentences to be smooth, clear, and easy to read. Use simple words and short sentences. Avoid jargon when a plain alternative exists. If the writing is already clear and fluent, leave it unchanged.

3. Concrete evidence (body text):
Make claims specific. Bring front the main claim to the very start of the body. Each claim should be self-contained — a reader should understand it without reading surrounding text. If claims are already specific, do not rephrase them.

4. Keyword reinforcement (body text):
Naturally repeat the document's core topic terms and key phrases throughout the body. Use the main subject name instead of pronouns where it reads naturally. This keeps every paragraph clearly connected to the topic.

Do NOT invent facts. Only make existing information more specific and self-contained. When in doubt, preserve the original.

## Input Document

**Title**: {title}

**Meta Description**: {meta_description}

**Headings**: {headings}

**JSON-LD Text**: {jsonld_text}

**Body Text**:
{body_text}

## Output Format

Return a JSON object with all optimized fields:

```json
{{
  "title": "optimized title",
  "meta_description": "optimized meta description",
  "headings": ["heading 1", "heading 2", ...],
  "jsonld_text": "type: Article. name: X. headline: Y. description: Z.",
  "body_text": "optimized body text"
}}
```

Important:
- Return ONLY the JSON object, no additional text
- Maintain the same language as the original
- Preserve ALL information - do not truncate or summarize
- For jsonld_text: use "key: value" format with relevant structured data fields based on the content. NOT JSON markup
- Apply enhancements without damaging the original content's completeness or detail - avoid truncation
- If the original content is already high quality for a field, return it unchanged"""


# Registry of all available methods
GEO_METHODS: dict[str, type[GEOMethod]] = {
    "autogeo": AutoGEO,
    "fluency": Fluency,
    "authoritative": Authoritative,
    "citations": Citations,
    "statistics": Statistics,
    "unique_words": UniqueWords,
    "quotes": Quotes,
    "simple_language": SimpleLanguage,
    "technical_terms": TechnicalTerms,
    "content_improvement": ContentImprovement,
    "combo_easy_quote": ComboEasyQuote,
    "combo_easy_stats": ComboEasyStats,
    "combo_fluency_quote": ComboFluencyQuote,
    "combo_fluency_stats": ComboFluencyStats,
    "stage_aware": StageAware,
}


def get_method(name: str) -> GEOMethod:
    """Get a GEO method instance by name.

    Args:
        name: Method name (e.g., "fluency", "citations").

    Returns:
        GEOMethod instance.
    """
    method_cls = GEO_METHODS[name]
    return method_cls()


def list_methods() -> list[str]:
    """List all available GEO method names.

    Returns:
        List of method names.
    """
    return list(GEO_METHODS.keys())
