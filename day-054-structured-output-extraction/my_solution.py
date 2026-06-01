"""
Day 054: Structured Output Extraction — Your Implementation

Build a system that extracts structured, typed data from unstructured text
using schema-guided prompting patterns.

Key concepts to implement:
1. Schema definition and serialization for prompts
2. Prompt construction with anti-hallucination guardrails
3. Error-tolerant JSON parsing
4. Schema validation with type checking
5. Confidence-scored extraction results
6. Retry logic with error feedback

Hints:
- Start with the schema → prompt text conversion (it's the foundation)
- For JSON parsing, handle markdown fences FIRST, then trailing commas
- Validation should be recursive for nested objects
- Confidence scoring: explicit mention = high, inference = medium, absent = zero
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional


# =============================================================================
# Schemas (provided — use these for testing)
# =============================================================================

PERSON_SCHEMA: dict[str, Any] = {
    "name": "Person",
    "description": "Information about a person extracted from text",
    "properties": {
        "full_name": {
            "type": "string",
            "description": "The person's full name",
            "required": True,
        },
        "age": {
            "type": "integer",
            "description": "The person's age in years",
            "required": False,
        },
        "email": {
            "type": "string",
            "description": "Email address",
            "required": False,
        },
        "occupation": {
            "type": "string",
            "description": "Current job title or role",
            "required": False,
        },
        "skills": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of technical or professional skills mentioned",
            "required": False,
        },
        "address": {
            "type": "object",
            "description": "Physical address if mentioned",
            "required": False,
            "properties": {
                "city": {"type": "string", "description": "City name", "required": False},
                "state": {"type": "string", "description": "State or province", "required": False},
                "country": {"type": "string", "description": "Country", "required": False},
            },
        },
    },
}

JOB_POSTING_SCHEMA: dict[str, Any] = {
    "name": "JobPosting",
    "description": "Structured data from a job posting",
    "properties": {
        "title": {
            "type": "string",
            "description": "Job title",
            "required": True,
        },
        "company": {
            "type": "string",
            "description": "Company name",
            "required": True,
        },
        "salary_min": {
            "type": "integer",
            "description": "Minimum salary in USD",
            "required": False,
        },
        "salary_max": {
            "type": "integer",
            "description": "Maximum salary in USD",
            "required": False,
        },
        "location": {
            "type": "string",
            "description": "Job location",
            "required": False,
        },
        "remote": {
            "type": "boolean",
            "description": "Whether remote work is available",
            "required": False,
        },
        "requirements": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of job requirements",
            "required": False,
        },
        "experience_years": {
            "type": "integer",
            "description": "Required years of experience",
            "required": False,
        },
    },
}


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class FieldResult:
    """Result for a single extracted field, including confidence metadata."""
    value: Any
    confidence: float  # 0.0 to 1.0
    evidence: Optional[str] = None  # The text snippet supporting this value

    def is_reliable(self, threshold: float = 0.7) -> bool:
        """Check if this extraction meets a confidence threshold."""
        raise NotImplementedError("TODO: implement this")


@dataclass
class ExtractionResult:
    """Complete extraction result with per-field confidence and validation status."""
    schema_name: str
    fields: dict[str, FieldResult] = field(default_factory=dict)
    validation_errors: list[str] = field(default_factory=list)
    retry_count: int = 0
    success: bool = True

    def to_dict(self, min_confidence: float = 0.0) -> dict[str, Any]:
        """Convert to a plain dict, optionally filtering by confidence.

        Hint: iterate over self.fields, include only those meeting the threshold.
        """
        raise NotImplementedError("TODO: implement this")

    def summary(self) -> str:
        """Human-readable summary of extraction quality.

        Hint: count fields by confidence tier (high >= 0.8, med 0.4-0.8, low < 0.4).
        """
        raise NotImplementedError("TODO: implement this")


# =============================================================================
# Prompt Construction
# =============================================================================

def schema_to_prompt_text(schema: dict[str, Any], indent: int = 0) -> str:
    """
    Convert a schema dict into a clear text representation for prompt inclusion.

    Hint: iterate over schema["properties"], format each field with its type,
    required status, and description. Recurse for nested objects.
    """
    raise NotImplementedError("TODO: implement this")


def build_extraction_prompt(schema: dict[str, Any], text: str) -> str:
    """
    Construct a complete extraction prompt from schema + source text.

    Hint: include these sections in order:
    1. Role/task framing (you are an extraction system)
    2. Schema presentation (call schema_to_prompt_text)
    3. Source text
    4. Output format instructions (JSON with confidence scores)
    5. Anti-hallucination guardrails (use null for missing fields)
    """
    raise NotImplementedError("TODO: implement this")


def build_retry_prompt(
    schema: dict[str, Any],
    text: str,
    previous_output: str,
    errors: list[str],
) -> str:
    """
    Build a retry prompt including the previous failed attempt and specific errors.

    Hint: start with build_extraction_prompt, then append the error list
    and previous output so the model can fix specific issues.
    """
    raise NotImplementedError("TODO: implement this")


# =============================================================================
# JSON Parsing (Error-Tolerant)
# =============================================================================

def parse_llm_json(raw_output: str) -> dict[str, Any]:
    """
    Parse JSON from LLM output with tolerance for common issues.

    Strategy (try each in order, return first success):
    1. Direct json.loads
    2. Strip markdown code fences (```json ... ```) and retry
    3. Regex-extract the JSON object from surrounding text
    4. Fix trailing commas and retry

    Hint: re.sub(r"^```(?:json)?\\s*\\n?", "", text) strips opening fence
    """
    raise NotImplementedError("TODO: implement this")


# =============================================================================
# Schema Validation
# =============================================================================

def validate_field_type(value: Any, expected_type: str) -> tuple[bool, str]:
    """
    Validate a single field value against its expected type.

    Returns (is_valid, error_message).

    Hint: map type strings to Python types. Be permissive with int/float.
    None values are always valid (they mean "not found").
    """
    raise NotImplementedError("TODO: implement this")


def validate_extraction(
    extracted: dict[str, Any],
    schema: dict[str, Any],
) -> list[str]:
    """
    Validate extracted data against schema. Returns list of error messages.

    Checks:
    1. Required fields present and non-null
    2. Types match
    3. Nested objects validated recursively
    4. Array items type-checked

    Hint: the extracted dict has field_name -> {"value": ..., "confidence": ...} structure
    """
    raise NotImplementedError("TODO: implement this")


# =============================================================================
# Rule-Based Extractor
# =============================================================================

def extract_with_rules(text: str, schema: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """
    Extract fields from text using regex patterns.

    Returns dict of field_name -> {"value": ..., "confidence": float, "evidence": str|None}

    Hint: dispatch on field_name. For each field type, define regex patterns
    ordered from most specific (high confidence) to least specific (low confidence).
    Common patterns:
    - Name: "My name is X", "X is a ...", any capitalized two-word sequence
    - Age: "N years old", "age: N", "aged N"
    - Email: standard email regex
    - Skills: look for skill section headers, then split by commas/bullets
    """
    raise NotImplementedError("TODO: implement this")


# =============================================================================
# Main Extraction Pipeline
# =============================================================================

def extract_structured_data(
    text: str,
    schema: dict[str, Any],
    max_retries: int = 2,
    min_confidence: float = 0.0,
) -> ExtractionResult:
    """
    Main pipeline: extract → validate → retry if needed → return results.

    Steps:
    1. Build prompt (for debugging/future LLM use)
    2. Extract fields with extract_with_rules
    3. Validate against schema
    4. If errors, retry up to max_retries times
    5. Assemble ExtractionResult with FieldResult per field

    Hint: the retry loop re-extracts and re-validates each iteration.
    """
    raise NotImplementedError("TODO: implement this")


# =============================================================================
# Test Your Implementation
# =============================================================================

if __name__ == "__main__":
    # Test 1: Person extraction
    person_text = """
    Sarah Chen is a 28-year-old machine learning engineer at DeepMind in London.
    She specializes in reinforcement learning and computer vision, with strong skills
    in Python, PyTorch, and C++. You can reach her at sarah.chen@example.com.
    She's currently based in San Francisco, CA, United States.
    """
    print("=" * 60)
    print("Test 1: Person Extraction")
    print("=" * 60)
    result = extract_structured_data(person_text, PERSON_SCHEMA)
    print(result.summary())
    for name, fr in result.fields.items():
        print(f"  {name}: {fr.value} (confidence: {fr.confidence:.2f})")

    # Test 2: Job posting extraction
    job_text = """
    Senior Backend Engineer at Stripe

    Stripe is hiring a Senior Backend Engineer for payments infrastructure.
    Location: San Francisco, CA (remote-friendly).
    Requirements: 5+ years experience, Go, Python, distributed systems, AWS
    Compensation: $180k - $250k base salary.
    """
    print("\n" + "=" * 60)
    print("Test 2: Job Posting Extraction")
    print("=" * 60)
    result = extract_structured_data(job_text, JOB_POSTING_SCHEMA)
    print(result.summary())
    for name, fr in result.fields.items():
        print(f"  {name}: {fr.value} (confidence: {fr.confidence:.2f})")

    # Test 3: JSON parsing
    print("\n" + "=" * 60)
    print("Test 3: JSON Parsing")
    print("=" * 60)
    test_cases = [
        '{"a": 1}',
        '```json\n{"a": 1}\n```',
        '{"a": 1,}',
        'Result: {"a": 1} done.',
    ]
    for raw in test_cases:
        try:
            print(f"  {raw!r:50s} -> {parse_llm_json(raw)}")
        except (ValueError, NotImplementedError) as e:
            print(f"  {raw!r:50s} -> ERROR: {e}")

    # Test 4: Schema validation
    print("\n" + "=" * 60)
    print("Test 4: Prompt Generation")
    print("=" * 60)
    prompt = build_extraction_prompt(PERSON_SCHEMA, "Test text here.")
    print(prompt[:300] + "...")
