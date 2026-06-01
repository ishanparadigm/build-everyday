"""
Day 054: Structured Output Extraction

A complete system for extracting structured, validated data from unstructured text
using schema-guided prompting. This implementation works WITHOUT an LLM API by
simulating extraction with rule-based parsing — the architecture and patterns are
identical to what you'd use with a real LLM, making it easy to swap in an API later.

Key design decisions:
- Schema defined as Python dicts for simplicity (production systems use Pydantic/JSON Schema)
- Confidence scoring based on extraction evidence strength
- Retry logic with error feedback loop
- Graceful degradation: partial results over total failure
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional


# =============================================================================
# Schema Definition Layer
# =============================================================================

# Schemas are dicts describing the expected structure.
# Each field has: type, description, required (bool), and optionally: items (for arrays),
# properties (for nested objects), enum (for constrained values).
#
# Why dicts instead of classes? They serialize directly to JSON for prompt inclusion,
# and they mirror JSON Schema — the industry standard for this kind of work.

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
# Data Classes for Results
# =============================================================================


@dataclass
class FieldResult:
    """Result for a single extracted field, including confidence metadata."""
    value: Any
    confidence: float  # 0.0 to 1.0
    evidence: Optional[str] = None  # The text snippet that supports this value

    def is_reliable(self, threshold: float = 0.7) -> bool:
        """Check if this extraction meets a confidence threshold."""
        return self.confidence >= threshold


@dataclass
class ExtractionResult:
    """Complete extraction result with per-field confidence and validation status."""
    schema_name: str
    fields: dict[str, FieldResult] = field(default_factory=dict)
    validation_errors: list[str] = field(default_factory=list)
    retry_count: int = 0
    success: bool = True

    def to_dict(self, min_confidence: float = 0.0) -> dict[str, Any]:
        """Convert to a plain dict, optionally filtering by confidence."""
        result = {}
        for name, field_result in self.fields.items():
            if field_result.confidence >= min_confidence:
                result[name] = field_result.value
        return result

    def summary(self) -> str:
        """Human-readable summary of extraction quality."""
        total = len(self.fields)
        high_conf = sum(1 for f in self.fields.values() if f.confidence >= 0.8)
        med_conf = sum(1 for f in self.fields.values() if 0.4 <= f.confidence < 0.8)
        low_conf = sum(1 for f in self.fields.values() if 0.0 < f.confidence < 0.4)
        missing = sum(1 for f in self.fields.values() if f.value is None)
        return (
            f"Extraction '{self.schema_name}': {total} fields | "
            f"High: {high_conf}, Med: {med_conf}, Low: {low_conf}, Missing: {missing} | "
            f"Retries: {self.retry_count} | Errors: {len(self.validation_errors)}"
        )


# =============================================================================
# Prompt Construction
# =============================================================================

def schema_to_prompt_text(schema: dict[str, Any], indent: int = 0) -> str:
    """
    Convert a schema dict into a clear text representation for prompt inclusion.

    Why a custom format instead of raw JSON Schema? Because LLMs respond better to
    natural language descriptions alongside type info. We want the model to understand
    the INTENT of each field, not just its type.
    """
    lines = []
    prefix = "  " * indent

    for field_name, field_def in schema.get("properties", {}).items():
        type_str = field_def["type"]
        required = "REQUIRED" if field_def.get("required") else "optional"
        desc = field_def.get("description", "")

        if type_str == "object" and "properties" in field_def:
            lines.append(f"{prefix}- {field_name} ({type_str}, {required}): {desc}")
            # Recurse into nested object
            lines.append(schema_to_prompt_text(field_def, indent + 1))
        elif type_str == "array":
            item_type = field_def.get("items", {}).get("type", "any")
            lines.append(f"{prefix}- {field_name} (array of {item_type}, {required}): {desc}")
        else:
            lines.append(f"{prefix}- {field_name} ({type_str}, {required}): {desc}")

    return "\n".join(lines)


def build_extraction_prompt(schema: dict[str, Any], text: str) -> str:
    """
    Construct a complete extraction prompt.

    The prompt structure is deliberate:
    1. Role and task framing (sets extraction, not generation mode)
    2. Schema presentation (what to extract)
    3. Source text (where to extract from)
    4. Output format instructions (how to return results)
    5. Anti-hallucination guardrails (critical for reliability)
    """
    schema_text = schema_to_prompt_text(schema)

    prompt = f"""You are a precise data extraction system. Your task is to extract structured data from the provided text according to the schema below.

SCHEMA: {schema['name']}
{schema['description']}

Fields to extract:
{schema_text}

SOURCE TEXT:
---
{text}
---

INSTRUCTIONS:
1. Extract ONLY information that is explicitly stated or strongly implied in the text.
2. For each field, provide a confidence score (0.0 to 1.0):
   - 1.0: Explicitly stated in text
   - 0.7-0.9: Strongly implied or requires minor inference
   - 0.3-0.6: Partially supported, some guessing required
   - 0.0: Not found in text — use null for the value
3. Do NOT hallucinate or invent values. If information is not in the text, use null.
4. Return valid JSON in this exact format:

{{
  "fields": {{
    "field_name": {{"value": <extracted_value>, "confidence": <float>, "evidence": "<supporting text snippet>"}},
    ...
  }}
}}

Return ONLY the JSON object, no other text."""

    return prompt


def build_retry_prompt(
    schema: dict[str, Any],
    text: str,
    previous_output: str,
    errors: list[str],
) -> str:
    """
    Build a retry prompt that includes the previous failed attempt and specific errors.

    Why include the failed output? Because the model can often fix specific issues
    more efficiently than re-extracting from scratch. This is analogous to a compiler
    error message — you fix the specific issue, not rewrite the whole program.
    """
    error_text = "\n".join(f"  - {e}" for e in errors)

    return f"""{build_extraction_prompt(schema, text)}

IMPORTANT: Your previous attempt had errors:
{error_text}

Previous (incorrect) output:
{previous_output}

Please fix ONLY the issues listed above and return corrected JSON."""


# =============================================================================
# JSON Parsing (Error-Tolerant)
# =============================================================================

def parse_llm_json(raw_output: str) -> dict[str, Any]:
    """
    Parse JSON from LLM output with tolerance for common formatting issues.

    LLMs frequently wrap JSON in markdown code fences, add trailing commas,
    or include explanatory text before/after the JSON. This parser handles
    all of these cases.

    Strategy:
    1. Try direct parse (fast path for well-behaved output)
    2. Strip markdown fences and retry
    3. Extract JSON object with regex and retry
    4. Fix common syntax errors and retry
    5. Give up and raise ValueError
    """
    # Fast path: try direct parse
    try:
        return json.loads(raw_output)
    except json.JSONDecodeError:
        pass

    # Strip markdown code fences: ```json ... ``` or ``` ... ```
    cleaned = re.sub(r"^```(?:json)?\s*\n?", "", raw_output.strip())
    cleaned = re.sub(r"\n?```\s*$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Try to extract a JSON object from surrounding text
    # This handles cases where the LLM adds explanation before/after
    match = re.search(r"\{[\s\S]*\}", cleaned)
    if match:
        json_str = match.group(0)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            # Fix trailing commas (e.g., {"a": 1,} → {"a": 1})
            fixed = re.sub(r",\s*([}\]])", r"\1", json_str)
            try:
                return json.loads(fixed)
            except json.JSONDecodeError:
                pass

    raise ValueError(f"Could not parse JSON from output: {raw_output[:200]}...")


# =============================================================================
# Schema Validation
# =============================================================================

def validate_field_type(value: Any, expected_type: str) -> tuple[bool, str]:
    """
    Validate a single field's type against the schema.

    Returns (is_valid, error_message).
    We're permissive with numeric types (int/float interchangeable) because
    LLMs often return "25" as a string or 25.0 as a float when we want an int.
    """
    if value is None:
        return True, ""  # None is always valid (means "not found")

    type_map = {
        "string": str,
        "integer": (int, float),  # Accept float if it's a whole number
        "number": (int, float),
        "boolean": bool,
        "array": list,
        "object": dict,
    }

    expected = type_map.get(expected_type)
    if expected is None:
        return True, ""  # Unknown type, skip validation

    if not isinstance(value, expected):
        return False, f"Expected {expected_type}, got {type(value).__name__}"

    # Special case: "integer" type but got float — check if it's a whole number
    if expected_type == "integer" and isinstance(value, float):
        if value != int(value):
            return False, f"Expected integer, got float {value}"

    return True, ""


def validate_extraction(
    extracted: dict[str, Any],
    schema: dict[str, Any],
) -> list[str]:
    """
    Validate extracted data against schema. Returns list of error messages.

    Validation checks:
    1. Required fields must be present and non-null
    2. Field types must match schema
    3. Nested objects are validated recursively
    4. Array items are type-checked
    """
    errors = []
    properties = schema.get("properties", {})

    for field_name, field_def in properties.items():
        field_data = extracted.get(field_name)

        # Check required fields
        if field_def.get("required") and field_data is None:
            errors.append(f"Required field '{field_name}' is missing or null")
            continue

        if field_data is None:
            continue

        # Extract the actual value (might be wrapped in FieldResult-style dict)
        value = field_data.get("value") if isinstance(field_data, dict) and "value" in field_data else field_data

        if value is None:
            if field_def.get("required"):
                errors.append(f"Required field '{field_name}' has null value")
            continue

        # Type validation
        is_valid, error_msg = validate_field_type(value, field_def["type"])
        if not is_valid:
            errors.append(f"Field '{field_name}': {error_msg}")

        # Nested object validation
        if field_def["type"] == "object" and isinstance(value, dict) and "properties" in field_def:
            nested_errors = validate_extraction(value, field_def)
            errors.extend(f"{field_name}.{e}" for e in nested_errors)

        # Array item validation
        if field_def["type"] == "array" and isinstance(value, list):
            item_type = field_def.get("items", {}).get("type")
            if item_type:
                for i, item in enumerate(value):
                    is_valid, error_msg = validate_field_type(item, item_type)
                    if not is_valid:
                        errors.append(f"Field '{field_name}[{i}]': {error_msg}")

    return errors


# =============================================================================
# Rule-Based Extractor (Simulates LLM Extraction)
# =============================================================================
# In production, you'd call an LLM API here. This rule-based extractor
# demonstrates the SAME interface and patterns — schemas in, structured data out —
# using regex and heuristics instead of an API call.

def extract_with_rules(text: str, schema: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """
    Rule-based extraction that simulates LLM behavior.

    Each extractor pattern returns (value, confidence, evidence).
    This is the component you'd swap out for an actual LLM call.
    The rest of the pipeline (validation, retry, confidence scoring) stays the same.
    """
    text_lower = text.lower()
    results: dict[str, dict[str, Any]] = {}

    for field_name, field_def in schema.get("properties", {}).items():
        value, confidence, evidence = _extract_single_field(
            text, text_lower, field_name, field_def
        )
        results[field_name] = {
            "value": value,
            "confidence": confidence,
            "evidence": evidence,
        }

    return results


def _extract_single_field(
    text: str,
    text_lower: str,
    field_name: str,
    field_def: dict[str, Any],
) -> tuple[Any, float, Optional[str]]:
    """
    Extract a single field using pattern matching.

    Returns (value, confidence, evidence_snippet).
    Patterns are ordered from most specific to least specific — higher specificity
    means higher confidence in the extraction.
    """
    field_type = field_def["type"]

    # --- Name extraction ---
    if field_name in ("full_name", "name"):
        # Pattern: "My name is X" or "I'm X" or "X is a ..."
        patterns = [
            (r"(?:my name is|i'?m|i am)\s+([A-Z][a-z]+ [A-Z][a-z]+(?:\s[A-Z][a-z]+)?)", 0.95),
            (r"^([A-Z][a-z]+ [A-Z][a-z]+(?:\s[A-Z][a-z]+)?)\s+is\s+a", 0.9),
            (r"([A-Z][a-z]+ [A-Z][a-z]+(?:\s[A-Z][a-z]+)?)", 0.6),  # Any capitalized name
        ]
        for pattern, conf in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1), conf, match.group(0)[:80]
        return None, 0.0, None

    # --- Age extraction ---
    if field_name == "age":
        patterns = [
            (r"(\d{1,3})\s*(?:years?\s*old|year-old|yo\b)", 0.95),
            (r"age[:\s]+(\d{1,3})", 0.95),
            (r"aged?\s+(\d{1,3})", 0.9),
        ]
        for pattern, conf in patterns:
            match = re.search(pattern, text_lower)
            if match:
                age = int(match.group(1))
                if 0 < age < 150:  # Sanity check
                    return age, conf, match.group(0)[:80]
        return None, 0.0, None

    # --- Email extraction ---
    if field_name == "email":
        match = re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", text)
        if match:
            return match.group(0), 0.98, match.group(0)
        return None, 0.0, None

    # --- Occupation/title extraction ---
    if field_name in ("occupation", "title"):
        patterns = [
            (r"(?:works?\s+as\s+(?:a|an)\s+)(.+?)(?:\.|,|$)", 0.9),
            (r"(?:is\s+(?:a|an)\s+)(.+?)(?:\s+at\s+|\s+who|\s+with|\.|,|$)", 0.8),
            (r"(?:position|role|title)[:\s]+(.+?)(?:\.|,|$)", 0.9),
            # Job posting title patterns
            (r"(?:hiring|looking for|seeking)\s+(?:a|an)\s+(.+?)(?:\.|,|$)", 0.85),
            (r"^(.+?)\s+(?:position|role|opening)", 0.85),
        ]
        for pattern, conf in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value = match.group(1).strip()
                if len(value) < 100:  # Sanity check
                    return value, conf, match.group(0)[:80]
        return None, 0.0, None

    # --- Company extraction ---
    if field_name == "company":
        patterns = [
            (r"(?:at|@)\s+([A-Z][\w\s&.]+?)(?:\.|,|\s+is|\s+are|\s+in|\s+we|$)", 0.9),
            (r"([A-Z][\w\s&.]+?)\s+is\s+(?:hiring|looking|seeking)", 0.85),
            (r"(?:join|company)[:\s]+([A-Z][\w\s&.]+?)(?:\.|,|$)", 0.85),
        ]
        for pattern, conf in patterns:
            match = re.search(pattern, text)
            if match:
                value = match.group(1).strip()
                if 1 < len(value) < 80:
                    return value, conf, match.group(0)[:80]
        return None, 0.0, None

    # --- Skills extraction (array) ---
    if field_name in ("skills", "requirements"):
        skills_found = []
        # Look for explicit skill lists
        skill_section = re.search(
            r"(?:skills?|requirements?|proficient|experienced? (?:in|with)|technologies)[:\s]+"
            r"(.+?)(?:\n\n|\Z)",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if skill_section:
            section_text = skill_section.group(1)
            # Split by commas, semicolons, "and", or bullet points
            items = re.split(r"[,;]\s*|\s+and\s+|\n\s*[-*]\s*|\n", section_text)
            skills_found = [s.strip().rstrip(".") for s in items if s.strip() and len(s.strip()) < 50]

        # Also look for known technology keywords
        known_skills = [
            "Python", "JavaScript", "TypeScript", "React", "Node.js", "Go", "Rust",
            "Java", "C\\+\\+", "SQL", "Docker", "Kubernetes", "AWS", "GCP", "Azure",
            "machine learning", "deep learning", "NLP", "computer vision",
            "Solidity", "blockchain", "DeFi", "smart contracts",
            "ROS", "robotics", "control systems", "SLAM",
        ]
        for skill in known_skills:
            if re.search(r"\b" + skill + r"\b", text, re.IGNORECASE):
                clean_skill = skill.replace("\\+\\+", "++")
                if clean_skill not in skills_found:
                    skills_found.append(clean_skill)

        if skills_found:
            return skills_found, 0.85, f"Found {len(skills_found)} skills"
        return None, 0.0, None

    # --- Address extraction (nested object) ---
    if field_name == "address" and field_type == "object":
        address = {}
        # City, State pattern
        match = re.search(r"([A-Z][a-z]+(?:\s[A-Z][a-z]+)?),\s*([A-Z]{2})\b", text)
        if match:
            address["city"] = match.group(1)
            address["state"] = match.group(2)
        # Country
        countries = ["United States", "USA", "UK", "Canada", "Germany", "France", "Japan", "India"]
        for country in countries:
            if country.lower() in text.lower():
                address["country"] = country
                break
        # City from "in <City>" or "based in <City>"
        if "city" not in address:
            match = re.search(r"(?:in|based in|from)\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)", text)
            if match:
                address["city"] = match.group(1)

        if address:
            return address, 0.75, f"Found address components: {list(address.keys())}"
        return None, 0.0, None

    # --- Salary extraction ---
    if field_name in ("salary_min", "salary_max"):
        # Match patterns like "$120k-$160k", "$120,000 - $160,000", "$150k"
        patterns = [
            r"\$(\d{2,3})[,.]?(\d{3})?\s*[-–to]+\s*\$(\d{2,3})[,.]?(\d{3})?",
            r"\$(\d{2,3})k\s*[-–to]+\s*\$(\d{2,3})k",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                groups = match.groups()
                if len(groups) == 4:  # Full number format
                    low = int(groups[0]) * (1000 if groups[1] is None else 1) + (int(groups[1]) if groups[1] else 0)
                    high = int(groups[2]) * (1000 if groups[3] is None else 1) + (int(groups[3]) if groups[3] else 0)
                else:  # "k" format
                    low = int(groups[0]) * 1000
                    high = int(groups[1]) * 1000
                if field_name == "salary_min":
                    return low, 0.9, match.group(0)
                else:
                    return high, 0.9, match.group(0)

        # Single salary mention
        match = re.search(r"\$(\d{2,3})k", text, re.IGNORECASE)
        if match:
            val = int(match.group(1)) * 1000
            return val, 0.7, match.group(0)

        return None, 0.0, None

    # --- Boolean extraction (e.g., remote) ---
    if field_name == "remote" and field_type == "boolean":
        remote_positive = ["remote", "work from home", "wfh", "distributed", "anywhere"]
        remote_negative = ["on-site", "onsite", "in-office", "in office", "no remote"]
        for neg in remote_negative:
            if neg in text.lower():
                return False, 0.9, neg
        for pos in remote_positive:
            if pos in text.lower():
                return True, 0.9, pos
        return None, 0.0, None

    # --- Location extraction ---
    if field_name == "location":
        match = re.search(
            r"(?:location|based in|located in|office in)[:\s]+([A-Za-z\s,]+?)(?:\.|;|\n|$)",
            text,
            re.IGNORECASE,
        )
        if match:
            return match.group(1).strip(), 0.85, match.group(0)[:80]
        # Fallback: City, State pattern
        match = re.search(r"([A-Z][a-z]+(?:\s[A-Z][a-z]+)?),\s*([A-Z]{2})\b", text)
        if match:
            return match.group(0), 0.7, match.group(0)
        return None, 0.0, None

    # --- Experience years extraction ---
    if field_name == "experience_years":
        match = re.search(r"(\d+)\+?\s*years?\s*(?:of\s+)?experience", text, re.IGNORECASE)
        if match:
            return int(match.group(1)), 0.9, match.group(0)
        return None, 0.0, None

    # --- Default: no extraction ---
    return None, 0.0, None


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
    Main extraction pipeline: extract → validate → retry if needed → return results.

    This is the function downstream code calls. It handles the full lifecycle:
    1. Build prompt (for logging/debugging, even though we use rule-based extraction)
    2. Extract fields
    3. Validate against schema
    4. Retry on validation errors (up to max_retries)
    5. Assemble results with confidence scores

    Args:
        text: The unstructured text to extract from
        schema: The extraction schema
        max_retries: Maximum retry attempts for validation failures
        min_confidence: Minimum confidence threshold for included fields

    Returns:
        ExtractionResult with per-field confidence and validation status
    """
    # Step 1: Build the prompt (useful for debugging and future LLM integration)
    prompt = build_extraction_prompt(schema, text)

    # Step 2: Extract (rule-based; swap this for LLM API call in production)
    raw_fields = extract_with_rules(text, schema)

    # Step 3: Validate
    errors = validate_extraction(raw_fields, schema)
    retry_count = 0

    # Step 4: Retry loop
    while errors and retry_count < max_retries:
        retry_count += 1
        # In production: build_retry_prompt() and call LLM again
        # Here we just re-extract (rule-based doesn't benefit from retry, but the pattern matters)
        raw_fields = extract_with_rules(text, schema)
        errors = validate_extraction(raw_fields, schema)

    # Step 5: Assemble results
    result = ExtractionResult(
        schema_name=schema["name"],
        validation_errors=errors,
        retry_count=retry_count,
        success=len(errors) == 0,
    )

    for field_name, field_data in raw_fields.items():
        fr = FieldResult(
            value=field_data["value"],
            confidence=field_data["confidence"],
            evidence=field_data.get("evidence"),
        )
        if fr.confidence >= min_confidence:
            result.fields[field_name] = fr

    return result


# =============================================================================
# Coercion Utilities
# =============================================================================

def coerce_value(value: Any, target_type: str) -> Any:
    """
    Attempt to coerce a value to the target type.

    LLMs often return numbers as strings or floats when you want ints.
    This function handles the most common mismatches gracefully.
    """
    if value is None:
        return None

    try:
        if target_type == "integer":
            if isinstance(value, str):
                return int(float(value))
            return int(value)
        elif target_type == "number":
            return float(value)
        elif target_type == "string":
            return str(value)
        elif target_type == "boolean":
            if isinstance(value, str):
                return value.lower() in ("true", "yes", "1")
            return bool(value)
    except (ValueError, TypeError):
        return value

    return value


# =============================================================================
# Demo: End-to-End Extraction
# =============================================================================

def demo_person_extraction() -> ExtractionResult:
    """Demonstrate extraction from a bio/resume-style text."""
    text = """
    Sarah Chen is a 28-year-old machine learning engineer at DeepMind in London.
    She specializes in reinforcement learning and computer vision, with strong skills
    in Python, PyTorch, and C++. Sarah holds a PhD from Stanford and previously
    worked at Google Brain. You can reach her at sarah.chen@example.com.
    She's currently based in San Francisco, CA, United States.
    """
    print("=" * 70)
    print("DEMO 1: Person Extraction")
    print("=" * 70)
    print(f"\nSource text:\n{text.strip()}\n")

    # Show the prompt that would be sent to an LLM
    prompt = build_extraction_prompt(PERSON_SCHEMA, text)
    print(f"Generated prompt (first 500 chars):\n{prompt[:500]}...\n")

    # Run extraction
    result = extract_structured_data(text, PERSON_SCHEMA)

    print(f"\n{result.summary()}\n")
    print("Extracted fields:")
    for name, fr in result.fields.items():
        status = "OK" if fr.is_reliable() else "LOW CONF"
        print(f"  {name:15s} = {fr.value!r:40s} [conf: {fr.confidence:.2f}] [{status}]")
        if fr.evidence:
            print(f"  {'':15s}   evidence: \"{fr.evidence}\"")

    print(f"\nAs dict (conf >= 0.7): {result.to_dict(min_confidence=0.7)}")

    return result


def demo_job_posting_extraction() -> ExtractionResult:
    """Demonstrate extraction from a job posting."""
    text = """
    Senior Backend Engineer at Stripe

    Stripe is hiring a Senior Backend Engineer to work on our payments infrastructure.
    Location: San Francisco, CA (remote-friendly).

    Requirements:
    - 5+ years of experience in backend development
    - Strong proficiency in Go, Python, or Java
    - Experience with distributed systems and microservices
    - Familiarity with AWS or GCP cloud platforms
    - Knowledge of SQL and NoSQL databases

    Compensation: $180k - $250k base salary plus equity.
    """
    print("\n" + "=" * 70)
    print("DEMO 2: Job Posting Extraction")
    print("=" * 70)
    print(f"\nSource text:\n{text.strip()}\n")

    result = extract_structured_data(text, JOB_POSTING_SCHEMA)

    print(f"\n{result.summary()}\n")
    print("Extracted fields:")
    for name, fr in result.fields.items():
        status = "OK" if fr.is_reliable() else "LOW CONF"
        print(f"  {name:20s} = {fr.value!r:40s} [conf: {fr.confidence:.2f}] [{status}]")
        if fr.evidence:
            print(f"  {'':20s}   evidence: \"{fr.evidence}\"")

    print(f"\nAs dict (all): {result.to_dict()}")

    return result


def demo_partial_extraction() -> ExtractionResult:
    """Demonstrate graceful handling when text has minimal information."""
    text = """
    Quick note: met John at the conference. He mentioned something about
    working with robots but I didn't catch the details.
    """
    print("\n" + "=" * 70)
    print("DEMO 3: Partial Extraction (Sparse Text)")
    print("=" * 70)
    print(f"\nSource text:\n{text.strip()}\n")

    result = extract_structured_data(text, PERSON_SCHEMA)

    print(f"\n{result.summary()}\n")
    print("Extracted fields:")
    for name, fr in result.fields.items():
        status = "OK" if fr.is_reliable() else "LOW CONF" if fr.confidence > 0 else "MISSING"
        print(f"  {name:15s} = {fr.value!r:40s} [conf: {fr.confidence:.2f}] [{status}]")

    print(f"\nAs dict (conf >= 0.7): {result.to_dict(min_confidence=0.7)}")
    print("  ^ Notice: sparse text yields few high-confidence fields. This is CORRECT")
    print("    behavior — the system admits what it doesn't know rather than hallucinating.")

    return result


def demo_json_parsing() -> None:
    """Demonstrate error-tolerant JSON parsing."""
    print("\n" + "=" * 70)
    print("DEMO 4: Error-Tolerant JSON Parsing")
    print("=" * 70)

    test_cases = [
        ("Clean JSON", '{"name": "Alice", "age": 30}'),
        ("Markdown-wrapped", '```json\n{"name": "Bob", "age": 25}\n```'),
        ("Trailing comma", '{"name": "Charlie", "age": 35,}'),
        ("Text around JSON", 'Here is the result: {"name": "Diana", "age": 28} Hope that helps!'),
    ]

    for label, raw in test_cases:
        try:
            parsed = parse_llm_json(raw)
            print(f"  {label:25s} -> {parsed}")
        except ValueError as e:
            print(f"  {label:25s} -> FAILED: {e}")


if __name__ == "__main__":
    # Run all demos to show the system working end-to-end
    demo_person_extraction()
    demo_job_posting_extraction()
    demo_partial_extraction()
    demo_json_parsing()

    print("\n" + "=" * 70)
    print("All demos completed successfully.")
    print("=" * 70)
