# Day 054: Structured Output Extraction

## Overview

Build a system that extracts structured, typed data from unstructured text using LLMs with schema-guided prompting. This is one of the most practically valuable LLM patterns — every production AI system eventually needs to turn messy human language into clean, validated data structures that downstream code can consume reliably.

In the real world, structured extraction powers everything from resume parsers to medical record digitization to financial document processing. The challenge isn't just getting the LLM to output JSON — it's handling schema validation, partial extraction, confidence scoring, nested structures, and graceful degradation when the text doesn't contain what you're looking for.

## Core Concepts

### Schema-Guided Prompting

The fundamental idea: instead of hoping the LLM returns data in the right shape, you **tell it exactly what shape you need** by embedding the schema in the prompt. This is more than just "return JSON" — it means:

1. **Defining the schema formally** (using JSON Schema, Pydantic models, or similar) so it's unambiguous
2. **Including the schema in the prompt** so the model knows every field, its type, whether it's required, and what values are valid
3. **Validating the output** against the schema to catch hallucinated fields, wrong types, or missing required data

The key insight: LLMs are probabilistic text generators. Without a schema constraint, they'll happily invent fields, use inconsistent types, or omit data. The schema acts as both a guide for generation and a contract for validation.

### Extraction vs. Generation

There's a critical distinction between **extraction** (pulling facts that exist in the text) and **generation** (creating new content). A well-built extraction system should:

- Return `null` or a low confidence score for fields where the text provides no evidence
- Never hallucinate values that aren't supported by the source text
- Distinguish between "the text says X" and "the text doesn't mention this"

This is where naive approaches fail: a basic "extract these fields" prompt will often make up plausible-sounding values rather than admitting "not found."

### Confidence Scoring

For each extracted field, we can ask the model to rate its confidence. This isn't a probability in the statistical sense — it's the model's self-assessment of how clearly the source text supports the extracted value. Useful tiers:

- **High (0.9-1.0)**: Text explicitly states the value
- **Medium (0.5-0.8)**: Value is strongly implied or requires minor inference
- **Low (0.1-0.4)**: Value is guessed from context or partially supported
- **None (0.0)**: Field not found in text, returning null

### Nested Schema Handling

Real-world extraction rarely involves flat key-value pairs. You need to handle:

- **Nested objects**: An "address" field containing street, city, state, zip
- **Arrays of objects**: Multiple "work experiences," each with company, role, dates
- **Optional vs. required fields**: Knowing which missing fields are errors vs. expected
- **Union types**: A field that could be a string OR an object depending on context

### Validation and Error Recovery

When the LLM returns malformed output (and it will), you need a strategy:

1. **Parse the JSON** — handle common LLM mistakes like trailing commas, unquoted keys, markdown code fences
2. **Validate against schema** — check types, required fields, enum values
3. **Retry with error context** — if validation fails, send the error back to the model with the original text for a corrected attempt
4. **Graceful degradation** — return partial results with error annotations rather than failing entirely

## Step-by-Step Breakdown

### Step 1: Define the Schema System

Build a schema definition layer using Python dataclasses or dictionaries that can:
- Express field names, types, descriptions, and whether they're required
- Handle nested objects and arrays
- Serialize to a clear text representation for inclusion in prompts

**Why this matters**: Without a formal schema, you're relying on natural language descriptions of what you want, which are inherently ambiguous. "Extract the date" — what format? Is it required? What if there are multiple dates?

### Step 2: Build the Prompt Constructor

Create a function that takes a schema + source text and constructs a prompt that:
- Clearly presents the schema with field descriptions and types
- Instructs the model to extract ONLY from the provided text
- Asks for confidence scores per field
- Specifies the exact output format (JSON)

**What would go wrong without this**: Ad-hoc prompts lead to inconsistent output formats, hallucinated fields, and no way to programmatically validate results.

### Step 3: Implement JSON Parsing with Error Tolerance

LLMs often return JSON wrapped in markdown code fences, with trailing commas, or with other minor syntax issues. Build a parser that:
- Strips markdown formatting
- Handles common JSON errors
- Falls back to regex extraction for severely malformed output

### Step 4: Schema Validation

Validate the parsed JSON against your schema:
- Type checking (string, int, float, bool, list, dict)
- Required field presence
- Enum value validation
- Nested object recursive validation

### Step 5: Retry Logic with Error Feedback

When validation fails, construct a retry prompt that includes:
- The original text
- The failed output
- Specific validation errors
- Instructions to fix just the problematic fields

### Step 6: Confidence-Aware Result Assembly

Combine extracted values with their confidence scores into a result object that downstream code can filter on (e.g., "only use fields with confidence > 0.7").

## Learning Objectives

- Understand schema-guided prompting and why it's essential for reliable LLM output
- Build robust JSON parsing that handles real-world LLM output quirks
- Implement validation pipelines with retry logic
- Design confidence scoring for extraction quality assessment
- Handle nested and complex schemas in extraction tasks
- Learn patterns used in production document processing systems

## Going Deeper

- **Tool use / function calling**: Modern LLM APIs (Claude, GPT-4) support structured tool use where the model is constrained to output valid JSON matching a schema. This is more reliable than prompt-based extraction but less flexible.
- **Multi-pass extraction**: For complex documents, extract high-level structure first, then do focused extraction on each section.
- **Few-shot examples**: Including 2-3 examples of correct extractions dramatically improves accuracy, especially for domain-specific schemas.
- **Chunking strategies**: For documents longer than the context window, you need strategies for splitting text while preserving extraction context.
- **Evaluation**: Build test sets with known-correct extractions to measure precision/recall per field. This connects to Day 050's evaluation concepts.
- **Production systems**: Libraries like Instructor, Marvin, and LangChain's output parsers implement these patterns at scale.
