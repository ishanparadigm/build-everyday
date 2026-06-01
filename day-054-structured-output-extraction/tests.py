"""
Day 054: Structured Output Extraction — Test Suite

Run with: python3 -m pytest tests.py -v
Or:       python3 tests.py
"""

import unittest
from my_solution import (
    FieldResult,
    ExtractionResult,
    PERSON_SCHEMA,
    JOB_POSTING_SCHEMA,
    schema_to_prompt_text,
    build_extraction_prompt,
    build_retry_prompt,
    parse_llm_json,
    validate_field_type,
    validate_extraction,
    extract_with_rules,
    extract_structured_data,
)


class TestFieldResult(unittest.TestCase):
    """Tests for FieldResult confidence checking."""

    def test_reliable_high_confidence(self):
        fr = FieldResult(value="Alice", confidence=0.9)
        self.assertTrue(fr.is_reliable(threshold=0.7))

    def test_unreliable_low_confidence(self):
        fr = FieldResult(value="maybe", confidence=0.3)
        self.assertFalse(fr.is_reliable(threshold=0.7))

    def test_custom_threshold(self):
        fr = FieldResult(value="test", confidence=0.5)
        self.assertTrue(fr.is_reliable(threshold=0.4))
        self.assertFalse(fr.is_reliable(threshold=0.6))


class TestExtractionResult(unittest.TestCase):
    """Tests for ExtractionResult data assembly."""

    def test_to_dict_includes_all(self):
        result = ExtractionResult(schema_name="Test")
        result.fields["name"] = FieldResult("Alice", 0.9)
        result.fields["age"] = FieldResult(30, 0.5)
        d = result.to_dict(min_confidence=0.0)
        self.assertEqual(d["name"], "Alice")
        self.assertEqual(d["age"], 30)

    def test_to_dict_filters_by_confidence(self):
        result = ExtractionResult(schema_name="Test")
        result.fields["name"] = FieldResult("Alice", 0.9)
        result.fields["age"] = FieldResult(30, 0.3)
        d = result.to_dict(min_confidence=0.7)
        self.assertIn("name", d)
        self.assertNotIn("age", d)

    def test_summary_contains_schema_name(self):
        result = ExtractionResult(schema_name="Person")
        result.fields["name"] = FieldResult("Alice", 0.9)
        summary = result.summary()
        self.assertIn("Person", summary)


class TestJsonParsing(unittest.TestCase):
    """Tests for error-tolerant JSON parsing."""

    def test_clean_json(self):
        self.assertEqual(parse_llm_json('{"a": 1}'), {"a": 1})

    def test_markdown_fenced(self):
        raw = '```json\n{"name": "Bob"}\n```'
        self.assertEqual(parse_llm_json(raw), {"name": "Bob"})

    def test_trailing_comma(self):
        raw = '{"name": "Charlie", "age": 35,}'
        parsed = parse_llm_json(raw)
        self.assertEqual(parsed["name"], "Charlie")
        self.assertEqual(parsed["age"], 35)

    def test_surrounding_text(self):
        raw = 'Here is the result: {"x": 42} Hope that helps!'
        self.assertEqual(parse_llm_json(raw), {"x": 42})

    def test_invalid_json_raises(self):
        with self.assertRaises(ValueError):
            parse_llm_json("this is not json at all")


class TestValidation(unittest.TestCase):
    """Tests for schema validation."""

    def test_valid_string(self):
        is_valid, _ = validate_field_type("hello", "string")
        self.assertTrue(is_valid)

    def test_invalid_type(self):
        is_valid, error = validate_field_type("not a number", "integer")
        self.assertFalse(is_valid)
        self.assertIn("integer", error.lower())

    def test_none_always_valid(self):
        is_valid, _ = validate_field_type(None, "string")
        self.assertTrue(is_valid)

    def test_validate_required_field_missing(self):
        extracted = {"full_name": None}
        errors = validate_extraction(extracted, PERSON_SCHEMA)
        self.assertTrue(any("full_name" in e for e in errors))

    def test_validate_all_valid(self):
        extracted = {
            "full_name": {"value": "Alice Smith", "confidence": 0.9},
            "age": {"value": 30, "confidence": 0.8},
        }
        errors = validate_extraction(extracted, PERSON_SCHEMA)
        self.assertEqual(len(errors), 0)


class TestPromptConstruction(unittest.TestCase):
    """Tests for prompt building."""

    def test_schema_to_prompt_includes_fields(self):
        text = schema_to_prompt_text(PERSON_SCHEMA)
        self.assertIn("full_name", text)
        self.assertIn("REQUIRED", text)
        self.assertIn("age", text)

    def test_build_prompt_includes_text(self):
        prompt = build_extraction_prompt(PERSON_SCHEMA, "John is 25.")
        self.assertIn("John is 25", prompt)
        self.assertIn("Person", prompt)

    def test_retry_prompt_includes_errors(self):
        prompt = build_retry_prompt(
            PERSON_SCHEMA, "test text", '{"bad": "json"}', ["field 'name' missing"]
        )
        self.assertIn("field 'name' missing", prompt)
        self.assertIn("bad", prompt)


class TestExtraction(unittest.TestCase):
    """Tests for the extraction pipeline."""

    def test_person_name_extracted(self):
        text = "Sarah Chen is a machine learning engineer."
        result = extract_structured_data(text, PERSON_SCHEMA)
        name_field = result.fields.get("full_name")
        self.assertIsNotNone(name_field)
        self.assertIn("Sarah", name_field.value)

    def test_email_extracted(self):
        text = "Contact me at alice@example.com for details."
        result = extract_structured_data(text, PERSON_SCHEMA)
        email_field = result.fields.get("email")
        self.assertIsNotNone(email_field)
        self.assertEqual(email_field.value, "alice@example.com")

    def test_age_extracted(self):
        text = "Bob is 35 years old and lives in Denver."
        result = extract_structured_data(text, PERSON_SCHEMA)
        age_field = result.fields.get("age")
        self.assertIsNotNone(age_field)
        self.assertEqual(age_field.value, 35)

    def test_missing_fields_have_zero_confidence(self):
        text = "Just a random sentence with no useful info."
        result = extract_structured_data(text, PERSON_SCHEMA)
        for name, fr in result.fields.items():
            if fr.value is None:
                self.assertEqual(fr.confidence, 0.0)

    def test_job_posting_salary(self):
        text = "We offer $120k - $180k for this role at Acme Corp. Senior Developer position."
        result = extract_structured_data(text, JOB_POSTING_SCHEMA)
        sal_min = result.fields.get("salary_min")
        sal_max = result.fields.get("salary_max")
        if sal_min and sal_min.value is not None:
            self.assertEqual(sal_min.value, 120000)
        if sal_max and sal_max.value is not None:
            self.assertEqual(sal_max.value, 180000)


if __name__ == "__main__":
    unittest.main()
