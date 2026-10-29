"""
FISSION — tests/test_schema_utils.py
======================================
Unit tests for shared/schema_utils.py.

Why this module matters more than its size suggests: it sits directly in
the HMAC signing path (parsed tool call -> canonicalise -> sign -> verify).
A silent miscoercion here doesn't just produce a wrong value — it produces
a wrong value that gets cryptographically signed as if it were correct,
so the executor has no way to detect the mistake downstream. This file
had zero test coverage before this session; writing it directly found a
real bug (see TestBoolIsNotInt below) that inspection alone had missed
across every prior session's read-throughs of this file.

Covers:
  - coerce_value: identity pass-through for already-correct types
  - coerce_value: str/int/float/bool coercion from compatible input
  - coerce_value: bool must never silently pass as int/float (the bug)
  - coerce_value: rejection of incompatible/malformed input
  - coerce_value: numeric string edge cases (commas, whitespace, signs)
  - normalise_arguments: schema-driven coercion, unknown-tool rejection,
    unknown-field passthrough (by design — B4 TCIV decides on those)

Run with:
    python -m pytest tests/test_schema_utils.py -v
or:
    python tests/test_schema_utils.py
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shared.schema_utils import (
    coerce_value,
    normalise_arguments,
    ArgumentNormalisationError,
)


class TestCoerceValueIdentity(unittest.TestCase):
    """Already-correct-type values should pass through unchanged."""

    def test_str_identity(self):
        self.assertEqual(coerce_value("hello", str), "hello")

    def test_int_identity(self):
        self.assertEqual(coerce_value(42, int), 42)

    def test_float_identity(self):
        self.assertEqual(coerce_value(3.14, float), 3.14)

    def test_bool_identity_true(self):
        self.assertIs(coerce_value(True, bool), True)

    def test_bool_identity_false(self):
        self.assertIs(coerce_value(False, bool), False)


class TestBoolIsNotInt(unittest.TestCase):
    """
    The critical edge case: bool is a subclass of int in Python, so
    isinstance(True, int) is True. A naive coercer that checks
    isinstance(value, expected_type) first will silently accept a bool
    where an int was expected — found as a real bug during this test
    file's development, not from reading the code.
    """

    def test_true_rejected_as_int(self):
        with self.assertRaises(ArgumentNormalisationError):
            coerce_value(True, int)

    def test_false_rejected_as_int(self):
        with self.assertRaises(ArgumentNormalisationError):
            coerce_value(False, int)

    def test_true_rejected_as_float(self):
        with self.assertRaises(ArgumentNormalisationError):
            coerce_value(True, float)

    def test_false_rejected_as_float(self):
        with self.assertRaises(ArgumentNormalisationError):
            coerce_value(False, float)

    def test_int_one_rejected_as_bool(self):
        # The inverse direction: a real int should not silently become
        # a bool either, even though bool(1) == True in plain Python.
        with self.assertRaises(ArgumentNormalisationError):
            coerce_value(1, bool)

    def test_int_zero_rejected_as_bool(self):
        with self.assertRaises(ArgumentNormalisationError):
            coerce_value(0, bool)


class TestCoerceValueToStr(unittest.TestCase):

    def test_int_to_str(self):
        self.assertEqual(coerce_value(42, str), "42")

    def test_float_to_str(self):
        self.assertEqual(coerce_value(3.14, str), "3.14")

    def test_bool_to_str(self):
        # str is not one of the numeric types the bool-guard blocks —
        # str(True) == 'True' is legitimate, expected Python behaviour.
        self.assertEqual(coerce_value(True, str), "True")


class TestCoerceValueToInt(unittest.TestCase):

    def test_numeric_string(self):
        self.assertEqual(coerce_value("150", int), 150)

    def test_numeric_string_with_commas(self):
        self.assertEqual(coerce_value("1,500", int), 1500)

    def test_numeric_string_with_whitespace(self):
        self.assertEqual(coerce_value("  42  ", int), 42)

    def test_negative_numeric_string(self):
        self.assertEqual(coerce_value("-7", int), -7)

    def test_explicit_positive_sign_string(self):
        self.assertEqual(coerce_value("+7", int), 7)

    def test_integral_float(self):
        self.assertEqual(coerce_value(150.0, int), 150)

    def test_non_integral_float_rejected(self):
        with self.assertRaises(ArgumentNormalisationError):
            coerce_value(150.5, int)

    def test_non_numeric_string_rejected(self):
        with self.assertRaises(ArgumentNormalisationError):
            coerce_value("not a number", int)

    def test_empty_string_rejected(self):
        with self.assertRaises(ArgumentNormalisationError):
            coerce_value("", int)

    def test_float_looking_string_rejected_as_int(self):
        # "3.14" should not silently truncate to 3 — that would be a
        # silent-data-loss coercion in a security-sensitive path.
        with self.assertRaises(ArgumentNormalisationError):
            coerce_value("3.14", int)

    def test_none_rejected(self):
        with self.assertRaises(ArgumentNormalisationError):
            coerce_value(None, int)

    def test_list_rejected(self):
        with self.assertRaises(ArgumentNormalisationError):
            coerce_value([1, 2, 3], int)


class TestCoerceValueToFloat(unittest.TestCase):

    def test_int_to_float(self):
        self.assertEqual(coerce_value(5, float), 5.0)

    def test_numeric_string(self):
        self.assertEqual(coerce_value("3.14", float), 3.14)

    def test_integral_looking_string(self):
        self.assertEqual(coerce_value("5", float), 5.0)

    def test_numeric_string_with_commas(self):
        self.assertEqual(coerce_value("1,500.50", float), 1500.50)

    def test_leading_decimal_string(self):
        self.assertEqual(coerce_value(".5", float), 0.5)

    def test_non_numeric_string_rejected(self):
        with self.assertRaises(ArgumentNormalisationError):
            coerce_value("abc", float)

    def test_none_rejected(self):
        with self.assertRaises(ArgumentNormalisationError):
            coerce_value(None, float)


class TestCoerceValueToBool(unittest.TestCase):

    def test_true_variants(self):
        for v in ("true", "True", "TRUE", "1", "yes", "y", "  true  "):
            with self.subTest(v=v):
                self.assertIs(coerce_value(v, bool), True)

    def test_false_variants(self):
        for v in ("false", "False", "FALSE", "0", "no", "n", "  false  "):
            with self.subTest(v=v):
                self.assertIs(coerce_value(v, bool), False)

    def test_ambiguous_string_rejected(self):
        with self.assertRaises(ArgumentNormalisationError):
            coerce_value("maybe", bool)

    def test_int_rejected(self):
        with self.assertRaises(ArgumentNormalisationError):
            coerce_value(1, bool)

    def test_none_rejected(self):
        with self.assertRaises(ArgumentNormalisationError):
            coerce_value(None, bool)


class TestNormaliseArguments(unittest.TestCase):
    """Schema-driven coercion against real registered_tools schemas."""

    def test_summarise_document_string_max_length_coerced(self):
        result = normalise_arguments(
            "summarise_document",
            {"document_id": "doc_001", "max_length": "150"},
        )
        self.assertEqual(result, {"document_id": "doc_001", "max_length": 150})
        self.assertIsInstance(result["max_length"], int)

    def test_summarise_document_bool_max_length_rejected(self):
        # The exact real-world shape of the bug this file exists to catch:
        # a VLM or adversarial input supplying {"max_length": true} must
        # not silently sign a boolean into a field the executor expects
        # to be a word-count integer.
        with self.assertRaises(ArgumentNormalisationError):
            normalise_arguments(
                "summarise_document",
                {"document_id": "doc_001", "max_length": True},
            )

    def test_unknown_tool_rejected(self):
        with self.assertRaises(ArgumentNormalisationError):
            normalise_arguments("delete_everything", {"foo": "bar"})

    def test_unknown_field_passed_through_unchanged(self):
        # By design (see module docstring): unknown fields are NOT
        # coerced or dropped here — B4 TCIV decides whether to warn or
        # block on them. normalise_arguments must not silently swallow
        # or "fix" a field it doesn't recognise.
        result = normalise_arguments(
            "summarise_document",
            {"document_id": "doc_001", "max_length": 100, "unexpected_field": 12345},
        )
        self.assertIn("unexpected_field", result)
        self.assertEqual(result["unexpected_field"], 12345)

    def test_send_slack_message_string_fields_untouched(self):
        args = {"channel": "#eng", "message": "hello"}
        result = normalise_arguments("send_slack_message", args)
        self.assertEqual(result, args)

    def test_query_database_string_fields_untouched(self):
        args = {"query": "SELECT * FROM logs", "database": "logs"}
        result = normalise_arguments("query_database", args)
        self.assertEqual(result, args)

    def test_empty_arguments_dict(self):
        result = normalise_arguments("summarise_document", {})
        self.assertEqual(result, {})

    def test_result_is_new_dict_not_mutated_input(self):
        original = {"document_id": "doc_001", "max_length": "150"}
        original_copy = dict(original)
        normalise_arguments("summarise_document", original)
        self.assertEqual(original, original_copy, "input dict must not be mutated in place")


if __name__ == "__main__":
    unittest.main(verbosity=2)

