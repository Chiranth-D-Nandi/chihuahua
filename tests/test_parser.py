"""
FISSION — tests/test_parser.py
================================
Unit tests for orchestrator/response_parser.py.

Covers every LLaVA formatting quirk handled by ResponseParser:
  - Happy path (clean JSON)
  - Markdown code fences  (```json ... ``` and ``` ... ```)
  - Underscore escaping   \_  →  _
  - Markdown escape artifacts  \* \_ \` \[ \]  →  bare characters
  - Valid JSON escapes left untouched  \n \t \r \\ \"
  - Array-wrapped response  [{ ... }]
  - tool_calls key wrapping  {"tool_calls": [{ ... }]}
  - Multiple JSON objects in one response (take first)
  - Trailing garbage after valid JSON
  - Missing 'tool' key  →  None
  - Empty arguments dict  (valid)
  - Completely unparseable output  →  None
  - Empty string input  →  None
  - Whitespace-only input  →  None

Run with:
    python -m pytest tests/test_parser.py -v
or:
    python tests/test_parser.py
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from orchestrator.response_parser import ResponseParser, ParsedToolCall


class TestResponseParserHappyPath(unittest.TestCase):

    def setUp(self):
        self.parser = ResponseParser()

    def test_clean_json(self):
        """Minimal valid tool call — no quirks."""
        raw = '{"tool": "summarise_document", "arguments": {"document_id": "doc-42", "max_length": 200}}'
        result = self.parser.parse(raw)
        self.assertIsNotNone(result)
        self.assertEqual(result.tool_name, "summarise_document")
        self.assertEqual(result.arguments["document_id"], "doc-42")
        self.assertEqual(result.arguments["max_length"], 200)

    def test_raw_output_preserved(self):
        """raw_output must be the original unmodified string."""
        raw = '{"tool": "send_slack_message", "arguments": {"channel": "#eng", "message": "hello"}}'
        result = self.parser.parse(raw)
        self.assertEqual(result.raw_output, raw)

    def test_leading_trailing_whitespace(self):
        """Whitespace around valid JSON is stripped cleanly."""
        raw = '  \n  {"tool": "query_database", "arguments": {"query": "SELECT 1", "database": "logs"}}  \n  '
        result = self.parser.parse(raw)
        self.assertIsNotNone(result)
        self.assertEqual(result.tool_name, "query_database")


class TestResponseParserCodeFences(unittest.TestCase):

    def setUp(self):
        self.parser = ResponseParser()

    def test_json_code_fence(self):
        """```json ... ``` wrapper is stripped."""
        raw = '```json\n{"tool": "summarise_document", "arguments": {"document_id": "x", "max_length": 100}}\n```'
        result = self.parser.parse(raw)
        self.assertIsNotNone(result)
        self.assertEqual(result.tool_name, "summarise_document")

    def test_plain_code_fence(self):
        """``` ... ``` wrapper (no language tag) is stripped."""
        raw = '```\n{"tool": "summarise_document", "arguments": {"document_id": "x", "max_length": 100}}\n```'
        result = self.parser.parse(raw)
        self.assertIsNotNone(result)
        self.assertEqual(result.tool_name, "summarise_document")

    def test_code_fence_with_extra_whitespace(self):
        """Code fence with blank lines inside still parses."""
        raw = '```json\n\n{"tool": "send_slack_message", "arguments": {"channel": "#ops", "message": "hi"}}\n\n```'
        result = self.parser.parse(raw)
        self.assertIsNotNone(result)
        self.assertEqual(result.tool_name, "send_slack_message")


class TestResponseParserEscapeArtifacts(unittest.TestCase):

    def setUp(self):
        self.parser = ResponseParser()

    def test_underscore_escape_in_value(self):
        r"""LLaVA emits \_ inside string values — must become _."""
        raw = '{"tool": "summarise\\_document", "arguments": {"document\\_id": "doc\\_99"}}'
        result = self.parser.parse(raw)
        self.assertIsNotNone(result)
        self.assertEqual(result.tool_name, "summarise_document")
        self.assertIn("document_id", result.arguments)

    def test_asterisk_escape_in_sql_value(self):
        r"""LLaVA emits \* in SQL strings — must become * before JSON parse."""
        raw = '{"tool": "query_database", "arguments": {"query": "SELECT \\* FROM logs", "database": "logs"}}'
        result = self.parser.parse(raw)
        self.assertIsNotNone(result)
        self.assertEqual(result.arguments["query"], "SELECT * FROM logs")

    def test_backtick_escape_in_value(self):
        r"""LLaVA emits \` — must become `."""
        raw = r'{"tool": "send_slack_message", "arguments": {"channel": "#eng", "message": "run \`ls\`"}}'
        result = self.parser.parse(raw)
        self.assertIsNotNone(result)
        self.assertEqual(result.arguments["message"], "run `ls`")

    def test_bracket_escapes_in_value(self):
        r"""LLaVA emits \[ and \] — must become [ and ]."""
        raw = r'{"tool": "send_slack_message", "arguments": {"channel": "#eng", "message": "\[note\]"}}'
        result = self.parser.parse(raw)
        self.assertIsNotNone(result)
        self.assertEqual(result.arguments["message"], "[note]")

    def test_valid_json_escapes_untouched(self):
        r"""Real JSON escapes \n \t \r \\ \" must survive normalisation."""
        raw = '{"tool": "send_slack_message", "arguments": {"channel": "#eng", "message": "line1\\nline2\\ttabbed"}}'
        result = self.parser.parse(raw)
        self.assertIsNotNone(result)
        # \n and \t should be interpreted as newline and tab by JSON decoder
        self.assertIn("\n", result.arguments["message"])
        self.assertIn("\t", result.arguments["message"])

    def test_backslash_escape_untouched(self):
        r"""Double backslash \\ must remain a single backslash after parse."""
        raw = '{"tool": "send_slack_message", "arguments": {"channel": "#eng", "message": "C:\\\\Users\\\\file"}}'
        result = self.parser.parse(raw)
        self.assertIsNotNone(result)
        self.assertIn("C:\\", result.arguments["message"])


class TestResponseParserWrapping(unittest.TestCase):

    def setUp(self):
        self.parser = ResponseParser()

    def test_array_wrapped(self):
        """[{ ... }] — unwraps to first element."""
        raw = '[{"tool": "summarise_document", "arguments": {"document_id": "d1", "max_length": 50}}]'
        result = self.parser.parse(raw)
        self.assertIsNotNone(result)
        self.assertEqual(result.tool_name, "summarise_document")

    def test_tool_calls_key(self):
        """{"tool_calls": [{ ... }]} — unwraps to first call."""
        raw = '{"tool_calls": [{"tool": "send_slack_message", "arguments": {"channel": "#ops", "message": "ok"}}]}'
        result = self.parser.parse(raw)
        self.assertIsNotNone(result)
        self.assertEqual(result.tool_name, "send_slack_message")

    def test_tool_calls_key_empty_list(self):
        """{"tool_calls": []} — no calls, returns None."""
        raw = '{"tool_calls": []}'
        result = self.parser.parse(raw)
        self.assertIsNone(result)

    def test_multiple_json_objects(self):
        """Two JSON objects in one string — takes first via raw_decode."""
        raw = '{"tool": "summarise_document", "arguments": {"document_id": "first", "max_length": 10}} {"tool": "query_database", "arguments": {"query": "SELECT 1", "database": "logs"}}'
        result = self.parser.parse(raw)
        self.assertIsNotNone(result)
        self.assertEqual(result.tool_name, "summarise_document")
        self.assertEqual(result.arguments["document_id"], "first")

    def test_trailing_garbage_after_json(self):
        """Valid JSON followed by non-JSON text — raw_decode takes the JSON part."""
        raw = '{"tool": "query_database", "arguments": {"query": "SELECT 1", "database": "logs"}} Sure, I ran that query for you!'
        result = self.parser.parse(raw)
        self.assertIsNotNone(result)
        self.assertEqual(result.tool_name, "query_database")


class TestResponseParserFailureCases(unittest.TestCase):

    def setUp(self):
        self.parser = ResponseParser()

    def test_missing_tool_key(self):
        """Valid JSON but no 'tool' key — returns None."""
        raw = '{"arguments": {"document_id": "x", "max_length": 10}}'
        result = self.parser.parse(raw)
        self.assertIsNone(result)

    def test_empty_tool_name(self):
        """tool key present but empty string — returns None."""
        raw = '{"tool": "", "arguments": {"document_id": "x"}}'
        result = self.parser.parse(raw)
        self.assertIsNone(result)

    def test_completely_unparseable(self):
        """Plain English response with no JSON — returns None."""
        raw = "I will summarise document 42 with a maximum of 200 words."
        result = self.parser.parse(raw)
        self.assertIsNone(result)

    def test_empty_string(self):
        """Empty string input — returns None."""
        result = self.parser.parse("")
        self.assertIsNone(result)

    def test_whitespace_only(self):
        """Whitespace-only input — returns None."""
        result = self.parser.parse("   \n\t  ")
        self.assertIsNone(result)

    def test_json_array_empty(self):
        """Empty array [] — returns None."""
        raw = "[]"
        result = self.parser.parse(raw)
        self.assertIsNone(result)

    def test_json_scalar(self):
        """JSON scalar (not an object) — returns None."""
        raw = '"just a string"'
        result = self.parser.parse(raw)
        self.assertIsNone(result)

    def test_truncated_json(self):
        """Truncated/malformed JSON — returns None."""
        raw = '{"tool": "summarise_document", "arguments": {"document_id":'
        result = self.parser.parse(raw)
        self.assertIsNone(result)


class TestResponseParserEdgeCases(unittest.TestCase):

    def setUp(self):
        self.parser = ResponseParser()

    def test_empty_arguments_dict(self):
        """arguments: {} is valid — should parse successfully."""
        raw = '{"tool": "summarise_document", "arguments": {}}'
        result = self.parser.parse(raw)
        self.assertIsNotNone(result)
        self.assertEqual(result.tool_name, "summarise_document")
        self.assertEqual(result.arguments, {})

    def test_missing_arguments_key(self):
        """No 'arguments' key at all — arguments defaults to {}."""
        raw = '{"tool": "summarise_document"}'
        result = self.parser.parse(raw)
        self.assertIsNotNone(result)
        self.assertEqual(result.tool_name, "summarise_document")
        self.assertEqual(result.arguments, {})

    def test_combined_quirks(self):
        """Code fence + underscore escape + asterisk escape all at once."""
        raw = '```json\n{"tool": "query\\_database", "arguments": {"query": "SELECT \\* FROM logs", "database": "logs"}}\n```'
        result = self.parser.parse(raw)
        self.assertIsNotNone(result)
        self.assertEqual(result.tool_name, "query_database")
        self.assertEqual(result.arguments["query"], "SELECT * FROM logs")

    def test_code_fence_plus_array_wrap(self):
        """Code fence wrapping an array-wrapped tool call."""
        raw = '```json\n[{"tool": "send_slack_message", "arguments": {"channel": "#eng", "message": "deploy done"}}]\n```'
        result = self.parser.parse(raw)
        self.assertIsNotNone(result)
        self.assertEqual(result.tool_name, "send_slack_message")


if __name__ == "__main__":
    unittest.main(verbosity=2)

