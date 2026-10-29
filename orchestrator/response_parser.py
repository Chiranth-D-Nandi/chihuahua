"""
FISSION — orchestrator/response_parser.py
==========================================
Parses VLM text output into structured tool calls.

Handles all known LLaVA formatting quirks:
  - Markdown code fences (```json ... ```)
  - Underscore escaping (\\_ → _)
  - Markdown escape artifacts (\\* \\_ \\` \\[ \\] → bare chars)
  - Valid JSON escapes left untouched (\\n \\t \\\\ \\")
  - Array-wrapped responses
  - tool_calls key wrapping
  - Multiple JSON objects in one response (takes first)
  - Trailing garbage after valid JSON

ABHYUDAYA — response parsing.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("fission.orchestrator.parser")


@dataclass
class ParsedToolCall:
    tool_name: Optional[str]
    arguments: dict = field(default_factory=dict)
    raw_output: str = ""
    parse_error: Optional[str] = None

    @property
    def is_valid(self) -> bool:
        return self.tool_name is not None and self.tool_name != "none"


class ResponseParser:
    """
    Parses VLM text output into ParsedToolCall objects.
    """

    # Valid tool names (from config)
    _VALID_TOOLS: frozenset = frozenset([
        "summarise_document", "send_slack_message", "query_database", "none"
    ])

    def __init__(self):
        self.last_error: Optional[str] = None

    def parse(self, raw: Optional[str]) -> Optional[ParsedToolCall]:
        """
        Parse VLM raw output into a ParsedToolCall.

        Returns None if parsing fails entirely. Sets self.last_error on failure.
        """
        if not raw or not raw.strip():
            self.last_error = "empty_input"
            return None

        self.last_error = None
        text = raw.strip()

        # ── Step 1: strip markdown code fences ─────────────────────────
        text = re.sub(r"^```json\s*\n?", "", text)
        text = re.sub(r"^```\s*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
        text = text.strip()

        # ── Step 2: fix escape artifacts before JSON parse ────────────
        # LLaVA emits markdown escapes inside string values
        text = re.sub(r'\\([_*`\[\]])', r'\1', text)

        # ── Step 3: fix raw newlines inside string values ──────────────
        # Strategy: walk the string and identify "inside string" segments
        # (between unescaped " quotes), replacing bare newlines within them
        # with \n so the JSON decoder can parse it.
        in_string = False
        result = []
        i = 0
        while i < len(text):
            ch = text[i]
            if ch == '"' and (i == 0 or text[i - 1] != '\\'):
                in_string = not in_string
                result.append(ch)
            elif ch == '\n' and in_string:
                result.append('\\n')
            elif ch == '\r':
                pass   # strip CR
            else:
                result.append(ch)
            i += 1
        text = ''.join(result)

        # ── Step 4: try to extract JSON ───────────────────────────────
        # Try whole-string parse first
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            # Try extracting first { ... } block
            try:
                obj = json.loads(self._extract_first_json_object(text))
            except Exception:
                self.last_error = "json_decode_error"
                logger.warning(f"Parse failed: {text[:100]!r}")
                return ParsedToolCall(
                    tool_name=None,
                    raw_output=raw,
                    parse_error=self.last_error,
                )

        # ── Step 5: unwrap container types ────────────────────────────
        if isinstance(obj, list):
            if len(obj) == 0:
                self.last_error = "empty_array"
                return None
            obj = obj[0]

        if isinstance(obj, dict) and "tool_calls" in obj:
            calls = obj["tool_calls"]
            if isinstance(calls, list) and len(calls) > 0:
                obj = calls[0]

        # ── Step 6: extract tool name ─────────────────────────────────
        tool_name = None
        for key in ("tool", "tool_name", "function", "name"):
            if key in obj:
                tool_name = str(obj[key]).strip().lower()
                break

        if not tool_name:
            self.last_error = "missing_tool_key"
            return ParsedToolCall(tool_name=None, raw_output=raw, parse_error=self.last_error)

        # ── Step 7: extract arguments ─────────────────────────────────
        arguments = {}
        if "arguments" in obj and isinstance(obj["arguments"], dict):
            arguments = {k: v for k, v in obj["arguments"].items()}
        elif len(obj) > 1:
            # Flat key-value style
            for k, v in obj.items():
                if k not in ("tool", "tool_name", "function", "name"):
                    arguments[str(k)] = v

        return ParsedToolCall(
            tool_name=tool_name,
            arguments=arguments,
            raw_output=raw,
        )

    @staticmethod
    def _extract_first_json_object(text: str) -> str:
        """Extract the first complete {...} JSON object from text."""
        depth = 0
        start = -1
        in_string = False
        for i, ch in enumerate(text):
            if ch == '"' and (i == 0 or text[i - 1] != '\\'):
                in_string = not in_string
            if not in_string:
                if ch == '{':
                    if depth == 0:
                        start = i
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0 and start != -1:
                        return text[start:i + 1]
        raise ValueError("No complete JSON object found")

