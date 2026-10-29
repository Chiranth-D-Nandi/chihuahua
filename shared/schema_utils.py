"""
FISSION — shared/schema_utils.py
================================
Shared helpers for tool argument canonicalisation.

Why this exists:
    VLMs often emit structurally correct JSON with values encoded as
    strings, e.g. {"max_length": "150"} instead of {"max_length": 150}.
    B4 TCIV performs strict type validation, so the orchestrator should
    canonicalise arguments *before* signing and dispatching them.

This keeps the HMAC contract stable:
    parsed tool call -> canonical arguments -> HMAC sign -> executor verify
"""

from __future__ import annotations

import re
from typing import Any

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.config import cfg


class ArgumentNormalisationError(ValueError):
    """Raised when a tool argument cannot be coerced into its schema type."""


_INT_RE = re.compile(r"^[+-]?\d+$")
_FLOAT_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$")


def coerce_value(value: Any, expected_type: type) -> Any:
    """Coerce a raw argument value into the configured schema type."""
    # Bool must be checked before the generic isinstance() shortcut below,
    # for numeric expected_types specifically. bool is a subclass of int
    # in Python, so isinstance(True, int) is True: without this guard
    # first, `coerce_value(True, int)` would hit the generic
    # `isinstance(value, expected_type): return value` line and silently
    # return True unchanged instead of raising. The explicit "boolean
    # cannot be coerced to int" check further down existed already, but
    # was unreachable dead code for expected_type=int specifically — this
    # was found via testing, not visible from reading the function alone.
    #
    # Scoped to {int, float} only, not a blanket "isinstance(value, bool)"
    # check — str(True) == "True" is legitimate Python behaviour and an
    # earlier version of this fix incorrectly blocked bool-to-str too,
    # caught by tests/test_schema_utils.py::TestCoerceValueToStr.
    if isinstance(value, bool) and expected_type in (int, float):
        raise ArgumentNormalisationError(
            f"boolean cannot be coerced to {expected_type.__name__}"
        )

    if isinstance(value, expected_type):
        return value

    if expected_type is str:
        return str(value)

    if expected_type is int:
        if isinstance(value, float):
            if value.is_integer():
                return int(value)
            raise ArgumentNormalisationError(f"non-integral float '{value}' cannot be coerced to int")
        if isinstance(value, str):
            cleaned = value.strip().replace(",", "")
            if _INT_RE.fullmatch(cleaned):
                return int(cleaned)
        raise ArgumentNormalisationError(f"value '{value}' cannot be coerced to int")

    if expected_type is float:
        if isinstance(value, int):
            return float(value)
        if isinstance(value, str):
            cleaned = value.strip().replace(",", "")
            if _FLOAT_RE.fullmatch(cleaned):
                return float(cleaned)
        raise ArgumentNormalisationError(f"value '{value}' cannot be coerced to float")

    if expected_type is bool:
        if isinstance(value, str):
            cleaned = value.strip().lower()
            if cleaned in {"true", "1", "yes", "y"}:
                return True
            if cleaned in {"false", "0", "no", "n"}:
                return False
        raise ArgumentNormalisationError(f"value '{value}' cannot be coerced to bool")

    if not isinstance(value, expected_type):
        raise ArgumentNormalisationError(
            f"value '{value}' is not of expected type {expected_type.__name__}"
        )
    return value


def normalise_arguments(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """
    Canonicalise arguments for a registered tool.

    Known schema fields are coerced into their configured Python types.
    Unknown fields are passed through unchanged so B4 TCIV can decide
    whether to warn/block on them.
    """
    tool_cfg = cfg.mcp.registered_tools.get(tool_name)
    if tool_cfg is None:
        raise ArgumentNormalisationError(f"unknown tool '{tool_name}'")

    schema = tool_cfg.get("schema", {})
    normalised: dict[str, Any] = {}

    for key, value in arguments.items():
        expected_type = schema.get(key)
        if expected_type is None:
            normalised[key] = value
        else:
            normalised[key] = coerce_value(value, expected_type)

    return normalised

