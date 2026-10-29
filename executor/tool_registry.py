"""
FISSION — executor/tool_registry.py
=====================================
Loads tool definitions from config and provides
the mock tool implementations.

CHIRANTH — executor side.
Tools accept **kwargs to handle LLaVA argument name variance.
"""

import logging
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.config import cfg

log = logging.getLogger("fission.tool_registry")


# ══════════════════════════════════════════════════════════════
#  Mock tool implementations
# ══════════════════════════════════════════════════════════════

def _summarise_document(**kwargs) -> str:
    doc_id     = (kwargs.get("document_id")
                  or kwargs.get("doc_id")
                  or kwargs.get("id", "unknown"))
    max_length = (kwargs.get("max_length")
                  or kwargs.get("max_words", 150))
    return (
        f"[MOCK] Document '{doc_id}' summarised in {max_length} words. "
        f"Key findings: pipeline operational, metrics within bounds."
    )


def _send_slack_message(**kwargs) -> str:
    channel = kwargs.get("channel", "#general")
    message = (kwargs.get("message")
               or kwargs.get("text")
               or kwargs.get("msg", ""))
    return f"[MOCK] Message posted to {channel}: '{str(message)[:80]}'"


def _query_database(**kwargs) -> str:
    query    = kwargs.get("query", "")
    database = (kwargs.get("database")
                or kwargs.get("db", "default"))
    return f"[MOCK] Query on '{database}': returned 3 rows for '{str(query)[:60]}'"


# ── Tool function dispatch map ───────────────────────────────────────────

TOOL_FUNCTIONS = {
    "summarise_document": _summarise_document,
    "send_slack_message":  _send_slack_message,
    "query_database":      _query_database,
}


def get_tool_fn(tool_name: str):
    """Return the tool function for a given name, or None if not found."""
    return TOOL_FUNCTIONS.get(tool_name)


def get_schema(tool_name: str) -> dict:
    """Return the argument schema for a tool."""
    tool = cfg.mcp.registered_tools.get(tool_name, {})
    return tool.get("schema", {})


def get_arg_semantics(tool_name: str) -> dict:
    """Return the argument semantics for B3 Layer 2 (ASV)."""
    tool = cfg.mcp.registered_tools.get(tool_name, {})
    return tool.get("arg_semantics", {})

