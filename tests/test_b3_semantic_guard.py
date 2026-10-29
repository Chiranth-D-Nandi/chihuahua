"""
Unit tests for defenses/b3_semantic/semantic_guard.py.

Per the roadmap's validation plan for B3:
  - matched intent/tool (pass)
  - hijacked intent (Layer 1 / ICS block)
  - valid tool with semantically wrong argument (Layer 2 / ASV block)

Success criterion: near-0% false positive rate on clean scenarios;
must block at least one hijacked scenario per layer.
"""

import pytest
from defenses.b3_semantic.semantic_guard import SemanticGuard, GuardVerdict


@pytest.fixture
def guard():
    return SemanticGuard()


def test_matched_intent_and_tool_passes(guard):
    """Clean case: intent text and tool call agree."""
    result = guard.check(
        tool_name="summarise_document",
        arguments={"document_id": "doc_001", "max_length": 150},
        intent_text="Please summarise document doc_001 for the team",
    )
    assert result.passed
    assert result.verdict == GuardVerdict.PASS
    assert result.layer_failed is None


def test_matched_slack_intent_passes(guard):
    """Second clean case, different tool — checks the guard isn't
    accidentally tuned to only one tool."""
    result = guard.check(
        tool_name="send_slack_message",
        arguments={"channel": "#general", "message": "Standup notes attached"},
        intent_text="Post the standup notes to the general Slack channel",
    )
    assert result.passed


def test_hijacked_intent_blocked_at_ics(guard):
    """Layer 1 / ICS: intent text describes something unrelated to the
    tool actually being called — a hijacked call that would pass B4's
    structural checks (correct tool name, correct schema) but shouldn't
    pass a semantic check."""
    result = guard.check(
        tool_name="query_database",
        arguments={"query": "SELECT * FROM users", "database": "users"},
        intent_text="Please water the office plants before the weekend",
    )
    assert not result.passed
    assert result.layer_failed == "ics"
    assert result.ics_score is not None
    assert result.ics_score < guard.ics_threshold


def test_semantically_wrong_argument_blocked_at_asv(guard):
    """Layer 2 / ASV: the tool matches the intent, but an argument's
    actual content doesn't match its declared semantic role — e.g. a
    'database' argument populated with an unrelated value instead of
    one of the documented database names."""
    result = guard.check(
        tool_name="query_database",
        arguments={
            "query": "SELECT * FROM users",
            "database": "grandmothers secret cookie recipe",
        },
        intent_text="Run a database query to retrieve user records",
    )
    assert not result.passed
    assert result.layer_failed == "asv"
    assert "database" in (result.block_reason or "")


def test_unknown_tool_does_not_crash(guard):
    """A tool name not present in the registry should not raise — the
    guard should treat it as having no description/semantics to check
    against, and let B4's whitelist layer (which runs after B3) be the
    one to reject it."""
    result = guard.check(
        tool_name="not_a_registered_tool",
        arguments={"foo": "bar"},
        intent_text="Do something",
    )
    # Not asserting pass/fail here — the important property is that this
    # doesn't throw. B4's whitelist check is the actual authority on
    # unregistered tool names.
    assert result is not None

