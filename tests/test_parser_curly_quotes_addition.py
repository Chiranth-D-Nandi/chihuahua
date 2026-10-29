# Add this test case to the existing tests/test_parser.py.
# Covers the Session 11 fix: curly/smart quotes inside a string value
# that previously desynced the raw-newline state machine and produced
# an "unterminated_string_raw_newline" failure.

def test_parse_handles_curly_quotes_in_string_value():
    parser = ResponseParser()

    # Smart/curly quotes (U+2018/2019 single, U+201C/201D double) inside
    # an argument value, as LLaVA occasionally emits them verbatim.
    raw = (
        '{"tool": "send_slack_message", "arguments": '
        '{"channel": "#general", "message": '
        '\u201cPlease review the \u2018Q3 summary\u2019 doc before EOD\u201d}}'
    )

    result = parser.parse(raw)

    assert result is not None, f"Expected successful parse, got last_error={parser.last_error}"
    assert result.tool_name == "send_slack_message"
    assert "Q3 summary" in result.arguments["message"]


def test_parse_handles_curly_quotes_before_raw_newline():
    parser = ResponseParser()

    # Regression case: a curly quote followed later by a raw newline
    # inside what should be the same string value. Before the Session 11
    # fix, the curly quote wasn't recognized as a string boundary, so the
    # newline-escaping pass lost track of "inside string" state and this
    # raised json.JSONDecodeError: Unterminated string starting at...
    raw = (
        '{"tool": "send_slack_message", "arguments": '
        '{"channel": "#general", "message": "Status update \u2014 '
        'see \u201cattached notes\u201d\nsecond line of the message"}}'
    )

    result = parser.parse(raw)

    assert result is not None, f"Expected successful parse, got last_error={parser.last_error}"
    assert result.tool_name == "send_slack_message"

