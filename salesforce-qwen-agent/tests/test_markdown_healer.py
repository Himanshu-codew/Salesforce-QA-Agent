"""
Formatting-resilience tests.

Verifies the Markdown Healer (inside finalize_user_response) guarantees
that the user receives clean, valid GFM even when the LLM/synthesizer
produces malformed Markdown: broken tables, unmatched asterisks, no-space
bullets, and missing table separators.
"""

import pytest

from agent.agent import finalize_user_response


def _rows_of(md):
    """Return the table's row lines (lines with a pipe)."""
    return [l for l in md.split("\n") if "|" in l]


def test_well_formed_table_is_preserved_verbatim():
    md = "### Accounts Found\n\n| Name | Industry |\n| --- | --- |\n| Acme | Tech |\n\n**Total: 1**"
    out = finalize_user_response(md)
    assert "| Name | Industry |" in out
    assert "| --- | --- |" in out
    assert "| Acme | Tech |" in out
    # No leaked artifacts
    assert '"arguments"' not in out
    assert "<tool_call>" not in out


def test_broken_table_without_outer_pipes_is_rebuilt():
    # LLM dropped leading/trailing pipes.
    malformed = "Here are the accounts:\nName | Industry\nAcme Corp | Tech\n\nDone."
    out = finalize_user_response(malformed)
    rows = _rows_of(out)
    assert len(rows) >= 2
    # Strict GFM rows now have outer pipes.
    assert rows[0].startswith("|") and rows[0].endswith("|")
    # Separator inserted.
    assert any("| --- " in r for r in rows)
    # Surrounding natural language preserved.
    assert "Here are the accounts:" in out
    assert "Done." in out


def test_table_missing_separator_row_is_fixed():
    malformed = "| Name | Size |\n| Acme | Big |\n| Globex | Medium |"
    out = finalize_user_response(malformed)
    rows = _rows_of(out)
    assert any("| --- " in r for r in rows), "separator row should be added"
    assert "| Acme | Big |" in out
    assert "| Globex | Medium |" in out


def test_mangled_reference_table_is_rebuilt():
    # Simulates LLM rewriting [reference_table] with broken pipes.
    malformed = "### Accounts Found\nName|Industry\n---|---\nAcme|Tech\nGlobex|Cloud\n"
    out = finalize_user_response(malformed)
    assert "### Accounts Found" in out
    rows = _rows_of(out)
    # Every data row is strict GFM (outer pipes present).
    for r in rows:
        assert r.startswith("|") and r.endswith("|"), f"row not strict GFM: {r!r}"
    assert "| Acme | Tech |" in out
    assert "| Globex | Cloud |" in out


def test_unmatched_bold_asterisks_are_removed():
    out = finalize_user_response("The **accounts are ready")
    assert "**" not in out
    assert "The accounts are ready" in out


def test_stray_italic_asterisks_are_balanced():
    out = finalize_user_response("This is italic* and trailing*")
    assert out.count("*") % 2 == 0, "asterisk count must be even"


def test_bullets_without_spacing_are_normalized():
    out = finalize_user_response("*First item\n*Second item")
    assert "- First item" in out
    assert "- Second item" in out
    assert not any(l.startswith("*") for l in out.split("\n") if l.strip())


def test_plus_bullets_are_normalized():
    out = finalize_user_response("+First\n+ Second")
    assert "- First" in out
    assert "- Second" in out


def test_bold_heading_is_not_mistaken_for_bullet():
    out = finalize_user_response("**bold heading** stays")
    assert out == "**bold heading** stays"


def test_mixed_content_heals_everything():
    malformed = (
        "### Top Accounts\n\n"
        "* Acme — Tech\n"
        "* Globex — Cloud\n\n"
        "| Account | Value |\n"
        "| --- | --- |\n"
        "| Acme | 100 |\n\n"
        "All done**"
    )
    out = finalize_user_response(malformed)
    assert "- Acme — Tech" in out
    assert "- Globex — Cloud" in out
    assert "| Account | Value |" in out
    assert "| Acme | 100 |" in out
    assert not out.endswith("**")
    # Trail-of-asterisk cleaned.
    assert "All done" in out


def test_clean_output_passes_last_mile_guard():
    # Healing must never reintroduce something flagged as a raw tool artifact.
    out = finalize_user_response("Here are your accounts:\n\n| Name |\n| --- |\n| Acme |")
    assert "I processed your request" not in out  # not replaced by fallback
    assert "Acme" in out
