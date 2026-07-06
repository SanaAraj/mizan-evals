"""Tests for tool-call extraction: prompt construction and robust parsing."""

from __future__ import annotations

from mizan.tools.extract import (
    build_tool_prompt,
    parse_native,
    parse_prompt_output,
    to_canonical_json,
)
from mizan.tools.registry import all_tools


def test_prompt_lists_every_tool_in_registry_order() -> None:
    prompt = build_tool_prompt(all_tools(), "What's the weather in Riyadh?")
    for tool in all_tools():
        assert tool.name in prompt
    # First-listed tool is the first registry tool.
    assert prompt.index(all_tools()[0].name) < prompt.index(all_tools()[1].name)
    assert prompt.rstrip().endswith("JSON:")


def test_prompt_is_stable_across_calls() -> None:
    a = build_tool_prompt(all_tools(), "hi")
    b = build_tool_prompt(all_tools(), "hi")
    assert a == b


def test_parse_native_roundtrip() -> None:
    raw = to_canonical_json("get_weather", {"location": "Riyadh"})
    result = parse_native(raw)
    assert result.chosen_tool == "get_weather"
    assert result.arguments == {"location": "Riyadh"}
    assert result.parse_error is None


def test_parse_native_no_call() -> None:
    result = parse_native(to_canonical_json(None, {}))
    assert result.chosen_tool is None
    assert result.parse_error is None
    assert not result.failed


def test_parse_native_invalid_json_is_parse_error() -> None:
    result = parse_native("not json at all")
    assert result.chosen_tool is None
    assert result.parse_error is not None
    assert result.failed


def test_parse_prompt_bare_json() -> None:
    result = parse_prompt_output('{"tool": "web_search", "arguments": {"query": "x"}}')
    assert result.chosen_tool == "web_search"
    assert result.arguments == {"query": "x"}


def test_parse_prompt_with_code_fence_and_prose() -> None:
    call = '{"tool": "calculator", "arguments": {"expression": "2+2"}}'
    raw = f"Sure! Here is the call:\n```json\n{call}\n```\nHope that helps."
    result = parse_prompt_output(raw)
    assert result.chosen_tool == "calculator"
    assert result.arguments == {"expression": "2+2"}


def test_parse_prompt_no_call() -> None:
    result = parse_prompt_output('The user is chatting. {"tool": null}')
    assert result.chosen_tool is None
    assert result.parse_error is None


def test_parse_prompt_alternate_key_spellings() -> None:
    result = parse_prompt_output('{"name": "get_directions", "parameters": {"origin": "A"}}')
    assert result.chosen_tool == "get_directions"
    assert result.arguments == {"origin": "A"}


def test_parse_prompt_no_json_is_parse_error() -> None:
    result = parse_prompt_output("I cannot help with that.")
    assert result.parse_error is not None
    assert result.failed


def test_parse_prompt_missing_tool_field_is_parse_error() -> None:
    result = parse_prompt_output('{"arguments": {"x": 1}}')
    assert result.parse_error is not None


def test_parse_prompt_nested_braces_in_arguments() -> None:
    raw = (
        '{"tool": "translate_text", "arguments": {"text": "a {b} c", "target_language": "English"}}'
    )
    result = parse_prompt_output(raw)
    assert result.chosen_tool == "translate_text"
    assert result.arguments["text"] == "a {b} c"


def test_parse_prompt_unregistered_tool_is_not_a_parse_error() -> None:
    # An invented tool name parses fine; the scorer flags it as hallucinated.
    result = parse_prompt_output('{"tool": "teleport", "arguments": {}}')
    assert result.chosen_tool == "teleport"
    assert result.parse_error is None
