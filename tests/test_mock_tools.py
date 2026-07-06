"""Tests for the mock backend's deterministic tool-calling behaviour."""

from __future__ import annotations

from mizan.llm.base import GenerationParams
from mizan.llm.mock import MockBackend
from mizan.tools.extract import build_tool_prompt, parse_native, parse_prompt_output
from mizan.tools.registry import all_tools, is_registered


def test_native_tool_call_is_deterministic_and_registered() -> None:
    backend = MockBackend("mock")
    tools = [t.to_openai_tool() for t in all_tools()]
    first = backend.generate_tool_call("book a table for four", tools, GenerationParams())
    second = backend.generate_tool_call("book a table for four", tools, GenerationParams())
    assert first == second

    result = parse_native(first)
    # Either a no-call or one of the registered tools; never an invented name.
    assert result.chosen_tool is None or is_registered(result.chosen_tool)


def test_native_default_supports_tools_prompt_variant_does_not() -> None:
    assert MockBackend("m").supports_native_tools is True
    assert MockBackend("m", native_tools=False).supports_native_tools is False


def test_prompt_mode_output_parses_as_a_tool_decision() -> None:
    backend = MockBackend("mock", native_tools=False)
    prompt = build_tool_prompt(all_tools(), "what's 3 times 4?")
    raw = backend.generate(prompt, GenerationParams())
    result = parse_prompt_output(raw)
    assert result.parse_error is None
    assert result.chosen_tool is None or is_registered(result.chosen_tool)


def test_non_tool_prompt_returns_plain_digest() -> None:
    backend = MockBackend("mock")
    out = backend.generate("just a normal question", GenerationParams())
    assert out.startswith("[mock:mock]")
