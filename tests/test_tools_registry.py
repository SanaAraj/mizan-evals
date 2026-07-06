"""Tests for the tool registry and the JSON-Schema argument validator."""

from __future__ import annotations

import pytest

from mizan.tools import (
    TOOLS,
    all_tools,
    get_tool,
    is_registered,
    tool_names,
    validate_arguments,
)
from mizan.tools.schema import ToolSpec

EXPECTED_TOOLS = {
    "get_weather",
    "convert_currency",
    "book_restaurant",
    "set_reminder",
    "web_search",
    "calculator",
    "translate_text",
    "create_calendar_event",
    "get_directions",
    "convert_units",
}


def test_registry_holds_exactly_the_ten_expected_tools() -> None:
    assert set(TOOLS) == EXPECTED_TOOLS
    assert len(all_tools()) == 10


def test_all_tools_order_is_stable_and_matches_names() -> None:
    assert [spec.name for spec in all_tools()] == tool_names()
    # Two calls return the same order (prompts must be stable across runs).
    assert tool_names() == tool_names()


def test_get_tool_and_is_registered() -> None:
    assert get_tool("get_weather") is TOOLS["get_weather"]
    assert get_tool("teleport") is None
    assert is_registered("calculator")
    assert not is_registered("teleport")


def test_every_tool_required_list_is_a_subset_of_properties() -> None:
    for spec in all_tools():
        assert set(spec.required) <= set(spec.properties)


def test_to_openai_tool_shape() -> None:
    spec = get_tool("get_weather")
    assert spec is not None
    rendered = spec.to_openai_tool()
    assert rendered["type"] == "function"
    assert rendered["function"]["name"] == "get_weather"
    assert rendered["function"]["parameters"] == spec.parameters


def test_spec_rejects_non_object_parameters() -> None:
    with pytest.raises(ValueError, match="parameters.type must be 'object'"):
        ToolSpec(name="x", description="d", parameters={"type": "string"})


def test_spec_rejects_required_not_in_properties() -> None:
    with pytest.raises(ValueError, match="required names not in properties"):
        ToolSpec(
            name="x",
            description="d",
            parameters={
                "type": "object",
                "properties": {"a": {"type": "string"}},
                "required": ["b"],
            },
        )


def test_spec_rejects_unsupported_property_type() -> None:
    with pytest.raises(ValueError, match="unsupported type"):
        ToolSpec(
            name="x",
            description="d",
            parameters={"type": "object", "properties": {"a": {"type": "date"}}},
        )


def test_validate_arguments_accepts_valid_call() -> None:
    spec = get_tool("get_weather")
    assert spec is not None
    assert validate_arguments(spec, {"location": "Riyadh", "unit": "celsius"}) == []


def test_validate_arguments_flags_missing_required() -> None:
    spec = get_tool("convert_currency")
    assert spec is not None
    errors = validate_arguments(spec, {"amount": 10, "from_currency": "SAR"})
    assert any("missing required argument 'to_currency'" in e for e in errors)


def test_validate_arguments_flags_unknown_argument() -> None:
    spec = get_tool("web_search")
    assert spec is not None
    errors = validate_arguments(spec, {"query": "x", "limit": 5})
    assert any("unknown argument 'limit'" in e for e in errors)


def test_validate_arguments_flags_wrong_type() -> None:
    spec = get_tool("book_restaurant")
    assert spec is not None
    errors = validate_arguments(
        spec,
        {"restaurant_name": "Najd", "party_size": "four", "date": "tomorrow", "time": "20:00"},
    )
    assert any("party_size" in e and "integer" in e for e in errors)


def test_validate_arguments_rejects_bool_as_integer() -> None:
    spec = get_tool("book_restaurant")
    assert spec is not None
    errors = validate_arguments(
        spec,
        {"restaurant_name": "Najd", "party_size": True, "date": "tomorrow", "time": "20:00"},
    )
    assert any("party_size" in e and "boolean" in e for e in errors)


def test_validate_arguments_flags_enum_violation() -> None:
    spec = get_tool("get_directions")
    assert spec is not None
    errors = validate_arguments(
        spec, {"origin": "Riyadh", "destination": "Jeddah", "mode": "teleport"}
    )
    assert any("mode" in e and "one of" in e for e in errors)
