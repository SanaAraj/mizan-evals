"""Tests for native function-calling on the OpenAI-compatible backend."""

from __future__ import annotations

import json

import httpx

from mizan.llm.base import GenerationParams
from mizan.llm.openai_compatible import OpenAICompatibleBackend
from mizan.tools.extract import parse_native
from mizan.tools.registry import all_tools


def _backend(handler) -> OpenAICompatibleBackend:
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return OpenAICompatibleBackend(
        "gpt-x", base_url="https://api.test/v1", api_key="sk-test", client=client
    )


def test_backend_advertises_native_tools() -> None:
    assert _backend(lambda r: httpx.Response(200, json={})).supports_native_tools is True


def test_sends_tools_and_canonicalizes_tool_call() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": json.dumps({"location": "Riyadh"}),
                                    }
                                }
                            ],
                        }
                    }
                ]
            },
        )

    backend = _backend(handler)
    tools = [t.to_openai_tool() for t in all_tools()]
    raw = backend.generate_tool_call("weather in Riyadh?", tools, GenerationParams(seed=1))

    assert seen["body"]["tool_choice"] == "auto"
    assert len(seen["body"]["tools"]) == 10
    result = parse_native(raw)
    assert result.chosen_tool == "get_weather"
    assert result.arguments == {"location": "Riyadh"}


def test_no_tool_calls_becomes_no_call() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "just chatting"}}]})

    raw = _backend(handler).generate_tool_call("hi", [], GenerationParams())
    result = parse_native(raw)
    assert result.chosen_tool is None
    assert result.parse_error is None


def test_malformed_provider_arguments_degrade_to_empty() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {"function": {"name": "calculator", "arguments": "{not valid json"}}
                            ]
                        }
                    }
                ]
            },
        )

    raw = _backend(handler).generate_tool_call("2+2", [], GenerationParams())
    result = parse_native(raw)
    assert result.chosen_tool == "calculator"
    assert result.arguments == {}
