"""Turning raw model output into a structured tool call.

Two extraction paths feed one representation, :class:`ToolCallResult`:

- **Native** — for backends with a function-calling API (GPT, Qwen). The backend
  emits its tool call already serialized into this module's canonical JSON
  contract (``{"tool": <name|null>, "arguments": {...}}``); :func:`parse_native`
  reads it back.
- **Prompt** — for backends without a tools API (e.g. ALLaM). We render the tool
  schemas into a strict instruction with :func:`build_tool_prompt`, and
  :func:`parse_prompt_output` robustly recovers the JSON object from free-form
  text (stripping code fences and surrounding prose).

A no-tool decision (``{"tool": null}``) is a *valid* parse, not an error — it is
exactly how a model should answer a distractor. A ``parse_error`` is reserved for
output that could not be interpreted at all; such items are scored as incorrect
and never dropped.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from mizan.tools.schema import ToolSpec

# Accepted key spellings, so a model that says "name"/"tool_name" or
# "parameters"/"args" instead of the canonical "tool"/"arguments" still parses.
_TOOL_KEYS = ("tool", "tool_name", "name")
_ARG_KEYS = ("arguments", "parameters", "args")


class ToolCallResult(BaseModel):
    """The structured outcome of extracting a tool call from model output."""

    model_config = ConfigDict(extra="forbid")

    chosen_tool: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    parse_error: str | None = None
    raw: str = ""

    @property
    def failed(self) -> bool:
        """Whether extraction failed to yield a usable decision."""
        return self.parse_error is not None


def to_canonical_json(tool: str | None, arguments: dict[str, Any] | None) -> str:
    """Serialize a tool decision into this module's canonical JSON contract.

    Used by native backends so their cached raw output has one stable shape.
    """
    return json.dumps({"tool": tool, "arguments": arguments or {}}, ensure_ascii=False)


def build_tool_prompt(tools: list[ToolSpec], utterance: str) -> str:
    """Render a strict prompt asking a non-tools model to choose a tool as JSON.

    The tool order is the registry order, so the prompt is byte-stable across runs
    (and therefore cache-stable).
    """
    lines = [
        "You are a function-calling engine. Decide whether exactly one of the "
        "available tools should be called to satisfy the user's request.",
        "",
        "Available tools (JSON Schema for each):",
    ]
    for i, tool in enumerate(tools, start=1):
        schema = json.dumps(tool.parameters, ensure_ascii=False, sort_keys=True)
        lines.append(f"{i}. {tool.name} — {tool.description}")
        lines.append(f"   parameters: {schema}")
    lines += [
        "",
        "Respond with a single JSON object and nothing else, in one of these forms:",
        '  to call a tool: {"tool": "<tool_name>", "arguments": { ... }}',
        '  if no tool applies: {"tool": null}',
        "",
        "Rules:",
        "- Use only a tool name listed above.",
        "- Include only arguments defined by that tool's schema.",
        "- Output raw JSON only: no explanations, no markdown, no code fences.",
        "",
        f"User request: {utterance}",
        "JSON:",
    ]
    return "\n".join(lines)


def _first(mapping: dict[str, Any], keys: tuple[str, ...]) -> tuple[bool, Any]:
    """Return ``(found, value)`` for the first present key in ``keys``."""
    for key in keys:
        if key in mapping:
            return True, mapping[key]
    return False, None


def _interpret(obj: Any, raw: str) -> ToolCallResult:
    """Interpret a parsed JSON value as a tool decision."""
    if not isinstance(obj, dict):
        return ToolCallResult(
            parse_error=f"expected a JSON object, got {type(obj).__name__}", raw=raw
        )

    found_tool, tool = _first(obj, _TOOL_KEYS)
    if not found_tool:
        return ToolCallResult(parse_error="no 'tool' field in output", raw=raw)

    if tool is None:
        return ToolCallResult(chosen_tool=None, arguments={}, raw=raw)
    if not isinstance(tool, str) or not tool.strip():
        return ToolCallResult(
            parse_error=f"'tool' must be a non-empty string or null, got {tool!r}", raw=raw
        )

    _, arguments = _first(obj, _ARG_KEYS)
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, dict):
        return ToolCallResult(
            parse_error=f"'arguments' must be an object, got {type(arguments).__name__}", raw=raw
        )
    return ToolCallResult(chosen_tool=tool.strip(), arguments=arguments, raw=raw)


def parse_native(raw: str) -> ToolCallResult:
    """Parse a native backend's canonical-JSON tool call."""
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as exc:
        return ToolCallResult(parse_error=f"invalid native JSON: {exc}", raw=raw)
    return _interpret(obj, raw)


def _extract_json_object(text: str) -> str | None:
    """Return the first balanced top-level ``{...}`` block in ``text``, if any."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def parse_prompt_output(raw: str) -> ToolCallResult:
    """Robustly recover a tool decision from a prompt-based model's free text.

    Handles a bare JSON object, an object wrapped in prose, and ```` ```json ````
    code fences. Output with no recoverable JSON object is a ``parse_error``.
    """
    text = raw.strip()
    # Fast path: the whole response is JSON.
    try:
        return _interpret(json.loads(text), raw)
    except json.JSONDecodeError:
        pass

    block = _extract_json_object(text)
    if block is None:
        return ToolCallResult(parse_error="no JSON object found in output", raw=raw)
    try:
        return _interpret(json.loads(block), raw)
    except json.JSONDecodeError as exc:
        return ToolCallResult(parse_error=f"malformed JSON object in output: {exc}", raw=raw)
