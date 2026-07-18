"""The fixed set of ten mock tools the tool-calling eval selects between.

The tools are invented for the benchmark - no tool calls a real service - but
their schemas are realistic: required/optional split, typed parameters, and
``enum``s where a real API would constrain a value. The set is deliberately
diverse so that intents can be unambiguous (one clearly-correct tool per intent)
while still leaving room for a model to pick a plausible wrong neighbour
(``get_directions`` vs ``get_weather`` both take a place; ``convert_currency``
vs ``convert_units`` both convert).

The registry is frozen at import time; :func:`all_tools` returns the canonical
order used when rendering tools to a model so that prompts are stable across runs.
"""

from __future__ import annotations

from mizan.tools.schema import ToolSpec

_TOOL_SPECS: list[ToolSpec] = [
    ToolSpec(
        name="get_weather",
        description="Get the current or forecast weather for a location on a given day.",
        parameters={
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City or place name, e.g. 'Riyadh'.",
                },
                "date": {
                    "type": "string",
                    "description": "Day to forecast, e.g. 'today', 'tomorrow', or an ISO date.",
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature unit.",
                },
            },
            "required": ["location"],
        },
    ),
    ToolSpec(
        name="convert_currency",
        description="Convert a monetary amount from one currency to another.",
        parameters={
            "type": "object",
            "properties": {
                "amount": {"type": "number", "description": "Amount to convert."},
                "from_currency": {
                    "type": "string",
                    "description": "Source currency as an ISO 4217 code, e.g. 'SAR'.",
                },
                "to_currency": {
                    "type": "string",
                    "description": "Target currency as an ISO 4217 code, e.g. 'USD'.",
                },
            },
            "required": ["amount", "from_currency", "to_currency"],
        },
    ),
    ToolSpec(
        name="book_restaurant",
        description="Reserve a table at a restaurant for a party at a given date and time.",
        parameters={
            "type": "object",
            "properties": {
                "restaurant_name": {"type": "string", "description": "Name of the restaurant."},
                "city": {"type": "string", "description": "City the restaurant is in."},
                "party_size": {"type": "integer", "description": "Number of people."},
                "date": {"type": "string", "description": "Reservation date, e.g. 'tomorrow'."},
                "time": {"type": "string", "description": "Reservation time, e.g. '20:00'."},
            },
            "required": ["restaurant_name", "party_size", "date", "time"],
        },
    ),
    ToolSpec(
        name="set_reminder",
        description="Create a personal reminder for a task at a specific time.",
        parameters={
            "type": "object",
            "properties": {
                "task": {"type": "string", "description": "What to be reminded about."},
                "datetime": {
                    "type": "string",
                    "description": "When to fire the reminder, e.g. 'tomorrow 09:00'.",
                },
            },
            "required": ["task", "datetime"],
        },
    ),
    ToolSpec(
        name="web_search",
        description="Search the web for open-ended, up-to-date, or factual information.",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query."},
            },
            "required": ["query"],
        },
    ),
    ToolSpec(
        name="calculator",
        description="Evaluate an arithmetic expression and return the numeric result.",
        parameters={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Arithmetic expression, e.g. '15 * 1.14'.",
                },
            },
            "required": ["expression"],
        },
    ),
    ToolSpec(
        name="translate_text",
        description="Translate a piece of text into a target language.",
        parameters={
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "The text to translate."},
                "target_language": {
                    "type": "string",
                    "description": "Language to translate into, e.g. 'English'.",
                },
                "source_language": {
                    "type": "string",
                    "description": "Language of the source text, if known.",
                },
            },
            "required": ["text", "target_language"],
        },
    ),
    ToolSpec(
        name="create_calendar_event",
        description="Add an event to the user's calendar.",
        parameters={
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Event title."},
                "date": {"type": "string", "description": "Event date, e.g. 'tomorrow'."},
                "start_time": {"type": "string", "description": "Start time, e.g. '14:30'."},
                "duration_minutes": {"type": "integer", "description": "Length in minutes."},
                "location": {"type": "string", "description": "Where the event is held."},
            },
            "required": ["title", "date", "start_time"],
        },
    ),
    ToolSpec(
        name="get_directions",
        description="Get travel directions between two places.",
        parameters={
            "type": "object",
            "properties": {
                "origin": {"type": "string", "description": "Starting place."},
                "destination": {"type": "string", "description": "Destination place."},
                "mode": {
                    "type": "string",
                    "enum": ["driving", "walking", "transit", "cycling"],
                    "description": "Mode of travel.",
                },
            },
            "required": ["origin", "destination"],
        },
    ),
    ToolSpec(
        name="convert_units",
        description="Convert a physical quantity from one unit of measure to another.",
        parameters={
            "type": "object",
            "properties": {
                "value": {"type": "number", "description": "The quantity to convert."},
                "from_unit": {
                    "type": "string",
                    "description": "Unit to convert from, e.g. 'kilometer'.",
                },
                "to_unit": {
                    "type": "string",
                    "description": "Unit to convert to, e.g. 'mile'.",
                },
            },
            "required": ["value", "from_unit", "to_unit"],
        },
    ),
]

# Canonical, import-time-frozen registry keyed by tool name.
TOOLS: dict[str, ToolSpec] = {spec.name: spec for spec in _TOOL_SPECS}


def all_tools() -> list[ToolSpec]:
    """Return every tool in canonical (registry) order."""
    return list(_TOOL_SPECS)


def tool_names() -> list[str]:
    """Return the names of every registered tool, in canonical order."""
    return [spec.name for spec in _TOOL_SPECS]


def get_tool(name: str) -> ToolSpec | None:
    """Return the tool named ``name``, or ``None`` if it is not registered."""
    return TOOLS.get(name)


def is_registered(name: str) -> bool:
    """Return whether ``name`` is a known tool (used to flag hallucinated tools)."""
    return name in TOOLS
