"""The tool-schema model and an argument validator.

Tools are described in the same shape a chat model's function-calling API
expects: a ``name``, a human ``description``, and a JSON-Schema ``parameters``
object. Keeping the on-disk shape identical to the wire shape means the native
function-calling backends can pass a :class:`ToolSpec` straight through without a
translation layer, and the prompt-based fallback can render the exact same schema
into text.

:func:`validate_arguments` is a deliberately small JSON-Schema checker: it covers
only the constructs the tool registry actually uses (``object`` with typed
``properties``, ``required``, and string ``enum``s). It is used to sanity-check
gold arguments in tests, not to police live model output - a model that emits an
argument violating the schema is simply scored as wrong by the tool-calling
scorer.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

# JSON-Schema ``type`` names mapped to the Python types we accept for them. Bools
# are excluded from ``number``/``integer`` because ``bool`` is a subclass of
# ``int`` in Python and ``True`` is never a valid numeric argument here.
_JSON_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "number": (int, float),
    "integer": (int,),
    "boolean": (bool,),
    "array": (list,),
    "object": (dict,),
}


class ToolSpec(BaseModel):
    """A single callable tool, described as an OpenAI-style function schema.

    ``parameters`` is a JSON-Schema ``object``: it must carry a ``properties``
    mapping and may carry a ``required`` list. The model is validated on
    construction so a malformed registry fails loudly at import time rather than
    mid-run.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    parameters: dict[str, Any]

    @model_validator(mode="after")
    def _check_parameters(self) -> ToolSpec:
        params = self.parameters
        if params.get("type") != "object":
            raise ValueError(f"tool {self.name!r}: parameters.type must be 'object'")
        properties = params.get("properties")
        if not isinstance(properties, dict) or not properties:
            raise ValueError(f"tool {self.name!r}: parameters.properties must be a non-empty map")
        for prop_name, schema in properties.items():
            prop_type = schema.get("type") if isinstance(schema, dict) else None
            if prop_type not in _JSON_TYPES:
                raise ValueError(
                    f"tool {self.name!r}: property {prop_name!r} has unsupported type {prop_type!r}"
                )
        required = params.get("required", [])
        if not isinstance(required, list):
            raise ValueError(f"tool {self.name!r}: parameters.required must be a list")
        unknown = [r for r in required if r not in properties]
        if unknown:
            raise ValueError(f"tool {self.name!r}: required names not in properties: {unknown}")
        return self

    @property
    def properties(self) -> dict[str, Any]:
        """The parameter property schemas, keyed by argument name."""
        return self.parameters["properties"]

    @property
    def required(self) -> list[str]:
        """The names of the required arguments, in schema order."""
        return list(self.parameters.get("required", []))

    def to_openai_tool(self) -> dict[str, Any]:
        """Render this spec as an OpenAI ``tools`` array entry."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def validate_arguments(spec: ToolSpec, arguments: dict[str, Any]) -> list[str]:
    """Return a list of human-readable schema violations for ``arguments``.

    An empty list means the arguments satisfy the tool's schema. The checks are
    intentionally shallow (top-level properties only): required-present, known
    property name, JSON type, and string ``enum`` membership.
    """
    errors: list[str] = []
    properties = spec.properties
    for name in spec.required:
        if name not in arguments:
            errors.append(f"missing required argument {name!r}")

    for name, value in arguments.items():
        schema = properties.get(name)
        if schema is None:
            errors.append(f"unknown argument {name!r}")
            continue
        expected = schema["type"]
        allowed_py = _JSON_TYPES[expected]
        # Guard the int/bool overlap: a bool is not a valid number or integer.
        if isinstance(value, bool) and expected in {"number", "integer"}:
            errors.append(f"argument {name!r} must be {expected}, got boolean")
            continue
        if not isinstance(value, allowed_py):
            errors.append(f"argument {name!r} must be {expected}, got {type(value).__name__}")
            continue
        enum = schema.get("enum")
        if enum is not None and value not in enum:
            errors.append(f"argument {name!r} must be one of {enum}, got {value!r}")

    return errors
