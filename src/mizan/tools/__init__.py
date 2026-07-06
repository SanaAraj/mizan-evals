"""Mock tool schemas, argument normalization, and tool-call extraction.

This package holds everything specific to the tool-calling eval: the fixed
registry of ten tool schemas (:mod:`mizan.tools.registry`), a small JSON-Schema
validator (:mod:`mizan.tools.schema`), the alias-based argument normalization
layer (:mod:`mizan.tools.normalize`), and the native/prompt extraction paths
(:mod:`mizan.tools.extract`).
"""

from __future__ import annotations

from mizan.tools.registry import (
    TOOLS,
    all_tools,
    get_tool,
    is_registered,
    tool_names,
)
from mizan.tools.schema import ToolSpec, validate_arguments

__all__ = [
    "TOOLS",
    "ToolSpec",
    "all_tools",
    "get_tool",
    "is_registered",
    "tool_names",
    "validate_arguments",
]
