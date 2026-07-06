"""Schemas for run metadata and per-item results.

These records make a run reproducible and dated: every run folder carries the
resolved configuration, the model ids under test, and a UTC timestamp. The
per-item result shape is defined now so that scoring milestones can populate it
without a schema change.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from mizan.config import RunConfig
from mizan.schemas import Language, TaskType


class RunMetadata(BaseModel):
    """Identifying, reproducibility-relevant metadata for a single run."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    name: str
    created_at: str
    mizan_version: str
    model_ids: list[str]
    config: RunConfig


class ItemResult(BaseModel):
    """The outcome of evaluating one item, in one language, with one model.

    ``error`` records a backend/transport failure (the call never returned usable
    output). ``parse_error`` is distinct: the call returned, but its output could
    not be parsed into a tool call — such items are scored as incorrect, never
    dropped, and counted separately so parser fragility is visible. ``chosen_tool``
    records the extracted tool name (or ``None`` for a no-call) so failures can be
    inspected by item id.
    """

    model_config = ConfigDict(extra="forbid")

    item_id: str
    model_id: str
    language: Language
    task_type: TaskType
    output: str | None = None
    scores: dict[str, float] = Field(default_factory=dict)
    cached: bool = False
    error: str | None = None
    parse_error: str | None = None
    chosen_tool: str | None = None


class RunResult(BaseModel):
    """A complete run: its metadata, per-item results, and aggregate summary."""

    model_config = ConfigDict(extra="forbid")

    metadata: RunMetadata
    items: list[ItemResult] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)
