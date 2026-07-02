"""Schemas for parallel English / MSA / Gulf-dialect evaluation items.

The central idea of the harness is that one *intent* is expressed in parallel
across three languages so that model behaviour can be compared like-for-like.
An :class:`EvalItem` therefore holds one :class:`ItemVariant` per language plus
a single :class:`GoldLabel` describing the correct outcome (relevant documents
for retrieval, or the expected tool call for agent tool-calling).
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Language(StrEnum):
    """Languages an evaluation item is written in.

    ``MSA`` is Modern Standard Arabic; ``GULF`` is Gulf dialect. English is the
    pivot language and is required on every item.
    """

    EN = "en"
    MSA = "msa"
    GULF = "gulf"


class TaskType(StrEnum):
    """The kind of capability an item probes."""

    RETRIEVAL = "retrieval"
    FAITHFULNESS = "faithfulness"
    ANSWER_QUALITY = "answer_quality"
    TOOL_CALLING = "tool_calling"


class ItemVariant(BaseModel):
    """One language rendering of an evaluation item.

    ``query`` is the user utterance in this language. ``reference_answer`` is the
    gold answer *in the same language*, used by the answer-quality and
    faithfulness judges; it is optional for pure retrieval or tool-calling items.
    """

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1)
    reference_answer: str | None = None


class ExpectedToolCall(BaseModel):
    """The gold tool call for a tool-calling item.

    ``arguments`` are stored in a single canonical (English) form; how argument
    values should be localised per language is an open methodology question and
    is intentionally not modelled yet.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)


class GoldLabel(BaseModel):
    """Language-independent gold outcome shared by all variants of an item."""

    model_config = ConfigDict(extra="forbid")

    relevant_doc_ids: list[str] = Field(default_factory=list)
    expected_tool: ExpectedToolCall | None = None


class EvalItem(BaseModel):
    """A single evaluation item expressed in parallel across languages.

    Invariants (enforced on construction):

    - ``variants`` must include English and be non-empty.
    - A ``retrieval`` item must have at least one relevant document.
    - A ``tool_calling`` item must have an expected tool call.
    - An ``answer_quality`` or ``faithfulness`` item must supply a reference
      answer for every variant; ``faithfulness`` additionally needs context
      documents to check the answer against.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    task_type: TaskType
    variants: dict[Language, ItemVariant]
    gold: GoldLabel = Field(default_factory=GoldLabel)
    tags: list[str] = Field(default_factory=list)
    source: str = "sample"
    notes: str | None = None

    @model_validator(mode="after")
    def _check_consistency(self) -> EvalItem:
        if not self.variants:
            raise ValueError(f"item {self.id!r}: must define at least one language variant")
        if Language.EN not in self.variants:
            raise ValueError(f"item {self.id!r}: English variant is required as the pivot")

        if self.task_type is TaskType.RETRIEVAL and not self.gold.relevant_doc_ids:
            raise ValueError(f"item {self.id!r}: retrieval items need gold.relevant_doc_ids")

        if self.task_type is TaskType.TOOL_CALLING and self.gold.expected_tool is None:
            raise ValueError(f"item {self.id!r}: tool_calling items need gold.expected_tool")

        if self.task_type in (TaskType.ANSWER_QUALITY, TaskType.FAITHFULNESS):
            missing = [
                lang.value for lang, v in self.variants.items() if v.reference_answer is None
            ]
            if missing:
                raise ValueError(
                    f"item {self.id!r}: {self.task_type.value} items need a reference_answer "
                    f"for every variant; missing for: {', '.join(sorted(missing))}"
                )
        if self.task_type is TaskType.FAITHFULNESS and not self.gold.relevant_doc_ids:
            raise ValueError(
                f"item {self.id!r}: faithfulness items need context in gold.relevant_doc_ids"
            )
        return self

    @property
    def languages(self) -> list[Language]:
        """Languages present on this item, in a stable order."""
        order = [Language.EN, Language.MSA, Language.GULF]
        return [lang for lang in order if lang in self.variants]
