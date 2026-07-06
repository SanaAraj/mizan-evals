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


class ReviewStatus(StrEnum):
    """How far an item's content has been quality-checked.

    Datasets must state their review provenance explicitly so published
    methodology claims can be tied to per-item state: ``UNREVIEWED`` is a raw
    draft, ``LLM_QA`` passed an LLM cross-lingual consistency and dialect QA
    pass, and ``NATIVE_REVIEWED`` was additionally reviewed by a native
    Arabic speaker.
    """

    UNREVIEWED = "unreviewed"
    LLM_QA = "llm_qa"
    NATIVE_REVIEWED = "native_reviewed"


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
    """Language-independent gold outcome shared by all variants of an item.

    For a tool-calling item, exactly one of two mutually exclusive outcomes is
    gold: ``expected_tool`` names the call the model should make, or
    ``expected_no_tool`` marks a distractor intent where the correct behaviour is
    to call nothing. Distractors make the refusal / hallucinated-tool rates
    measurable rather than assumed.
    """

    model_config = ConfigDict(extra="forbid")

    relevant_doc_ids: list[str] = Field(default_factory=list)
    expected_tool: ExpectedToolCall | None = None
    expected_no_tool: bool = False


class EvalItem(BaseModel):
    """A single evaluation item expressed in parallel across languages.

    Invariants (enforced on construction):

    - ``variants`` must include English and be non-empty.
    - A ``retrieval`` item must have at least one relevant document.
    - A ``tool_calling`` item must have exactly one of an expected tool call or
      ``expected_no_tool`` (a no-tool distractor).
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
    review_status: ReviewStatus = ReviewStatus.UNREVIEWED
    notes: str | None = None

    @model_validator(mode="after")
    def _check_consistency(self) -> EvalItem:
        if not self.variants:
            raise ValueError(f"item {self.id!r}: must define at least one language variant")
        if Language.EN not in self.variants:
            raise ValueError(f"item {self.id!r}: English variant is required as the pivot")

        if self.task_type is TaskType.RETRIEVAL and not self.gold.relevant_doc_ids:
            raise ValueError(f"item {self.id!r}: retrieval items need gold.relevant_doc_ids")

        if self.task_type is TaskType.TOOL_CALLING:
            has_tool = self.gold.expected_tool is not None
            if has_tool == self.gold.expected_no_tool:
                raise ValueError(
                    f"item {self.id!r}: tool_calling items need exactly one of "
                    "gold.expected_tool (a positive intent) or gold.expected_no_tool=true "
                    "(a distractor)"
                )

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
