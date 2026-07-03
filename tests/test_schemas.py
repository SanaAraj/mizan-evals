"""Tests for the evaluation-item schema and its invariants."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from mizan.schemas import (
    EvalItem,
    ExpectedToolCall,
    GoldLabel,
    ItemVariant,
    Language,
    ReviewStatus,
    TaskType,
)


def _retrieval_item() -> EvalItem:
    return EvalItem(
        id="it-1",
        task_type=TaskType.RETRIEVAL,
        variants={
            Language.EN: ItemVariant(query="What is the capital of Saudi Arabia?"),
            Language.MSA: ItemVariant(query="ما هي عاصمة المملكة العربية السعودية؟"),
            Language.GULF: ItemVariant(query="وش عاصمة السعودية؟"),
        },
        gold=GoldLabel(relevant_doc_ids=["arwiki:الرياض#0"]),
    )


def test_valid_retrieval_item_round_trips_with_arabic_preserved() -> None:
    item = _retrieval_item()
    dumped = item.model_dump(mode="json")
    restored = EvalItem.model_validate(dumped)
    assert restored == item
    # Arabic content must survive serialisation untouched.
    assert restored.variants[Language.MSA].query == "ما هي عاصمة المملكة العربية السعودية؟"


def test_review_status_defaults_to_unreviewed() -> None:
    assert _retrieval_item().review_status is ReviewStatus.UNREVIEWED


def test_review_status_round_trips_from_json_value() -> None:
    dumped = _retrieval_item().model_dump(mode="json")
    dumped["review_status"] = "llm_qa"
    assert EvalItem.model_validate(dumped).review_status is ReviewStatus.LLM_QA


def test_review_status_rejects_unknown_value() -> None:
    dumped = _retrieval_item().model_dump(mode="json")
    dumped["review_status"] = "vibes_checked"
    with pytest.raises(ValidationError, match="review_status"):
        EvalItem.model_validate(dumped)


def test_languages_property_is_stable_order() -> None:
    item = _retrieval_item()
    assert item.languages == [Language.EN, Language.MSA, Language.GULF]


def test_english_variant_is_required() -> None:
    with pytest.raises(ValidationError, match="English variant is required"):
        EvalItem(
            id="it-no-en",
            task_type=TaskType.RETRIEVAL,
            variants={Language.MSA: ItemVariant(query="سؤال")},
            gold=GoldLabel(relevant_doc_ids=["d1"]),
        )


def test_retrieval_item_requires_relevant_docs() -> None:
    with pytest.raises(ValidationError, match="relevant_doc_ids"):
        EvalItem(
            id="it-no-gold",
            task_type=TaskType.RETRIEVAL,
            variants={Language.EN: ItemVariant(query="q")},
        )


def test_tool_calling_item_requires_expected_tool() -> None:
    with pytest.raises(ValidationError, match="expected_tool"):
        EvalItem(
            id="it-tool",
            task_type=TaskType.TOOL_CALLING,
            variants={Language.EN: ItemVariant(query="What's the weather in Dubai?")},
        )


def test_tool_calling_item_is_valid_with_expected_tool() -> None:
    item = EvalItem(
        id="it-tool-ok",
        task_type=TaskType.TOOL_CALLING,
        variants={Language.EN: ItemVariant(query="What's the weather in Dubai tomorrow?")},
        gold=GoldLabel(
            expected_tool=ExpectedToolCall(
                name="get_weather", arguments={"city": "Dubai", "date": "tomorrow"}
            )
        ),
    )
    assert item.gold.expected_tool is not None
    assert item.gold.expected_tool.arguments["city"] == "Dubai"


def test_answer_quality_requires_reference_answer_for_every_variant() -> None:
    with pytest.raises(ValidationError, match="reference_answer"):
        EvalItem(
            id="it-aq",
            task_type=TaskType.ANSWER_QUALITY,
            variants={
                Language.EN: ItemVariant(query="q", reference_answer="a"),
                Language.MSA: ItemVariant(query="س"),  # missing reference answer
            },
        )


def test_faithfulness_requires_context_documents() -> None:
    with pytest.raises(ValidationError, match="context"):
        EvalItem(
            id="it-faith",
            task_type=TaskType.FAITHFULNESS,
            variants={Language.EN: ItemVariant(query="q", reference_answer="a")},
        )


def test_unknown_field_is_rejected() -> None:
    with pytest.raises(ValidationError):
        EvalItem(
            id="it-extra",
            task_type=TaskType.RETRIEVAL,
            variants={Language.EN: ItemVariant(query="q")},
            gold=GoldLabel(relevant_doc_ids=["d1"]),
            difficulty="hard",  # type: ignore[call-arg]
        )


def test_empty_query_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ItemVariant(query="")
