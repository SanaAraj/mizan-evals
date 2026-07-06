"""Hard tests for the tool-calling scorer and its aggregation.

Every one of the five metrics is exercised with a hand-built scenario so the
numbers in the results table are traceable to a known-correct example.
"""

from __future__ import annotations

from mizan.results import ItemResult
from mizan.schemas import ExpectedToolCall, GoldLabel, Language, TaskType
from mizan.scoring.tool_calling import (
    ARG_ACCURACY,
    EXACT_MATCH,
    HALLUCINATED,
    IS_DISTRACTOR,
    LOCALIZED_CORRECT,
    REFUSED,
    TOOL_CORRECT,
    aggregate,
    score_tool_item,
)

WEATHER_GOLD = GoldLabel(
    expected_tool=ExpectedToolCall(
        name="get_weather", arguments={"location": "Riyadh", "date": "tomorrow"}
    )
)
DISTRACTOR_GOLD = GoldLabel(expected_no_tool=True)


def _item(scores: dict[str, float], **kw: object) -> ItemResult:
    base: dict[str, object] = {
        "item_id": "x",
        "model_id": "m",
        "language": Language.EN,
        "task_type": TaskType.TOOL_CALLING,
    }
    base.update(kw)
    return ItemResult(**base, scores=scores)  # type: ignore[arg-type]


# --- per-item scoring ------------------------------------------------------


def test_correct_tool_all_args_english() -> None:
    scores = score_tool_item(
        WEATHER_GOLD,
        chosen_tool="get_weather",
        arguments={"location": "Riyadh", "date": "tomorrow"},
    )
    assert scores[TOOL_CORRECT] == 1.0
    assert scores[ARG_ACCURACY] == 1.0
    assert scores[EXACT_MATCH] == 1.0
    assert scores[HALLUCINATED] == 0.0
    assert scores[LOCALIZED_CORRECT] == 0.0


def test_correct_tool_localized_args() -> None:
    scores = score_tool_item(
        WEATHER_GOLD,
        chosen_tool="get_weather",
        arguments={"location": "الرياض", "date": "بكرة"},
    )
    assert scores[ARG_ACCURACY] == 1.0
    # Both args are localizable and were produced in localized form.
    assert scores["localizable_correct"] == 2.0
    assert scores[LOCALIZED_CORRECT] == 2.0


def test_partial_arguments_give_partial_accuracy() -> None:
    scores = score_tool_item(
        WEATHER_GOLD,
        chosen_tool="get_weather",
        arguments={"location": "Riyadh", "date": "yesterday"},
    )
    assert scores[ARG_ACCURACY] == 0.5
    assert scores[EXACT_MATCH] == 0.0


def test_spurious_argument_breaks_exact_match_but_not_accuracy() -> None:
    scores = score_tool_item(
        WEATHER_GOLD,
        chosen_tool="get_weather",
        arguments={"location": "Riyadh", "date": "tomorrow", "unit": "celsius"},
    )
    assert scores[ARG_ACCURACY] == 1.0
    assert scores["spurious_args"] == 1.0
    assert scores[EXACT_MATCH] == 0.0


def test_wrong_tool_skips_argument_scoring() -> None:
    scores = score_tool_item(
        WEATHER_GOLD, chosen_tool="web_search", arguments={"query": "weather riyadh"}
    )
    assert scores[TOOL_CORRECT] == 0.0
    assert ARG_ACCURACY not in scores
    assert scores[EXACT_MATCH] == 0.0
    assert scores[HALLUCINATED] == 0.0  # web_search is a real tool


def test_hallucinated_tool_flagged() -> None:
    scores = score_tool_item(WEATHER_GOLD, chosen_tool="teleport", arguments={})
    assert scores[HALLUCINATED] == 1.0
    assert scores[TOOL_CORRECT] == 0.0


def test_parse_failure_scored_as_incorrect_no_call() -> None:
    scores = score_tool_item(WEATHER_GOLD, chosen_tool=None, arguments={}, failed=True)
    assert scores[TOOL_CORRECT] == 0.0
    assert scores[HALLUCINATED] == 0.0


def test_distractor_refusal() -> None:
    scores = score_tool_item(DISTRACTOR_GOLD, chosen_tool=None, arguments={})
    assert scores[IS_DISTRACTOR] == 1.0
    assert scores[REFUSED] == 1.0


def test_distractor_wrong_call_is_not_refusal() -> None:
    scores = score_tool_item(
        DISTRACTOR_GOLD, chosen_tool="get_weather", arguments={"location": "Riyadh"}
    )
    assert scores[REFUSED] == 0.0
    assert scores[HALLUCINATED] == 0.0


def test_distractor_failed_call_is_not_refusal() -> None:
    scores = score_tool_item(DISTRACTOR_GOLD, chosen_tool=None, arguments={}, failed=True)
    assert scores[REFUSED] == 0.0


# --- aggregation -----------------------------------------------------------


def test_aggregate_mixed_group() -> None:
    results = [
        # positive, fully correct, one localized arg
        _item(
            score_tool_item(
                WEATHER_GOLD,
                chosen_tool="get_weather",
                arguments={"location": "الرياض", "date": "tomorrow"},
            )
        ),
        # positive, correct tool, half args
        _item(
            score_tool_item(
                WEATHER_GOLD,
                chosen_tool="get_weather",
                arguments={"location": "Riyadh", "date": "yesterday"},
            )
        ),
        # positive, wrong tool
        _item(score_tool_item(WEATHER_GOLD, chosen_tool="web_search", arguments={})),
        # distractor, correctly refused
        _item(score_tool_item(DISTRACTOR_GOLD, chosen_tool=None, arguments={})),
        # distractor, wrongly called a tool
        _item(
            score_tool_item(DISTRACTOR_GOLD, chosen_tool="get_weather", arguments={"location": "x"})
        ),
    ]
    metrics = aggregate(results)

    assert metrics.n_positive == 3
    assert metrics.n_distractor == 2
    # 2 of 3 positives selected the right tool.
    assert metrics.correct_tool_rate == 2 / 3
    # arg accuracy averaged over the 2 correct-tool positives: (1.0 + 0.5) / 2
    assert metrics.n_arg_scored == 2
    assert metrics.argument_accuracy == 0.75
    # exact match over all positives: only the first is exact -> 1/3
    assert metrics.exact_match_rate == 1 / 3
    # 1 of 2 distractors refused.
    assert metrics.refusal_on_distractor_rate == 0.5
    # localization: 1 localized correct arg out of 3 localizable-correct args
    # (item1: location+date localizable, both correct, location localized;
    #  item2: only location correct+localizable, not localized) -> 1/3.
    assert metrics.localization_rate == 1 / 3
    assert metrics.hallucinated_tool_rate == 0.0


def test_aggregate_localization_none_when_no_localizable_correct() -> None:
    results = [
        _item(score_tool_item(WEATHER_GOLD, chosen_tool="web_search", arguments={})),
    ]
    metrics = aggregate(results)
    assert metrics.localization_rate is None
    assert metrics.argument_accuracy is None
    assert metrics.refusal_on_distractor_rate is None


def test_aggregate_counts_parse_errors() -> None:
    results = [
        _item(
            score_tool_item(WEATHER_GOLD, chosen_tool=None, arguments={}, failed=True),
            parse_error="x",
        ),
    ]
    metrics = aggregate(results)
    assert metrics.n_parse_errors == 1
    assert metrics.correct_tool_rate == 0.0
