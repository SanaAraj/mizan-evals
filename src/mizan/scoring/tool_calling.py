"""Scoring for the tool-calling eval: five metrics, per language.

The unit of scoring is one (item, language, model) triple. :func:`score_tool_item`
turns a single extracted tool call into a flat ``dict[str, float]`` of per-item
signals; :func:`aggregate` rolls a homogeneous group of those (one model, one
language) into a :class:`ToolCallMetrics` record. Splitting it this way keeps the
persisted ``results.json`` self-describing (every signal is a plain number on the
item) while keeping the aggregation logic in one tested place.

Metrics (all conditioned as noted):

- **correct_tool_rate** — over positive items, fraction where the model selected
  the gold tool. A parse failure or a no-call counts as incorrect.
- **argument_accuracy** — over positive items *where the correct tool was
  selected*, the mean fraction of gold arguments matched (after alias
  normalization). Arguments are meaningless when the wrong tool fired, so those
  items are excluded from this mean.
- **exact_match_rate** — over positive items, fraction where the tool and *every*
  gold argument matched with no spurious arguments. A strict secondary view.
- **hallucinated_tool_rate** — over all items, fraction where the model invoked a
  tool name that is not in the registry.
- **refusal_on_distractor_rate** — over distractor items, fraction where the model
  correctly called no tool.
- **localization_rate** — over correctly-matched, localizable arguments, the
  fraction the model left in localized (non-English) form. Reported per language
  as a behavioural finding, not an error.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict

from mizan.results import ItemResult
from mizan.schemas import GoldLabel
from mizan.tools.normalize import match_argument
from mizan.tools.registry import is_registered

# Keys written onto ItemResult.scores by score_tool_item. Kept as a named set so
# the aggregator and tests refer to one source of truth.
IS_DISTRACTOR = "is_distractor"
TOOL_CORRECT = "tool_correct"
HALLUCINATED = "hallucinated_tool"
EXACT_MATCH = "exact_match"
REFUSED = "refused"
ARG_ACCURACY = "arg_accuracy"
SPURIOUS_ARGS = "spurious_args"
LOCALIZABLE_CORRECT = "localizable_correct"
LOCALIZED_CORRECT = "localized_correct"


class ToolCallMetrics(BaseModel):
    """Aggregate tool-calling metrics for one model in one language.

    Rate fields are ``None`` when their denominator is empty (e.g. a run with no
    distractors has no ``refusal_on_distractor_rate``), so a missing measurement
    is never rendered as ``0.0``.
    """

    model_config = ConfigDict(extra="forbid")

    n_positive: int
    n_distractor: int
    n_arg_scored: int
    n_parse_errors: int
    correct_tool_rate: float | None = None
    argument_accuracy: float | None = None
    exact_match_rate: float | None = None
    hallucinated_tool_rate: float | None = None
    refusal_on_distractor_rate: float | None = None
    localization_rate: float | None = None


def score_tool_item(
    gold: GoldLabel,
    *,
    chosen_tool: str | None,
    arguments: Mapping[str, Any],
    failed: bool = False,
) -> dict[str, float]:
    """Score one extracted tool call against an item's gold label.

    Args:
        gold: the item's gold label (either an expected tool or ``expected_no_tool``).
        chosen_tool: the tool name the model selected, or ``None`` for no call
            (including when extraction or the backend call failed).
        arguments: the arguments the model supplied.
        failed: whether the call did not yield a usable decision (a parse failure
            or a backend error). Such an item is still scored — as incorrect, and
            never counted as a distractor refusal — rather than dropped.

    Returns:
        A flat mapping of per-item numeric signals for :func:`aggregate`.
    """
    is_distractor = gold.expected_no_tool
    hallucinated = chosen_tool is not None and not is_registered(chosen_tool)
    scores: dict[str, float] = {
        IS_DISTRACTOR: float(is_distractor),
        HALLUCINATED: float(hallucinated),
    }

    if is_distractor:
        # Correct behaviour is to call nothing. A failed call is not a refusal.
        scores[REFUSED] = float(chosen_tool is None and not failed)
        return scores

    expected = gold.expected_tool
    if expected is None:  # pragma: no cover - schema forbids this on positives
        raise ValueError("positive tool-calling item is missing gold.expected_tool")

    tool_correct = chosen_tool == expected.name
    scores[TOOL_CORRECT] = float(tool_correct)

    if not tool_correct:
        # Arguments are only scored when the right tool fired; exact match fails.
        scores[EXACT_MATCH] = 0.0
        return scores

    gold_args = expected.arguments
    matched = 0
    localizable_correct = 0
    localized_correct = 0
    for name, gold_value in gold_args.items():
        if name not in arguments:
            continue
        result = match_argument(gold_value, arguments[name])
        if result.matched:
            matched += 1
            if result.localizable:
                localizable_correct += 1
                if result.localized:
                    localized_correct += 1

    spurious = sum(1 for name in arguments if name not in gold_args)
    arg_accuracy = matched / len(gold_args) if gold_args else 1.0
    exact = matched == len(gold_args) and spurious == 0

    scores[ARG_ACCURACY] = arg_accuracy
    scores[SPURIOUS_ARGS] = float(spurious)
    scores[EXACT_MATCH] = float(exact)
    scores[LOCALIZABLE_CORRECT] = float(localizable_correct)
    scores[LOCALIZED_CORRECT] = float(localized_correct)
    return scores


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def aggregate(results: list[ItemResult]) -> ToolCallMetrics:
    """Aggregate per-item tool-calling scores for one (model, language) group.

    Expects every result to carry the score keys produced by
    :func:`score_tool_item`. Distractor and positive items are separated by the
    ``is_distractor`` marker so each metric uses the correct denominator.
    """
    positives = [r for r in results if r.scores.get(IS_DISTRACTOR, 0.0) == 0.0]
    distractors = [r for r in results if r.scores.get(IS_DISTRACTOR, 0.0) == 1.0]

    correct_tool = _mean([r.scores[TOOL_CORRECT] for r in positives if TOOL_CORRECT in r.scores])
    exact_match = _mean([r.scores[EXACT_MATCH] for r in positives if EXACT_MATCH in r.scores])
    arg_scored = [r.scores[ARG_ACCURACY] for r in positives if ARG_ACCURACY in r.scores]
    argument_accuracy = _mean(arg_scored)
    hallucinated = _mean([r.scores[HALLUCINATED] for r in results if HALLUCINATED in r.scores])
    refused = _mean([r.scores[REFUSED] for r in distractors if REFUSED in r.scores])

    localizable_total = sum(r.scores.get(LOCALIZABLE_CORRECT, 0.0) for r in positives)
    localized_total = sum(r.scores.get(LOCALIZED_CORRECT, 0.0) for r in positives)
    localization_rate = localized_total / localizable_total if localizable_total else None

    return ToolCallMetrics(
        n_positive=len(positives),
        n_distractor=len(distractors),
        n_arg_scored=len(arg_scored),
        n_parse_errors=sum(1 for r in results if r.parse_error is not None),
        correct_tool_rate=correct_tool,
        argument_accuracy=argument_accuracy,
        exact_match_rate=exact_match,
        hallucinated_tool_rate=hallucinated,
        refusal_on_distractor_rate=refused,
        localization_rate=localization_rate,
    )
