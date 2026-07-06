"""Generate Markdown results tables from a :class:`RunResult`.

Tables are written into the run folder so every published number is traceable to
a dated run. Both the retrieval and tool-calling tables are generated here.
"""

from __future__ import annotations

from mizan.results import ItemResult, RunResult
from mizan.schemas import TaskType
from mizan.scoring.tool_calling import ToolCallMetrics, aggregate


def _recall_key_order(keys: set[str]) -> list[str]:
    recall = sorted(
        (k for k in keys if k.startswith("recall@")),
        key=lambda k: int(k.split("@", 1)[1]),
    )
    tail = [k for k in ("mrr",) if k in keys]
    return recall + tail


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def retrieval_table(result: RunResult) -> str:
    """Return a Markdown table of mean retrieval metrics per system and language.

    Metrics are averaged over retrieval items. Returns a short placeholder when
    the run contains no retrieval results.
    """
    rows: list[ItemResult] = [
        r for r in result.items if r.task_type is TaskType.RETRIEVAL and r.scores
    ]
    if not rows:
        return "_No retrieval results in this run._"

    metrics = _recall_key_order({key for r in rows for key in r.scores})

    grouped: dict[tuple[str, str], list[ItemResult]] = {}
    for r in rows:
        grouped.setdefault((r.model_id, r.language.value), []).append(r)

    header = "| System | Language | " + " | ".join(_pretty(m) for m in metrics) + " | n |"
    sep = "|" + "---|" * (len(metrics) + 3)
    lines = [header, sep]
    for (system, language), group in grouped.items():
        cells = [f"{_mean([g.scores[m] for g in group]):.3f}" for m in metrics]
        lines.append(f"| {system} | {language} | " + " | ".join(cells) + f" | {len(group)} |")
    return "\n".join(lines)


def _pretty(metric: str) -> str:
    return "MRR" if metric == "mrr" else metric


def _fmt(value: float | None) -> str:
    """Format a rate as a fixed-width number, or ``-`` when unmeasured."""
    return "-" if value is None else f"{value:.3f}"


# Column header -> attribute on ToolCallMetrics, in table order.
_TOOL_COLUMNS: list[tuple[str, str]] = [
    ("correct-tool", "correct_tool_rate"),
    ("arg-acc", "argument_accuracy"),
    ("exact-match", "exact_match_rate"),
    ("halluc-tool", "hallucinated_tool_rate"),
    ("refusal", "refusal_on_distractor_rate"),
    ("localization", "localization_rate"),
]


def tool_calling_table(result: RunResult) -> str:
    """Return a Markdown table of tool-calling metrics per model and language.

    One row per (model, language); columns are the five headline metrics plus the
    strict exact-match rate. ``pos`` and ``dist`` count the positive and distractor
    items scored; ``parse-err`` counts extraction failures. Returns a short
    placeholder when the run has no tool-calling results.
    """
    rows = [r for r in result.items if r.task_type is TaskType.TOOL_CALLING]
    if not rows:
        return "_No tool-calling results in this run._"

    grouped: dict[tuple[str, str], list[ItemResult]] = {}
    for r in rows:
        grouped.setdefault((r.model_id, r.language.value), []).append(r)

    headers = [name for name, _ in _TOOL_COLUMNS]
    header = "| System | Language | " + " | ".join(headers) + " | pos | dist | parse-err |"
    sep = "|" + "---|" * (len(headers) + 5)
    lines = [header, sep]
    for (system, language), group in grouped.items():
        metrics: ToolCallMetrics = aggregate(group)
        cells = [_fmt(getattr(metrics, attr)) for _, attr in _TOOL_COLUMNS]
        lines.append(
            f"| {system} | {language} | "
            + " | ".join(cells)
            + f" | {metrics.n_positive} | {metrics.n_distractor} | {metrics.n_parse_errors} |"
        )
    return "\n".join(lines)
