"""Generate Markdown results tables from a :class:`RunResult`.

Tables are written into the run folder so every published number is traceable to
a dated run. Retrieval is supported now; the tool-calling table lands with that
eval.
"""

from __future__ import annotations

from mizan.results import ItemResult, RunResult
from mizan.schemas import TaskType


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
