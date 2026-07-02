"""Loading and validating evaluation items from JSONL files.

Items are stored one JSON object per line (``.jsonl``) so that a dataset can be
diffed and reviewed line by line - which matters because native-speaker review
of the Arabic content is part of the methodology.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from pydantic import ValidationError

from mizan.schemas import EvalItem


class DatasetError(ValueError):
    """Raised when a dataset file cannot be parsed or fails validation."""


def iter_items(path: str | Path) -> Iterator[EvalItem]:
    """Yield validated :class:`EvalItem` objects from a JSONL file.

    Blank lines are skipped. The line number is included in error messages so a
    reviewer can jump straight to the offending item.

    Raises:
        DatasetError: if the file is missing, a line is not valid JSON, an item
            fails schema validation, or two items share an ``id``.
    """
    path = Path(path)
    if not path.is_file():
        raise DatasetError(f"dataset file not found: {path}")

    seen_ids: set[str] = set()
    with path.open(encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
            try:
                item = EvalItem.model_validate(record)
            except ValidationError as exc:
                raise DatasetError(f"{path}:{lineno}: invalid evaluation item: {exc}") from exc
            if item.id in seen_ids:
                raise DatasetError(f"{path}:{lineno}: duplicate item id {item.id!r}")
            seen_ids.add(item.id)
            yield item


def load_items(path: str | Path) -> list[EvalItem]:
    """Load and validate every evaluation item from a JSONL file.

    Raises:
        DatasetError: if the file is missing/malformed or contains no items.
    """
    items = list(iter_items(path))
    if not items:
        raise DatasetError(f"dataset file is empty: {path}")
    return items
