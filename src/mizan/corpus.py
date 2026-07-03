"""The document corpus for retrieval evaluation.

A corpus is a JSONL file of :class:`Document` records. Document ids follow the
same ``arwiki:<title>#<chunk>`` convention used by the evaluation items' gold
labels, so a retriever's output can be scored directly against them.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class CorpusError(ValueError):
    """Raised when a corpus file cannot be parsed or fails validation."""


class Document(BaseModel):
    """One retrievable document (or document chunk)."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    text: str = Field(min_length=1)
    url: str | None = None


def load_documents(path: str | Path) -> list[Document]:
    """Load and validate a corpus from a JSONL file.

    Raises:
        CorpusError: if the file is missing, a line is malformed, a document
            fails validation, ids are duplicated, or the corpus is empty.
    """
    path = Path(path)
    if not path.is_file():
        raise CorpusError(f"corpus file not found: {path}")

    documents: list[Document] = []
    seen_ids: set[str] = set()
    with path.open(encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CorpusError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
            try:
                document = Document.model_validate(record)
            except ValidationError as exc:
                raise CorpusError(f"{path}:{lineno}: invalid document: {exc}") from exc
            if document.id in seen_ids:
                raise CorpusError(f"{path}:{lineno}: duplicate document id {document.id!r}")
            seen_ids.add(document.id)
            documents.append(document)

    if not documents:
        raise CorpusError(f"corpus file is empty: {path}")
    return documents
