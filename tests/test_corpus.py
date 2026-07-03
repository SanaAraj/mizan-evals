"""Tests for the corpus loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from mizan.corpus import CorpusError, load_documents

VALID = '{"id": "d1", "title": "الرياض", "text": "عاصمة السعودية."}'


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "corpus.jsonl"
    path.write_text(text, encoding="utf-8")
    return path


def test_loads_valid_corpus(tmp_path: Path) -> None:
    docs = load_documents(_write(tmp_path, f"{VALID}\n"))
    assert len(docs) == 1
    assert docs[0].id == "d1"
    assert docs[0].title == "الرياض"


def test_missing_file_raises() -> None:
    with pytest.raises(CorpusError, match="not found"):
        load_documents("nope.jsonl")


def test_empty_file_raises(tmp_path: Path) -> None:
    with pytest.raises(CorpusError, match="empty"):
        load_documents(_write(tmp_path, "\n\n"))


def test_invalid_json_reports_line(tmp_path: Path) -> None:
    with pytest.raises(CorpusError, match=":2: invalid JSON"):
        load_documents(_write(tmp_path, f"{VALID}\n{{bad\n"))


def test_missing_field_reports_line(tmp_path: Path) -> None:
    with pytest.raises(CorpusError, match=":1: invalid document"):
        load_documents(_write(tmp_path, '{"id": "d1", "title": "t"}'))


def test_duplicate_ids_rejected(tmp_path: Path) -> None:
    with pytest.raises(CorpusError, match="duplicate document id"):
        load_documents(_write(tmp_path, f"{VALID}\n{VALID}\n"))
