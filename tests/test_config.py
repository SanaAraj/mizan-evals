"""Tests for loading and validating YAML run configurations."""

from __future__ import annotations

from pathlib import Path

import pytest

from mizan.config import ConfigError, load_config
from mizan.schemas import Language, TaskType

VALID_YAML = """
name: sample-run
dataset: data/samples/eval_items.jsonl
models:
  - id: mock-small
    backend: mock
  - id: mock-large
    backend: mock
    params:
      temperature: 0.0
tasks:
  - retrieval
  - tool_calling
languages:
  - en
  - msa
  - gulf
retrieval_k: [1, 3, 5]
seed: 7
"""


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_loads_valid_config(tmp_path: Path) -> None:
    config = load_config(_write(tmp_path, VALID_YAML))
    assert config.name == "sample-run"
    assert config.model_ids == ["mock-small", "mock-large"]
    assert config.tasks == [TaskType.RETRIEVAL, TaskType.TOOL_CALLING]
    assert config.seed == 7


def test_defaults_are_applied(tmp_path: Path) -> None:
    minimal = """
name: minimal
dataset: data.jsonl
models:
  - id: m1
tasks:
  - retrieval
"""
    config = load_config(_write(tmp_path, minimal))
    assert config.languages == [Language.EN, Language.MSA, Language.GULF]
    assert config.retrieval_k == [1, 3, 5, 10]
    assert config.output_dir == "runs"
    assert config.models[0].backend == "mock"


def test_missing_file_raises() -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config("does/not/exist.yaml")


def test_non_mapping_yaml_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="must be a mapping"):
        load_config(_write(tmp_path, "- just\n- a\n- list\n"))


def test_missing_required_field_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="invalid run configuration"):
        load_config(_write(tmp_path, "name: x\nmodels: []\ntasks: [retrieval]\n"))


def test_unknown_field_is_rejected(tmp_path: Path) -> None:
    bad = VALID_YAML + "unexpected: true\n"
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, bad))


def test_duplicate_model_ids_rejected(tmp_path: Path) -> None:
    dup = """
name: dup
dataset: d.jsonl
models:
  - id: same
  - id: same
tasks:
  - retrieval
"""
    with pytest.raises(ConfigError, match="unique"):
        load_config(_write(tmp_path, dup))
