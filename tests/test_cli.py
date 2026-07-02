"""Tests for the ``mizan run`` CLI skeleton."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mizan.cli import main
from mizan.results import RunMetadata

CONFIG_TEMPLATE = """
name: cli test run
dataset: {dataset}
models:
  - id: mock-a
  - id: mock-b
tasks:
  - retrieval
languages:
  - en
  - msa
"""


def _write_config(tmp_path: Path, dataset: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(CONFIG_TEMPLATE.format(dataset=dataset), encoding="utf-8")
    return path


def test_run_creates_folder_and_metadata(tmp_path: Path) -> None:
    dataset = tmp_path / "items.jsonl"
    dataset.write_text("", encoding="utf-8")
    config = _write_config(tmp_path, str(dataset))
    out = tmp_path / "runs"

    code = main(["run", "--config", str(config), "--output-dir", str(out)])
    assert code == 0

    run_dirs = list(out.iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    # Run id ends with a slug derived from the config name.
    assert run_dir.name.endswith("-cli-test-run")

    metadata_path = run_dir / "metadata.json"
    assert metadata_path.is_file()
    metadata = RunMetadata.model_validate_json(metadata_path.read_text(encoding="utf-8"))
    assert metadata.model_ids == ["mock-a", "mock-b"]
    assert metadata.name == "cli test run"
    assert metadata.created_at.endswith("+00:00")  # UTC timestamp recorded


def test_run_reports_missing_dataset_but_succeeds(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    config = _write_config(tmp_path, "no/such/dataset.jsonl")
    code = main(["run", "--config", str(config), "--output-dir", str(tmp_path / "runs")])
    assert code == 0
    assert "not found" in capsys.readouterr().out


def test_run_with_invalid_config_returns_error(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    code = main(["run", "--config", str(tmp_path / "missing.yaml")])
    assert code == 2
    assert "error:" in capsys.readouterr().err


def test_output_dir_defaults_to_config_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _write_config(tmp_path, str(tmp_path / "d.jsonl"))
    monkeypatch.chdir(tmp_path)
    code = main(["run", "--config", str(config)])
    assert code == 0
    # Default output_dir is "runs", created relative to the working directory.
    assert (tmp_path / "runs").is_dir()


def test_metadata_json_is_parseable(tmp_path: Path) -> None:
    config = _write_config(tmp_path, str(tmp_path / "d.jsonl"))
    out = tmp_path / "runs"
    main(["run", "--config", str(config), "--output-dir", str(out)])
    metadata_path = next(out.glob("*/metadata.json"))
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert payload["config"]["name"] == "cli test run"
