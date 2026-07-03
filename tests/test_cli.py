"""Tests for the ``mizan run`` CLI."""

from __future__ import annotations

from pathlib import Path

import pytest

from mizan.cli import main
from mizan.results import RunMetadata, RunResult

CONFIG_TEMPLATE = """
name: cli test run
dataset: {dataset}
models:
  - id: mock-a
  - id: mock-b
tasks:
  - answer_quality
languages:
  - en
  - msa
"""

DATASET_LINE = (
    '{"id": "c1", "task_type": "answer_quality", '
    '"variants": {"en": {"query": "q", "reference_answer": "a"}, '
    '"msa": {"query": "\\u0633", "reference_answer": "\\u062c"}}}'
)


def _write_config(tmp_path: Path, dataset: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(CONFIG_TEMPLATE.format(dataset=dataset), encoding="utf-8")
    return path


def _dataset(tmp_path: Path) -> str:
    path = tmp_path / "items.jsonl"
    path.write_text(DATASET_LINE + "\n", encoding="utf-8")
    return str(path)


def _run(tmp_path: Path, config: Path, *extra: str) -> int:
    return main(
        [
            "run",
            "--config",
            str(config),
            "--output-dir",
            str(tmp_path / "runs"),
            "--cache-dir",
            str(tmp_path / "cache"),
            *extra,
        ]
    )


def test_dry_run_creates_metadata_but_no_results(tmp_path: Path) -> None:
    config = _write_config(tmp_path, _dataset(tmp_path))
    assert _run(tmp_path, config, "--dry-run") == 0

    run_dir = next((tmp_path / "runs").iterdir())
    assert run_dir.name.endswith("-cli-test-run")
    metadata = RunMetadata.model_validate_json((run_dir / "metadata.json").read_text("utf-8"))
    assert metadata.model_ids == ["mock-a", "mock-b"]
    assert metadata.created_at.endswith("+00:00")
    assert not (run_dir / "results.json").exists()


def test_full_run_writes_results(tmp_path: Path) -> None:
    config = _write_config(tmp_path, _dataset(tmp_path))
    assert _run(tmp_path, config) == 0

    results_path = next((tmp_path / "runs").glob("*/results.json"))
    result = RunResult.model_validate_json(results_path.read_text("utf-8"))
    # 1 item x 2 languages x 2 models = 4 results, all with output.
    assert result.summary["n_results"] == 4
    assert all(r.output is not None for r in result.items)


def test_full_run_missing_dataset_errors(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    config = _write_config(tmp_path, "no/such/dataset.jsonl")
    assert _run(tmp_path, config) == 2
    assert "error:" in capsys.readouterr().err


def test_invalid_config_returns_error(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    assert (
        main(["run", "--config", str(tmp_path / "missing.yaml"), "--output-dir", str(tmp_path)])
        == 2
    )
    assert "error:" in capsys.readouterr().err


def test_output_dir_defaults_to_config_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _write_config(tmp_path, _dataset(tmp_path))
    monkeypatch.chdir(tmp_path)
    assert main(["run", "--config", str(config), "--dry-run"]) == 0
    assert (tmp_path / "runs").is_dir()
