"""Run configuration loaded and validated from YAML.

A config describes *what* to evaluate: which models, over which dataset, on
which tasks and languages. It does not perform any work itself - the CLI turns a
validated :class:`RunConfig` into a run folder.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from mizan.schemas import Language, TaskType


class ConfigError(ValueError):
    """Raised when a run configuration is missing or invalid."""


class ModelSpec(BaseModel):
    """One system under test.

    ``backend`` selects the client implementation (e.g. ``mock``; real backends
    such as ``openai`` or ``hf`` arrive with later milestones). ``params`` are
    passed through to the backend at call time.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    backend: str = "mock"
    params: dict[str, Any] = Field(default_factory=dict)


class RunConfig(BaseModel):
    """A validated evaluation run configuration."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    dataset: str
    models: list[ModelSpec] = Field(min_length=1)
    tasks: list[TaskType] = Field(min_length=1)
    languages: list[Language] = Field(default_factory=lambda: list(Language))
    retrieval_k: list[int] = Field(default_factory=lambda: [1, 3, 5, 10])
    seed: int = 0
    output_dir: str = "runs"

    @model_validator(mode="after")
    def _check_consistency(self) -> RunConfig:
        if len(self.model_ids) != len(set(self.model_ids)):
            raise ValueError("model ids must be unique within a run")
        if any(k <= 0 for k in self.retrieval_k):
            raise ValueError("retrieval_k values must be positive integers")
        return self

    @property
    def model_ids(self) -> list[str]:
        """The ids of every model under test, in configured order."""
        return [m.id for m in self.models]


def load_config(path: str | Path) -> RunConfig:
    """Load and validate a run configuration from a YAML file.

    Raises:
        ConfigError: if the file is missing, is not valid YAML, is not a mapping,
            or does not satisfy the :class:`RunConfig` schema.
    """
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: invalid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: top-level YAML must be a mapping, got {type(raw).__name__}")
    try:
        return RunConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"{path}: invalid run configuration: {exc}") from exc
