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

    ``backend`` selects the client implementation: ``mock`` (offline), ``openai``
    (any OpenAI-compatible ``/chat/completions`` endpoint), or ``hf`` (a model
    served through the Hugging Face client).

    ``id`` should be the provider's exact, ideally dated/pinned model id (e.g.
    ``gpt-5.5-2026-04-23`` or ``Qwen/Qwen2.5-7B-Instruct``); it is recorded in run
    metadata. ``base_url`` overrides the endpoint for OpenAI-compatible providers.
    ``api_key_env`` names the environment variable holding the API key (so keys
    never live in the config). ``revision`` pins a Hugging Face commit/tag for
    reproducibility. ``params`` are decoding params passed to the backend.

    ``tool_mode`` selects how tool calls are extracted: ``native`` uses the
    backend's function-calling API, ``prompt`` uses the strict JSON prompt-based
    fallback, and ``auto`` (the default) picks native when the backend supports it
    and prompt otherwise.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    backend: str = "mock"
    base_url: str | None = None
    api_key_env: str | None = None
    revision: str | None = None
    tool_mode: str = "auto"
    params: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_tool_mode(self) -> ModelSpec:
        allowed = {"auto", "native", "prompt"}
        if self.tool_mode not in allowed:
            raise ValueError(
                f"model {self.id!r}: tool_mode must be one of {sorted(allowed)}, "
                f"got {self.tool_mode!r}"
            )
        return self


class RetrieverSpec(BaseModel):
    """One retrieval system under test.

    ``type`` selects the implementation (currently ``bm25``; dense retrieval is a
    later milestone). ``id`` labels the system in the results table.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    type: str = "bm25"


class RunConfig(BaseModel):
    """A validated evaluation run configuration."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    dataset: str
    models: list[ModelSpec] = Field(min_length=1)
    tasks: list[TaskType] = Field(min_length=1)
    languages: list[Language] = Field(default_factory=lambda: list(Language))
    retrieval_k: list[int] = Field(default_factory=lambda: [1, 3, 5, 10])
    corpus: str | None = None
    retrievers: list[RetrieverSpec] = Field(default_factory=list)
    seed: int = 0
    output_dir: str = "runs"

    @model_validator(mode="after")
    def _check_consistency(self) -> RunConfig:
        if len(self.model_ids) != len(set(self.model_ids)):
            raise ValueError("model ids must be unique within a run")
        if len(self.retriever_ids) != len(set(self.retriever_ids)):
            raise ValueError("retriever ids must be unique within a run")
        if any(k <= 0 for k in self.retrieval_k):
            raise ValueError("retrieval_k values must be positive integers")
        if self.retrievers and not self.corpus:
            raise ValueError("a corpus is required when retrievers are configured")
        return self

    @property
    def model_ids(self) -> list[str]:
        """The ids of every model under test, in configured order."""
        return [m.id for m in self.models]

    @property
    def retriever_ids(self) -> list[str]:
        """The ids of every retriever under test, in configured order."""
        return [r.id for r in self.retrievers]


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
