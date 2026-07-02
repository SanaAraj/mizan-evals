"""Backend abstraction and shared value objects for the LLM layer."""

from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel, ConfigDict, Field


class GenerationParams(BaseModel):
    """Decoding parameters that also participate in the cache key.

    Two calls with identical params and prompt to the same model id share a
    cache entry, which is what makes runs resumable.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    temperature: float = Field(default=0.0, ge=0.0)
    max_tokens: int = Field(default=512, gt=0)
    seed: int | None = None


class LLMResponse(BaseModel):
    """A single generation, annotated with where it came from."""

    model_config = ConfigDict(extra="forbid")

    text: str
    model_id: str
    cached: bool = False


class Backend(ABC):
    """A text-generation backend for one model.

    Implementations are intentionally thin: caching, retries and result recording
    live in :class:`~mizan.llm.client.CachedLLMClient`, so a backend only has to
    turn a prompt into text.
    """

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Stable identifier for the underlying model (used in cache keys)."""

    @abstractmethod
    def generate(self, prompt: str, params: GenerationParams) -> str:
        """Return the model's completion for ``prompt`` under ``params``."""
