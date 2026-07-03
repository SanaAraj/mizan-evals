"""LLM client layer: a backend abstraction, concrete backends, and a disk cache."""

from mizan.llm.base import Backend, BackendError, GenerationParams, LLMResponse
from mizan.llm.client import CachedLLMClient
from mizan.llm.factory import build_backend
from mizan.llm.huggingface import HuggingFaceBackend
from mizan.llm.mock import MockBackend
from mizan.llm.openai_compatible import OpenAICompatibleBackend

__all__ = [
    "Backend",
    "BackendError",
    "CachedLLMClient",
    "GenerationParams",
    "HuggingFaceBackend",
    "LLMResponse",
    "MockBackend",
    "OpenAICompatibleBackend",
    "build_backend",
]
