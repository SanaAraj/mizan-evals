"""Construct a backend from a :class:`ModelSpec`, reading keys from the environment.

Keys never live in configs: a spec names the environment variable to read
(``api_key_env``), and this factory resolves it at construction time.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

from mizan.config import ModelSpec
from mizan.llm.base import Backend, BackendError
from mizan.llm.huggingface import HuggingFaceBackend
from mizan.llm.mock import MockBackend
from mizan.llm.openai_compatible import OpenAICompatibleBackend

DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"


def build_backend(spec: ModelSpec, env: Mapping[str, str] | None = None) -> Backend:
    """Build the backend named by ``spec.backend``.

    Args:
        spec: the model specification from the run config.
        env: environment mapping to read keys from (defaults to ``os.environ``).

    Raises:
        BackendError: for an unknown backend or a missing required API key.
    """
    env = os.environ if env is None else env

    if spec.backend == "mock":
        return MockBackend(model_id=spec.id)

    if spec.backend == "openai":
        base_url = spec.base_url or env.get("OPENAI_BASE_URL") or DEFAULT_OPENAI_BASE_URL
        key_var = spec.api_key_env or "OPENAI_API_KEY"
        api_key = env.get(key_var)
        if not api_key:
            raise BackendError(
                f"model {spec.id!r}: OpenAI-compatible backend needs an API key; "
                f"set the {key_var} environment variable"
            )
        return OpenAICompatibleBackend(spec.id, base_url=base_url, api_key=api_key)

    if spec.backend == "hf":
        key_var = spec.api_key_env or "HF_TOKEN"
        token = env.get(key_var)
        return HuggingFaceBackend(spec.id, token=token, revision=spec.revision)

    raise BackendError(
        f"model {spec.id!r}: unknown backend {spec.backend!r} (expected 'mock', 'openai', or 'hf')"
    )
