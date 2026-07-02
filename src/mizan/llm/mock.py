"""A deterministic mock backend for tests and offline development.

The mock never calls a network service, so the whole harness is runnable and
testable without API keys. Output is a deterministic function of the model id,
prompt and decoding params, which lets tests assert on exact strings and lets
the cache be exercised meaningfully.
"""

from __future__ import annotations

import hashlib

from mizan.llm.base import Backend, GenerationParams


class MockBackend(Backend):
    """A backend that returns canned or deterministically-synthesised text.

    Args:
        model_id: identifier reported to the client and used in cache keys.
        scripted: optional exact ``prompt -> response`` overrides, useful for
            asserting on realistic content in tests.

    The ``calls`` counter records how many times :meth:`generate` actually ran,
    so tests can prove that a cache hit avoided invoking the backend.
    """

    def __init__(self, model_id: str = "mock-1", scripted: dict[str, str] | None = None) -> None:
        self._model_id = model_id
        self._scripted = dict(scripted or {})
        self.calls = 0

    @property
    def model_id(self) -> str:
        return self._model_id

    def generate(self, prompt: str, params: GenerationParams) -> str:
        self.calls += 1
        if prompt in self._scripted:
            return self._scripted[prompt]
        digest = hashlib.sha256(
            f"{self._model_id}\x00{prompt}\x00{params.temperature}\x00{params.seed}".encode()
        ).hexdigest()[:16]
        return f"[mock:{self._model_id}] {digest}"
