"""Backend for models served through the Hugging Face inference client.

Used for open models that are not exposed via an OpenAI-compatible endpoint
(e.g. ALLaM). ``revision`` is recorded for reproducibility; note that serverless
Hugging Face inference tracks the model repo's current revision.
"""

from __future__ import annotations

from typing import Any

from mizan.llm.base import Backend, BackendError, GenerationParams


class HuggingFaceBackend(Backend):
    """Generate text via ``huggingface_hub.InferenceClient.chat_completion``.

    Args:
        model_id: Hugging Face repo id (or a dedicated endpoint URL).
        token: Hugging Face access token.
        revision: commit/tag recorded in run metadata for reproducibility.
        client: optional pre-built client exposing ``chat_completion`` (inject a
            fake in tests); a real ``InferenceClient`` is created when omitted.
        timeout: request timeout in seconds for the default client.
    """

    def __init__(
        self,
        model_id: str,
        *,
        token: str | None = None,
        revision: str | None = None,
        client: Any | None = None,
        timeout: float = 120.0,
    ) -> None:
        self._model_id = model_id
        self._revision = revision
        if client is not None:
            self._client = client
        else:
            from huggingface_hub import InferenceClient

            self._client = InferenceClient(model=model_id, token=token, timeout=timeout)

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def revision(self) -> str | None:
        return self._revision

    def generate(self, prompt: str, params: GenerationParams) -> str:
        kwargs: dict[str, Any] = {
            "messages": [{"role": "user", "content": prompt}],
            "model": self._model_id,
            "max_tokens": params.max_tokens,
        }
        # Some TGI backends reject temperature=0; use greedy decoding instead.
        if params.temperature > 0:
            kwargs["temperature"] = params.temperature
        if params.seed is not None:
            kwargs["seed"] = params.seed

        try:
            response = self._client.chat_completion(**kwargs)
        except Exception as exc:  # noqa: BLE001 - surface any client failure uniformly
            raise BackendError(f"{self._model_id}: inference call failed: {exc}") from exc

        try:
            content = response.choices[0].message.content
        except (AttributeError, IndexError, TypeError) as exc:
            raise BackendError(
                f"{self._model_id}: unexpected response shape: {response!r}"
            ) from exc
        if content is None:
            raise BackendError(f"{self._model_id}: response contained no content")
        return str(content)
