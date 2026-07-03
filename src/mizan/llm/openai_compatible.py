"""Backend for any OpenAI-compatible ``/chat/completions`` endpoint.

This one class serves the frontier GPT system-under-test on ``api.openai.com`` and
any open model exposed through an OpenAI-compatible provider (e.g. Qwen on
DeepInfra) - they differ only by ``base_url``, API key, and model id.
"""

from __future__ import annotations

import httpx

from mizan.llm.base import Backend, BackendError, GenerationParams


class OpenAICompatibleBackend(Backend):
    """Call a chat-completions endpoint over HTTP.

    Args:
        model_id: exact provider model id (recorded in run metadata).
        base_url: endpoint root, e.g. ``https://api.openai.com/v1``.
        api_key: bearer token for the provider.
        client: optional pre-built ``httpx.Client`` (inject a mock transport in
            tests); a default client is created when omitted.
        timeout: request timeout in seconds for the default client.
    """

    def __init__(
        self,
        model_id: str,
        *,
        base_url: str,
        api_key: str,
        client: httpx.Client | None = None,
        timeout: float = 60.0,
    ) -> None:
        self._model_id = model_id
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._client = client or httpx.Client(timeout=timeout)

    @property
    def model_id(self) -> str:
        return self._model_id

    def generate(self, prompt: str, params: GenerationParams) -> str:
        payload: dict[str, object] = {
            "model": self._model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": params.temperature,
            "max_tokens": params.max_tokens,
        }
        if params.seed is not None:
            payload["seed"] = params.seed

        try:
            response = self._client.post(
                f"{self._base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json=payload,
            )
        except httpx.HTTPError as exc:
            raise BackendError(f"{self._model_id}: request failed: {exc}") from exc

        if response.status_code != 200:
            raise BackendError(
                f"{self._model_id}: HTTP {response.status_code} from {self._base_url}: "
                f"{response.text[:500]}"
            )

        return self._extract_text(response.json())

    def _extract_text(self, body: dict) -> str:
        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise BackendError(f"{self._model_id}: unexpected response shape: {body!r}") from exc
        if content is None:
            raise BackendError(f"{self._model_id}: response contained no content")
        return str(content)
