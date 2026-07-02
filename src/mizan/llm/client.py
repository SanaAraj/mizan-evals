"""A disk-cached, resumable wrapper around any :class:`Backend`.

Every generation is written to a JSON file keyed by a hash of the model id,
prompt and decoding params. Re-running an interrupted evaluation therefore skips
work that already completed and never re-charges an API - the core requirement
for reproducible, resumable LLM runs.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from mizan.llm.base import Backend, GenerationParams, LLMResponse


class CachedLLMClient:
    """Wrap a backend with an on-disk cache.

    Args:
        backend: the underlying model backend.
        cache_dir: directory for cache files; created if missing.
    """

    def __init__(self, backend: Backend, cache_dir: str | Path) -> None:
        self.backend = backend
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_key(self, prompt: str, params: GenerationParams) -> str:
        payload = json.dumps(
            {
                "model_id": self.backend.model_id,
                "prompt": prompt,
                "temperature": params.temperature,
                "max_tokens": params.max_tokens,
                "seed": params.seed,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def generate(
        self,
        prompt: str,
        params: GenerationParams | None = None,
    ) -> LLMResponse:
        """Return a completion for ``prompt``, serving from cache when possible.

        A corrupt cache entry is treated as a miss and transparently rewritten,
        so a partially-written file from an interrupted run cannot wedge a rerun.
        """
        params = params or GenerationParams()
        key = self._cache_key(prompt, params)
        path = self._cache_path(key)

        cached_text = self._read_cache(path)
        if cached_text is not None:
            return LLMResponse(text=cached_text, model_id=self.backend.model_id, cached=True)

        text = self.backend.generate(prompt, params)
        self._write_cache(path, prompt, params, text)
        return LLMResponse(text=text, model_id=self.backend.model_id, cached=False)

    def _read_cache(self, path: Path) -> str | None:
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return str(data["text"])
        except (json.JSONDecodeError, KeyError, OSError):
            return None

    def _write_cache(
        self,
        path: Path,
        prompt: str,
        params: GenerationParams,
        text: str,
    ) -> None:
        record = {
            "model_id": self.backend.model_id,
            "prompt": prompt,
            "params": params.model_dump(),
            "text": text,
        }
        # Write to a temp file in the same directory and atomically rename, so a
        # crash mid-write can never leave a half-written cache entry behind.
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, path)
