"""A deterministic mock backend for tests and offline development.

The mock never calls a network service, so the whole harness is runnable and
testable without API keys. Plain-text output is a deterministic function of the
model id, prompt and decoding params, which lets tests assert on exact strings
and lets the cache be exercised meaningfully.

The mock also implements both tool-calling paths so the full tool-calling
pipeline can be run end-to-end offline (Phase 3B): it picks a tool
deterministically from the utterance. The choice is arbitrary - the mock is not a
system under test and its accuracy is meaningless - but it is stable, which is all
the pipeline-structure check needs.
"""

from __future__ import annotations

import hashlib
import re

from mizan.llm.base import Backend, GenerationParams
from mizan.tools.extract import to_canonical_json

# Matches the "1. tool_name - description" lines rendered by build_tool_prompt,
# so the prompt-mode mock can recover the offered tool names.
_PROMPT_TOOL_LINE = re.compile(r"^\s*\d+\.\s+([a-z_]+)\s+-", re.MULTILINE)


class MockBackend(Backend):
    """A backend that returns canned or deterministically-synthesised output.

    Args:
        model_id: identifier reported to the client and used in cache keys.
        scripted: optional exact ``prompt -> response`` overrides, useful for
            asserting on realistic content in tests.
        native_tools: whether this mock advertises native tool calling. Set
            ``False`` to drive the prompt-based fallback path in an offline run.

    The ``calls`` counter records how many times :meth:`generate` actually ran,
    so tests can prove that a cache hit avoided invoking the backend.
    """

    def __init__(
        self,
        model_id: str = "mock-1",
        scripted: dict[str, str] | None = None,
        native_tools: bool = True,
    ) -> None:
        self._model_id = model_id
        self._scripted = dict(scripted or {})
        self._native_tools = native_tools
        self.calls = 0

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def supports_native_tools(self) -> bool:
        return self._native_tools

    def _digest_int(self, text: str) -> int:
        return int(hashlib.sha256(f"{self._model_id}\x00{text}".encode()).hexdigest(), 16)

    def generate(self, prompt: str, params: GenerationParams) -> str:
        self.calls += 1
        if prompt in self._scripted:
            return self._scripted[prompt]
        # A tool-calling prompt gets a deterministic, parseable JSON decision so
        # the prompt-fallback path yields a complete offline results column.
        tool_names = _PROMPT_TOOL_LINE.findall(prompt)
        if tool_names and "JSON:" in prompt:
            name, arguments = self._pick_tool(prompt, tool_names)
            return to_canonical_json(name, arguments)
        digest = hashlib.sha256(
            f"{self._model_id}\x00{prompt}\x00{params.temperature}\x00{params.seed}".encode()
        ).hexdigest()[:16]
        return f"[mock:{self._model_id}] {digest}"

    def generate_tool_call(
        self, utterance: str, tools: list[dict], params: GenerationParams
    ) -> str:
        self.calls += 1
        names = [t["function"]["name"] for t in tools]
        required: dict[str, list[str]] = {
            t["function"]["name"]: t["function"]["parameters"].get("required", []) for t in tools
        }
        name, _ = self._pick_tool(utterance, names)
        arguments = {arg: "mock" for arg in required.get(name or "", [])} if name else {}
        return to_canonical_json(name, arguments)

    def _pick_tool(self, seed_text: str, names: list[str]) -> tuple[str | None, dict[str, str]]:
        """Deterministically pick one offered tool, or a no-call, from ``seed_text``."""
        if not names:
            return None, {}
        h = self._digest_int(seed_text)
        # Roughly one in seven utterances is answered with a no-call, so the
        # offline table exercises the refusal/no-call branch too.
        if h % 7 == 0:
            return None, {}
        return names[h % len(names)], {}
