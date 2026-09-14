"""
LLM Provider abstraction and Ollama implementation.

Provides protocol-driven LLM generation with native support for local models
such as Qwen3 8B, handling chat formatting, reasoning/thinking trace extraction,
token usage tracking, and robust error management.
"""
from __future__ import annotations

import logging
import re
import time
from typing import Any, Dict, List, Optional, Protocol, Sequence, runtime_checkable
import httpx

from apps.agent.models import ChatMessage, LLMResponse

logger = logging.getLogger(__name__)


# =====================================================================
# Exception Hierarchy
# =====================================================================

class LLMError(Exception):
    """Base exception for all LLM errors."""
    pass


class LLMConnectionError(LLMError):
    """Raised when communication with the LLM server fails."""
    pass


class LLMTimeoutError(LLMConnectionError):
    """Raised when an LLM request times out."""
    pass


class LLMModelNotFoundError(LLMError):
    """Raised when the requested model is not found on the server."""
    pass


# =====================================================================
# Protocol Definition
# =====================================================================

@runtime_checkable
class LLMProvider(Protocol):
    """Protocol defining the interface for LLM generation providers."""

    @property
    def model_name(self) -> str:
        """The identifier of the underlying model (e.g. 'qwen3:8b')."""
        ...

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a completion for a prompt."""
        ...

    def chat(
        self,
        messages: Sequence[ChatMessage],
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate an assistant completion given a sequence of ChatMessage objects."""
        ...


# =====================================================================
# Concrete Ollama Implementation (Qwen3 8B)
# =====================================================================

class OllamaLLM:
    """
    LLM Provider connected to a local or remote Ollama server.

    Optimized for Qwen3 8B reasoning models:
    - Connects to `/api/chat` for native role-based conversation turns.
    - Extracts `thinking` traces from both Ollama's `thinking` field and inline `<think>` tags.
    - Extracts token usage (`prompt_eval_count`, `eval_count`) and total duration.
    - Supports custom `httpx.Client` injection for offline testing.
    """

    THINK_TAG_REGEX = re.compile(r"<think>(.*?)</think>", re.DOTALL)

    def __init__(
        self,
        model_name: str = "qwen3:8b",
        base_url: str = "http://localhost:11434",
        timeout: float = 120.0,
        client: Optional[httpx.Client] = None,
    ) -> None:
        self._model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client = client

    @property
    def model_name(self) -> str:
        return self._model_name

    def _get_client(self) -> httpx.Client:
        """Return injected client or instantiate a temporary one."""
        if self._client is not None:
            return self._client
        return httpx.Client(base_url=self.base_url, timeout=self.timeout)

    def chat(
        self,
        messages: Sequence[ChatMessage],
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Send a conversation turn to Ollama /api/chat.

        Args:
            messages: Sequence of ChatMessage objects.
            temperature: Sampling temperature (0.0 for deterministic answers).
            max_tokens: Maximum number of tokens to predict.
            kwargs: Additional generation options passed to Ollama.

        Returns:
            LLMResponse containing synthesized content, thinking trace, and usage metrics.
        """
        if not messages:
            raise ValueError("Messages list cannot be empty.")

        options: Dict[str, Any] = {"temperature": float(temperature)}
        if max_tokens is not None:
            options["num_predict"] = int(max_tokens)
        options.update(kwargs)

        msg_payload = [{"role": m.role, "content": m.content} for m in messages]
        payload = {
            "model": self._model_name,
            "messages": msg_payload,
            "stream": False,
            "options": options,
        }

        client = self._get_client()
        should_close = self._client is None
        start_time = time.perf_counter()

        try:
            url = f"{self.base_url}/api/chat"
            response = client.post(url, json=payload)
        except (httpx.ConnectError, httpx.NetworkError) as e:
            raise LLMConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. Is Ollama running? Error: {e}"
            ) from e
        except httpx.TimeoutException as e:
            raise LLMTimeoutError(
                f"Timeout ({self.timeout}s) waiting for response from Ollama at {self.base_url}."
            ) from e
        except httpx.RequestError as e:
            raise LLMConnectionError(f"HTTP request to Ollama failed: {e}") from e
        finally:
            if should_close:
                client.close()

        elapsed_seconds = round(time.perf_counter() - start_time, 4)

        if response.status_code == 404:
            err_text = response.text.lower()
            if "not found" in err_text or "model" in err_text:
                raise LLMModelNotFoundError(
                    f"Model '{self._model_name}' not found on Ollama server at {self.base_url}. "
                    f"Run `ollama pull {self._model_name}` to install it."
                )
            raise LLMError(f"Ollama endpoint not found: {response.text}")
        elif response.status_code != 200:
            raise LLMError(
                f"Ollama server returned HTTP {response.status_code}: {response.text}"
            )

        try:
            data = response.json()
        except Exception as e:
            raise LLMError(f"Failed to parse JSON response from Ollama: {e}") from e

        # Extract message content
        msg = data.get("message", {})
        raw_content = msg.get("content", "")
        thinking = msg.get("thinking")

        # If thinking is embedded inline via <think> tags (e.g. in some Ollama versions)
        if not thinking and "<think>" in raw_content:
            match = self.THINK_TAG_REGEX.search(raw_content)
            if match:
                thinking = match.group(1).strip()
                raw_content = self.THINK_TAG_REGEX.sub("", raw_content).strip()

        # Extract token usage and duration metrics
        prompt_tokens = data.get("prompt_eval_count", 0)
        completion_tokens = data.get("eval_count", 0)
        total_duration_ns = data.get("total_duration")
        if total_duration_ns is not None:
            total_duration_sec = round(total_duration_ns / 1e9, 4)
        else:
            total_duration_sec = elapsed_seconds

        return LLMResponse(
            content=raw_content.strip(),
            thinking=thinking.strip() if thinking else None,
            model=data.get("model", self._model_name),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            duration_seconds=total_duration_sec,
            raw_response=data,
        )

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Generate completion for a single prompt string.

        Args:
            prompt: User prompt string.
            system_prompt: Optional system prompt to guide response.
            temperature: Sampling temperature.
            kwargs: Additional options.
        """
        messages: List[ChatMessage] = []
        if system_prompt and system_prompt.strip():
            messages.append(ChatMessage(role="system", content=system_prompt.strip()))
        messages.append(ChatMessage(role="user", content=prompt.strip()))
        return self.chat(messages=messages, temperature=temperature, **kwargs)
