# pytorch-researcher/pytorch_researcher/src/pytorch_tools/llm.py
"""LLM client abstractions for the PyTorch Researcher tools.

This module provides a small, dependency-light abstraction for interacting with
an LLM. The goal is to allow tools to depend on a minimal interface and enable
easy mocking in tests. The default concrete implementation performs direct
HTTP POSTs to a "<base_url>/chat/completions" endpoint using the Python
standard library (urllib) so there are no extra runtime dependencies.

Public classes:
- BaseLLMClient: abstract interface used by tools.
- HTTPLLMClient: default HTTP-only client used for local/testing LLM endpoints.
- DisabledLLMClient: a sentinel client that raises on use (useful when LLM usage
  is intentionally disabled).
- LLMClientError: exception type raised on client errors.

Design notes:
- Return shape: the `call` method returns a dict with a "raw" key containing the
  parsed JSON response (matching the shape expected by existing tools).
- Retries: HTTPLLMClient implements a simple exponential backoff retry loop.
- Timeouts and temperature are supported as parameters on `call`.
"""

from __future__ import annotations

import logging
import time
from typing import Any

try:
    from litellm import completion
    from litellm.exceptions import APIError
except ImportError as e:
    raise ImportError(f"Required dependency 'litellm' not found: {e}") from e

logger = logging.getLogger(__name__)


class LLMClientError(Exception):
    """Raised for LLM client errors (network, parse, etc.)."""


class BaseLLMClient:
    """Minimal LLM client interface used by assembler and other tools.

    Implementations must provide the `call` method below.

    Method:
        call(prompt: str, temperature: float = 0.0, timeout: int = 300) -> Dict[str, Any]
            Execute the LLM call and return a dictionary with at least a "raw"
            key containing the parsed response object.
    """

    def call(
        self, prompt: str, temperature: float = 0.0, timeout: int = 300
    ) -> dict[str, Any]:
        raise NotImplementedError(
            "BaseLLMClient.call must be implemented by subclasses"
        )


class DisabledLLMClient(BaseLLMClient):
    """A sentinel client that signals LLM usage is disabled.

    Calling `call` will raise `LLMClientError`. Tools can check for this if they
    need to provide alternative behavior.
    """

    def call(
        self, prompt: str, temperature: float = 0.0, timeout: int = 300
    ) -> dict[str, Any]:
        raise LLMClientError("LLM usage is disabled (DisabledLLMClient was provided)")


class LiteLLMClient(BaseLLMClient):
    """LLM client using LiteLLM for unified provider interface.

    This client uses LiteLLM to provide a unified interface for different LLM providers
    while maintaining compatibility with the existing API. It supports HTTP-only providers
    like Ollama, OpenAI, Anthropic, and any other provider supported by LiteLLM.

    Parameters
    ----------
    base_url: str
        Base URL of the LLM HTTP API (e.g., "http://localhost:11434/v1").
    model_name: str
        Model name to request (e.g., "gpt-oss:20b", "openai/gpt-4").
    api_key: Optional[str]
        Optional API key for the LLM provider.
    max_retries: int
        Number of retries on transient failures (default: 2).
    retry_backoff: float
        Base backoff seconds used for exponential backoff (default: 1.0).

    """

    def __init__(
        self,
        base_url: str,
        model_name: str = "gpt-5.1-mini",
        api_key: str | None = None,
        max_retries: int = 2,
        retry_backoff: float = 1.0,
    ) -> None:
        if not base_url:
            raise ValueError("base_url must be provided for LiteLLMClient")
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.api_key = api_key
        self.max_retries = int(max_retries)
        self.retry_backoff = float(retry_backoff)

    def call(
        self, prompt: str, temperature: float = 0.0, timeout: int = 300
    ) -> dict[str, Any]:
        """Perform an LLM call using LiteLLM and return {"raw": parsed_response}.

        This method uses LiteLLM to provide a unified interface for different LLM providers
        while maintaining the existing API contract. Retries on transient errors using
        exponential backoff. Raises LLMClientError on fatal errors.
        """
        # Prepare messages in the format expected by LiteLLM
        messages = [
            {
                "role": "system",
                "content": "You are an expert assistant that writes runnable PyTorch nn.Module code.",
            },
            {"role": "user", "content": prompt},
        ]

        # Prepare LiteLLM completion call parameters
        model_name = self.model_name
        completion_kwargs = {
            "model": model_name,
            "messages": messages,
            "temperature": float(temperature),
        }

        # Configure API base URL and API key for LiteLLM based on endpoint
        if self.base_url:
            # Use 'api_base' instead of 'base_url' for LiteLLM compatibility
            completion_kwargs["api_base"] = self.base_url

            # Enhanced logging for debugging
            logger.info(f"🚀 LiteLLM CALL START - Model: {model_name}")
            logger.info(f"📝 Base URL: {self.base_url}")
            logger.info(f"🔑 Using API Key: {'Yes' if self.api_key else 'No'}")

            # Special handling for different providers
            if "openrouter.ai" in self.base_url.lower():
                completion_kwargs["custom_llm_provider"] = "openai"
                logger.info("Using OpenRouter with provider: openai")
                if self.api_key:
                    completion_kwargs["api_key"] = self.api_key
            elif "localhost" in self.base_url or "127.0.0.1" in self.base_url:
                # For local endpoints like Ollama - add openai/ prefix for compatibility
                if "openai/" not in model_name:
                    model_name = f"openai/{model_name}"
                    completion_kwargs["model"] = model_name
                    logger.info(f"Added openai prefix to local model: {model_name}")
                completion_kwargs["custom_llm_provider"] = "openai"
                if self.api_key:
                    completion_kwargs["api_key"] = self.api_key
                else:
                    completion_kwargs["api_key"] = "local"
                logger.info("Using local endpoint with provider: openai")
            else:
                # For other endpoints (OpenAI, etc.) - respect exact model name
                completion_kwargs["custom_llm_provider"] = "openai"
                logger.info("Using custom endpoint with provider: openai")
                if self.api_key:
                    completion_kwargs["api_key"] = self.api_key

        # Handle timeout - LiteLLM uses timeout parameter directly
        if timeout:
            completion_kwargs["timeout"] = timeout

        # Handle retries - LiteLLM supports num_retries parameter
        if self.max_retries:
            completion_kwargs["num_retries"] = self.max_retries

        last_exc: Exception | None = None
        for attempt in range(1 + self.max_retries):
            try:
                # Use LiteLLM completion function
                logger.debug(f"Calling LiteLLM with model: {model_name}, timeout: {timeout}")
                response = completion(**completion_kwargs)

                # Log the raw response for debugging
                logger.debug(f"Raw LiteLLM response: {response}")

                # Extract the response content following the existing pattern
                if hasattr(response, 'choices') and response.choices:
                    choice = response.choices[0]
                    if hasattr(choice, 'message') and choice.message:
                        content = choice.message.content
                    else:
                        content = str(choice)
                else:
                    # Fallback for other response formats
                    content = str(response)

                logger.debug(f"Extracted content (length {len(content) if content else 0}): {content[:500]!r}...")

                # Return in the format expected by the existing code
                parsed_response = {"content": content}
                if hasattr(response, 'choices'):
                    parsed_response["choices"] = [
                        {"message": {"content": content}}
                    ]

                return {"raw": parsed_response}

            except APIError as exc:
                # LiteLLM-specific errors
                raise LLMClientError(f"LiteLLM API error: {exc}") from exc
            except Exception as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    wait = self.retry_backoff * (2**attempt)
                    logger.debug(
                        "LLM call failed (attempt %d/%d): %s; retrying in %.1fs",
                        attempt + 1,
                        self.max_retries + 1,
                        exc,
                        wait,
                    )
                    time.sleep(wait)
                    continue
                break

        raise LLMClientError(f"LLM request failed after {self.max_retries + 1} attempts: {last_exc}")


__all__ = [
    "BaseLLMClient",
    "DisabledLLMClient",
    "LLMClientError",
    "LiteLLMClient",
]
