# pytorch-researcher/pytorch_researcher/src/pytorch_tools/llm.py
"""
LLM client abstractions for the PyTorch Researcher tools.

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

import json
import logging
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class LLMClientError(Exception):
    """Raised for LLM client errors (network, parse, etc.)."""


class BaseLLMClient:
    """
    Minimal LLM client interface used by assembler and other tools.

    Implementations must provide the `call` method below.

    Method:
        call(prompt: str, temperature: float = 0.0, timeout: int = 300) -> Dict[str, Any]
            Execute the LLM call and return a dictionary with at least a "raw"
            key containing the parsed response object.
    """

    def call(
        self, prompt: str, temperature: float = 0.0, timeout: int = 300
    ) -> Dict[str, Any]:
        raise NotImplementedError(
            "BaseLLMClient.call must be implemented by subclasses"
        )


class DisabledLLMClient(BaseLLMClient):
    """
    A sentinel client that signals LLM usage is disabled.

    Calling `call` will raise `LLMClientError`. Tools can check for this if they
    need to provide alternative behavior.
    """

    def call(
        self, prompt: str, temperature: float = 0.0, timeout: int = 300
    ) -> Dict[str, Any]:
        raise LLMClientError("LLM usage is disabled (DisabledLLMClient was provided)")


class HTTPLLMClient(BaseLLMClient):
    """
    Simple HTTP-only LLM client.

    This client performs POST requests to "{base_url}/chat/completions" and
    expects a JSON response. It intentionally uses the standard library so the
    project doesn't require external HTTP dependencies for the MVP.

    Parameters
    ----------
    base_url: str
        Base URL of the LLM HTTP API (e.g., "http://localhost:11434/v1").
    model_name: str
        Model name to request (e.g., "gpt-oss:20b").
    api_key: Optional[str]
        Optional API key to include in the Authorization header as Bearer.
    max_retries: int
        Number of retries on transient failures (default: 2).
    retry_backoff: float
        Base backoff seconds used for exponential backoff (default: 1.0).
    """

    def __init__(
        self,
        base_url: str,
        model_name: str = "gpt-5.1-mini",
        api_key: Optional[str] = None,
        max_retries: int = 2,
        retry_backoff: float = 1.0,
    ) -> None:
        if not base_url:
            raise ValueError("base_url must be provided for HTTPLLMClient")
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.api_key = api_key
        self.max_retries = int(max_retries)
        self.retry_backoff = float(retry_backoff)

    def call(
        self, prompt: str, temperature: float = 0.0, timeout: int = 300
    ) -> Dict[str, Any]:
        """
        Perform an HTTP POST to the LLM endpoint and return {"raw": parsed_json}.

        Retries on transient errors using exponential backoff. Raises
        LLMClientError on fatal errors.
        """
        try:
            import urllib.error as _urlerr  # type: ignore
            import urllib.request as _urlreq  # type: ignore
        except Exception as e:
            raise LLMClientError(f"Missing HTTP support libraries: {e}") from e

        endpoint = self.base_url + "/chat/completions"
        messages = [
            {
                "role": "system",
                "content": "You are an expert assistant that writes runnable PyTorch nn.Module code.",
            },
            {"role": "user", "content": prompt},
        ]
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": float(temperature),
        }
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        last_exc: Optional[Exception] = None
        for attempt in range(1 + self.max_retries):
            try:
                req = _urlreq.Request(
                    endpoint, data=data, headers=headers, method="POST"
                )
                with _urlreq.urlopen(req, timeout=timeout) as resp:
                    body = resp.read().decode("utf-8")
                    parsed = json.loads(body)
                    return {"raw": parsed}
            except _urlerr.HTTPError as he:  # type: ignore
                # Try to include body for diagnostics
                try:
                    err_body = he.read().decode("utf-8")
                except Exception:
                    err_body = "<no body>"
                raise LLMClientError(
                    f"HTTP error from LLM endpoint: {getattr(he, 'code', '')} {getattr(he, 'reason', '')} - {err_body}"
                ) from he
            except Exception as exc:
                last_exc = exc
                wait = self.retry_backoff * (2**attempt)
                logger.debug(
                    "HTTP LLM call failed (attempt %d/%d): %s; retrying in %.1fs",
                    attempt + 1,
                    self.max_retries + 1,
                    exc,
                    wait,
                )
                time.sleep(wait)
                continue

        raise LLMClientError(f"LLM HTTP request failed after retries: {last_exc}")


__all__ = [
    "BaseLLMClient",
    "HTTPLLMClient",
    "DisabledLLMClient",
    "LLMClientError",
]
