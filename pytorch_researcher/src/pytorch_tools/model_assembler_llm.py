# pytorch-researcher/pytorch_researcher/src/pytorch_tools/model_assembler_llm.py
"""
Minimal HTTP-only LLM-backed PyTorch Model Assembler (MVP)

This simplified module provides a basic LLM integration that ONLY uses direct
HTTP POST requests to a `<base_url>/chat/completions` endpoint (useful for
local LLMs like Ollama). It intentionally avoids supporting the `openai`
Python SDK or injected clients to keep the MVP surface small and predictable.

Key points:
- Direct HTTP POST to "{base_url}/chat/completions".
- Increased HTTP timeout to 300 seconds to support slow local model generation.
- Expects the LLM to return a JSON-compatible response similar to:
  { "choices": [ { "message": { "content": "<assistant text>" } } ] }
- The assistant text is expected to be a JSON object (string) containing a
  `"python"` key with the generated source. Fallback heuristics attempt to
  extract fenced code blocks or python-like text.
- Validates generated code with ast.parse and requires at least one class def.
- Falls back to the deterministic assembler when LLM generation is disabled or fails.
"""

from __future__ import annotations

import ast
import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional

# LLM client abstractions (factored out for DRYness and easier testing)
from pytorch_researcher.src.pytorch_tools.llm import (
    BaseLLMClient,
    DisabledLLMClient,
    HTTPLLMClient,
    LLMClientError,
)

logger = logging.getLogger(__name__)

# Try to import the programmatic assembler as a fallback (best-effort).
try:
    from pytorch_researcher.src.pytorch_tools.model_assembler import (
        ModelAssemblerError,
        ModelConfig,
        assemble_model_code,
        save_model_code,
    )
except Exception as exc:  # pragma: no cover - defensive import
    assemble_model_code = None  # type: ignore
    save_model_code = None  # type: ignore
    ModelConfig = None  # type: ignore
    ModelAssemblerError = Exception  # type: ignore
    _fallback_import_error = exc
else:
    _fallback_import_error = None

# Simple regex to extract python fenced blocks
_CODE_FENCE_RE = re.compile(
    r"```(?:python)?\s*(?P<code>.*?)```", re.DOTALL | re.IGNORECASE
)


class LLMModelAssemblerError(Exception):
    """Raised for errors in the LLM-backed assembler pipeline."""


class LLMModelAssembler:
    """
    Minimal HTTP-only LLM assembler.

    Parameters
    ----------
    model_name: str
        Name of the LLM model to request (e.g., "gpt-oss:20b").
    base_url: str
        Base URL for the LLM HTTP API (e.g., "http://localhost:11434/v1").
    api_key: Optional[str]
        API key to include in Authorization header (Bearer).
    max_retries: int
        Number of retries for transient errors (exponential backoff).
    retry_backoff: float
        Base backoff seconds used for exponential backoff between retries.
    prompt_template: Optional[str]
        Template used to ask the LLM to produce a structured JSON response.
    """

    def __init__(
        self,
        model_name: str = "gpt-5.1-mini",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        max_retries: int = 2,
        retry_backoff: float = 1.0,
        prompt_template: Optional[str] = None,
        llm_client: Optional[BaseLLMClient] = None,
    ):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/") if base_url else None
        self.api_key = api_key
        self.max_retries = int(max_retries)
        self.retry_backoff = float(retry_backoff)
        self.prompt_template = prompt_template or DEFAULT_PROMPT_TEMPLATE

        # Configure the LLM client with the following preference:
        # 1) an explicit `llm_client` argument (injected),
        # 2) construct a default `HTTPLLMClient` if `base_url` is provided,
        # 3) otherwise, leave `self.llm_client` as None (LLM usage disabled).
        if llm_client is not None:
            self.llm_client = llm_client
        elif self.base_url:
            try:
                self.llm_client = HTTPLLMClient(
                    base_url=self.base_url,
                    model_name=self.model_name,
                    api_key=self.api_key,
                    max_retries=self.max_retries,
                    retry_backoff=self.retry_backoff,
                )
            except Exception as exc:
                logger.warning(
                    "Failed to construct HTTPLLMClient: %s. LLM usage will be disabled.",
                    exc,
                )
                self.llm_client = None
        else:
            self.llm_client = None
            logger.warning(
                "No base_url provided for LLMModelAssembler: HTTP-only assembler requires base_url if no llm_client is supplied."
            )

    def _call_llm(
        self, prompt: str, temperature: float = 0.0, timeout: int = 300
    ) -> Dict[str, Any]:
        """
        Delegate LLM calls to the injected LLM client.

        This method is intentionally thin: it expects `self.llm_client` to be set
        (either via constructor injection or by constructing a default HTTP client).
        Tests can inject a mock client implementing the same `call` method to
        avoid network access.
        """
        if not hasattr(self, "llm_client") or self.llm_client is None:
            raise LLMModelAssemblerError(
                "No LLM client configured for LLMModelAssembler."
            )

        try:
            return self.llm_client.call(
                prompt, temperature=temperature, timeout=timeout
            )
        except Exception as exc:
            raise LLMModelAssemblerError(f"LLM client error: {exc}") from exc

    def _extract_code_from_response(self, raw_response: Dict[str, Any]) -> str:
        """
        Extract Python source code from the LLM response.

        Strategy:
        - Look for typical 'choices' -> first -> 'message' -> 'content' shapes.
        - Attempt to parse content as JSON and prefer 'python' / 'code' keys.
        - Fallback to fenced code block extraction or heuristic detection.
        """
        raw = raw_response.get("raw")
        text: Optional[str] = None

        # Common json shape: { "choices": [ { "message": { "content": "..." } } ] }
        try:
            if isinstance(raw, dict) and "choices" in raw:
                choices = raw.get("choices") or []
                if choices:
                    first = choices[0]
                    # Newer shape has message.content
                    if isinstance(first, dict):
                        msg = first.get("message") or first.get("text") or {}
                        if isinstance(msg, dict):
                            text = msg.get("content") or msg.get("text")
                        elif isinstance(msg, str):
                            text = msg
                    elif isinstance(first, str):
                        text = first
        except Exception:
            text = None

        if not text:
            # last resort stringify
            try:
                text = json.dumps(raw) if not isinstance(raw, str) else raw
            except Exception:
                text = str(raw)

        if not text:
            raise LLMModelAssemblerError(
                "Could not extract assistant text from LLM response."
            )

        text = text.strip()

        # Try parse as JSON object in the assistant text
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                for key in ("python", "python_code", "code", "source"):
                    if (
                        key in parsed
                        and isinstance(parsed[key], str)
                        and parsed[key].strip()
                    ):
                        code_candidate = parsed[key].strip()
                        mf = _CODE_FENCE_RE.search(code_candidate)
                        if mf:
                            return mf.group("code").strip()
                        return code_candidate
                # fallback to content/body
                for key in ("content", "body"):
                    if (
                        key in parsed
                        and isinstance(parsed[key], str)
                        and parsed[key].strip()
                    ):
                        mf = _CODE_FENCE_RE.search(parsed[key])
                        if mf:
                            return mf.group("code").strip()
                        return parsed[key].strip()
        except Exception:
            # not JSON, continue to other heuristics
            pass

        # Fenced code block in raw text
        mf = _CODE_FENCE_RE.search(text)
        if mf:
            return mf.group("code").strip()

        # Heuristic: check for python indicators
        py_indicators = ("import ", "def ", "class ", "torch", "nn.Module", "return ")
        if any(tok in text for tok in py_indicators):
            return text

        # Last resort: return full text
        return text

    def _validate_source(self, src: str) -> None:
        """
        Validate that the source is syntactically valid Python and contains at least one class def.
        """
        try:
            tree = ast.parse(src)
        except SyntaxError as e:
            raise LLMModelAssemblerError(
                f"Generated source has syntax error: {e}"
            ) from e

        has_class = any(isinstance(node, ast.ClassDef) for node in tree.body)
        if not has_class:
            raise LLMModelAssemblerError(
                "Generated source does not contain a class definition."
            )

    def assemble_from_config(
        self,
        model_config: Dict[str, Any],
        output_path: str,
        use_llm: bool = True,
        temperature: float = 0.0,
        prompt_addendum: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate model source from model_config, write to output_path.
        Returns a dict describing the result.
        """
        out_path = str(Path(output_path).resolve())

        try:
            config_json = json.dumps(model_config, indent=2)
        except Exception:
            config_json = str(model_config)

        # Avoid .format; simple replace for the placeholder
        prompt = self.prompt_template.replace("{model_config}", config_json)
        if prompt_addendum:
            prompt = prompt + "\n\n" + prompt_addendum

        # Attempt LLM generation if requested
        if use_llm:
            try:
                raw = self._call_llm(prompt, temperature=temperature, timeout=300)
                code = self._extract_code_from_response(raw)
                self._validate_source(code)
                # Save
                if save_model_code is not None:
                    save_model_code(out_path, code, overwrite=True)
                else:
                    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
                    Path(out_path).write_text(code, encoding="utf-8")
                logger.info("Model assembled via LLM and saved to %s", out_path)
                return {
                    "path": out_path,
                    "via": "llm",
                    "source": code,
                    "llm_response": raw,
                }
            except Exception as exc:
                logger.warning(
                    "LLM assembler failed: %s. Falling back to deterministic assembler.",
                    exc,
                )

        # Fallback deterministic assembler
        if assemble_model_code is None or save_model_code is None:
            if _fallback_import_error is not None:
                raise LLMModelAssemblerError(
                    f"Programmatic assembler not available: {_fallback_import_error}"
                )
            else:
                raise LLMModelAssemblerError(
                    "Programmatic assembler not available and LLM failed."
                )

        try:
            # Ensure the programmatic ModelConfig class is available.
            # The earlier checks guarantee assemble_model_code/save_model_code exist,
            # but static type checkers may still consider ModelConfig None, so guard explicitly.
            if ModelConfig is None:
                raise LLMModelAssemblerError(
                    "Programmatic ModelConfig class not available for fallback assembler."
                )

            # If the caller already provided a ModelConfig instance, use it directly.
            if isinstance(model_config, ModelConfig):
                cfg = model_config
            else:
                # Coerce dict-like configs into a ModelConfig safely.
                if isinstance(model_config, dict):
                    class_name = model_config.get("class_name") or "AssembledModel"
                    input_shape = model_config.get("input_shape")
                    layers = list(model_config.get("layers", []))
                    docstring = model_config.get("docstring")
                    cfg = ModelConfig(
                        class_name=str(class_name),
                        input_shape=input_shape,
                        layers=layers,
                        docstring=docstring,
                    )
                else:
                    # Unknown shape: create a minimal ModelConfig to allow assembly.
                    cfg = ModelConfig(class_name="AssembledModel", layers=[])

            src = assemble_model_code(cfg)
            save_model_code(out_path, src, overwrite=True)
            logger.info("Model assembled via fallback and saved to %s", out_path)
            return {
                "path": out_path,
                "via": "fallback",
                "source": src,
                "llm_response": None,
            }
        except Exception as exc:
            logger.error("Fallback assembler failed: %s", exc)
            raise LLMModelAssemblerError(f"Fallback assembler failed: {exc}") from exc


# Default prompt template asking for a structured JSON response with a "python" key
DEFAULT_PROMPT_TEMPLATE = """You are an expert PyTorch engineer. Given the JSON model configuration below, generate a SINGLE, structured JSON response containing the keys described in the schema. DO NOT include any additional text outside the JSON. The JSON must be valid and parseable.

Model configuration (JSON):
{model_config}

Output JSON schema (must follow exactly):
{
  "python": "<string with the complete Python source for the model class (no surrounding triple-backticks)>",
  "metadata": {
    "class_name": "<name of the generated class, e.g. AssembledModel>",
    "assumptions": { "<any assumptions made>": "<values>" },
    "notes": "<short human-readable summary, optional>"
  }
}

Requirements:
- The value of `python` must be a single string containing only Python source code that imports `torch` and `torch.nn as nn`, and defines a subclass of `nn.Module`.
- The class should be named `AssembledModel` unless a `class_name` is provided in the configuration; reflect the actual class name in `metadata.class_name`.
- Do NOT include triple-backtick fences in the `python` field; the code should be raw text.
- Keep the implementation simple, correct, and runnable on CPU. Provide reasonable defaults for unspecified parameters and place any such assumptions under `metadata.assumptions`.
- Do not include explanatory text outside the JSON object. The response must be exactly one JSON object matching the schema above.

If you cannot follow these instructions exactly, return an error object of the form:
{ "error": "<reason>" }
"""

# Convenience default assembler instance
_default_assembler: Optional[LLMModelAssembler] = None


def assemble_from_config(
    model_config: Dict[str, Any],
    output_path: str,
    use_llm: bool = True,
    llm_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Wrapper that creates a default LLMModelAssembler with llm_kwargs and calls it.
    """
    global _default_assembler
    if _default_assembler is None:
        llm_kwargs = llm_kwargs or {}
        _default_assembler = LLMModelAssembler(**llm_kwargs)
    return _default_assembler.assemble_from_config(
        model_config, output_path, use_llm=use_llm
    )
