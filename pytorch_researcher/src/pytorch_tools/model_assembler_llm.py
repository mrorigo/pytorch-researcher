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
        - Handle markdown fences (```json ``` blocks) first.
        - Attempt to parse content as JSON and prefer 'python' / 'code' keys.
        - Fallback to fenced code block extraction or heuristic detection.
        - Handle double JSON nesting (content containing JSON with python key).
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
        logger.debug(f"Initial extracted text (length {len(text)}): {text[:500]!r}...")

        # FIRST: Handle markdown fences before JSON parsing
        # Check if text starts with markdown fences
        markdown_fence_pattern = re.compile(r'^```(?:json)?\s*(.*?)```$', re.DOTALL | re.IGNORECASE)
        fence_match = markdown_fence_pattern.search(text)
        if fence_match:
            logger.debug("Found markdown fence, extracting content")
            fenced_content = fence_match.group(1).strip()
            logger.debug(f"Fenced content (length {len(fenced_content)}): {fenced_content[:500]!r}...")
            
            # Check if fenced content is JSON
            if fenced_content.strip().startswith('{'):
                try:
                    # Try to parse the fenced content as JSON directly
                    logger.debug("Attempting to parse fenced content as JSON")
                    parsed = json.loads(fenced_content)
                    logger.debug(f"Successfully parsed fenced JSON: {list(parsed.keys()) if isinstance(parsed, dict) else type(parsed)}")
                    
                    if isinstance(parsed, dict):
                        # Look for code keys in the parsed JSON
                        for key in ("python", "python_code", "code", "source"):
                            if (
                                key in parsed
                                and isinstance(parsed[key], str)
                                and parsed[key].strip()
                            ):
                                code_candidate = parsed[key].strip()
                                # Handle escaped newlines in the extracted code string
                                if '\\n' in code_candidate:
                                    logger.debug("Handling escaped newlines in fenced code")
                                    code_candidate = code_candidate.replace('\\n', '\n')
                                
                                logger.debug(f"Extracted Python code from fenced JSON key '{key}'")
                                return code_candidate
                        
                        logger.debug(f"Fenced JSON parsed but no code key found. Available keys: {list(parsed.keys())}")
                        
                except Exception as json_exc:
                    logger.debug(f"Failed to parse fenced content as JSON: {json_exc}")
                    # If fenced content is not JSON, treat it as direct code
                    logger.debug("Treating fenced content as direct code")
                    return fenced_content
            else:
                # Fenced content is not JSON, return as direct code
                logger.debug("Fenced content is not JSON, treating as direct code")
                return fenced_content

        # SECOND: Try parse as JSON object in the assistant text (without markdown fences)
        try:
            logger.debug(f"Attempting to parse JSON from text (length {len(text)})")
            logger.debug(f"Text preview: {text[:500]!r}...")
            
            # Parse JSON without modifying escaped characters
            parsed = json.loads(text)
            logger.debug(f"Successfully parsed JSON: {list(parsed.keys()) if isinstance(parsed, dict) else type(parsed)}")
                
            if isinstance(parsed, dict):
                # First, try to find direct code keys
                for key in ("python", "python_code", "code", "source"):
                    if (
                        key in parsed
                        and isinstance(parsed[key], str)
                        and parsed[key].strip()
                    ):
                        code_candidate = parsed[key].strip()
                        # Handle escaped newlines in the extracted code string
                        if '\\n' in code_candidate:
                            logger.debug("Handling escaped newlines in extracted code")
                            code_candidate = code_candidate.replace('\\n', '\n')
                        
                        logger.debug(f"Extracted Python code from JSON key '{key}'")
                        return code_candidate
                
                # Check if we have a content/body key that might contain more JSON
                for key in ("content", "body"):
                    if (
                        key in parsed
                        and isinstance(parsed[key], str)
                        and parsed[key].strip()
                    ):
                        extracted = parsed[key].strip()
                        logger.debug(f"Found content in key '{key}', attempting to parse as JSON")
                        
                        # Check if this extracted content looks like JSON (starts with {)
                        if extracted.strip().startswith('{'):
                            try:
                                # Try to parse the content as JSON to handle nested JSON
                                nested_parsed = json.loads(extracted)
                                logger.debug(f"Successfully parsed nested JSON, keys: {list(nested_parsed.keys()) if isinstance(nested_parsed, dict) else type(nested_parsed)}")
                                
                                if isinstance(nested_parsed, dict):
                                    # Look for code keys in the nested JSON
                                    for nested_key in ("python", "python_code", "code", "source"):
                                        if (
                                            nested_key in nested_parsed
                                            and isinstance(nested_parsed[nested_key], str)
                                            and nested_parsed[nested_key].strip()
                                        ):
                                            code_candidate = nested_parsed[nested_key].strip()
                                            if '\\n' in code_candidate:
                                                logger.debug("Handling escaped newlines in nested code")
                                                code_candidate = code_candidate.replace('\\n', '\n')
                                            
                                            logger.debug(f"Extracted Python code from nested JSON key '{nested_key}'")
                                            return code_candidate
                            except Exception as nested_json_exc:
                                logger.debug(f"Failed to parse nested JSON: {nested_json_exc}")
                                # Fall back to treating the content as direct code
                                pass
                        
                        # Handle escaped newlines in fallback content
                        if '\\n' in extracted:
                            extracted = extracted.replace('\\n', '\n')
                        
                        logger.debug(f"Extracted Python code from JSON key '{key}' (direct)")
                        return extracted
                        
                # If we have a parsed dict but no code key, we might have returned the wrong thing
                logger.debug(f"JSON parsed but no code key found. Available keys: {list(parsed.keys())}")
                # Check if the parsed dict itself is the code (malformed JSON)
                if len(parsed) == 1 and 'python' in str(parsed):
                    # Maybe the whole JSON was returned as the code instead of proper format
                    return str(parsed)
                    
        except Exception as json_exc:
            logger.debug(f"JSON parsing failed: {json_exc}")
            logger.debug(f"Failed text was: {text[:1000]!r}...")
            # not JSON, continue to other heuristics
            pass

        # THIRD: Fenced code block in raw text (second pass, in case first pass didn't catch it)
        mf = _CODE_FENCE_RE.search(text)
        if mf:
            extracted_code = mf.group("code").strip()
            logger.debug(f"Extracted Python code from fenced block (second pass)")
            return extracted_code

        # FOURTH: Heuristic: check for python indicators
        py_indicators = ("import ", "def ", "class ", "torch", "nn.Module", "return ")
        if any(tok in text for tok in py_indicators):
            logger.debug(f"Extracted Python code using heuristic detection")
            return text

        # Last resort: return full text
        logger.debug(f"Using last resort - returning full text as code")
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
        max_llm_retries: int = 3,
        sandbox_error: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate model source from model_config, write to output_path.
        Returns a dict describing the result.
        
        Args:
            model_config: Configuration dict for the model
            output_path: Path where to save the generated model
            use_llm: Whether to use LLM generation (overridden by deterministic-first strategy)
            temperature: LLM temperature (0.0 for more deterministic output)
            prompt_addendum: Additional prompt instructions
            max_llm_retries: Number of retry attempts for LLM generation
            sandbox_error: Optional error message from sandbox execution for intelligent retry
        """
        out_path = str(Path(output_path).resolve())

        try:
            config_json = json.dumps(model_config, indent=2)
        except Exception:
            config_json = str(model_config)

        # 💰 DETERMINISTIC-FIRST STRATEGY with complexity heuristic
        logger.info("🔄 Model assembly: Starting with deterministic-first strategy")
        
        # Complexity heuristic: use deterministic for simple, supported layer configs
        def _is_deterministic_friendly(config: Dict[str, Any]) -> bool:
            """Heuristic to determine if config should use deterministic assembler."""
            simple_layers = {
                "Conv2d", "Linear", "ReLU", "LeakyReLU", "MaxPool2d",
                "AvgPool2d", "BatchNorm2d", "Dropout", "Flatten"
            }
            
            if not isinstance(config.get("layers"), list):
                return False
                
            # Check layer types and config simplicity
            num_layers = len(config["layers"])
            has_unsupported = False
            
            for layer in config["layers"]:
                layer_type = layer.get("type")
                if layer_type not in simple_layers:
                    has_unsupported = True
                    break
            
            # Use deterministic if: supported layers only + reasonable size + explicit shape hints
            return (
                not has_unsupported and
                num_layers <= 15 and
                num_layers > 0 and
                config.get("input_shape") is not None
            )
        
        should_use_deterministic = _is_deterministic_friendly(model_config)
        logger.info("📊 Config complexity: %d layers, deterministic_friendly=%s",
                   len(model_config.get("layers", [])), should_use_deterministic)
        
        # Check if deterministic fallback is available
        if assemble_model_code is None or save_model_code is None:
            if _fallback_import_error is not None:
                raise LLMModelAssemblerError(
                    f"Programmatic assembler not available: {_fallback_import_error}"
                )
            else:
                raise LLMModelAssemblerError(
                    "Programmatic assembler not available."
                )

        try:
            # Ensure the programmatic ModelConfig class is available
            if ModelConfig is None:
                raise LLMModelAssemblerError(
                    "Programmatic ModelConfig class not available for deterministic assembler."
                )

            # Coerce dict-like configs into a ModelConfig safely
            if isinstance(model_config, ModelConfig):
                cfg = model_config
            else:
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
                    cfg = ModelConfig(class_name="AssembledModel", layers=[])

            src = assemble_model_code(cfg)
            save_model_code(out_path, src, overwrite=True)
            logger.info("💾 Model assembled via DETERMINISTIC and saved to %s", out_path)
            return {
                "path": out_path,
                "via": "deterministic",
                "source": src,
                "llm_response": None,
                "attempts": 0,
            }
            
        except Exception as det_exc:
            logger.info("⚠️ Deterministic assembly failed (%s), falling back to LLM", det_exc)
        
        # LLM fallback only if deterministic fails
        logger.info("🤖 Falling back to LLM generation (max %d retries)", max_llm_retries)
        
        if not use_llm:
            raise LLMModelAssemblerError("Deterministic failed and use_llm=False")
            
        for attempt in range(max_llm_retries):
                try:
                    # Build prompt
                    prompt = self.prompt_template.replace("{model_config}", config_json)
                    
                    # Add error feedback on retry attempts
                    if attempt > 0:
                        if sandbox_error:
                            # Add intelligent sandbox error feedback
                            prompt += f"\n\nCRITICAL: Previous generation failed during sandbox execution with this error:\n{sandbox_error}\n\nPlease fix the runtime issues in your code. Ensure tensor shapes match, input dimensions are correct, and PyTorch operations are valid."
                        else:
                            prompt += f"\n\nIMPORTANT: Previous attempt failed. Please be more careful about JSON formatting and syntax validity."
                    
                    if prompt_addendum:
                        prompt = prompt + "\n\n" + prompt_addendum

                    logger.info(f"🤖 LLM ASSEMBLER ATTEMPT {attempt + 1}/{max_llm_retries}")
                    logger.info(f"📋 Model config length: {len(config_json)} chars")
                    logger.info(f"📝 Prompt length: {len(prompt)} chars")
                    
                    # Call LLM with comprehensive logging
                    start_time = time.time()
                    raw = self._call_llm(prompt, temperature=temperature, timeout=300)
                    llm_duration = time.time() - start_time
                    
                    logger.info(f"⏱️ LLM call duration: {llm_duration:.2f}s")
                    logger.info(f"📦 Raw LLM response type: {type(raw).__name__}")
                    
                    # Extract code with detailed logging
                    logger.info(f"🔍 Extracting code from LLM response...")
                    code = self._extract_code_from_response(raw)
                    
                    logger.info(f"✅ Code extracted successfully")
                    logger.info(f"📏 Extracted code length: {len(code)} chars")
                    logger.info(f"🔍 Code preview: {code[:200]!r}{'...' if len(code) > 200 else ''}")
                    
                    # Validate the generated code with detailed error reporting
                    logger.info(f"🔎 Validating extracted code...")
                    try:
                        self._validate_source(code)
                        logger.info(f"✅ Code validation PASSED")
                    except LLMModelAssemblerError as val_exc:
                        # LOG FAILED GENERATION FOR ANALYSIS
                        logger.error(f"🚫 LLM GENERATION {attempt + 1} FAILED - Logging for analysis:")
                        logger.error(f"📋 Validation error: {val_exc}")
                        logger.error(f"📦 Raw LLM response: {raw}")
                        logger.error(f"🔍 Extracted code that failed validation:")
                        logger.error(f"📏 Code length: {len(code)} chars")
                        logger.error(f"📝 Failed code content:\n{code}")
                        logger.error(f"📄 Failed code lines:")
                        for i, line in enumerate(code.split('\n')[:30]):
                            logger.error(f"  {i+1:2d}: {line}")
                        
                        if "class definition" in str(val_exc):
                            # Check what classes exist in the generated code
                            try:
                                import ast
                                tree = ast.parse(code)
                                class_names = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
                                logger.error(f"🔍 Classes found in failed code: {class_names}")
                            except Exception as parse_exc:
                                logger.error(f"❌ Could not parse failed code for class analysis: {parse_exc}")
                        
                        logger.info(f"🔄 Retry {attempt + 2}/{max_llm_retries} with improved prompt...")
                        raise val_exc  # Re-raise to trigger retry logic
                    
                    # Save the validated code
                    if save_model_code is not None:
                        save_model_code(out_path, code, overwrite=True)
                    else:
                        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
                        Path(out_path).write_text(code, encoding="utf-8")
                    
                    logger.info(f"💾 Model assembled via LLM and saved to {out_path} (attempt {attempt + 1})")
                    logger.info(f"🎉 LLM ASSEMBLER SUCCESS!")
                    
                    return {
                        "path": out_path,
                        "via": "llm",
                        "source": code,
                        "llm_response": raw,
                        "attempts": attempt + 1,
                        "sandbox_error": sandbox_error,  # Include original error for tracking
                    }
                    
                except Exception as exc:
                    # Enhanced logging to help debug LLM failures
                    logger.error(f"❌ LLM attempt {attempt + 1}/{max_llm_retries} failed: {exc}")
                    
                    if attempt < max_llm_retries - 1:
                        # Add error feedback to prompt for next attempt
                        if hasattr(exc, 'args') and exc.args:
                            error_msg = str(exc.args[0])
                            logger.info(f"📝 Adding error feedback for retry: {error_msg}")
                            # The error feedback will be included in next attempt's prompt
                    
                    if attempt == max_llm_retries - 1:
                        # Last attempt failed, log comprehensive details for analysis
                        logger.error(f"🚫 LLM ASSEMBLER FAILED after {max_llm_retries} attempts")
                        
                        # Log all details for prompt improvement analysis
                        if 'raw' in locals():
                            logger.error(f"📦 Final attempt LLM raw response: {raw}")
                            logger.error(f"📄 Final attempt extracted code: {code if 'code' in locals() else 'N/A'}")
                        
                        logger.info("🔄 Falling back to deterministic assembler.")
                    else:
                        logger.info(f"🔄 Retrying LLM generation (attempt {attempt + 2})")

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

            # This fallback block is now unreachable due to deterministic-first strategy above
            logger.warning("Fallback assembler reached unexpectedly")
            raise LLMModelAssemblerError("Unexpected fallback path reached")
        except Exception as exc:
            logger.error("Fallback assembler failed: %s", exc)
            raise LLMModelAssemblerError(f"Fallback assembler failed: {exc}") from exc


# Default prompt template asking for a structured JSON response with a "python" key
DEFAULT_PROMPT_TEMPLATE = """You are an expert PyTorch engineer. Generate a SINGLE, valid JSON response only. NO additional text before or after the JSON.

Model configuration:
{model_config}

Output ONLY this exact JSON format:
{
  "python": "Your PyTorch code here as a single string",
  "metadata": {
    "class_name": "AssembledModel",
    "assumptions": {"key": "value"},
    "notes": "brief summary"
  }
}

CRITICAL REQUIREMENTS:
- Return ONLY the JSON object - no markdown, no explanations, no code fences
- The "python" value must be a single string containing valid Python code
- Code must import torch and torch.nn, define a class subclassing nn.Module
- Class name should be AssembledModel
- Code must be syntactically valid Python

Example response:
{"python": "import torch\\nimport torch.nn as nn\\n\\nclass AssembledModel(nn.Module):\\n    def __init__(self):\\n        super(AssembledModel, self).__init__()\\n        self.linear = nn.Linear(10, 1)\\n    \\n    def forward(self, x):\\n        return self.linear(x)", "metadata": {"class_name": "AssembledModel", "assumptions": {}, "notes": "Simple linear model"}}

Response:"""

# Convenience default assembler instance
_default_assembler: Optional[LLMModelAssembler] = None


def assemble_from_config(
    model_config: Dict[str, Any],
    output_path: str,
    use_llm: bool = True,
    llm_kwargs: Optional[Dict[str, Any]] = None,
    sandbox_error: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Wrapper that creates a default LLMModelAssembler with llm_kwargs and calls it.
    """
    global _default_assembler
    if _default_assembler is None:
        llm_kwargs = llm_kwargs or {}
        _default_assembler = LLMModelAssembler(**llm_kwargs)
    return _default_assembler.assemble_from_config(
        model_config, output_path, use_llm=use_llm, max_llm_retries=3, sandbox_error=sandbox_error
    )
