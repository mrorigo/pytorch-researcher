# pytorch-researcher/pytorch_researcher/src/planning_llm/client.py
"""
Planning LLM client (HTTP-only) and Orchestrator.

This module provides:
- PlanningLLMClient: an HTTP-only client for chat-completion endpoints. It
  posts to "<base_url>/chat/completions" and returns parsed, validated JSON
  according to the orchestrator's expected structured schemas.
- Orchestrator: a lightweight orchestrator class that ties PlanningLLMClient
  with pluggable tool callables (assembler, summarizer, evaluator, sandbox runner,
  registry writer) and implements the iteration loop.

Design goals:
- Keep the LLM transport HTTP-only (no external SDK dependency).
- Keep the client small and testable; higher-level prompt templates live in the
  orchestrator or the caller.
- Make all external integrations (assembler, summarizer, evaluator, sandbox,
  registry writer) pluggable callables passed to Orchestrator so unit tests
  can inject mocks easily.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Iterable, List, Optional, Sequence

# Import the repository-local HTTP LLM client abstraction (DRY)
try:
    from pytorch_researcher.src.pytorch_tools.llm import HTTPLLMClient, LLMClientError
except Exception:  # pragma: no cover - defensive import for environments without module
    HTTPLLMClient = None  # type: ignore
    LLMClientError = Exception  # type: ignore

logger = logging.getLogger(__name__)

# Export public API
__all__ = [
    "PlanningLLMClient",
    "PlanningLLMClientError",
    "Orchestrator",
    "OrchestratorError",
]


class PlanningLLMClientError(Exception):
    """Raised for planning LLM client errors."""


class PlanningLLMClient:
    """
    HTTP-only Planning LLM client wrapper.

    This client posts chat-completion styled requests to "<base_url>/chat/completions"
    and returns parsed structured JSON results for higher-level orchestrator use.

    Example usage:
        client = PlanningLLMClient(base_url="http://localhost:11434/v1", model="gpt-oss:20b")
        proposal = client.propose_initial_config(goal="Make a small CIFAR10 CNN", constraints={...})
        decision = client.decide_next_action(goal, registry, latest_result)
    """

    DEFAULT_SYSTEM_PROMPT = (
        "You are the planning LLM for an ML research agent. "
        "When asked for structured output, return a single valid JSON object and nothing else."
    )

    def __init__(
        self,
        base_url: str,
        model: str = "gpt-5.1-mini",
        api_key: Optional[str] = None,
        timeout: int = 300,
        max_retries: int = 2,
        retry_backoff: float = 1.0,
        system_prompt: Optional[str] = None,
    ) -> None:
        if not base_url:
            raise ValueError("base_url is required for PlanningLLMClient")
        if HTTPLLMClient is None:
            # If the repo's HTTPLLMClient is not importable, we can't proceed
            raise RuntimeError("HTTPLLMClient is not available in this environment")
        self._http_client = HTTPLLMClient(
            base_url=base_url,
            model_name=model,
            api_key=api_key,
            max_retries=max_retries,
            retry_backoff=retry_backoff,
        )
        self.timeout = int(timeout)
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT

    # low-level helper -------------------------------------------------------

    def _call(
        self, prompt: str, temperature: float = 0.0, timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Perform an assistant call and return a dict {"raw": parsed_response}.
        Wrap transport exceptions in PlanningLLMClientError.
        """
        try:
            return self._http_client.call(
                prompt, temperature=temperature, timeout=timeout or self.timeout
            )
        except LLMClientError as exc:
            raise PlanningLLMClientError(f"LLM transport error: {exc}") from exc
        except Exception as exc:
            raise PlanningLLMClientError(f"Unexpected transport error: {exc}") from exc

    @staticmethod
    def _extract_assistant_text(raw: Any) -> str:
        """
        Extract assistant textual content from common chat-completion shapes.
        Returns a best-effort string (may be JSON text).
        """
        text: Optional[str] = None
        try:
            if isinstance(raw, dict) and "choices" in raw:
                choices = raw.get("choices") or []
                if choices:
                    first = choices[0]
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

        if text is None:
            try:
                text = json.dumps(raw)
            except Exception:
                text = str(raw)
        return text

    # structured interactions ------------------------------------------------

    def propose_initial_config(
        self, goal: str, constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Ask the planning LLM to propose an initial `model_config`.

        Expected assistant JSON schema (example):
        {
            "action": "propose",
            "model_config": { ... },
            "notes": "...",
            "metadata": { "assumptions": {...} }
        }

        Returns the parsed dict. Raises PlanningLLMClientError on parse/validation errors.
        """
        prompt_parts: List[str] = [
            self.system_prompt(),
            "Task: Propose an initial model_config for the research goal below.",
            "Return EXACTLY one JSON object and nothing else, following this schema:",
            '{ "action": "propose", "model_config": { /* dict */ }, "notes": "<optional>", "metadata": { "assumptions": {...} } }',
            "",
            "Goal:",
            goal,
        ]
        if constraints:
            prompt_parts.extend(
                ["", "Constraints (JSON):", json.dumps(constraints, indent=2)]
            )
        prompt = "\n".join(prompt_parts)
        raw = self._call(prompt, temperature=0.0, timeout=self.timeout)
        assistant_text = self._extract_assistant_text(raw.get("raw"))
        try:
            parsed = json.loads(assistant_text)
        except Exception as exc:
            raise PlanningLLMClientError(
                f"Failed to parse assistant JSON for proposal: {exc}. Raw: {assistant_text!r}"
            ) from exc

        if (
            not isinstance(parsed, dict)
            or parsed.get("action") != "propose"
            or not isinstance(parsed.get("model_config"), dict)
        ):
            raise PlanningLLMClientError(
                f"Assistant returned invalid proposal schema: {parsed!r}"
            )
        return parsed

    def decide_next_action(
        self,
        goal: str,
        registry: Sequence[Dict[str, Any]],
        latest_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Ask the planning LLM to decide the next action.

        Expected assistant JSON schema:
        {
          "action": "refine" | "achieve" | "stop",
          "reason": "<explanation>",
          "next_config": { ... }  // required when action == "refine"
        }

        Returns the parsed dict; raises PlanningLLMClientError on parse/validation errors.
        """
        prompt_parts: List[str] = [
            self.system_prompt(),
            "Task: Based on the goal, the experiment registry, and the latest result, decide the NEXT ACTION.",
            "Return EXACTLY one JSON object with keys: action (refine|achieve|stop), reason (string), next_config (optional dict if refine).",
            "",
            "Goal:",
            goal,
            "",
            "Latest result (JSON):",
            json.dumps(latest_result, indent=2),
            "",
            "Experiment registry (most recent entries):",
            json.dumps(list(registry)[-8:], indent=2),
            "",
            "Return only the JSON object — no extra commentary.",
        ]
        prompt = "\n".join(prompt_parts)
        raw = self._call(prompt, temperature=0.0, timeout=self.timeout)
        assistant_text = self._extract_assistant_text(raw.get("raw"))
        try:
            parsed = json.loads(assistant_text)
        except Exception as exc:
            raise PlanningLLMClientError(
                f"Failed to parse assistant JSON for decision: {exc}. Raw: {assistant_text!r}"
            ) from exc

        if not isinstance(parsed, dict) or "action" not in parsed:
            raise PlanningLLMClientError(
                f"Assistant decision missing required 'action' key: {parsed!r}"
            )
        if parsed.get("action") == "refine" and "next_config" not in parsed:
            raise PlanningLLMClientError(
                "Assistant requested 'refine' but did not provide 'next_config'."
            )
        return parsed

    # small helper to provide consistent system prompt formatting
    def system_prompt(self) -> str:
        return (
            self.system_prompt
            if isinstance(self.system_prompt, str)
            else str(self.DEFAULT_SYSTEM_PROMPT)
        )


# Orchestrator -------------------------------------------------------


class OrchestratorError(Exception):
    """Raised for orchestrator-level failures."""


class Orchestrator:
    """
    High-level Orchestrator.

    All external behaviors are injected as callables to improve testability.

    Expected callables:
      - assembler(model_config: dict, output_path: str, use_llm: bool=True, llm_kwargs: Optional[dict]=None) -> dict
      - summarizer(model_path: str, class_name: Optional[str], input_size: tuple) -> dict
      - evaluator(model_path: str, model_config: dict) -> dict
      - sandbox_runner(model_path: str, class_name: Optional[str], input_size: tuple, timeout: int) -> dict
      - registry_writer(entry: dict) -> None
    """

    def __init__(
        self,
        planning_client: PlanningLLMClient,
        assembler: Any,
        summarizer: Any,
        evaluator: Any,
        sandbox_runner: Any,
        registry_writer: Any,
        max_iterations: int = 5,
        target_accuracy: float = 0.70,
    ) -> None:
        self.planning_client = planning_client
        self.assembler = assembler
        self.summarizer = summarizer
        self.evaluator = evaluator
        self.sandbox_runner = sandbox_runner
        self.registry_writer = registry_writer
        self.max_iterations = int(max_iterations)
        self.target_accuracy = float(target_accuracy)

    def run(
        self, goal: str, workdir: str, keep_artifacts: bool = True
    ) -> Dict[str, Any]:
        """
        Run the Orchestrator loop.

        Args:
            goal: high-level research goal string.
            workdir: path where per-iteration artifacts (models, experiments) are written.
            keep_artifacts: whether artifacts should be kept (caller may remove workdir).

        Returns:
            A dict with details of the run, iterations and final decision.
        """
        run_report: Dict[str, Any] = {"goal": goal, "iterations": []}
        # 1) request initial proposal
        proposal = self.planning_client.propose_initial_config(
            goal=goal, constraints=None
        )
        model_config = proposal.get("model_config")
        run_report["planning_proposal"] = proposal

        for it in range(1, self.max_iterations + 1):
            iter_entry: Dict[str, Any] = {"iteration": it, "model_config": model_config}
            iteration_dir = Path(workdir) / f"iter_{it}"
            iteration_dir.mkdir(parents=True, exist_ok=True)
            model_path = str((iteration_dir / "model.py").resolve())

            # 2) assemble model (allow assembler to decide whether to call its own LLM)
            try:
                asm_res = self.assembler(
                    model_config, model_path, use_llm=False, llm_kwargs=None
                )
                iter_entry["assemble"] = asm_res
            except Exception as e:
                iter_entry["assemble_error"] = str(e)
                run_report["iterations"].append(iter_entry)
                # Ask planning LLM whether to stop
                decision = self.planning_client.decide_next_action(
                    goal, run_report["iterations"], iter_entry
                )
                run_report["last_decision"] = decision
                if decision.get("action") == "stop":
                    run_report["final_status"] = "stopped_due_to_assembly_error"
                    break
                else:
                    # if refine suggested, continue loop with provided config
                    model_config = decision.get("next_config", model_config)
                    continue

            # 3) sandbox validate and summarize
            if self.sandbox_runner is not None:
                try:
                    sb_res = self.sandbox_runner(
                        model_path,
                        class_name=model_config.get("class_name"),
                        input_size=(1, 3, 32, 32),
                        timeout=120,
                    )
                    iter_entry["sandbox"] = sb_res
                    if sb_res.get("success") and self.summarizer is not None:
                        try:
                            summ = self.summarizer(
                                model_path,
                                class_name=model_config.get("class_name"),
                                input_size=(1, 3, 32, 32),
                            )
                            iter_entry["summary"] = summ
                        except Exception as e:
                            iter_entry["summary_error"] = str(e)
                    elif not sb_res.get("success"):
                        iter_entry["sandbox_error"] = sb_res.get("error") or sb_res.get(
                            "stderr"
                        )
                except Exception as e:
                    iter_entry["sandbox_error"] = str(e)
            else:
                iter_entry["sandbox_error"] = "sandbox runner not configured"

            # 4) quick evaluation (only if sandbox passed)
            if (
                iter_entry.get("sandbox", {}).get("success", False)
                and self.evaluator is not None
            ):
                try:
                    eval_res = self.evaluator(model_path, model_config)
                    iter_entry["evaluation"] = eval_res
                except Exception as e:
                    iter_entry["evaluation_error"] = str(e)
            else:
                if "evaluation_error" not in iter_entry:
                    iter_entry["evaluation_error"] = (
                        "sandbox failed or evaluator missing"
                    )

            # 5) persist iteration to registry (best-effort)
            try:
                if self.registry_writer is not None:
                    self.registry_writer(iter_entry)
                    iter_entry["persisted"] = True
            except Exception as e:
                iter_entry["persist_error"] = str(e)

            run_report["iterations"].append(iter_entry)

            # 6) planning decision
            latest_result = (
                iter_entry.get("evaluation")
                or iter_entry.get("summary")
                or iter_entry.get("sandbox")
                or {}
            )
            try:
                decision = self.planning_client.decide_next_action(
                    goal=goal,
                    registry=run_report["iterations"],
                    latest_result=latest_result,
                )
                run_report["last_decision"] = decision
            except Exception as e:
                # If planning LLM fails, stop for safety
                run_report["final_status"] = f"planning_error: {e}"
                break

            action = decision.get("action")
            if action == "achieve":
                run_report["final_status"] = "achieved"
                break
            if action == "stop":
                run_report["final_status"] = "stopped"
                break
            if action == "refine":
                model_config = decision.get("next_config") or model_config
                # continue to next iteration
                continue

            # unknown action -> stop
            run_report["final_status"] = f"unknown_action:{action}"
            break

        else:
            run_report["final_status"] = "max_iterations_reached"

        return run_report


# helpers for type hints referencing Path without importing Path at module top-level
from pathlib import Path  # placed here to avoid circular top-level import issues
