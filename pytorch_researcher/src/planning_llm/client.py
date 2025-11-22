#!/usr/bin/env python3
"""Planning LLM client and Orchestrator.

This module provides:
- PlanningLLMClient: A unified LLM client using LiteLLM for chat-completion endpoints.
  It supports various providers (OpenAI, Anthropic, local, etc.) and returns parsed,
  validated JSON according to the orchestrator's expected structured schemas.
- Orchestrator: a lightweight orchestrator class that ties PlanningLLMClient
  with pluggable tool callables (assembler, summarizer, evaluator, sandbox runner,
  registry writer) and implements the iteration loop.

Design goals:
- Use LiteLLM for unified interface across different LLM providers.
- Keep the client small and testable; higher-level prompt templates live in the
  orchestrator or the caller.
- Make all external integrations (assembler, summarizer, evaluator, sandbox,
  registry writer) pluggable callables passed to Orchestrator so unit tests
  can inject mocks easily.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

# Import the repository-local HTTP LLM client abstraction (DRY)
try:
    from pytorch_researcher.src.pytorch_tools.llm import LiteLLMClient, LLMClientError
except Exception:  # pragma: no cover - defensive import for environments without module
    LiteLLMClient = None  # type: ignore
    LLMClientError = Exception  # type: ignore

# Import LLM logger
try:
    from pytorch_researcher.src.pytorch_tools.llm_logger import (
        get_llm_logger,
        log_llm_call,
    )
except ImportError:
    # Fallback if logger not available
    get_llm_logger = None
    log_llm_call = None

# Configure logger to inherit from parent for consistent logging levels
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Set to DEBUG to ensure all levels propagate
logger.propagate = True  # Propagate logs to parent logger for consistent output

# Export public API
__all__ = [
    "Orchestrator",
    "OrchestratorError",
    "PlanningLLMClient",
    "PlanningLLMClientError",
]


class PlanningLLMClientError(Exception):
    """Raised for planning LLM client errors."""


class PlanningLLMClient(LiteLLMClient):
    """Planning LLM client using LiteLLM for unified provider interface.

    This client uses LiteLLM to provide a unified interface for different LLM providers
    and returns parsed structured JSON results for higher-level orchestrator use.

    The client supports any LLM provider supported by LiteLLM including OpenAI,
    Anthropic, local endpoints like Ollama, and many others.

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
        api_key: str | None = None,
        timeout: int = 300,
        max_retries: int = 2,
        retry_backoff: float = 1.0,
        system_prompt: str | None = None,
        run_id: str | None = None,
    ) -> None:
        """Initialize the planning LLM client."""
        super().__init__(
            base_url=base_url,
            model_name=model,
            api_key=api_key,
            max_retries=max_retries,
            retry_backoff=retry_backoff,
        )
        self.timeout = int(timeout)
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self.run_id = run_id

        # Initialize LLM logger
        self.llm_logger = get_llm_logger() if get_llm_logger else None

    def _call(
        self, prompt: str, temperature: float = 0.0, timeout: int | None = None
    ) -> dict[str, Any]:
        """Perform an assistant call using the parent LiteLLMClient.
        Wrap transport exceptions in PlanningLLMClientError.
        """
        try:
            # Use parent's call method
            # Note: parent call method handles logging and retries
            return self.call(prompt, temperature=temperature, timeout=timeout or self.timeout)
        except LLMClientError as exc:
            # Re-raise as PlanningLLMClientError for backward compatibility
            raise PlanningLLMClientError(f"LLM client error: {exc}") from exc
        except Exception as exc:
            raise PlanningLLMClientError(f"Unexpected LLM error: {exc}") from exc

    @staticmethod
    def _extract_assistant_text(raw: Any) -> str:
        """Extract assistant textual content from common chat-completion shapes.
        Returns a best-effort string (may be JSON text).
        """
        text: str | None = None
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

    def _format_memory_context(self, memory_context: list[dict[str, Any]]) -> str:
        """Format memory context for LLM consumption.

        This method provides fallback formatting if memory manager is not available.
        When memory manager is available, it should be preferred for formatting.

        Args:
            memory_context: List of memory context items with metadata

        Returns:
            Formatted memory context string for injection into prompts

        """
        if not memory_context:
            return ""

        prompt = "=== RELEVANT RESEARCH INSIGHTS ===\n"
        prompt += "Use these insights to inform your decisions:\n\n"

        # Prioritize essential memories and organize by category
        essential_memories = []
        regular_memories = []

        for context_item in memory_context:
            category = context_item.get("category_primary", "")
            if category.startswith("essential_"):
                essential_memories.append(context_item)
            else:
                regular_memories.append(context_item)

        # Add essential memories first (higher priority)
        for context_item in essential_memories:
            category = context_item.get("category_primary", "research")
            content = context_item.get("searchable_content", "") or context_item.get("summary", "")
            prompt += f"[{category.upper()}] {content}\n"

        # Add regular memories
        for context_item in regular_memories:
            category = context_item.get("category_primary", "research")
            content = context_item.get("searchable_content", "") or context_item.get("summary", "")
            prompt += f"- [{category}] {content}\n"

        prompt += "\n=== END RESEARCH INSIGHTS ===\n"
        return prompt

    def get_system_prompt(self) -> str:
        """Return the configured system prompt string."""
        return (
            self.system_prompt
            if isinstance(self.system_prompt, str)
            else str(self.DEFAULT_SYSTEM_PROMPT)
        )

    def propose_initial_config(
        self,
        goal: str,
        constraints: dict[str, Any] | None = None,
        memory_context: list[dict[str, Any]] | None = None,
        iteration: int | None = None
    ) -> dict[str, Any]:
        """Ask the planning LLM to propose an initial `model_config` and evaluation configuration.

        Expected assistant JSON schema (example):
        {
            "action": "propose",
            "model_config": { ... },
            "evaluation_config": {
                "dataset_name": "mnist",
                "subset_size": 512,
                "epochs": 1,
                "batch_size": 32,
                "target_accuracy": 0.7
            },
            "notes": "...",
            "metadata": { "assumptions": {...} }
        }

        Args:
            goal: The research goal
            constraints: Optional constraints for the research
            memory_context: Optional memory context to enhance planning decisions
            iteration: Optional iteration number for logging

        Returns:
            The parsed dict. Raises PlanningLLMClientError on parse/validation errors.

        """
        # Set current call type for logging
        self._current_call_type = "propose_initial_config"

        prompt_parts: list[str] = [
            self.get_system_prompt(),
            "Task: Propose an initial model_config and evaluation configuration for the research goal below.",
            "Return EXACTLY one JSON object and nothing else, following this schema:",
            '{ "action": "propose", "model_config": { /* dict */ }, "evaluation_config": { "dataset_name": "string", "subset_size": 512, "epochs": 1, "batch_size": 32, "target_accuracy": 0.7 }, "notes": "<optional>", "metadata": { "assumptions": {...} } }',
            "",
            "Goal:",
            goal,
        ]

        # Inject memory context if provided
        if memory_context:
            memory_prompt = self._format_memory_context(memory_context)
            prompt_parts.insert(3, memory_prompt)  # Insert after schema, before goal

        if constraints:
            prompt_parts.extend(
                ["", "Constraints (JSON):", json.dumps(constraints, indent=2)]
            )
        prompt = "\n".join(prompt_parts)
        raw = self._call(prompt, temperature=0.0, timeout=self.timeout)
        assistant_text = self._extract_assistant_text(raw.get("raw"))
        try:
            # Handle markdown-formatted JSON responses
            cleaned_text = assistant_text.strip()
            if cleaned_text.startswith('```json'):
                # Remove markdown code fences
                lines = cleaned_text.split('\n')
                if lines[0].startswith('```'):
                    lines = lines[1:]  # Remove first line
                if lines and lines[-1].strip() == '```':
                    lines = lines[:-1]  # Remove last line
                assistant_text = '\n'.join(lines)
            elif cleaned_text.startswith('```'):
                # Remove generic code fences
                lines = cleaned_text.split('\n')
                if lines[0].startswith('```'):
                    lines = lines[1:]  # Remove first line
                if lines and lines[-1].strip() == '```':
                    lines = lines[:-1]  # Remove last line
                assistant_text = '\n'.join(lines)

            # Handle Python-style tuples/commas in JSON (convert to JSON arrays)
            # Replace patterns like (1, 2, 3) with [1, 2, 3]
            import re
            assistant_text = re.sub(r'\(\s*(\d+(?:\s*,\s*\d+)*)\s*\)', r'[\1]', assistant_text)

            # Also handle single numbers in parentheses like (0.5) -> [0.5]
            assistant_text = re.sub(r'\(\s*([\d.]+)\s*\)', r'[\1]', assistant_text)

            parsed = json.loads(assistant_text)
        except Exception as exc:
            # Enhanced error logging for JSON parsing issues
            logger.error(f"JSON parsing failed for proposal. Error: {exc}")
            logger.error(f"Raw assistant text (length {len(assistant_text)}): {assistant_text[:1000]!r}")
            logger.error(f"Assistant text ends with: ...{assistant_text[-200:]!r}")

            # Try to identify the issue
            if "Unterminated string" in str(exc):
                logger.error("ISSUE: Unterminated string detected in JSON response")
                # Try to find where the string starts and ends
                lines = assistant_text.split('\n')
                for i, line in enumerate(lines):
                    if line.strip().startswith('"') and not line.strip().endswith('"'):
                        logger.error(f"Problematic line {i+1}: {line!r}")

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

        # Set default evaluation_config if not provided
        if "evaluation_config" not in parsed:
            parsed["evaluation_config"] = {
                "dataset_name": "cifar10",  # fallback to current default
                "subset_size": 512,
                "epochs": 1,
                "batch_size": 32,
                "target_accuracy": 0.7
            }

        return parsed

    def decide_next_action(
        self,
        goal: str,
        registry: Sequence[dict[str, Any]],
        latest_result: dict[str, Any],
        original_proposal: dict[str, Any] | None = None,
        memory_context: list[dict[str, Any]] | None = None,
        iteration: int | None = None
    ) -> dict[str, Any]:
        """Ask the planning LLM to decide the next action.

        Expected assistant JSON schema:
        {
          "action": "refine" | "achieve" | "stop",
          "reason": "<explanation>",
          "next_config": { ... }  // required when action == "refine"
        }

        Args:
            goal: The research goal
            registry: Experiment registry with previous iterations
            latest_result: Latest iteration result
            original_proposal: Optional original proposal to preserve input_shape
            memory_context: Optional memory context to enhance decision making
            iteration: Optional iteration number for logging

        Returns:
            The parsed dict; raises PlanningLLMClientError on parse/validation errors.

        """
        # Set current call type for logging
        self._current_call_type = "decide_next_action"

        prompt_parts: list[str] = [
            self.get_system_prompt(),
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
        ]

        # Inject memory context if provided
        if memory_context:
            memory_prompt = self._format_memory_context(memory_context)
            prompt_parts.insert(4, memory_prompt)  # Insert after action schema, before goal

        prompt_parts.append("Return only the JSON object — no extra commentary.")
        prompt = "\n".join(prompt_parts)
        raw = self._call(prompt, temperature=0.0, timeout=self.timeout)
        assistant_text = self._extract_assistant_text(raw.get("raw"))
        try:
            # Handle markdown-formatted JSON responses
            cleaned_text = assistant_text.strip()
            if cleaned_text.startswith('```json'):
                # Remove markdown code fences
                lines = cleaned_text.split('\n')
                if lines[0].startswith('```'):
                    lines = lines[1:]  # Remove first line
                if lines and lines[-1].strip() == '```':
                    lines = lines[:-1]  # Remove last line
                assistant_text = '\n'.join(lines)
            elif cleaned_text.startswith('```'):
                # Remove generic code fences
                lines = cleaned_text.split('\n')
                if lines[0].startswith('```'):
                    lines = lines[1:]  # Remove first line
                if lines and lines[-1].strip() == '```':
                    lines = lines[:-1]  # Remove last line
                assistant_text = '\n'.join(lines)

            # Handle Python-style tuples/commas in JSON (convert to JSON arrays)
            # Replace patterns like (1, 2, 3) with [1, 2, 3]
            import re
            assistant_text = re.sub(r'\(\s*(\d+(?:\s*,\s*\d+)*)\s*\)', r'[\1]', assistant_text)

            # Also handle single numbers in parentheses like (0.5) -> [0.5]
            assistant_text = re.sub(r'\(\s*([\d.]+)\s*\)', r'[\1]', assistant_text)

            parsed = json.loads(assistant_text)
        except Exception as exc:
            # Enhanced error logging for JSON parsing issues
            logger.error(f"JSON parsing failed for decision. Error: {exc}")
            logger.error(f"Raw assistant text (length {len(assistant_text)}): {assistant_text[:1000]!r}")
            logger.error(f"Assistant text ends with: ...{assistant_text[-200:]!r}")

            # Try to identify the issue
            if "Unterminated string" in str(exc):
                logger.error("ISSUE: Unterminated string detected in JSON response")
                # Try to find where the string starts and ends
                lines = assistant_text.split('\n')
                for i, line in enumerate(lines):
                    if line.strip().startswith('"') and not line.strip().endswith('"'):
                        logger.error(f"Problematic line {i+1}: {line!r}")

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

        # Preserve input_shape from original proposal across iterations
        if (parsed.get("action") == "refine" and
            original_proposal and
            "next_config" in parsed and
            isinstance(parsed["next_config"], dict)):

            original_input_shape = original_proposal.get("model_config", {}).get("input_shape")
            if original_input_shape and "input_shape" not in parsed["next_config"]:
                parsed["next_config"]["input_shape"] = original_input_shape
                logger.debug(f"Preserved input_shape from original proposal: {original_input_shape}")

        return parsed


class OrchestratorError(Exception):
    """Raised for orchestrator-level failures."""


class Orchestrator:
    """High-level Orchestrator with optional memory integration.

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
        memory_manager: Any = None,
    ) -> None:
        """Initialize the orchestrator with injected tooling."""
        self.planning_client = planning_client
        self.assembler = assembler
        self.summarizer = summarizer
        self.evaluator = evaluator
        self.sandbox_runner = sandbox_runner
        self.registry_writer = registry_writer
        self.max_iterations = int(max_iterations)
        self.target_accuracy = float(target_accuracy)
        self.memory_manager = memory_manager

    def _determine_research_phase(self, registry: Sequence[dict[str, Any]]) -> str:
        """Determine current research phase for targeted memory queries.

        Args:
            registry: Experiment registry with previous iterations

        Returns:
            Current research phase string ('planning', 'architecture', 'evaluation', etc.)

        """
        if not registry:
            return "planning"

        recent_iterations = list(registry)[-3:]

        # Check if we're in architecture refinement phase
        assembly_errors = [i for i in recent_iterations if i.get("assemble_error")]
        if assembly_errors:
            return "architecture"

        # Check if we're in evaluation phase
        evaluations = [i for i in recent_iterations if i.get("evaluation")]
        if evaluations:
            return "evaluation"

        # Default to planning
        return "planning"

    def _get_memory_context_for_phase(self, goal: str, research_phase: str) -> list[dict[str, Any]]:
        """Get memory context for the current research phase.

        Args:
            goal: The research goal
            research_phase: Current research phase

        Returns:
            List of relevant memory context items

        """
        if not self.memory_manager or not getattr(self.memory_manager, 'enabled', False):
            return []

        try:
            memories = self.memory_manager.get_smart_research_context(goal, research_phase)
            # Track memory usage for this phase
            if not hasattr(self, '_memory_usage_by_phase'):
                self._memory_usage_by_phase = {}
            if research_phase not in self._memory_usage_by_phase:
                self._memory_usage_by_phase[research_phase] = []

            # Store memory IDs used in this phase
            for memory in memories:
                memory_id = memory.get('id', 'unknown')
                if memory_id not in self._memory_usage_by_phase[research_phase]:
                    self._memory_usage_by_phase[research_phase].append(memory_id)

            logger.info(f"🧠 Memory context retrieved for {research_phase} phase: {len(memories)} memories")
            return memories
        except Exception as e:
            logger.warning(f"Failed to get memory context for phase '{research_phase}': {e}")
            return []

    def _enhanced_decision_making(
        self,
        goal: str,
        registry: Sequence[dict[str, Any]],
        latest_result: dict[str, Any],
        original_proposal: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Enhanced decision making with memory context.

        Args:
            goal: The research goal
            registry: Experiment registry with previous iterations
            latest_result: Latest iteration result
            original_proposal: Optional original proposal

        Returns:
            Planning LLM decision with memory-enhanced context

        """
        # Get memory context if available
        memory_context = []
        if self.memory_manager and getattr(self.memory_manager, 'enabled', False):
            research_phase = self._determine_research_phase(registry)
            memory_context = self._get_memory_context_for_phase(goal, research_phase)
            if memory_context:
                logger.info(f"🧠 Enhanced decision making with {len(memory_context)} memory insights for {research_phase} phase")

        # Enhanced decision with memory context
        return self.planning_client.decide_next_action(
            goal=goal,
            registry=registry,
            latest_result=latest_result,
            original_proposal=original_proposal,
            memory_context=memory_context
        )

    def _enhance_proposal_with_memory(
        self,
        proposal: dict[str, Any],
        memory_context: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Enhance initial proposal with relevant memory insights.

        Args:
            proposal: Original proposal from planning LLM
            memory_context: Relevant memory context to inject

        Returns:
            Enhanced proposal with memory insights

        """
        if not memory_context:
            return proposal

        enhanced_proposal = proposal.copy()

        # Add memory context to notes or metadata
        notes = enhanced_proposal.get("notes", "")
        memory_notes = "=== RELEVANT RESEARCH INSIGHTS ===\n"

        for context_item in memory_context:
            category = context_item.get("category_primary", "research")
            content = context_item.get("searchable_content", "") or context_item.get("summary", "")
            memory_notes += f"[{category.upper()}] {content}\n"

        memory_notes += "\n=== END RESEARCH INSIGHTS ===\n"

        enhanced_proposal["notes"] = memory_notes + notes
        enhanced_proposal["memory_enhanced"] = True

        logger.info(f"📝 Enhanced proposal with {len(memory_context)} memory insights")

        return enhanced_proposal

    def run(
        self,
        goal: str,
        workdir: str,
        keep_artifacts: bool = True,
        existing_proposal: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Run the Orchestrator loop.

        Args:
            goal: high-level research goal string.
            workdir: path where per-iteration artifacts (models, experiments) are written.
            keep_artifacts: whether artifacts should be kept (caller may remove workdir).
            existing_proposal: Optional pre-existing proposal to use instead of requesting a new one.

        Returns:
            A dict with details of the run, iterations and final decision.

        """
        run_report: dict[str, Any] = {"goal": goal, "iterations": []}

        logger.info(f"🎯 ORCHESTRATOR STARTED - Goal: {goal}")
        logger.info("📊 Planning initial model configuration...")

        # Use existing proposal or request a new one
        if existing_proposal:
            logger.info("🤖 Step 1: Using existing proposal from orchestrator")
            proposal = existing_proposal
            model_config = proposal.get("model_config")
            evaluation_config = proposal.get("evaluation_config")
        else:
            # 1) request initial proposal
            logger.info("🤖 Step 1: Requesting initial proposal from planning LLM...")

            # Get memory context for initial planning if enabled
            memory_context = []
            if self.memory_manager and getattr(self.memory_manager, 'enabled', False):
                memory_context = self._get_memory_context_for_phase(goal, "planning")

            proposal = self.planning_client.propose_initial_config(
                goal=goal, constraints=None, memory_context=memory_context
            )

            # Enhance proposal with memory insights if available
            if memory_context:
                proposal = self._enhance_proposal_with_memory(proposal, memory_context)

            model_config = proposal.get("model_config")
            evaluation_config = proposal.get("evaluation_config")  # Extract evaluation config for input sizing

        run_report["planning_proposal"] = proposal

        if model_config is None:
            raise OrchestratorError("Planning LLM returned proposal without model_config")

        logger.info(f"✅ Step 1: Initial proposal received - Architecture: {model_config.get('architecture', 'Unknown')}")

        for it in range(1, self.max_iterations + 1):
            logger.info(f"🔄 === ITERATION {it}/{self.max_iterations} ===")
            iter_entry: dict[str, Any] = {"iteration": it, "model_config": model_config}
            iteration_dir = Path(workdir) / f"iter_{it}"
            iteration_dir.mkdir(parents=True, exist_ok=True)
            model_path = str((iteration_dir / "model.py").resolve())

            # 2) assemble model (first attempt)
            logger.info(f"🔨 Step 2.{it}: Assembling model from configuration...")
            try:
                asm_res = self.assembler(
                    model_config, model_path, use_llm=True, llm_kwargs=None
                )
                iter_entry["assemble"] = asm_res
                logger.info(f"✅ Step 2.{it}: Model assembled successfully (via {asm_res.get('via', 'unknown')})")
            except Exception as e:
                logger.error(f"❌ Step 2.{it}: Model assembly failed: {e}")
                iter_entry["assemble_error"] = str(e)
                run_report["iterations"].append(iter_entry)

                # Ask planning LLM whether to stop (with memory context if available)
                logger.info(f"🤔 Step 2.{it}: Asking planning LLM what to do after assembly failure...")
                decision = self._enhanced_decision_making(
                    goal, run_report["iterations"], iter_entry,
                    original_proposal=run_report.get("planning_proposal")
                )
                run_report["last_decision"] = decision
                logger.info(f"🎯 Step 2.{it}: Planning LLM decision: {decision.get('action', 'unknown')}")

                if decision.get("action") == "stop":
                    run_report["final_status"] = "stopped_due_to_assembly_error"
                    logger.info("🛑 Stopping due to assembly error as requested by planning LLM")
                    break
                else:
                    # if refine suggested, continue loop with provided config
                    model_config = decision.get("next_config", model_config)
                    logger.info(f"🔧 Step 2.{it}: Continuing with refined configuration")
                    continue

            # 3) sandbox validate and summarize
            if self.sandbox_runner is not None:
                logger.info(f"🔬 Step 3.{it}: Running sandbox validation...")
                try:
                    sb_res = self.sandbox_runner(
                        model_path,
                        class_name=model_config.get("class_name"),
                        input_size=_get_input_size_from_config(model_config or {}, evaluation_config),
                        timeout=120,
                    )
                    iter_entry["sandbox"] = sb_res

                    if sb_res.get("success"):
                        logger.info(f"✅ Step 3.{it}: Sandbox validation PASSED")
                        if self.summarizer is not None:
                            logger.info(f"📝 Step 3.{it}: Generating model summary...")
                            try:
                                summ = self.summarizer(
                                    model_path,
                                    class_name=model_config.get("class_name"),
                                    input_size=_get_input_size_from_config(model_config or {}, evaluation_config),
                                )
                                iter_entry["summary"] = summ
                                logger.info(f"✅ Step 3.{it}: Model summary generated successfully")
                            except Exception as e:
                                logger.warning(f"⚠️ Step 3.{it}: Summary generation failed: {e}")
                                iter_entry["summary_error"] = str(e)
                        else:
                            logger.info(f"📝 Step 3.{it}: Summarizer not configured")
                    else:
                        logger.warning(f"❌ Step 3.{it}: Sandbox validation FAILED")
                        sandbox_error = sb_res.get("error") or sb_res.get("stderr")
                        iter_entry["sandbox_error"] = sandbox_error

                        # INTELLIGENT RETRY: If sandbox failed, retry assembly with error feedback
                        logger.info(f"🔄 Step 3.{it}: Sandbox failed - retrying assembly with error feedback...")
                        try:
                            retry_asm_res = self.assembler(
                                model_config, model_path, use_llm=True, llm_kwargs=None, sandbox_error=sandbox_error
                            )
                            iter_entry["assemble"] = retry_asm_res
                            logger.info(f"✅ Step 3.{it}: Model reassembled successfully with error feedback (via {retry_asm_res.get('via', 'unknown')})")

                            # Run sandbox again with the improved model
                            logger.info(f"🔬 Step 3.{it}: Re-running sandbox validation with improved model...")
                            retry_sb_res = self.sandbox_runner(
                                model_path,
                                class_name=model_config.get("class_name"),
                                input_size=_get_input_size_from_config(model_config or {}, evaluation_config),
                                timeout=120,
                            )
                            iter_entry["sandbox"] = retry_sb_res

                            if retry_sb_res.get("success"):
                                logger.info(f"✅ Step 3.{it}: Sandbox validation PASSED on retry!")
                            else:
                                logger.warning(f"❌ Step 3.{it}: Sandbox still failed after intelligent retry")
                                iter_entry["sandbox_error"] = retry_sb_res.get("error") or retry_sb_res.get("stderr")

                        except Exception as retry_e:
                            logger.error(f"❌ Step 3.{it}: Intelligent retry assembly failed: {retry_e}")
                            # Continue with original error

                except Exception as e:
                    logger.error(f"❌ Step 3.{it}: Sandbox execution failed: {e}")
                    iter_entry["sandbox_error"] = str(e)
            else:
                iter_entry["sandbox_error"] = "sandbox runner not configured"
                logger.warning(f"⚠️ Step 3.{it}: Sandbox runner not configured")

            # 4) quick evaluation (only if sandbox passed)
            if (
                iter_entry.get("sandbox", {}).get("success", False)
                and self.evaluator is not None
            ):
                logger.info(f"📊 Step 4.{it}: Running quick evaluation...")
                try:
                    eval_res = self.evaluator(model_path, model_config)
                    iter_entry["evaluation"] = eval_res
                    logger.info(f"✅ Step 4.{it}: Quick evaluation completed")
                except Exception as e:
                    logger.warning(f"⚠️ Step 4.{it}: Quick evaluation failed: {e}")
                    iter_entry["evaluation_error"] = str(e)
            else:
                logger.info(f"📊 Step 4.{it}: Skipping evaluation (sandbox failed or evaluator missing)")
                if "evaluation_error" not in iter_entry:
                    iter_entry["evaluation_error"] = (
                        "sandbox failed or evaluator missing"
                    )

            # 5) persist iteration to registry (best-effort)
            logger.info(f"💾 Step 5.{it}: Persisting iteration to registry...")
            try:
                if self.registry_writer is not None:
                    self.registry_writer(iter_entry)
                    iter_entry["persisted"] = True
                    logger.info(f"✅ Step 5.{it}: Iteration persisted successfully")
                else:
                    logger.warning(f"⚠️ Step 5.{it}: Registry writer not configured")
            except Exception as e:
                logger.warning(f"⚠️ Step 5.{it}: Failed to persist iteration: {e}")
                iter_entry["persist_error"] = str(e)

            run_report["iterations"].append(iter_entry)

            # 6) planning decision
            latest_result = (
                iter_entry.get("evaluation")
                or iter_entry.get("summary")
                or iter_entry.get("sandbox")
                or {}
            )
            logger.info(f"🤔 Step 6.{it}: Asking planning LLM for next action...")
            try:
                decision = self._enhanced_decision_making(
                    goal=goal,
                    registry=run_report["iterations"],
                    latest_result=latest_result,
                    original_proposal=run_report.get("planning_proposal"),
                )
                run_report["last_decision"] = decision
                action = decision.get("action")
                logger.info(f"🎯 Step 6.{it}: Planning LLM decision: {action}")

                if action == "achieve":
                    logger.info("🏆 Goal achieved! Planning LLM says we can stop.")
                    run_report["final_status"] = "achieved"
                    break
                if action == "stop":
                    logger.info("🛑 Planning LLM says to stop.")
                    run_report["final_status"] = "stopped"
                    break
                if action == "refine":
                    model_config = decision.get("next_config") or model_config
                    logger.info("🔧 Planning LLM suggests refinement, continuing to next iteration")
                    # continue to next iteration
                    continue

                # unknown action -> stop
                logger.warning(f"⚠️ Unknown action '{action}' received, stopping")
                run_report["final_status"] = f"unknown_action:{action}"
                break

            except Exception as e:
                # If planning LLM fails, stop for safety
                logger.error(f"❌ Step 6.{it}: Planning LLM failed: {e}")
                run_report["final_status"] = f"planning_error: {e}"
                break

        else:
            logger.info(f"⏰ Reached maximum iterations ({self.max_iterations})")
            run_report["final_status"] = "max_iterations_reached"

        logger.info(f"🏁 ORCHESTRATOR COMPLETED - Final status: {run_report.get('final_status')}")
        return run_report

    def run_with_proposal(
        self,
        goal: str,
        workdir: str,
        keep_artifacts: bool = True,
        initial_proposal: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Run the Orchestrator loop with an initial proposal.

        This method allows starting the research with a pre-configured proposal,
        which is useful when the agent_orchestrator has already obtained an
        initial proposal from the planning LLM.

        Args:
            goal: high-level research goal string.
            workdir: path where per-iteration artifacts (models, experiments) are written.
            keep_artifacts: whether artifacts should be kept (caller may remove workdir).
            initial_proposal: Optional initial proposal from planning LLM with model_config and evaluation_config.

        Returns:
            A dict with details of the run, iterations and final decision.

        """
        # Run the orchestrator with the existing proposal
        return self.run(goal=goal, workdir=workdir, keep_artifacts=keep_artifacts, existing_proposal=initial_proposal)


def _get_input_size_from_config(model_config: dict[str, Any], evaluation_config: dict[str, Any] | None = None) -> tuple:
    """Extract the correct input size tuple from model configuration for sandbox testing.

    Uses a prioritized approach:
    1. Extract from model_config input_shape (if available)
    2. Infer from dataset name in evaluation_config
    3. Fallback to CIFAR-10 format

    Args:
        model_config: Dictionary containing model configuration with layers
        evaluation_config: Optional evaluation configuration with dataset_name

    Returns:
        tuple: Input size tuple including batch dimension, e.g., (1, 3, 32, 32) for CIFAR-10
               or (1, 1, 28, 28) for MNIST

    """
    # Dataset-specific input sizes as primary fallback
    dataset_input_sizes = {
        "mnist": (1, 1, 28, 28),
        "fashion_mnist": (1, 1, 28, 28),
        "cifar10": (1, 3, 32, 32),
        "cifar100": (1, 3, 32, 32),
        "imagenet": (1, 3, 224, 224),
    }

    # 1. Try to extract from model_config first
    try:
        if isinstance(model_config, dict) and "layers" in model_config:
            # Look for input_shape in the first layer (common pattern)
            for layer in model_config["layers"]:
                if isinstance(layer, dict):
                    input_shape = layer.get("input_shape")
                    if input_shape and isinstance(input_shape, (list, tuple)) and len(input_shape) >= 3:
                        # Convert [height, width, channels] to [batch, channels, height, width] for PyTorch
                        height, width, channels = input_shape[-3:]
                        result = (1, channels, height, width)
                        logger.debug(f"Extracted input shape from layer config: {result}")
                        return result

            # Alternative: look for input_shape in the model_config directly
            input_shape = model_config.get("input_shape")
            if input_shape and isinstance(input_shape, (list, tuple)) and len(input_shape) >= 3:
                height, width, channels = input_shape[-3:]
                result = (1, channels, height, width)
                logger.debug(f"Extracted input shape from model_config: {result}")
                return result
    except Exception as e:
        logger.warning(f"Failed to extract input shape from model config: {e}")

    # 2. Infer from dataset name if provided
    if evaluation_config and isinstance(evaluation_config, dict):
        dataset_name = evaluation_config.get("dataset_name", "").lower()
        if dataset_name in dataset_input_sizes:
            result = dataset_input_sizes[dataset_name]
            logger.debug(f"Inferred input size from dataset '{dataset_name}': {result}")
            return result

    # 3. Fallback to CIFAR-10 format
    default_input_size = (1, 3, 32, 32)
    logger.debug(f"Using default input size: {default_input_size}")
    return default_input_size
