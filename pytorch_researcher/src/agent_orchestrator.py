#!/usr/bin/env python3
"""
ML Research Agent Orchestrator - baseline implementation

This module provides the primary CLI entrypoint for the research agent. It orchestrates
the planning LLM, LLM-backed model assembler, sandbox validator, summarizer,
quick evaluator, and a persistent registry for iterative research.

This implementation assumes core components are present and will fail fast if
dependencies are missing, keeping the MVP lean and auditable.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

# Planning LLM and high-level orchestrator
from pytorch_researcher.src.planning_llm.client import (
    Orchestrator,
    PlanningLLMClient,
)

# LLM-backed assembler wrapper
from pytorch_researcher.src.pytorch_tools.model_assembler_llm import (
    assemble_from_config,
)

# Summarizer, sandbox and quick evaluator components
from pytorch_researcher.src.pytorch_tools.model_summary import summarize_model_from_path
from pytorch_researcher.src.pytorch_tools.quick_evaluator import (
    QuickEvalConfig,
    quick_evaluate_once,
)
from pytorch_researcher.src.tools.sandbox.sandbox_runner import run_sandboxed_harness

# Core utilities (project scaffold + simple file helpers)
from pytorch_researcher.src.utils import (
    create_pytorch_project_scaffold,
    read_file,
    write_file,
)

LOG = logging.getLogger("agent.orchestrator")


def _ensure_logger(verbose: bool = False) -> None:
    """Ensure the root logger is configured for console output."""
    if not LOG.handlers:
        handler = logging.StreamHandler(sys.stdout)
        fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
        handler.setFormatter(logging.Formatter(fmt))
        LOG.addHandler(handler)
    LOG.setLevel(logging.DEBUG if verbose else logging.INFO)


def _registry_path_for(project_root: str) -> Path:
    """Return the canonical path to the registry.json within a project."""
    return Path(project_root) / "experiments" / "registry.json"


def _append_registry(project_root: str, entry: Dict[str, Any]) -> None:
    """Append a new entry to the project's registry.json."""
    reg = _registry_path_for(project_root)
    try:
        current = json.loads(read_file(str(reg)))
    except Exception:
        current = []
    current.append(entry)
    write_file(str(reg), json.dumps(current, indent=2), overwrite=True)


def _build_evaluator_callable() -> callable:
    """Return a callable for model evaluation suitable for the orchestrator.

    The callable signature is `(model_path: str, model_config: dict) -> dict`.
    It dynamically imports the generated model, instantiates the class named
    in `model_config["class_name"]` (or `AssembledModel`), and runs a
    one-shot quick evaluation using `quick_evaluate_once`.
    """

    def _evaluator(model_path: str, model_config: Dict[str, Any]) -> Dict[str, Any]:
        try:
            # Import module dynamically (using __import__ to avoid top-level import errors)
            # This ensures modules are loaded into a distinct namespace per evaluation.
            import importlib.util as _il

            # Generate a unique module name for each evaluation to prevent conflicts
            mod_name = f"eval_mod_{int(datetime.utcnow().timestamp())}"
            spec = _il.spec_from_file_location(mod_name, model_path)
            assert (
                spec is not None and spec.loader is not None
            ), "Failed to create module spec."
            module = _il.module_from_spec(spec)
            spec.loader.exec_module(module)  # type: ignore

            # Determine class name from config, fallback to 'AssembledModel'
            cls = None
            if isinstance(model_config, dict):
                cls_name = model_config.get("class_name")
                if cls_name:
                    cls = getattr(module, cls_name, None)
            if cls is None:
                cls = getattr(module, "AssembledModel", None)
            if cls is None:
                raise RuntimeError("Could not locate model class in generated module")

            # Instantiate model and run quick evaluation
            inst = cls()
            cfg = QuickEvalConfig(epochs=1, subset_size=512, batch_size=32)
            # direct call, as quick_evaluate_once is now guaranteed to be imported
            return quick_evaluate_once(inst, cfg)
        except Exception as exc:
            return {"error": str(exc)}

    return _evaluator


def run(
    *,
    goal: str,
    project_name: str | None = None,
    keep: bool = False,
    llm_base_url: str,
    llm_model: str = "gpt-oss:20b",
    llm_api_key: str | None = None,
    max_iter: int = 5,
    target_accuracy: float = 0.7,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run the planning-driven orchestrator and return a run report.

    This function sets up the project scaffold, wires the necessary components,
    and executes the iterative research loop. It assumes core components are
    importable and will fail fast (raise) if any are missing.

    Args:
        goal: The high-level research goal for the agent.
        project_name: Optional name for the project directory. If None, a timestamped name is generated.
        keep: If True, keep the created project directory and artifacts after the run.
        llm_base_url: Base URL for the Planning LLM HTTP endpoint (e.g., http://localhost:11434/v1).
        llm_model: Model name for the Planning LLM.
        llm_api_key: Optional API key for the Planning LLM.
        max_iter: Maximum number of iterations for the research loop.
        target_accuracy: Target accuracy for the quick evaluation to consider the goal achieved.
        verbose: If True, enable debug logging.

    Returns:
        A dictionary containing `project_root` and the orchestrator's `report`.
    """
    _ensure_logger(verbose)
    results: Dict[str, Any] = {}
    project_root = None

    try:
        if not project_name:
            timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
            project_name = f"run_{timestamp}"

        project_root = create_pytorch_project_scaffold(project_name)
        LOG.info("Created project scaffold at %s", project_root)
        results["project_root"] = project_root

        # Instantiate planning LLM client
        planning_client = PlanningLLMClient(
            base_url=llm_base_url, model=llm_model, api_key=llm_api_key
        )

        # Wire assembler, summarizer, sandbox, and evaluator callables
        assembler_callable = assemble_from_config
        summarizer_callable = summarize_model_from_path
        sandbox_callable = run_sandboxed_harness
        evaluator_callable = _build_evaluator_callable()

        # Registry writer for persisting iteration data
        registry_writer = lambda entry: _append_registry(project_root, entry)

        # Instantiate orchestrator
        orchestrator = Orchestrator(
            planning_client=planning_client,
            assembler=assembler_callable,
            summarizer=summarizer_callable,
            evaluator=evaluator_callable,
            sandbox_runner=sandbox_callable,
            registry_writer=registry_writer,
            max_iterations=max_iter,
            target_accuracy=target_accuracy,
        )

        LOG.info("Starting orchestrator loop (goal=%r)", goal)
        report = orchestrator.run(goal=goal, workdir=project_root, keep_artifacts=keep)
        LOG.info(
            "Orchestrator finished with final_status=%s", report.get("final_status")
        )

        # Persist a top-level run record
        run_record = {
            "run_id": f"run-{datetime.utcnow().isoformat()}",
            "timestamp": datetime.utcnow().isoformat(),
            "goal": goal,
            "report": report,
        }
        _append_registry(project_root, run_record)

        results["report"] = report
        results["success"] = True
        return results

    except (
        Exception
    ) as exc:  # pragma: no cover - top-level orchestration errors surfaced to caller
        LOG.exception("Orchestrator run failed: %s", exc)
        results["error"] = str(exc)
        results["success"] = False
        return results

    finally:
        # Cleanup unless requested otherwise
        if project_root and not keep:
            try:
                shutil.rmtree(project_root)
                LOG.info("Removed project artifacts at %s", project_root)
                results["cleanup"] = "removed"
            except Exception as cleanup_exc:
                LOG.warning(
                    "Failed to cleanup project directory %s: %s",
                    project_root,
                    cleanup_exc,
                )
                results["cleanup_error"] = str(cleanup_exc)
        elif project_root and keep:
            LOG.info("Keeping project artifacts at %s", project_root)
            results["cleanup"] = "kept"


def _parse_cli_args() -> argparse.Namespace:
    """Parse command-line arguments for the orchestrator."""
    p = argparse.ArgumentParser(
        prog="ml_research_agent_orchestrator",
        description="ML Research Agent Orchestrator",
    )
    p.add_argument("--goal", "-g", required=True, help="High-level research goal")
    p.add_argument("--name", "-n", default=None, help="Optional project/run name")
    p.add_argument("--keep", action="store_true", help="Keep artifacts after run")
    p.add_argument(
        "--llm-base-url",
        required=True,
        help="Planning LLM base URL (e.g. http://localhost:11434/v1)",
    )
    p.add_argument("--llm-model", default="gpt-oss:20b", help="Planning LLM model name")
    p.add_argument("--llm-api-key", default=None, help="Optional LLM API key")
    p.add_argument("--max-iter", type=int, default=5, help="Maximum iterations")
    p.add_argument(
        "--target-accuracy",
        type=float,
        default=0.7,
        help="Target accuracy for stopping",
    )
    p.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")
    return p.parse_args()


def main() -> None:
    """Main entrypoint for the orchestrator CLI."""
    args = _parse_cli_args()
    _ensure_logger(args.verbose)

    try:
        result = run(
            goal=args.goal,
            project_name=args.name,
            keep=args.keep,
            llm_base_url=args.llm_base_url,
            llm_model=args.llm_model,
            llm_api_key=args.llm_api_key,
            max_iter=args.max_iter,
            target_accuracy=args.target_accuracy,
            verbose=args.verbose,
        )
    except Exception as exc:
        LOG.exception("Orchestrator run encountered a fatal error: %s", exc)
        print("\n=== ML Research Agent Orchestrator Run ===")
        print("Status: ERROR")
        print(str(exc))
        sys.exit(1)

    print("\n=== ML Research Agent Orchestrator Run ===")
    print(f"project_root: {result.get('project_root')}")
    print("Status: SUCCESS")
    print("report:", result.get("report"))


if __name__ == "__main__":
    main()
