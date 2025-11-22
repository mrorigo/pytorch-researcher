#!/usr/bin/env python3
"""Manual Memory Context Manager for PyTorch Research Agent

This module provides complete manual control over Memori memory operations
without auto-interception, enabling selective memory context injection
into LLM calls.
"""

from __future__ import annotations

import logging
from typing import Any

# Try to import ConfigManager, fall back to basic configuration if not available
from memori import ConfigManager, Memori

logger = logging.getLogger(__name__)


class ManualMemoryContextManager:
    """Manual memory context manager for controlled memory operations.

    This class provides complete manual control over Memori operations
    without auto-interception, allowing precise control over when and
    how memory is retrieved and injected into LLM calls.
    """

    def __init__(self, memori_instance: Memori):
        """Initialize the manual memory context manager.

        Args:
            memori_instance: Memori instance configured for manual operation

        """
        self.memori = memori_instance
        self.enabled = False
        self.namespace = memori_instance.namespace
        logger.info(f"ManualMemoryContextManager initialized for namespace: {self.namespace}")

    def enable_manual_mode(self):
        """Enable manual memory operations without auto-interception."""
        self.enabled = True
        logger.info("Manual memory mode enabled - all operations are explicit")

    def disable_manual_mode(self):
        """Disable manual memory operations."""
        self.enabled = False
        logger.info("Manual memory mode disabled")

    def get_research_context(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Get relevant research context explicitly.

        NOTE: This method is currently not used in the main research pipeline.
        The smart research context method provides more intelligent context retrieval.

        Args:
            query: The search query for relevant context
            limit: Maximum number of context items to return

        Returns:
            List of relevant memory items with metadata

        """
        if not self.enabled:
            logger.debug("Manual memory mode disabled - returning empty context")
            return []

        try:
            context_items = self.memori.retrieve_context(query, limit=limit)
            logger.debug(f"Retrieved {len(context_items)} context items for query: '{query[:50]}...'")
            return context_items
        except Exception as e:
            logger.error(f"Failed to retrieve research context: {e}")
            return []

    def get_conscious_context(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get conscious/essential conversations from short-term memory.

        NOTE: This method is used internally by get_smart_research_context()
        for planning phase enhancements.

        Args:
            limit: Maximum number of essential conversations to return

        Returns:
            List of essential conversation memories

        """
        if not self.enabled:
            return []

        try:
            essential_conversations = self.memori.get_essential_conversations(limit=limit)
            logger.debug(f"Retrieved {len(essential_conversations)} essential conversations")
            return essential_conversations
        except Exception as e:
            logger.error(f"Failed to get conscious context: {e}")
            return []

    def record_research_insight(
        self,
        insight_type: str,
        content: str,
        importance: str = "medium",
        metadata: dict[str, Any] | None = None
    ) -> str:
        """Manually record research insights to memory.

        Args:
            insight_type: Type of insight (dataset_patterns, architecture_patterns, etc.)
            content: The insight content
            importance: Importance level (low, medium, high)
            metadata: Additional metadata

        Returns:
            Memory ID for tracking

        """
        if not self.enabled:
            logger.debug("Manual memory mode disabled - not recording insight")
            return ""

        try:
            # Create research-specific metadata
            research_metadata = {
                "source": "manual_research_insight",
                "insight_type": insight_type,
                "importance": importance,
                "recorded_by": "research_orchestrator",
                **(metadata or {})
            }

            # Format content with category prefix
            categorized_content = f"[{insight_type.upper()}] {content}"

            memory_id = self.memori.record_conversation(
                user_input=categorized_content,
                ai_output="Research insight recorded",
                metadata=research_metadata
            )

            logger.info(f"Recorded research insight of type '{insight_type}' with ID: {memory_id[:8]}...")
            return memory_id

        except Exception as e:
            logger.error(f"Failed to record research insight: {e}")
            return ""

    def analyze_and_record_research_session(self, run_report: dict[str, Any]) -> dict[str, str]:
        """Analyze completed research session and extract insights.

        Args:
            run_report: Complete run report from orchestrator

        Returns:
            Dictionary mapping insight types to memory IDs

        """
        if not self.enabled:
            logger.debug("Manual memory mode disabled - not analyzing session")
            return {}

        logger.info("Analyzing research session for insights...")
        recorded_insights = {}

        try:
            # Extract successful iterations (those that achieved target accuracy)
            successful_iterations = [
                iter_data for iter_data in run_report.get("iterations", [])
                if iter_data.get("evaluation", {}).get("accuracy", 0) > 0.7
            ]

            logger.info(f"Found {len(successful_iterations)} successful iterations")

            # Record architecture patterns from successful iterations
            for iteration in successful_iterations:
                model_config = iteration.get("model_config", {})
                eval_results = iteration.get("evaluation", {})

                if "architecture" in model_config:
                    architecture_insight = (
                        f"Successful {model_config['architecture']} architecture "
                        f"achieved {eval_results.get('accuracy', 0):.3f} accuracy"
                    )
                    memory_id = self.record_research_insight(
                        "architecture_patterns",
                        architecture_insight,
                        importance="high",
                        metadata={
                            "source_iteration": iteration.get("iteration"),
                            "accuracy": eval_results.get("accuracy"),
                            "dataset": eval_results.get("dataset_name")
                        }
                    )
                    recorded_insights[f"architecture_{iteration.get('iteration')}"] = memory_id

            # Record dataset effectiveness patterns
            eval_results = run_report.get("iterations", [{}])[-1].get("evaluation", {})
            if eval_results:
                dataset_insight = (
                    f"Dataset evaluation: {eval_results.get('dataset_name', 'unknown')} "
                    f"reached {eval_results.get('accuracy', 0):.3f} accuracy with target {eval_results.get('target_accuracy', 'N/A')}"
                )
                memory_id = self.record_research_insight(
                    "dataset_patterns",
                    dataset_insight,
                    importance="medium",
                    metadata=eval_results
                )
                recorded_insights["dataset_patterns"] = memory_id

            # Record research methodology insights
            final_status = run_report.get("final_status", "unknown")
            if final_status in ["achieved", "stopped"]:
                methodology_insight = (
                    f"Research methodology led to {final_status} outcome. "
                    f"Total iterations: {len(run_report.get('iterations', []))}"
                )
                memory_id = self.record_research_insight(
                    "research_methodologies",
                    methodology_insight,
                    importance="medium",
                    metadata={
                        "final_status": final_status,
                        "total_iterations": len(run_report.get("iterations", [])),
                        "goal": run_report.get("goal", "")
                    }
                )
                recorded_insights["research_methodologies"] = memory_id

            # Record failure analysis if applicable
            failed_iterations = [
                iter_data for iter_data in run_report.get("iterations", [])
                if iter_data.get("evaluation_error") or iter_data.get("sandbox_error")
            ]

            if failed_iterations:
                failure_analysis = f"Found {len(failed_iterations)} failed iterations with errors"
                memory_id = self.record_research_insight(
                    "failure_patterns",
                    failure_analysis,
                    importance="high",
                    metadata={
                        "failed_iterations": len(failed_iterations),
                        "total_iterations": len(run_report.get("iterations", [])),
                        "failure_rate": len(failed_iterations) / len(run_report.get("iterations", []))
                    }
                )
                recorded_insights["failure_patterns"] = memory_id

            logger.info(f"Successfully recorded {len(recorded_insights)} research insights")
            return recorded_insights

        except Exception as e:
            logger.error(f"Failed to analyze research session: {e}")
            return {}

    def get_smart_research_context(self, current_goal: str, research_phase: str) -> list[dict[str, Any]]:
        """Get intelligent memory context based on current research phase.

        Args:
            current_goal: The current research goal
            research_phase: Current phase of research (planning, architecture, evaluation, etc.)

        Returns:
            List of relevant memory contexts filtered and prioritized

        """
        if not self.enabled:
            return []

        try:
            # Define context queries based on research phase
            context_queries = {
                "planning": f"research planning patterns for {current_goal}",
                "architecture": f"successful model architectures for {current_goal}",
                "evaluation": f"evaluation strategies and metrics for {current_goal}",
                "failure_patterns": f"common failure modes in {current_goal} research",
                "dataset_recommendations": f"recommended datasets for {current_goal}",
                "methodology": f"effective research methodologies for {current_goal}",
            }

            # Get query for current phase
            query = context_queries.get(research_phase, f"research insights for {current_goal}")

            # Retrieve context
            relevant_contexts = self.memori.retrieve_context(query, limit=5)

            # Add essential conversations if in planning phase
            if research_phase == "planning":
                essential_conversations = self.get_conscious_context(limit=3)
                relevant_contexts.extend(essential_conversations)

            # Remove duplicates and limit total results
            seen_content = set()
            filtered_contexts = []
            for context in relevant_contexts:
                content = context.get("searchable_content", "") or context.get("summary", "")
                content_key = content.lower().strip()

                if content_key and content_key not in seen_content:
                    seen_content.add(content_key)
                    filtered_contexts.append(context)

                if len(filtered_contexts) >= 10:  # Limit total context to prevent overwhelming
                    break

            logger.debug(f"Smart context retrieval for {research_phase} phase: {len(filtered_contexts)} items")
            return filtered_contexts

        except Exception as e:
            logger.error(f"Failed to get smart research context: {e}")
            return []

    def format_memory_context_for_llm(self, memory_context: list[dict[str, Any]]) -> str:
        """Format memory context for LLM consumption.

        Args:
            memory_context: List of memory context items

        Returns:
            Formatted context string for LLM system prompt

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

    def get_memory_stats(self) -> dict[str, Any]:
        """Get memory statistics for monitoring.

        NOTE: This method is currently not used in the main research pipeline.
        Intended for future monitoring and debugging capabilities.

        Returns:
            Dictionary containing memory system statistics

        """
        if not self.enabled:
            return {"enabled": False}

        try:
            # Fallback implementation for v2.1.1 (no get_memory_stats method)
            stats = {
                "enabled": True,
                "mode": "manual",
                "conscious_ingest": True,
                "auto_ingest": False,
                "namespace": self.namespace,
                "api_version": "2.1.1"
            }
            return stats
        except Exception as e:
            logger.error(f"Failed to get memory stats: {e}")
            return {"enabled": True, "error": str(e)}

    def search_memories(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """Search memories with a custom query (using retrieve_context for v2.1.1 compatibility).

        NOTE: This method is currently not used in the main research pipeline.
        Provides direct search capability for debugging and advanced use cases.

        Args:
            query: Search query
            limit: Maximum number of results

        Returns:
            List of matching memories

        """
        if not self.enabled:
            return []

        try:
            return self.memori.retrieve_context(query, limit=limit)
        except Exception as e:
            logger.error(f"Failed to search memories: {e}")
            return []

    def clear_memory(self, memory_type: str = "all"):
        """Clear memory data (use with caution).

        NOTE: This method is currently not used in the main research pipeline.
        Intended for development and testing purposes only.

        Args:
            memory_type: Type of memory to clear ('short_term', 'long_term', 'all')

        """
        if not self.enabled:
            logger.warning("Manual memory mode disabled - cannot clear memory")
            return

        try:
            # Fallback implementation for v2.1.1 (no clear_memory method)
            # This is a placeholder - actual implementation would need database operations
            logger.warning(f"clear_memory not available in Memori v2.1.1, skipping {memory_type} memory clear")
            logger.info(f"Memory clear operation skipped for namespace: {self.namespace}")
        except Exception as e:
            logger.error(f"Failed to clear memory: {e}")


def create_manual_memory_manager(
    llm_base_url: str,
    llm_model: str,
    llm_api_key: str | None = None,
    verbose: bool = False
) -> ManualMemoryContextManager:
    """Create and configure a manual memory manager.

    Args:
        llm_base_url: Base URL for the LLM endpoint
        llm_model: Model name for the LLM
        llm_api_key: Optional API key for the LLM
        verbose: Enable verbose logging

    Returns:
        Configured ManualMemoryContextManager instance

    """
    try:
        # Load configuration (with fallback for different Memori versions)
        config = ConfigManager()
        config.auto_load()
        db_connection = config.get_setting("database_connection_string", default="sqlite:///pytorch_researcher_memori.db")
        openai_api_key_env = config.get_setting("OPENAI_API_KEY")

        # Base Memori configuration for Option A (conscious processing without auto-interception)
        memori_config = {
            "conscious_ingest": True,   # Enable conscious memory processing
            "auto_ingest": False,       # Disable auto-interception
            "database_connect": db_connection,
            "namespace": "ml_research",
            "verbose": verbose,
        }

        # Configure provider based on LLM endpoint
        if "localhost" in llm_base_url or "127.0.0.1" in llm_base_url:
            # For local endpoints (Ollama)
            memori_config.update({
                "model": llm_model,
                "base_url": llm_base_url,
                "api_key": "local",
            })
            logger.info(f"Configuring manual memory for local endpoint: {llm_base_url}")
        elif "openrouter.ai" in llm_base_url:
            # For OpenRouter endpoints
            openrouter_api_key = llm_api_key or openai_api_key_env
            if openrouter_api_key:
                memori_config.update({
                    "model": llm_model,
                    "base_url": llm_base_url,
                    "api_key": openrouter_api_key,
                })
                logger.info(f"Configuring manual memory for OpenRouter endpoint: {llm_base_url}")
            else:
                logger.warning("No API key provided for manual memory OpenRouter configuration")
        else:
            # For other cloud endpoints (OpenAI, etc.)
            cloud_api_key = llm_api_key or openai_api_key_env
            if cloud_api_key:
                memori_config.update({
                    "model": llm_model,
                    "api_key": cloud_api_key,
                })
                logger.info("Configuring manual memory for cloud OpenAI endpoint")
            else:
                logger.warning("No API key provided for manual memory cloud configuration")
                # Fallback to local configuration
                memori_config.update({
                    "model": llm_model,
                    "base_url": llm_base_url,
                    "api_key": "local",
                })

        # Create Memori instance configured for conscious research memory (manual mode)
        memori_instance = Memori(**memori_config)

        # Enable the Memori instance for memory operations
        try:
            memori_instance.enable()
            logger.info("🧠 MEMORI: Memori backend enabled successfully")
        except Exception as e:
            logger.warning(f"🧠 MEMORI: Failed to enable Memori backend: {e}")

        # Create and return manual memory manager
        memory_manager = ManualMemoryContextManager(memori_instance)
        # Enable manual mode for conscious research memory management
        memory_manager.enable_manual_mode()

        logger.info("Manual memory manager created and enabled for conscious research memory")
        return memory_manager

    except Exception as e:
        logger.error(f"Failed to create manual memory manager: {e}")
        # Return disabled manager as fallback
        disabled_memori = Memori()
        memory_manager = ManualMemoryContextManager(disabled_memori)
        return memory_manager
