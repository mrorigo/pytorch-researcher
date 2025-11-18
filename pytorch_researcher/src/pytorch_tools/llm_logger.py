#!/usr/bin/env python3
"""
LLM Interaction Logger

This module provides comprehensive logging for all LLM interactions in the system.
It captures prompts, responses, metadata, and performance metrics to help improve
the system, memory management, and prompt engineering.

Features:
- JSON-structured logs for easy analysis
- Comprehensive metadata capture
- Performance metrics tracking
- Error logging and debugging support
- Memory context logging
- Run-level and iteration-level organization
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import threading
import os


class LLMInteractionLogger:
    """
    Comprehensive LLM interaction logger.
    
    Captures all LLM calls with rich metadata for analysis and improvement.
    """
    
    def __init__(self, log_dir: str = "llm_logs"):
        """
        Initialize the LLM logger.
        
        Args:
            log_dir: Directory to store log files
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Thread-safe counter for log entries
        self._counter = 0
        self._lock = threading.Lock()
        
        # Configure logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def _get_next_id(self) -> int:
        """Get next unique ID for log entry (thread-safe)."""
        with self._lock:
            self._counter += 1
            return self._counter
    
    def log_llm_interaction(
        self,
        call_type: str,
        model_name: str,
        provider: str,
        messages: List[Dict[str, str]],
        response: Any,
        duration: float,
        temperature: float = 0.0,
        timeout: Optional[int] = None,
        error: Optional[str] = None,
        memory_context: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        run_id: Optional[str] = None,
        iteration: Optional[int] = None,
    ) -> str:
        """
        Log an LLM interaction with comprehensive details.
        
        Args:
            call_type: Type of call (e.g., 'propose_initial_config', 'decide_next_action')
            model_name: Name of the LLM model used
            provider: LLM provider (e.g., 'openai', 'openrouter', 'local')
            messages: List of messages sent to LLM
            response: Response from LLM
            duration: Time taken for the call in seconds
            temperature: Temperature parameter used
            timeout: Timeout parameter used
            error: Error message if call failed
            memory_context: Memory context injected (if any)
            metadata: Additional metadata
            run_id: Current run ID
            iteration: Current iteration number
            
        Returns:
            Log entry ID for reference
        """
        log_id = self._get_next_id()
        timestamp = datetime.now().isoformat()
        
        # Create structured log entry
        log_entry = {
            "log_id": log_id,
            "timestamp": timestamp,
            "call_type": call_type,
            "run_id": run_id,
            "iteration": iteration,
            "model": {
                "name": model_name,
                "provider": provider,
            },
            "request": {
                "messages": messages,
                "temperature": temperature,
                "timeout": timeout,
                "total_messages": len(messages),
            },
            "response": {
                "content": self._extract_response_content(response),
                "raw_response": self._serialize_response_for_logging(response),
                "response_type": type(response).__name__,
            },
            "performance": {
                "duration_seconds": duration,
                "timestamp_start": timestamp,
                "timestamp_end": datetime.now().isoformat(),
            },
            "metadata": metadata or {},
            "error": error,
        }
        
        # Add memory context if present
        if memory_context:
            log_entry["memory_context"] = {
                "enabled": True,
                "context_items": len(memory_context),
                "context_preview": [
                    {
                        "category": item.get("category_primary", "unknown"),
                        "content_preview": item.get("searchable_content", "")[:200] + "..." if len(item.get("searchable_content", "")) > 200 else item.get("searchable_content", ""),
                    }
                    for item in memory_context[:5]  # Limit to first 5 items
                ],
            }
        else:
            log_entry["memory_context"] = {"enabled": False}
        
        # Save to file
        self._save_log_entry(log_entry)
        
        # Log to console for debugging
        status = "ERROR" if error else "SUCCESS"
        self.logger.info(f"LLM {call_type} ({status}) - Model: {model_name}, Duration: {duration:.2f}s, Log ID: {log_id}")
        
        return f"log_{log_id}"
    
    def _extract_response_content(self, response: Any) -> Optional[str]:
            """Extract readable content from various LLM response formats."""
            try:
                # Handle LiteLLM response format
                if hasattr(response, 'choices') and response.choices:
                    choice = response.choices[0]
                    if hasattr(choice, 'message') and choice.message:
                        return choice.message.content
                    elif hasattr(choice, 'text'):
                        return choice.text
                
                # Handle dictionary responses
                if isinstance(response, dict):
                    if 'content' in response:
                        return response['content']
                    elif 'choices' in response and response['choices']:
                        choice = response['choices'][0]
                        if 'message' in choice and 'content' in choice['message']:
                            return choice['message']['content']
                        elif 'text' in choice:
                            return choice['text']
                
                # Handle string responses
                if isinstance(response, str):
                    return response
                
                # Fallback
                return str(response)
            except Exception as e:
                self.logger.warning(f"Failed to extract response content: {e}")
                return str(response)
                
    def _serialize_response_for_logging(self, response: Any) -> str:
        """
        Serialize response object for JSON logging.
        
        Args:
            response: The LLM response object
            
        Returns:
            JSON-serializable string representation of the response
        """
        try:
            # Handle LiteLLM response format
            if hasattr(response, 'choices') and response.choices:
                choice = response.choices[0]
                if hasattr(choice, 'message') and choice.message:
                    # For LiteLLM responses, serialize the full choice structure
                    return json.dumps({
                        "type": "litellm_response",
                        "choices": [{
                            "message": {
                                "content": choice.message.content,
                                "role": getattr(choice.message, 'role', 'assistant')
                            }
                        }]
                    }, ensure_ascii=False, indent=2)
            
            # Handle dictionary responses
            if isinstance(response, dict):
                return json.dumps(response, ensure_ascii=False, indent=2)
            
            # Handle string responses
            if isinstance(response, str):
                return response
            
            # Handle lists and other JSON-serializable types
            try:
                return json.dumps(response, ensure_ascii=False, indent=2)
            except (TypeError, ValueError):
                # Fallback for non-JSON-serializable objects
                return str(response)
                
        except Exception as e:
            self.logger.warning(f"Failed to serialize response for logging: {e}")
            return str(response)
    
    def _save_log_entry(self, log_entry: Dict[str, Any]) -> None:
        """Save log entry to file."""
        try:
            # Determine file path based on run_id and timestamp
            timestamp = log_entry['timestamp']
            run_id = log_entry.get('run_id', 'unknown_run')
            
            # Create run-specific subdirectory
            run_dir = self.log_dir / f"run_{run_id}"
            run_dir.mkdir(exist_ok=True)
            
            # Use date-based filename for organization
            date_str = timestamp.split('T')[0]  # YYYY-MM-DD
            filename = f"llm_interactions_{date_str}.jsonl"
            filepath = run_dir / filename
            
            # Append to file (JSONL format)
            with open(filepath, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
                
        except Exception as e:
            self.logger.error(f"Failed to save log entry: {e}")
    
    def create_run_summary(self, run_id: str) -> Dict[str, Any]:
        """
        Create a summary of all LLM interactions for a specific run.
        
        Args:
            run_id: The run ID to summarize
            
        Returns:
            Dictionary containing run summary statistics
        """
        run_dir = self.log_dir / f"run_{run_id}"
        if not run_dir.exists():
            return {"error": f"No logs found for run {run_id}"}
        
        summary = {
            "run_id": run_id,
            "total_calls": 0,
            "call_types": {},
            "models_used": {},
            "total_duration": 0.0,
            "errors": 0,
            "memory_enhanced_calls": 0,
            "average_duration": 0.0,
            "performance_metrics": {},
        }
        
        # Process all log files for this run
        for log_file in run_dir.glob("llm_interactions_*.jsonl"):
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            log_entry = json.loads(line)
                            summary["total_calls"] += 1
                            
                            # Track call types
                            call_type = log_entry.get("call_type", "unknown")
                            summary["call_types"][call_type] = summary["call_types"].get(call_type, 0) + 1
                            
                            # Track models
                            model_name = log_entry.get("model", {}).get("name", "unknown")
                            summary["models_used"][model_name] = summary["models_used"].get(model_name, 0) + 1
                            
                            # Track duration
                            duration = log_entry.get("performance", {}).get("duration_seconds", 0)
                            summary["total_duration"] += duration
                            
                            # Track errors
                            if log_entry.get("error"):
                                summary["errors"] += 1
                            
                            # Track memory enhancement
                            if log_entry.get("memory_context", {}).get("enabled"):
                                summary["memory_enhanced_calls"] += 1
                                
            except Exception as e:
                self.logger.error(f"Error processing log file {log_file}: {e}")
        
        # Calculate averages
        if summary["total_calls"] > 0:
            summary["average_duration"] = summary["total_duration"] / summary["total_calls"]
        
        # Save summary
        summary_file = run_dir / "llm_interaction_summary.json"
        try:
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"Failed to save summary: {e}")
        
        return summary
    
    def get_recent_interactions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get the most recent LLM interactions across all runs.
        
        Args:
            limit: Maximum number of interactions to return
            
        Returns:
            List of recent log entries
        """
        recent_entries = []
        
        # Process all run directories
        for run_dir in self.log_dir.glob("run_*"):
            for log_file in run_dir.glob("llm_interactions_*.jsonl"):
                try:
                    with open(log_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            if line.strip():
                                log_entry = json.loads(line)
                                recent_entries.append(log_entry)
                except Exception as e:
                    self.logger.error(f"Error reading log file {log_file}: {e}")
        
        # Sort by timestamp and return most recent
        recent_entries.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        return recent_entries[:limit]


# Global logger instance
_llm_logger: Optional[LLMInteractionLogger] = None


def get_llm_logger() -> LLMInteractionLogger:
    """Get or create the global LLM logger instance."""
    global _llm_logger
    if _llm_logger is None:
        _llm_logger = LLMInteractionLogger()
    return _llm_logger


def log_llm_call(
    call_type: str,
    model_name: str,
    provider: str,
    messages: List[Dict[str, str]],
    response: Any,
    duration: float,
    **kwargs
) -> str:
    """
    Convenience function to log an LLM call.
    
    Args:
        call_type: Type of call
        model_name: Model name
        provider: Provider name
        messages: Messages sent
        response: LLM response
        duration: Call duration
        **kwargs: Additional arguments for logging
        
    Returns:
        Log entry ID
    """
    logger = get_llm_logger()
    return logger.log_llm_interaction(
        call_type=call_type,
        model_name=model_name,
        provider=provider,
        messages=messages,
        response=response,
        duration=duration,
        **kwargs
    )


if __name__ == "__main__":
    # Example usage
    logger = LLMInteractionLogger()
    
    # Test logging
    test_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"}
    ]
    
    log_id = logger.log_llm_interaction(
        call_type="test_call",
        model_name="gpt-3.5-turbo",
        provider="openai",
        messages=test_messages,
        response={"choices": [{"message": {"content": "I'm doing well, thank you!"}}]},
        duration=1.23,
        temperature=0.7
    )
    
    print(f"Test log entry created: {log_id}")
    
    # Create summary
    summary = logger.create_run_summary("test_run")
    print(f"Run summary: {json.dumps(summary, indent=2)}")