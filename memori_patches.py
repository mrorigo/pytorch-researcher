#!/usr/bin/env python3
"""
Enhanced Memori patches for timeout and logging issues
"""

import asyncio
import json
import re
from typing import Any
from memori.database.search_service import SearchService
from memori.agents.memory_agent import MemoryAgent

# ===== FTS5 SANITIZATION PATCH (from previous implementation) =====

# Save original method
original_search_sqlite_fts = SearchService._search_sqlite_fts

def sanitize_fts5_query(query: str) -> str:
    """Sanitize a query for FTS5 to avoid syntax errors with special characters."""
    if not query or not query.strip():
        return ""
    
    # For very long queries (like system prompts), extract key meaningful terms
    if len(query.strip()) > 100:
        # Extract key terms that are likely to be searchable
        cleaned = re.sub(r'You are.*?agent\.', '', query, flags=re.IGNORECASE)
        cleaned = re.sub(r'When asked.*?nothing else\.', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'Return EXACTLY.*?follow this schema:.*?\{.*?\}', '', cleaned, flags=re.DOTALL)
        cleaned = re.sub(r'/\*.*?\*/', '', cleaned)  # Remove comment-like syntax
        
        # Extract key terms (words longer than 3 characters)
        words = re.findall(r'\b[a-zA-Z]{3,}\b', cleaned)
        
        if words:
            # Use key terms for searching, limit to avoid overly complex queries
            key_terms = words[:10]  # Limit to 10 key terms
            return " ".join(key_terms)
    
    # For shorter queries, remove problematic characters
    cleaned = query.strip()
    
    # Remove or escape FTS5 special characters that cause syntax errors
    cleaned = re.sub(r'[<>]', ' ', cleaned)
    
    # Remove other problematic FTS5 syntax characters but keep alphanumeric
    cleaned = re.sub(r'[^\w\s]', ' ', cleaned)
    
    # Collapse multiple spaces
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    return cleaned

def patched_search_sqlite_fts(
    self,
    query: str,
    namespace: str,
    category_filter: list[str] | None,
    limit: int,
    search_short_term: bool,
    search_long_term: bool,
) -> list[dict[str, Any]]:
    """Patched version that sanitizes queries to avoid FTS5 syntax errors."""
    try:
        from loguru import logger
        
        # Sanitize the query to prevent FTS5 syntax errors
        sanitized_query = sanitize_fts5_query(query)
        
        if sanitized_query != query:
            logger.debug(f"Query sanitized for FTS5: '{query[:50]}{'...' if len(query) > 50 else ''}' -> '{sanitized_query[:50]}{'...' if len(sanitized_query) > 50 else ''}'")
        
        if not sanitized_query:
            logger.debug("Query became empty after sanitization, returning empty results")
            return []
        
        # Use sanitized query in the original method
        return original_search_sqlite_fts(self, sanitized_query, namespace, category_filter, limit, search_short_term, search_long_term)

    except Exception as e:
        from loguru import logger
        logger.error(f"Error in patched SQLite FTS search: {e}")
        return []

# ===== MEMORY AGENT TIMEOUT PATCH =====

# Save original methods
original_process_conversation_async = MemoryAgent.process_conversation_async
original_process_with_fallback_parsing = MemoryAgent._process_with_fallback_parsing

async def patched_process_conversation_async(
    self,
    chat_id: str,
    user_input: str,
    ai_output: str,
    context: Any = None,
    existing_memories: list[str] | None = None,
) -> Any:
    """Patched version with increased timeout for memory processing."""
    try:
        from loguru import logger
        
        # Add timeout wrapper around the original processing
        # Use a generous timeout for local models (300 seconds = 5 minutes)
        timeout = 300.0
        
        logger.debug(f"Starting memory processing with {timeout}s timeout for chat {chat_id[:8]}...")
        
        try:
            # Run the original method with timeout
            result = await asyncio.wait_for(
                original_process_conversation_async(self, chat_id, user_input, ai_output, context, existing_memories),
                timeout=timeout
            )
            logger.debug(f"Memory processing completed successfully for {chat_id[:8]}...")
            return result
            
        except asyncio.TimeoutError:
            logger.warning(f"Memory processing timed out after {timeout}s for {chat_id[:8]}..., creating fallback memory")
            # Return a simple fallback memory instead of failing completely
            return self._create_fallback_memory(chat_id, user_input, ai_output)
            
    except Exception as e:
        from loguru import logger
        logger.error(f"Error in patched memory processing for {chat_id[:8]}...: {e}")
        return self._create_fallback_memory(chat_id, user_input, ai_output)

async def patched_process_with_fallback_parsing(
    self,
    chat_id: str,
    system_prompt: str,
    conversation_text: str,
    context_info: str,
) -> Any:
    """Patched version with timeout for fallback processing."""
    try:
        from loguru import logger
        
        # Add timeout to the fallback processing as well
        timeout = 300.0  # 5 minutes
        
        logger.debug(f"Starting fallback memory processing with {timeout}s timeout for {chat_id[:8]}...")
        
        try:
            # Enhanced system prompt for JSON output
            json_system_prompt = (
                system_prompt
                + "\n\nIMPORTANT: You MUST respond with a valid JSON object that matches this exact schema:\n"
            )
            json_system_prompt += self._get_json_schema_prompt()
            json_system_prompt += "\n\nRespond ONLY with the JSON object, no additional text or formatting."

            # Call regular chat completions with timeout
            completion = await asyncio.wait_for(
                self.async_client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": json_system_prompt},
                        {
                            "role": "user",
                            "content": f"Process this conversation for enhanced memory storage:\n\n{conversation_text}\n{context_info}",
                        },
                    ],
                    metadata=["INTERNAL_MEMORY_PROCESSING"],
                    temperature=0.1,
                    max_tokens=2000,
                    # Add timeout parameter to OpenAI call if supported
                    timeout=120,  # 2 minutes for the OpenAI call itself
                ),
                timeout=timeout
            )

            # Extract and parse JSON response
            response_text = completion.choices[0].message.content
            if not response_text:
                raise ValueError("Empty response from model")

            # Clean up response (remove markdown formatting if present)
            response_text = response_text.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            response_text = response_text.strip()

            # Parse JSON
            try:
                logger.debug(f"Attempting to parse JSON response (length {len(response_text)}) for {chat_id[:8]}...")
                logger.debug(f"Response preview: {response_text[:500]!r}...")
                logger.debug(f"Response end: ...{response_text[-200:]!r}")
                
                parsed_data = json.loads(response_text)
                logger.debug(f"Successfully parsed JSON: {type(parsed_data)}")
                
            except json.JSONDecodeError as e:
                # Enhanced error logging for JSON parsing issues
                logger.error(f"Failed to parse JSON response for {chat_id}: {e}")
                logger.error(f"Raw response length: {len(response_text)}")
                logger.error(f"Raw response start: {response_text[:1000]!r}")
                logger.error(f"Raw response end: ...{response_text[-200:]!r}")
                
                # Try to identify the issue
                if "Unterminated string" in str(e):
                    logger.error("ISSUE: Unterminated string detected in JSON response from memory processing")
                    # Try to find where the string starts and ends
                    lines = response_text.split('\n')
                    for i, line in enumerate(lines):
                        if line.strip().startswith('"') and not line.strip().endswith('"'):
                            logger.error(f"Problematic line {i+1}: {line!r}")
                
                return self._create_empty_long_term_memory(
                    chat_id, f"JSON parsing failed: {e}"
                )

            # Convert to ProcessedLongTermMemory object with validation and defaults
            processed_memory = self._create_memory_from_dict(parsed_data, chat_id)

            logger.debug(
                f"Successfully parsed memory using fallback method for {chat_id}"
            )
            return processed_memory

        except asyncio.TimeoutError:
            logger.warning(f"Fallback memory processing timed out after {timeout}s for {chat_id[:8]}...")
            return self._create_fallback_memory(chat_id, "", "")

    except Exception as e:
        from loguru import logger
        logger.error(f"Fallback memory processing failed for {chat_id}: {e}")
        return self._create_empty_long_term_memory(
            chat_id, f"Fallback processing failed: {e}"
        )

# Helper method for creating fallback memory
def create_fallback_memory(self, chat_id: str, user_input: str, ai_output: str) -> Any:
    """Create a simple fallback memory when processing fails."""
    try:
        from datetime import datetime
        from ..utils.pydantic_models import (
            MemoryClassification,
            MemoryImportanceLevel,
            ProcessedLongTermMemory,
        )
        
        content = f"User: {user_input}\nAssistant: {ai_output}" if user_input and ai_output else "Memory processing failed"
        summary = "Simple fallback memory created due to processing timeout"
        
        return ProcessedLongTermMemory(
            content=content[:1000],  # Limit content length
            summary=summary,
            classification=MemoryClassification.CONVERSATIONAL,
            importance=MemoryImportanceLevel.LOW,
            conversation_id=chat_id,
            confidence_score=0.1,
            classification_reason="Fallback memory due to processing timeout",
            promotion_eligible=False,
            extraction_timestamp=datetime.now(),
        )
    except Exception:
        # If we can't even create the fallback, return empty
        from datetime import datetime
        from ..utils.pydantic_models import (
            MemoryClassification,
            MemoryImportanceLevel,
            ProcessedLongTermMemory,
        )
        return ProcessedLongTermMemory(
            content="Processing failed",
            summary="Processing failed",
            classification=MemoryClassification.CONVERSATIONAL,
            importance=MemoryImportanceLevel.LOW,
            conversation_id=chat_id,
            confidence_score=0.0,
            classification_reason="Memory processing completely failed",
            promotion_eligible=False,
            extraction_timestamp=datetime.now(),
        )

# Add the helper method to the class
MemoryAgent._create_fallback_memory = create_fallback_memory

# ===== LLM ASSEMBLER LOGGING ENHANCEMENT =====

# This would require patching the model assembler, but we can create a more comprehensive logging patch
# For now, let's focus on the timeout issue which is more critical

# Apply all patches
SearchService._search_sqlite_fts = patched_search_sqlite_fts
MemoryAgent.process_conversation_async = patched_process_conversation_async
MemoryAgent._process_with_fallback_parsing = patched_process_with_fallback_parsing

print("✅ Applied comprehensive Memori patches:")
print("   - FTS5 query sanitization (prevents syntax errors)")
print("   - Increased timeouts for memory processing (300s)")
print("   - Fallback memory creation for timeout cases")
print("   - Enhanced error handling and logging")