"""
LLM Interface for AAAIM

Handles LLM interactions for annotation.
"""

import os
import re
import time
from typing import Dict, List, Tuple, Any
from openai import OpenAI, RateLimitError, APIError
import logging
from utils.constants import (
    EntityType,
    DEFAULT_MAX_RETRIES,
    DEFAULT_INITIAL_DELAY,
    DEFAULT_MAX_DELAY,
    OPENROUTER_BASE_URL,
    LLAMA_BASE_URL,
    GPT_MINI_MODEL,
    SYSTEM_PROMPT_AUTO,
    SYSTEM_PROMPT_CHEMICAL,
    SYSTEM_PROMPT_GENE,
    SYSTEM_PROMPT_PROTEIN,
    SYSTEM_PROMPT_REACTION,
)

SYSTEM_PROMPT = SYSTEM_PROMPT_AUTO

logger = logging.getLogger(__name__)

def _extract_llama_completion_message_text(response: Any) -> str:
    """Extract text from the Llama API client response shape used in this project.

    Historically, the Llama API path returned a response with a ``completion_message``
    dict containing ``{\"content\": {\"text\": \"...\"}}``.
    """
    if response is None or not hasattr(response, "completion_message"):
        return ""
    try:
        cm = response.completion_message
        if isinstance(cm, dict):
            content = cm.get("content") or {}
            if isinstance(content, dict):
                text = content.get("text")
                return text if isinstance(text, str) else ""
    except Exception:
        return ""
    return ""

def _extract_chat_text(response: Any) -> str:
    """Best-effort extraction of assistant text from an OpenAI chat completion response.

    Supports:
    - OpenAI python client: response.choices[0].message.content (string or list of parts)
    - Fallbacks for dict-like responses (rare in this codebase)
    """
    if response is None:
        return ""

    # OpenAI python client objects
    if hasattr(response, "choices") and getattr(response, "choices"):
        choice0 = response.choices[0]
        msg = getattr(choice0, "message", None)
        content = getattr(msg, "content", None) if msg is not None else None
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        # Some clients return a list of content parts; join text-like fields.
        if isinstance(content, list):
            parts: list[str] = []
            for p in content:
                if isinstance(p, str):
                    parts.append(p)
                elif isinstance(p, dict):
                    # Common shape: {"type":"text","text":"..."}
                    t = p.get("text")
                    if isinstance(t, str):
                        parts.append(t)
            return "\n".join(x for x in parts if x.strip())
        return str(content)

    # Dict-like fallback
    if isinstance(response, dict):
        try:
            choices = response.get("choices") or []
            if choices:
                msg = choices[0].get("message") or {}
                content = msg.get("content")
                return content if isinstance(content, str) else str(content or "")
        except Exception:
            return ""

    return ""


def get_system_prompt(entity_type: str | EntityType = EntityType.CHEMICAL) -> str:
    """
    Get the appropriate system prompt based on entity type.
    
    Args:
        entity_type: Type of entity ("chemical", "gene", "protein", "auto")
        
    Returns:
        System prompt string
    """
    if isinstance(entity_type, str):
        try:
            entity_type = EntityType(entity_type)
        except ValueError:
            logger.warning(f"Unknown entity type {entity_type}, using chemical prompt")
            entity_type = EntityType.CHEMICAL

    if entity_type == EntityType.AUTO:
        return SYSTEM_PROMPT_AUTO
    elif entity_type == EntityType.CHEMICAL:
        return SYSTEM_PROMPT_CHEMICAL
    elif entity_type == EntityType.GENE:
        return SYSTEM_PROMPT_GENE
    elif entity_type == EntityType.PROTEIN:
        return SYSTEM_PROMPT_PROTEIN
    elif entity_type == EntityType.REACTION:
        return SYSTEM_PROMPT_REACTION
    else:
        logger.warning(f"Unknown entity type {entity_type}, using chemical prompt")
        return SYSTEM_PROMPT_CHEMICAL

def _make_api_call_with_retry(client, model: str, messages: list, 
                               max_retries: int = DEFAULT_MAX_RETRIES,
                               initial_delay: float = DEFAULT_INITIAL_DELAY,
                               max_delay: float = DEFAULT_MAX_DELAY,
                               api_name: str = "API"):
    """
    Make an API call with retry logic for rate limit errors (429).
    
    Args:
        client: OpenAI client instance
        model: Model name
        messages: List of message dicts for the API
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds before first retry
        max_delay: Maximum delay between retries
        api_name: Name of the API for logging
        
    Returns:
        API response or None on failure
    """
    delay = initial_delay
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages
            )
            return response
            
        except RateLimitError as e:
            last_exception = e
            if attempt < max_retries:
                # Extract wait time from error message if available
                wait_time = delay
                error_msg = str(e)
                
                # Try to parse retry-after from error message
                if "retry after" in error_msg.lower():
                    try:
                        # Look for patterns like "retry after X seconds" or "try again in X"
                        import re
                        match = re.search(r'(\d+)\s*(?:seconds?|s)', error_msg.lower())
                        if match:
                            suggested_wait = int(match.group(1))
                            wait_time = max(wait_time, suggested_wait + 1)  # Add 1s buffer
                    except:
                        pass
                
                # Cap at max_delay
                wait_time = min(wait_time, max_delay)
                
                print(f"Rate limit error (429) from {api_name}. Attempt {attempt + 1}/{max_retries + 1}. "
                      f"Waiting {wait_time:.1f}s before retry...")
                time.sleep(wait_time)
                
                # Exponential backoff for next attempt
                delay = min(delay * 2, max_delay)
            else:
                print(f"Rate limit error (429) from {api_name}. Max retries ({max_retries}) exceeded.")
                
        except APIError as e:
            # Handle other API errors (500, 502, 503, etc.)
            last_exception = e
            if e.status_code in [500, 502, 503, 504] and attempt < max_retries:
                wait_time = min(delay, max_delay)
                print(f"API error ({e.status_code}) from {api_name}. Attempt {attempt + 1}/{max_retries + 1}. "
                      f"Waiting {wait_time:.1f}s before retry...")
                time.sleep(wait_time)
                delay = min(delay * 2, max_delay)
            else:
                print(f"API error from {api_name}: {e}")
                break
                
        except Exception as e:
            # Non-retryable error
            print(f"Error querying {api_name}: {e}")
            return None
    
    # All retries exhausted
    if last_exception:
        print(f"All retries exhausted for {api_name}. Last error: {last_exception}")
    return None


def query_llm(prompt: str, developer_prompt: str = None, model=GPT_MINI_MODEL, entity_type: str = "chemical",
              max_retries: int = DEFAULT_MAX_RETRIES, initial_delay: float = DEFAULT_INITIAL_DELAY):
    """
    Query the OpenAI LLM with the formatted prompt.
    Includes automatic retry with exponential backoff for rate limit errors (429).
    
    Args:
        prompt: The formatted prompt to send to the LLM
        developer_prompt: The system prompt (if None, will use appropriate prompt for entity_type)
        model: The model to use for the LLM, e.g., "meta-llama/llama-3.3-70b-instruct:free"
        entity_type: Type of entity for prompt selection if developer_prompt is None
        max_retries: Maximum number of retry attempts for rate limit errors (default: 5)
        initial_delay: Initial delay in seconds before first retry (default: 10)

    Returns:
        String response from LLM or empty string on error
    """
    if developer_prompt is None:
        developer_prompt = get_system_prompt(entity_type)
    
    messages = [
        {"role": "system", "content": developer_prompt},
        {"role": "user", "content": prompt}
    ]
    
    response = None
    if model.startswith("gpt"):
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = _make_api_call_with_retry(
            client, model, messages, 
            max_retries=max_retries, 
            initial_delay=initial_delay,
            api_name="OpenAI"
        )
    elif model.startswith("meta-llama"):
        client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=os.getenv("OPENROUTER_API_KEY"))
        response = _make_api_call_with_retry(
            client, model, messages,
            max_retries=max_retries,
            initial_delay=initial_delay,
            api_name="OpenRouter"
        )
    elif model.startswith("Llama"):
        client = OpenAI(base_url=LLAMA_BASE_URL, api_key=os.getenv("LLAMA_API_KEY"))
        response = _make_api_call_with_retry(
            client, model, messages,
            max_retries=max_retries,
            initial_delay=initial_delay,
            api_name="Llama API"
        )
    else:
        raise ValueError(f"Model {model} not supported")
    
    # Keep legacy Llama response extraction, but use the standard OpenAI structure
    # for non-Llama models.
    if str(model).startswith("Llama"):
        text = _extract_llama_completion_message_text(response) or _extract_chat_text(response)
    else:
        text = _extract_chat_text(response)
    if text and str(text).strip():
        return str(text)

    # Empty content: log minimal response metadata for debugging.
    try:
        finish_reason = None
        if response is not None and hasattr(response, "choices") and response.choices:
            finish_reason = getattr(response.choices[0], "finish_reason", None)
        logger.warning(
            "No response or empty response from LLM (model=%s, finish_reason=%s).",
            model,
            finish_reason,
        )
    except Exception:
        print("No response or empty response from LLM.")
    return ""

def query_llm_with_history(messages: list, model: str = GPT_MINI_MODEL,
                           max_retries: int = DEFAULT_MAX_RETRIES,
                           initial_delay: float = DEFAULT_INITIAL_DELAY) -> str:
    """
    Query the LLM with a full conversation history (multi-turn).
    
    Used by the feedback loop to send the original prompt, the LLM's prior
    response, and user feedback as a coherent conversation so the LLM can
    revise its output.
    
    Args:
        messages: List of message dicts (role/content) representing the full
                  conversation so far, including system, user, assistant, and
                  feedback turns.
        model: LLM model identifier.
        max_retries: Retry attempts for rate-limit / transient errors.
        initial_delay: Initial backoff delay in seconds.

    Returns:
        The assistant's response text, or empty string on failure.
    """
    response = None
    if model.startswith("gpt"):
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = _make_api_call_with_retry(
            client, model, messages,
            max_retries=max_retries, initial_delay=initial_delay,
            api_name="OpenAI"
        )
    elif model.startswith("meta-llama"):
        client = OpenAI(base_url=OPENROUTER_BASE_URL,
                        api_key=os.getenv("OPENROUTER_API_KEY"))
        response = _make_api_call_with_retry(
            client, model, messages,
            max_retries=max_retries, initial_delay=initial_delay,
            api_name="OpenRouter"
        )
    elif model.startswith("Llama"):
        client = OpenAI(base_url=LLAMA_BASE_URL,
                        api_key=os.getenv("LLAMA_API_KEY"))
        response = _make_api_call_with_retry(
            client, model, messages,
            max_retries=max_retries, initial_delay=initial_delay,
            api_name="Llama API"
        )
    else:
        raise ValueError(f"Model {model} not supported")

    if str(model).startswith("Llama"):
        text = _extract_llama_completion_message_text(response) or _extract_chat_text(response)
    else:
        text = _extract_chat_text(response)
    if text and str(text).strip():
        return str(text)
    logger.warning("No response or empty response from LLM (model=%s).", model)
    return ""


def parse_llm_response(
    response,
    entity_type: str | EntityType = EntityType.AUTO,
) -> Tuple[Dict[str, List[str]], Dict[str, str], str]:
    """
    Parse the LLM response to extract species synonyms and entity types in the format:
    SpeciesA (chemical): "name1", "name2", ...
    SpeciesB (gene): "name1", name2, ...
    Reason: ...
    
    Extended to support automatic entity type detection.
    
    Args:
        response: The raw response string from the LLM
        entity_type: The entity type being used ("auto" for automatic detection, 
                     or specific type like "chemical", "gene", "protein")
        
    Returns:
        Tuple containing:
        - Dictionary mapping species IDs to lists of synonyms
        - Dictionary mapping species IDs to entity types
        - Reason string
    """
    # Remove markdown code block syntax if present
    response = re.sub(r'```.*?\n', '', response)
    response = re.sub(r'```\s*$', '', response)
    
    # Initialize the dictionaries and reason
    synonyms_dict = {}
    entity_type_dict = {}
    reason = ""
    
    # Split response into lines
    lines = response.strip().split('\n')
    reason_start = None

    # Find the line where 'Reason:' starts
    for idx, line in enumerate(lines):
        if line.strip().lower().startswith('reason:'):
            reason_start = idx
            break

    if reason_start is not None:
        # Everything after 'Reason:' is the reason, including the rest of the lines
        reason_lines = lines[reason_start:]
        if reason_lines:
            # Remove the 'Reason:' prefix from the first line
            first_line = reason_lines[0]
            reason = first_line[first_line.lower().find('reason:') + len('reason:'):].strip()
            # Add the rest of the lines (if any)
            if len(reason_lines) > 1:
                reason += '\n' + '\n'.join(l.strip() for l in reason_lines[1:])
        # Only parse synonym lines before 'Reason:'
        lines = lines[:reason_start]

    if isinstance(entity_type, EntityType):
        entity_type_str = entity_type.value
    else:
        entity_type_str = str(entity_type)

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Try to parse with entity type format: "SpeciesA (entity_type): names..."
        entity_type_pattern = r'^([A-Za-z0-9_]+)\s*\((\w+)\):\s*(.+)$'
        entity_type_match = re.match(entity_type_pattern, line)
        
        if entity_type_match:
            # Format with entity type
            species_id = entity_type_match.group(1).strip()
            detected_type = entity_type_match.group(2).strip().lower() 
            names_str = entity_type_match.group(3).strip()
            # Only use detected type if in auto mode, otherwise use specified entity_type
            if entity_type_str == EntityType.AUTO.value:
                entity_type_dict[species_id] = detected_type
            else:
                entity_type_dict[species_id] = entity_type_str
        else:
            # Standard format without entity type: "SpeciesA: names..."
            parts = line.split(':', 1)
            if len(parts) != 2:
                continue
            species_id = parts[0].strip()
            names_str = parts[1].strip()
            # Use specified entity_type, or "unknown" only if in auto mode
            if entity_type_str == EntityType.AUTO.value:
                entity_type_dict[species_id] = "unknown"
            else:
                entity_type_dict[species_id] = entity_type_str

        # Extract all synonyms, handling both quoted and unquoted names
        names = []

        # First, extract all quoted items
        quoted_items = re.findall(r'"([^"]*)"', names_str)
        names.extend(quoted_items)

        # Remove quoted parts from the string for further processing
        processed_str = names_str
        for item in quoted_items:
            processed_str = processed_str.replace(f'"{item}"', '')

        # Now extract unquoted items by splitting on commas
        unquoted_parts = [part.strip() for part in processed_str.split(',')]
        for part in unquoted_parts:
            if part and not part.isspace():
                names.append(part)

        # Remove any empty strings that might have been added
        names = [name for name in names if name and not name.isspace()]
        if names:
            synonyms_dict[species_id] = names

    # Handle case where parsing failed
    if not synonyms_dict and not reason:
        print("Failed to parse response:")
        print(response)
        # Save the response with timestamp
        timestamp = int(time.time())
        error_file = f"error_response_{timestamp}.txt"
        with open(error_file, 'w') as f:
            f.write(str(response))
        print(f"Error response saved to: {error_file}")

    return synonyms_dict, entity_type_dict, reason 