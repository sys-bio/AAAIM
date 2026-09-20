"""
LLM Interface for AAAIM

Handles LLM interactions for annotation.
"""

import os
import re
import time
import requests
from typing import Callable, Dict, List, Tuple, Any, Optional
from openai import OpenAI, RateLimitError, APIError, APIConnectionError
import logging
from utils.constants import (
    EntityType,
    DEFAULT_MAX_RETRIES,
    DEFAULT_INITIAL_DELAY,
    DEFAULT_MAX_DELAY,
    OPENROUTER_BASE_URL,
    GPT_MINI_MODEL,
    SYSTEM_PROMPT_AUTO,
    SYSTEM_PROMPT_CHEMICAL,
    SYSTEM_PROMPT_GENE,
    SYSTEM_PROMPT_PROTEIN,
    SYSTEM_PROMPT_REACTION,
)

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

SYSTEM_PROMPT = SYSTEM_PROMPT_AUTO

logger = logging.getLogger(__name__)


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
                               api_name: str = "API",
                               max_completion_tokens: Optional[int] = None,
                               response_format: Optional[Dict[str, Any]] = None,
                               usage_callback: Optional[Callable[[Dict[str, Any]], None]] = None):
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
            request_kwargs = {"model": model, "messages": messages}
            if max_completion_tokens is not None:
                request_kwargs["max_completion_tokens"] = max_completion_tokens
            if response_format is not None:
                request_kwargs["response_format"] = response_format
            response = client.chat.completions.create(**request_kwargs)
            if usage_callback is not None:
                usage_callback(_extract_usage(response))
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
            status_code = getattr(e, "status_code", None)
            retryable = status_code in [500, 502, 503, 504] or isinstance(e, APIConnectionError)
            if retryable and attempt < max_retries:
                wait_time = min(delay, max_delay)
                print(f"API error ({status_code}) from {api_name}. Attempt {attempt + 1}/{max_retries + 1}. "
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


def _make_openrouter_api_call_with_retry(
    model: str,
    messages: list,
    max_retries: int = DEFAULT_MAX_RETRIES,
    initial_delay: float = DEFAULT_INITIAL_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    max_completion_tokens: Optional[int] = None,
    response_format: Optional[Dict[str, Any]] = None,
    usage_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
):
    """
    Make an OpenRouter chat completion request.

    Uses raw HTTP so provider-specific fields like ``reasoning_details`` are
    preserved and can be passed back unchanged in follow-up requests.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable is required for OpenRouter models")

    payload = {
        "model": model,
        "messages": messages,
        "reasoning": {"enabled": True},
    }
    if max_completion_tokens is not None:
        payload["max_tokens"] = max_completion_tokens
    if response_format is not None:
        payload["response_format"] = response_format

    delay = initial_delay
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            response = requests.post(
                f"{OPENROUTER_BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=120,
            )

            if response.status_code == 429 and attempt < max_retries:
                wait_time = min(delay, max_delay)
                print(
                    f"Rate limit error (429) from OpenRouter. Attempt {attempt + 1}/{max_retries + 1}. "
                    f"Waiting {wait_time:.1f}s before retry..."
                )
                time.sleep(wait_time)
                delay = min(delay * 2, max_delay)
                continue

            if response.status_code in [500, 502, 503, 504] and attempt < max_retries:
                wait_time = min(delay, max_delay)
                print(
                    f"API error ({response.status_code}) from OpenRouter. Attempt {attempt + 1}/{max_retries + 1}. "
                    f"Waiting {wait_time:.1f}s before retry..."
                )
                time.sleep(wait_time)
                delay = min(delay * 2, max_delay)
                continue

            response.raise_for_status()
            response_json = response.json()
            if usage_callback is not None:
                usage_callback(_extract_usage(response_json))
            return response_json

        except requests.RequestException as e:
            last_exception = e
            print(f"API error from OpenRouter: {e}")
            break

    if last_exception:
        print(f"All retries exhausted for OpenRouter. Last error: {last_exception}")
    return None


def _extract_response_text(response: Any) -> Optional[str]:
    """
    Extract assistant text from the response shapes used by supported chat APIs.

    OpenAI-compatible chat completions return ``choices[0].message.content``.
    """
    if response is None:
        return None

    if hasattr(response, "choices") and response.choices:
        message = response.choices[0].message
        content = getattr(message, "content", None)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                else:
                    text = getattr(item, "text", None)
                if text:
                    text_parts.append(text)
            if text_parts:
                return "".join(text_parts)

    return None


def _value_from_object_or_dict(value: Any, key: str, default: Any = None) -> Any:
    """Read one field from either an SDK object or a JSON dictionary."""
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _extract_usage(response: Any) -> Dict[str, Any]:
    """Normalize token usage returned by OpenAI-compatible chat APIs."""
    usage = _value_from_object_or_dict(response, "usage")
    if usage is None:
        return {}

    details = _value_from_object_or_dict(usage, "completion_tokens_details")
    prompt_details = _value_from_object_or_dict(usage, "prompt_tokens_details")
    result = {
        "input_tokens": int(_value_from_object_or_dict(usage, "prompt_tokens", 0) or 0),
        "output_tokens": int(_value_from_object_or_dict(usage, "completion_tokens", 0) or 0),
        "total_tokens": int(_value_from_object_or_dict(usage, "total_tokens", 0) or 0),
        "reasoning_tokens": int(
            _value_from_object_or_dict(details, "reasoning_tokens", 0) or 0
        ),
        "cached_input_tokens": int(
            _value_from_object_or_dict(prompt_details, "cached_tokens", 0) or 0
        ),
    }
    response_model = _value_from_object_or_dict(response, "model")
    if response_model:
        result["model"] = str(response_model)
    return result


def summarize_usage_records(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Sum normalized usage records from every LLM request in one workflow."""
    token_fields = (
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "reasoning_tokens",
        "cached_input_tokens",
    )
    summary = {field: sum(int(record.get(field, 0) or 0) for record in records) for field in token_fields}
    summary["requests"] = len(records)
    models = list(dict.fromkeys(record.get("model") for record in records if record.get("model")))
    if models:
        summary["models"] = models
    return summary


def _extract_assistant_message(response: Any) -> Optional[Dict[str, Any]]:
    """Extract an assistant message dict, preserving OpenRouter reasoning details."""
    if response is None:
        return None

    if isinstance(response, dict):
        choices = response.get("choices") or []
        if not choices:
            return None
        message = choices[0].get("message") or {}
        assistant_message = {
            "role": "assistant",
            "content": message.get("content") or "",
        }
        if message.get("reasoning_details") is not None:
            assistant_message["reasoning_details"] = message.get("reasoning_details")
        return assistant_message

    text = _extract_response_text(response)
    if text is None:
        return None
    return {"role": "assistant", "content": text}


def _is_openrouter_model(model: str) -> bool:
    return model.startswith("meta-llama") or model.startswith("openrouter/")


def query_llm_message(
    prompt: str,
    developer_prompt: str = None,
    model=GPT_MINI_MODEL,
    entity_type: str = "chemical",
    max_retries: int = DEFAULT_MAX_RETRIES,
    initial_delay: float = DEFAULT_INITIAL_DELAY,
    max_completion_tokens: Optional[int] = None,
    response_format: Optional[Dict[str, Any]] = None,
    usage_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """
    Query the configured LLM and return the assistant message dict.

    For OpenRouter reasoning requests, the returned dict includes
    ``reasoning_details`` when the provider returns it; pass that dict back
    unchanged in later message history.
    """
    if developer_prompt is None:
        developer_prompt = get_system_prompt(entity_type)

    messages = [
        {"role": "system", "content": developer_prompt},
        {"role": "user", "content": prompt}
    ]

    return query_llm_message_with_history(
        messages,
        model=model,
        max_retries=max_retries,
        initial_delay=initial_delay,
        max_completion_tokens=max_completion_tokens,
        response_format=response_format,
        usage_callback=usage_callback,
    )


def query_llm(prompt: str, developer_prompt: str = None, model=GPT_MINI_MODEL, entity_type: str = "chemical",
              max_retries: int = DEFAULT_MAX_RETRIES, initial_delay: float = DEFAULT_INITIAL_DELAY,
              max_completion_tokens: Optional[int] = None,
              response_format: Optional[Dict[str, Any]] = None,
              usage_callback: Optional[Callable[[Dict[str, Any]], None]] = None):
    """
    Query the configured LLM with the formatted prompt.
    Includes automatic retry with exponential backoff for rate limit errors (429).
    
    Args:
        prompt: The formatted prompt to send to the LLM
        developer_prompt: The system prompt (if None, will use appropriate prompt for entity_type)
        model: The model to use, e.g. "gpt-4o-mini" or
            "meta-llama/llama-3.3-70b-instruct:free"
        entity_type: Type of entity for prompt selection if developer_prompt is None
        max_retries: Maximum number of retry attempts for rate limit errors (default: 5)
        initial_delay: Initial delay in seconds before first retry (default: 10)

    Returns:
        String response from LLM or empty string on error
    """
    assistant_message = query_llm_message(
        prompt,
        developer_prompt,
        model=model,
        entity_type=entity_type,
        max_retries=max_retries,
        initial_delay=initial_delay,
        max_completion_tokens=max_completion_tokens,
        response_format=response_format,
        usage_callback=usage_callback,
    )
    text = assistant_message.get("content") if assistant_message else None
    if text:
        return text
    else:
        print("No response or empty response from LLM.")
        return ""

def query_llm_message_with_history(
    messages: list,
    model: str = GPT_MINI_MODEL,
    max_retries: int = DEFAULT_MAX_RETRIES,
    initial_delay: float = DEFAULT_INITIAL_DELAY,
    max_completion_tokens: Optional[int] = None,
    response_format: Optional[Dict[str, Any]] = None,
    usage_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """Query the LLM with full history and return the assistant message dict."""
    response = None
    if model.startswith("gpt"):
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = _make_api_call_with_retry(
            client, model, messages,
            max_retries=max_retries, initial_delay=initial_delay,
            api_name="OpenAI",
            max_completion_tokens=max_completion_tokens,
            response_format=response_format,
            usage_callback=usage_callback,
        )
    elif _is_openrouter_model(model):
        response = _make_openrouter_api_call_with_retry(
            model,
            messages,
            max_retries=max_retries,
            initial_delay=initial_delay,
            max_completion_tokens=max_completion_tokens,
            response_format=response_format,
            usage_callback=usage_callback,
        )
    else:
        raise ValueError(
            f"Model {model} not supported. Use an OpenAI model starting with "
            "'gpt' or an OpenRouter model starting with 'meta-llama' or 'openrouter/'."
        )

    return _extract_assistant_message(response) or {}


def query_llm_with_history(messages: list, model: str = GPT_MINI_MODEL,
                           max_retries: int = DEFAULT_MAX_RETRIES,
                           initial_delay: float = DEFAULT_INITIAL_DELAY,
                           max_completion_tokens: Optional[int] = None,
                           response_format: Optional[Dict[str, Any]] = None,
                           usage_callback: Optional[Callable[[Dict[str, Any]], None]] = None) -> str:
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
    assistant_message = query_llm_message_with_history(
        messages,
        model=model,
        max_retries=max_retries,
        initial_delay=initial_delay,
        max_completion_tokens=max_completion_tokens,
        response_format=response_format,
        usage_callback=usage_callback,
    )
    text = assistant_message.get("content") if assistant_message else None
    if text:
        return text
    else:
        print("No response or empty response from LLM.")
        return ""


_COMPONENT_TYPE_TAG = re.compile(r'\((chemical|protein|gene)\)', re.IGNORECASE)


def _extract_synonym_names(names_str: str) -> List[str]:
    """Pull quoted and unquoted comma-separated names from a synonym fragment."""
    names = []
    quoted_items = re.findall(r'"([^"]*)"', names_str)
    names.extend(quoted_items)
    processed_str = names_str
    for item in quoted_items:
        processed_str = processed_str.replace(f'"{item}"', '')
    for part in processed_str.split(','):
        part = part.strip().strip(';')
        if part and not part.isspace():
            names.append(part)
    return [name for name in names if name and not name.isspace()]


def parse_typed_components(names_str: str) -> List[Tuple[str, List[str]]]:
    """Parse 'names (type); names (type)' groups. Empty if no component type tags."""
    matches = list(_COMPONENT_TYPE_TAG.finditer(names_str))
    if not matches:
        return []
    components: List[Tuple[str, List[str]]] = []
    last = 0
    for match in matches:
        names_part = names_str[last:match.start()].strip().strip(';,').strip()
        names = _extract_synonym_names(names_part)
        if names:
            components.append((match.group(1).lower(), names))
        last = match.end()
    return components


def format_component_curated_name(components: List[Tuple[str, List[str]]]) -> str:
    """Display typed component groups, e.g. 'HRAS, KRAS (protein); GTP (chemical)'."""
    return "; ".join(f"{', '.join(names)} ({typ})" for typ, names in components)


def parse_llm_response(
    response,
    entity_type: str | EntityType = EntityType.AUTO,
) -> Tuple[Dict[str, List[str]], Dict[str, str], str, Dict[str, List[Tuple[str, List[str]]]]]:
    """
    Parse the LLM response to extract species synonyms and entity types in the format:
    SpeciesA (chemical): "name1", "name2", ...
    SpeciesB (complex): "name1", "name2" (protein); "name3" (chemical)
    Reason: ...
    
    Extended to support automatic entity type detection and per-component types.
    
    Args:
        response: The raw response string from the LLM
        entity_type: The entity type being used ("auto" for automatic detection, 
                     or specific type like "chemical", "gene", "protein")
        
    Returns:
        Tuple containing:
        - Dictionary mapping species IDs to lists of synonyms
        - Dictionary mapping species IDs to entity types
        - Reason string
        - Dictionary mapping species IDs to typed component groups
          (empty for species without per-component type tags)
    """
    # Remove markdown code block syntax if present
    response = re.sub(r'```.*?\n', '', response)
    response = re.sub(r'```\s*$', '', response)
    
    # Initialize the dictionaries and reason
    synonyms_dict = {}
    entity_type_dict = {}
    component_dict: Dict[str, List[Tuple[str, List[str]]]] = {}
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

        components = parse_typed_components(names_str)
        if components:
            component_dict[species_id] = components
            # Typed groups are reserved for physical complexes, including in a
            # forced gene/protein run.  Preserve that structure downstream
            # while the component tags still enforce the requested ontology.
            entity_type_dict[species_id] = EntityType.COMPLEX.value
            names = [name for _typ, group in components for name in group]
        else:
            names = _extract_synonym_names(names_str)
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

    return synonyms_dict, entity_type_dict, reason, component_dict 
