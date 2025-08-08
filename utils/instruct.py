from tokencost import calculate_cost_by_tokens
from typing import Type, Optional, List, Dict, Union
from pydantic import BaseModel
import warnings
import time
import traceback

# Filter out specific Pydantic warning about config keys
warnings.filterwarnings("ignore", message="Valid config keys have changed in V2:*")

from litellm import completion
import instructor

import utils.db.logging_db as logging_db

def format_vision_messages(
    images: List[str],  # base64 encoded images or URLs
    text: str,
    model: str,
    system_message: Optional[str] = None,
) -> List[Dict]:
    """Format vision messages based on model provider."""
    if any(x in model for x in ["claude-3"]):
        # Claude format: images first, then text
        content = [
            {
                "type": "image_url",
                "image_url": {
                    "url": (
                        f"data:image/png;base64,{img}"
                        if not img.startswith("http")
                        else img
                    )
                },
            }
            for img in images
        ] + [{"type": "text", "text": text}]
    else:
        # OpenAI format: text first, then images
        content = [{"type": "text", "text": text}] + [
            {
                "type": "image_url",
                "image_url": {
                    "url": (
                        img
                        if img.startswith("http")
                        else f"data:image/png;base64,{img}"
                    )
                },
            }
            for img in images
        ]

    messages = [{"role": "user", "content": content}]
    if system_message:
        messages.insert(0, {"role": "system", "content": system_message})

    return messages


def log_llm_usage(
    usage, llm_model: str, process_id: str = None, verbose: bool = False
) -> None:
    """Log LLM usage statistics and costs."""
    prompt_cost = calculate_cost_by_tokens(usage.prompt_tokens, llm_model, "input")
    completion_cost = calculate_cost_by_tokens(
        usage.completion_tokens, llm_model, "output"
    )

    # Calculate cache-related costs
    cache_creation_cost = None
    cache_read_cost = None
    if (
        hasattr(usage, "cache_creation_input_tokens")
        and usage.cache_creation_input_tokens
    ):
        cache_creation_cost = (
            calculate_cost_by_tokens(
                usage.cache_creation_input_tokens, llm_model, "input"
            )
            * 2
        )
    if hasattr(usage, "cache_read_input_tokens") and usage.cache_read_input_tokens:
        cache_read_cost = calculate_cost_by_tokens(
            usage.cache_read_input_tokens, llm_model, "cached"
        )

    if verbose:
        total_tokens = usage.prompt_tokens + usage.completion_tokens
        total_cost = (
            (prompt_cost or 0)
            + (completion_cost or 0)
            + (cache_creation_cost or 0)
            + (cache_read_cost or 0)
        )

        print("\n=== LLM Query Statistics ===")
        print(f"Model: {llm_model}")
        print("\nToken Usage:")
        print(f"  • Prompt tokens:      {usage.prompt_tokens:,}")
        print(f"  • Completion tokens:  {usage.completion_tokens:,}")
        print(f"  • Total tokens:       {total_tokens:,}")

        # Display cache tokens if present.
        if (
            hasattr(usage, "cache_creation_input_tokens")
            and usage.cache_creation_input_tokens
        ):
            print(f"  • Cache creation:     {usage.cache_creation_input_tokens:,}")
        if hasattr(usage, "cache_read_input_tokens") and usage.cache_read_input_tokens:
            print(f"  • Cache read:         {usage.cache_read_input_tokens:,}")

        print("\nCost Breakdown:")
        print(
            f"  • Prompt cost:        ${prompt_cost:.4f}"
            if prompt_cost
            else "  • Prompt cost:        N/A"
        )
        print(
            f"  • Completion cost:    ${completion_cost:.4f}"
            if completion_cost
            else "  • Completion cost:    N/A"
        )
        if cache_creation_cost is not None:
            print(f"  • Cache creation:     ${cache_creation_cost:.4f}")
        if cache_read_cost is not None:
            print(f"  • Cache read:         ${cache_read_cost:.4f}")

        print(
            f"  • Total cost:         ${total_cost:.4f}"
            if any([prompt_cost, completion_cost, cache_creation_cost, cache_read_cost])
            else "  • Total cost:         N/A"
        )
        print("========================\n")

    # Get cache token values, defaulting to 0 if not present
    cache_creation_tokens = getattr(usage, "cache_creation_input_tokens", 0) or 0
    cache_read_tokens = getattr(usage, "cache_read_input_tokens", 0) or 0

    logging_db.log_instructor_query(
        model_name=llm_model,
        process_id=process_id,
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        prompt_cost=prompt_cost,
        completion_cost=completion_cost,
        cache_creation_input_tokens=cache_creation_tokens,
        cache_read_input_tokens=cache_read_tokens,
        cache_creation_cost=cache_creation_cost,
        cache_read_cost=cache_read_cost,
    )


def run_instructor_query(
    system_message: Optional[str] = None,
    user_message: Optional[str] = None,
    model: Optional[Type[BaseModel]] = None,
    llm_model: str = "gpt-5",
    temperature: float = 1,
    process_id: str = None,
    messages: Optional[List[Dict]] = None,
    verbose: bool = False,
    workflow_id: Optional[str] = None,
    step_type: Optional[str] = None,
    step_metadata: Optional[Dict] = None,
    **kwargs,
) -> Union[BaseModel, str]:
    """Run a query with the instructor API and get a structured response using LiteLLM as unified interface."""
    # Validate that we have either messages or at least a user_message
    if messages is None and user_message is None:
        raise ValueError(
            "Either 'messages' parameter or 'user_message' parameter must be provided"
        )

    # Use provided messages if available (for image content).
    temperature = (
        None if any(x in llm_model for x in ["o1", "r1", "o3"]) else temperature
    )

    if messages is None:
        messages = [{"role": "user", "content": user_message}]
        if system_message is not None:
            messages.insert(0, {"role": "system", "content": system_message})

    max_retries = 4
    
    # Define retry delays: 0s, 30s, 60s, 300s (5 minutes)
    retry_delays = [0, 30, 60, 300]

    for attempt in range(max_retries):
        try:
            if model is None:
                response = completion(
                    model=llm_model,
                    temperature=temperature,
                    messages=messages,
                    **kwargs,
                )
                answer = response.choices[0].message.content.strip()
                usage = response.usage
            else:
                client = instructor.from_litellm(
                    completion, mode=instructor.Mode.TOOLS_STRICT
                )
                response, completion_obj = (
                    client.chat.completions.create_with_completion(
                        model=llm_model,
                        temperature=temperature,
                        messages=messages,
                        response_model=model,
                        **kwargs,
                    )
                )
                answer = response
                usage = completion_obj.usage
            break
        except Exception as e:
            import traceback
            print(f"\nError on attempt {attempt + 1}/{max_retries}:")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            print(traceback.format_exc())

            if attempt < max_retries - 1:
                delay = retry_delays[attempt + 1]
                if delay > 0:
                    print(f"\nWaiting {delay} seconds before retry...")
                    time.sleep(delay)
            else:
                print("\nAll retry attempts failed. Full traceback:")
                print(traceback.format_exc())
                raise e

    # Log usage statistics and costs
    log_llm_usage(usage, llm_model, process_id, verbose)

    # Log workflow step if workflow context provided
    if workflow_id and step_type:
        try:
            logging_db.log_workflow_step(
                workflow_id=workflow_id,
                step_type=step_type,
                system_prompt=system_message or "",
                user_prompt=user_message or "",
                structured_response=answer if model else None,
                step_metadata=step_metadata,
            )
        except Exception as log_error:
            # Don't fail the main operation if logging fails
            print(f"Warning: Failed to log workflow step: {str(log_error)}")

    return answer


def add_cache_control(
    messages: List[Dict], cache_message_index: int = 0, llm_model: str = "gpt-5"
) -> List[Dict]:
    """Add cache control to messages for prompt caching support."""
    # Check if model supports caching (Claude models only)
    if not any(provider in llm_model.lower() for provider in ["claude", "anthropic"]):
        return messages

    # Validate cache_message_index
    if cache_message_index >= len(messages) or cache_message_index < 0:
        return messages

    # Create a copy to avoid mutating the original
    cached_messages = [msg.copy() for msg in messages]

    # Add cache control to the specified message
    cached_messages[cache_message_index]["cache_control"] = {"type": "ephemeral"}

    return cached_messages