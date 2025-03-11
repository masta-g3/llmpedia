from tokencost import calculate_cost_by_tokens
from typing import Type, Optional, List, Dict, Union
from pydantic import BaseModel
import warnings

# Filter out specific Pydantic warning about config keys
warnings.filterwarnings('ignore', message='Valid config keys have changed in V2:*')

from litellm import completion, InternalServerError
import instructor
import time

import utils.db.logging_db as logging_db


def format_vision_messages(
    images: List[str],  # base64 encoded images or URLs
    text: str,
    model: str,
    system_message: Optional[str] = None
) -> List[Dict]:
    """Format vision messages based on model provider."""
    if any(x in model for x in ['claude-3']):
        # Claude format: images first, then text
        content = [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img}" if not img.startswith('http') else img}
            } for img in images
        ] + [{"type": "text", "text": text}]
    else:
        # OpenAI format: text first, then images
        content = [
            {"type": "text", "text": text}
        ] + [
            {
                "type": "image_url",
                "image_url": {"url": img if img.startswith('http') else f"data:image/png;base64,{img}"}
            } for img in images
        ]
    
    messages = [{"role": "user", "content": content}]
    if system_message:
        messages.insert(0, {"role": "system", "content": system_message})
    
    return messages


def run_instructor_query(
    system_message: str,
    user_message: str,
    model: Optional[Type[BaseModel]] = None,
    llm_model: str = "gpt-4",
    temperature: float = 0.5,
    process_id: str = None,
    messages: Optional[List[Dict]] = None,
    verbose: bool = False,
    **kwargs
) -> Union[BaseModel, str]:
    """Run a query with the instructor API and get a structured response using LiteLLM as unified interface."""
    # Use provided messages if available (for image content), otherwise construct from user_message
    if messages is None:
        # Special handling for o1/r1 models - prepend system message to user message
        if any(x in llm_model for x in ['o1', 'r1', 'o3']):
            combined_message = f"{system_message}\n\n{user_message}" if system_message else user_message
            messages = [{"role": "user", "content": combined_message}]
            temperature = None
        else:
            messages = [{"role": "user", "content": user_message}]
            if system_message is not None:
                messages.insert(0, {"role": "system", "content": system_message})

    max_retries = 3
    retry_delay = 5  # seconds
    
    for attempt in range(max_retries):
        try:
            if model is None:
                response = completion(
                    model=llm_model,
                    temperature=temperature,
                    messages=messages,
                    **kwargs
                )
                answer = response.choices[0].message.content.strip()
                usage = response.usage
            else:
                client = instructor.from_litellm(completion, mode=instructor.Mode.TOOLS_STRICT)
                response, completion_obj = client.chat.completions.create_with_completion(
                    model=llm_model,
                    temperature=temperature,
                    messages=messages,
                    response_model=model,
                    **kwargs
                )
                answer = response
                usage = completion_obj.usage
            break
        except Exception as e:
            if (attempt < max_retries - 1 
                and ("overloaded_error" in str(e) or isinstance(e, InternalServerError))):
                time.sleep(retry_delay * (attempt + 1))
                continue
            else:
                raise e

    ## Log usage.
    try:
        prompt_cost = calculate_cost_by_tokens(usage.prompt_tokens, llm_model, "input")
        completion_cost = calculate_cost_by_tokens(usage.completion_tokens, llm_model, "output")
    except Exception as e:
        prompt_cost = None
        completion_cost = None

    if verbose:
        total_tokens = usage.prompt_tokens + usage.completion_tokens
        total_cost = (prompt_cost or 0) + (completion_cost or 0)
        
        print("\n=== LLM Query Statistics ===")
        print(f"Model: {llm_model}")
        print("\nToken Usage:")
        print(f"  • Prompt tokens:      {usage.prompt_tokens:,}")
        print(f"  • Completion tokens:  {usage.completion_tokens:,}")
        print(f"  • Total tokens:       {total_tokens:,}")
        print("\nCost Breakdown:")
        print(f"  • Prompt cost:        ${prompt_cost:.4f}" if prompt_cost else "  • Prompt cost:        N/A")
        print(f"  • Completion cost:    ${completion_cost:.4f}" if completion_cost else "  • Completion cost:    N/A")
        print(f"  • Total cost:         ${total_cost:.4f}" if prompt_cost and completion_cost else "  • Total cost:         N/A")
        print("========================\n")

    logging_db.log_instructor_query(
        model_name=llm_model,
        process_id=process_id,
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        prompt_cost=prompt_cost,
        completion_cost=completion_cost
    )

    return answer
