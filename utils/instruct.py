from typing import Type, Optional, List, Dict, Union
from pydantic import BaseModel
import warnings
import logging
import time

# Filter out specific Pydantic warning about config keys
warnings.filterwarnings('ignore', message='Valid config keys have changed in V2:*')

from litellm import completion, InternalServerError
import instructor

import utils.db.logging_db as logging_db

logger = logging.getLogger("llmpedia_app")

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
                logger.error(f"Error in LLM query: {str(e)}")
                raise e

    # Log usage information
    if verbose:
        logger.info(f"LLM Query ({llm_model}): Prompt tokens={usage.prompt_tokens}, Completion tokens={usage.completion_tokens}")

    # Log to API if available
    try:
        logging_db.log_qna_db(
            user_question=f"[API CALL] {process_id or 'unknown_process'}",
            response=f"Model: {llm_model}, Tokens: {usage.prompt_tokens + usage.completion_tokens}"
        )
    except Exception as log_error:
        logger.warning(f"Failed to log API usage: {str(log_error)}")

    return answer