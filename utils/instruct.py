from typing import Type, Optional
from pydantic import BaseModel, Field, model_validator
import instructor
from anthropic import Anthropic
from openai import OpenAI


def run_instructor_query(
    system_message: str,
    user_message: str,
    model: Optional[Type[BaseModel]] = None,
    llm_model: str = "claude-3-haiku-20240307",
    temperature: float = 0.5,
):
    """Run a query with the instructor API and get a structured response."""
    model_type = "OpenAI" if "gpt" in llm_model else "Anthropic"
    if model_type == "Anthropic":
        client = Anthropic()
        response = create_anthropic_message(
            client, system_message, user_message, model, llm_model, temperature
        )
    elif model_type == "OpenAI":
        client = OpenAI()
        response = create_openai_message(
            client, system_message, user_message, model, llm_model, temperature
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    return response


def create_anthropic_message(
    client, system_message, user_message, model, llm_model, temperature
):
    """Create a message with the Anthropic client, with an optional Pydantic model."""
    if model is None:
        response = client.messages.create(
            max_tokens=4096,
            model=llm_model,
            system=system_message,
            temperature=temperature,
            messages=[
                {"role": "user", "content": user_message},
            ],
        )
        answer = response.content[0].text
    else:
        client = instructor.from_anthropic(client)
        response = client.messages.create(
            max_tokens=4096,
            max_retries=3,
            model=llm_model,
            temperature=temperature,
            system=system_message,
            messages=[
                {"role": "user", "content": user_message},
            ],
            response_model=model,
        )
        answer = response
    return answer


def create_openai_message(
    client, system_message, user_message, model, llm_model, temperature
):
    """Create a message with the OpenAI client, with an optional Pydantic model."""
    if model is None:
        response = client.chat.completions.create(
            model=llm_model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
        )
        answer = response.choices[0].message.content
    else:
        client = instructor.from_openai(client)
        response = client.chat.completions.create(
            model=llm_model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            response_model=model,
        )
        answer = response
    return answer
