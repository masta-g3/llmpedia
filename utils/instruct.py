from tokencost import calculate_cost_by_tokens
from typing import Type, Optional
from pydantic import BaseModel
from anthropic import Anthropic
from openai import OpenAI
from groq import Groq
import instructor
import os

import utils.db as db


def run_instructor_query(
    system_message: str,
    user_message: str,
    model: Optional[Type[BaseModel]] = None,
    llm_model: str = "gpt-4o",
    temperature: float = 0.5,
    process_id: str = None,
):
    """Run a query with the instructor API and get a structured response."""
    model_type = ("OpenAI" if ("gpt" in llm_model or "o1" in llm_model) 
                  else "Groq" if "llama" in llm_model 
                  else "Anthropic")
    
    if model_type == "Anthropic":
        client = Anthropic()
        response, usage = create_anthropic_message(
            client, system_message, user_message, model, llm_model, temperature
        )
    elif model_type == "OpenAI":
        client = OpenAI()
        if "o1" in llm_model:
            user_message = system_message + "\n\n" + user_message
            system_message = None
            model = None
            temperature = 1.0
        response, usage = create_openai_message(
            client, system_message, user_message, model, llm_model, temperature
        )
    elif model_type == "Groq":
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        model = None
        response, usage = create_groq_message(
            client, system_message, user_message, model, llm_model, temperature
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    ## Log usage.
    try:
        prompt_cost = calculate_cost_by_tokens(usage["prompt_tokens"], llm_model, "input")
        completion_cost = calculate_cost_by_tokens(usage["completion_tokens"], llm_model, "output")
    except Exception as e:
        # print(f"Error calculating cost: {e}")
        prompt_cost = None
        completion_cost = None
    db.log_instructor_query(
        model_name=llm_model,
        process_id=process_id,
        prompt_tokens=usage["prompt_tokens"],
        completion_tokens=usage["completion_tokens"],
        prompt_cost=prompt_cost,
        completion_cost=completion_cost
    )

    return response


def create_anthropic_message(
    client, system_message, user_message, model, llm_model, temperature
):
    """Create a message with the Anthropic client, with an optional Pydantic model."""
    max_tokens = 4096 if "opus" in llm_model else 8192
    if model is None:
        response = client.messages.create(
            max_tokens=max_tokens,
            model=llm_model,
            system=system_message,
            temperature=temperature,
            messages=[
                {"role": "user", "content": user_message},
            ],
        )
        answer = response.content[0].text.strip()
        usage = response.to_dict()["usage"]
    else:
        client = instructor.from_anthropic(client)
        response, completion = client.messages.create_with_completion(
            max_tokens=max_tokens,
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
        usage = completion.to_dict()["usage"]
    usage = {
        "prompt_tokens": usage.get("input_tokens", 0),
        "completion_tokens": usage.get("output_tokens", 0)
    }
    return answer, usage


def create_openai_message(
    client, system_message, user_message, model, llm_model, temperature
):
    """Create a message with the OpenAI client, with an optional Pydantic model."""
    content = [
        {"role": "user", "content": user_message},
    ]
    if system_message is not None:
        content.insert(0, {"role": "system", "content": system_message})
    if model is None:
        response = client.chat.completions.create(
            model=llm_model,
            temperature=temperature,
            messages=content
        )
        answer = response.choices[0].message.content.strip()
        usage = response.to_dict()["usage"]
    else:
        client = instructor.from_openai(client, mode=instructor.Mode.TOOLS_STRICT)
        response, completion = client.chat.completions.create_with_completion(
            model=llm_model,
            temperature=temperature,
            messages=content,
            response_model=model,
        )
        answer = response
        usage = completion.to_dict()["usage"]
    return answer, usage


def create_groq_message(
    client, system_message, user_message, model, llm_model, temperature
):
    """"""
    content = [
        {"role": "user", "content": user_message},
    ]
    if system_message is not None:
        content.insert(0, {"role": "system", "content": system_message})
    if model is None:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            temperature=temperature,
            messages=content
        )
        answer = response.choices[0].message.content.strip()
        usage = response.dict()["usage"]
    else:
        client = instructor.from_groq(client, mode=instructor.Mode.TOOLS_STRICT)
        response, completion = client.chat.completions.create_with_completion(
            model="llama-3.3-70b-versatile",
            temperature=temperature,
            messages=content,
            response_model=model,
        )
        answer = response
        usage = completion.dict()["usage"]
    return answer, usage    
