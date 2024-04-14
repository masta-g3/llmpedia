from typing import Type, Optional
from pydantic import BaseModel, Field, model_validator
import instructor
from anthropic import Anthropic

base_client = Anthropic()


def run_instructor_query(
    system_message: str,
    user_message: str,
    model: Optional[Type[BaseModel]] = None,
    llm_model: str = "claude-3-haiku-20240307",
):
    """Run a query with the instructor API and get a structured response."""
    if model is None:
        client = Anthropic()
        response = client.messages.create(
            max_tokens=4096,
            model=llm_model,
            system=system_message,
            messages=[
                {"role": "user", "content": user_message},
            ],
        )
        answer = response.content[0].text
    else:
        client = instructor.from_anthropic(Anthropic())
        response = client.messages.create(
            max_tokens=4096,
            max_retries=3,
            model=llm_model,
            messages=[
                {
                    "role": "system",
                    "content": system_message,
                },
                {"role": "user", "content": user_message},
            ],
            response_model=model,
        )
        answer = response
    return answer
