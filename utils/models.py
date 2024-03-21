from langchain_openai import ChatOpenAI
from langchain_together import Together
from langchain_anthropic import ChatAnthropic
import os

together_key = os.getenv("TOGETHER_API_KEY")


def get_mlx_model(
    model_name: str = "una-cybertron-7b-v2-bf16-4bit-mlx",
    chat_template_name: [str, None] = "chatml.jinja",
):
    """Load MLX model + tokenizer and apply chat template."""
    from mlx_lm import load
    mlx_model, mlx_tokenizer = load(f"mlx-community/{model_name}")
    if chat_template_name is not None:
        chat_template = open(f"utils/{chat_template_name}").read()
        chat_template = chat_template.replace("    ", "").replace("\n", "")
        mlx_tokenizer.chat_template = chat_template
    return mlx_model, mlx_tokenizer


llm_map = {
    ## Closed AI.
    "GPT-3.5-Turbo-JSON": ChatOpenAI(
        model_name="gpt-3.5-turbo-0125", temperature=0.0
    ).bind(response_format={"type": "json_object"}),
    "GPT-3.5-Turbo": ChatOpenAI(model_name="gpt-3.5-turbo-0125", temperature=0.0),
    "GPT-3.5-Turbo-HT": ChatOpenAI(model_name="gpt-3.5-turbo-0125", temperature=0.8),
    "GPT-4": ChatOpenAI(model_name="gpt-4", temperature=0.0),
    "GPT-4-Turbo": ChatOpenAI(model_name="gpt-4-0125-preview", temperature=0.0),
    "GPT-4-Turbo-JSON": ChatOpenAI(
        model_name="gpt-4-0125-preview", temperature=0.0
    ).bind(response_format={"type": "json_object"}),
    ## Open AI.
    "openchat": Together(
        model="openchat/openchat-3.5-1210",
        temperature=0.0,
        together_api_key=together_key,
    ),
    "gemma": Together(
        model="google/gemma-7b-it",
        temperature=0.05,
        together_api_key=together_key,
    ),
    "mistral": Together(
        model="mistralai/Mistral-7B-Instruct-v0.2",
        temperature=0.0,
        together_api_key=together_key,
    ),
    "hermes": Together(
        model="NousResearch/Nous-Hermes-2-Mistral-7B-DPO",
        temperature=0.0,
        together_api_key=together_key,
    ),
    "open-orca": Together(
        model="Open-Orca/Mistral-7B-OpenOrca",
        temperature=0.0,
        together_api_key=together_key,
    ),
    "qwen-7": Together(
        model="Qwen/Qwen1.5-7B-Chat",
        temperature=0.0,
        together_api_key=together_key,
    ),
    "yi": Together(
        model="zero-one-ai/Yi-34B-Chat",
        temperature=0.0,
        together_api_key=together_key,
    ),
    "mixtral": Together(
        model="mistralai/Mixtral-8x7B-Instruct-v0.1",
        temperature=0.0,
        together_api_key=together_key,
    ),
    "mixtral-dpo": Together(
        model="NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
        temperature=0.0,
        together_api_key=together_key,
    ),
    "mixtral-sft": Together(
        model="NousResearch/Nous-Hermes-2-Mixtral-8x7B-SFT",
        temperature=0.0,
        together_api_key=together_key,
    ),
    "qwen-14": Together(
        model="Qwen/Qwen1.5-14B-Chat",
        temperature=0.0,
        together_api_key=together_key,
    ),
    ## Local model.
    "local": ChatOpenAI(model_name="local", temperature=0.0),
    ## Anthropic.
    "claude-haiku": ChatAnthropic(temperature=0, model_name="claude-3-haiku-20240307"),
    "claude-sonnet": ChatAnthropic(
        temperature=0, model_name="claude-3-sonnet-20240229"
    ),
    "claude-opus": ChatAnthropic(temperature=0, model_name="claude-3-opus-20240229"),
}
