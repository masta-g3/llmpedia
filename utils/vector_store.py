import demjson3
import pandas as pd
import json
import os

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts.chat import ChatPromptTemplate
from langchain.chains import LLMChain
from mlx_lm import generate

from langchain.chains.openai_functions import (
    create_structured_output_chain,
)
from utils.custom_langchain import NewCohereEmbeddings, NewPGVector

import tiktoken

import utils.custom_langchain as clc
import utils.db as db
import utils.prompts as ps
import utils.app_utils as au
from utils.models import llm_map

CONNECTION_STRING = (
    f"postgresql+psycopg2://{db.db_params['user']}:{db.db_params['password']}"
    f"@{db.db_params['host']}:{db.db_params['port']}/{db.db_params['dbname']}"
)

token_encoder = tiktoken.encoding_for_model("gpt-3.5-turbo")


def validate_openai_env():
    """Validate that the API base is not set to local."""
    api_base = os.environ.get("OPENAI_API_BASE", "")
    false_base = "http://localhost:1234/v1"
    assert api_base != false_base, "API base is not set to local."


###################
## SUMMARIZATION ##
###################

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=750, chunk_overlap=22
)


def recursive_summarize_by_parts(
    paper_title: str,
    document: str,
    max_tokens=400,
    model="local",
    mlx_model=None,
    mlx_tokenizer=None,
    verbose=False,
):
    """Recursively apply the summarize_by_segments function to a document."""
    ori_token_count = len(token_encoder.encode(document))
    token_count = ori_token_count + 0
    if verbose:
        print(f"Starting tokens: {ori_token_count}")
    summaries_dict = {}
    token_dict = {}
    retry_once = True
    i = 1

    while token_count > max_tokens:
        if verbose:
            print("------------------------")
            print(f"Summarization iteration {i}...")
        document = summarize_by_parts(
            paper_title, document, model, mlx_model, mlx_tokenizer, verbose
        )

        token_diff = token_count - len(token_encoder.encode(document))
        token_count = len(token_encoder.encode(document))
        frac = token_count / ori_token_count
        summaries_dict[i] = document
        token_dict[i] = token_count
        i += 1
        if verbose:
            print(f"Total tokens: {token_count}")
            print(f"Compression: {frac:.2f}")

        if token_diff < 50:
            if retry_once:
                retry_once = False
                continue
            else:
                print("Cannot compress further. Stopping.")
                break

    return summaries_dict, token_dict


def summarize_by_parts(
    paper_title: str,
    document: str,
    model="mlx",
    mlx_model=None,
    mlx_tokenizer=None,
    verbose=False,
):
    """Summarize a paper by segments."""
    doc_chunks = text_splitter.create_documents([document])
    summary_notes = ""
    st_time = pd.Timestamp.now()
    for idx, current_chunk in enumerate(doc_chunks):
        summary_notes += (
            au.numbered_to_bullet_list(
                summarize_doc_chunk_mlx(
                    paper_title, current_chunk, mlx_model, mlx_tokenizer
                )
                if model == "mlx"
                else summarize_doc_chunk(paper_title, current_chunk, model)
            )
            + "\n"
        )
        if verbose:
            time_elapsed = pd.Timestamp.now() - st_time
            print(
                f"{idx+1}/{len(doc_chunks)}: {time_elapsed.total_seconds():.2f} seconds"
            )
            st_time = pd.Timestamp.now()

    return summary_notes


def summarize_doc_chunk(paper_title: str, document: str, model="local"):
    """Summarize a paper by segments."""
    summarizer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", ps.SUMMARIZE_BY_PARTS_SYSTEM_PROMPT),
            ("human", ps.SUMMARIZE_BY_PARTS_USER_PROMPT),
        ]
    )
    chain = LLMChain(llm=llm_map[model], prompt=summarizer_prompt, verbose=False)
    summary = chain.invoke(dict(paper_title=paper_title, content=document))["text"]
    return summary


def summarize_doc_chunk_mlx(paper_title: str, document: str, mlx_model, mlx_tokenizer):
    """Summarize a paper by segments with MLX models."""
    messages = [
        ("system", ps.SUMMARIZE_BY_PARTS_SYSTEM_PROMPT.format(paper_title=paper_title)),
        ("user", ps.SUMMARIZE_BY_PARTS_USER_PROMPT.format(content=document)),
    ]
    messages = [{"role": role, "content": content} for role, content in messages]

    prompt = mlx_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    summary = generate(
        mlx_model,
        mlx_tokenizer,
        prompt=prompt,
        max_tokens=500,
        # repetition_penalty=1.05,
        verbose=False,
    )

    return summary


def verify_llm_paper(paper_content: str, model="GPT-3.5-Turbo-JSON"):
    """Verify if a paper is about LLMs via LLMChain."""
    llm_paper_check_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", ps.LLM_PAPER_CHECK_TEMPLATE),
            ("human", "{paper_content}"),
            ("human", ps.LLM_PAPER_CHECK_FMT_TEMPLATE),
        ]
    )
    llm_chain = LLMChain(
        llm=llm_map[model], prompt=llm_paper_check_prompt, verbose=False
    )
    is_llm_paper = llm_chain.invoke(dict(paper_content=paper_content))["text"]
    is_llm_paper = is_llm_paper.replace("\n", "")
    is_llm_paper = demjson3.decode(is_llm_paper)
    return is_llm_paper


def review_llm_paper(paper_content: str, model="GPT-3.5-Turbo"):
    """Review a paper via LLMChain."""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", ps.SUMMARIZER_SYSTEM_PROMPT),
            ("human", ps.SUMMARIZER_HUMAN_REMINDER),
        ]
    )
    parser = clc.CustomFixParser(pydantic_schema=ps.PaperReview)
    chain = create_structured_output_chain(
        ps.PaperReview, llm_map[model], prompt, output_parser=parser, verbose=False
    )
    review = chain.invoke(dict(paper_content=paper_content))["function"]
    return review


def convert_notes_to_narrative(paper_title, notes, model="GPT-3.5-Turbo"):
    """Convert notes to narrative via LLMChain."""
    narrative_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", ps.NARRATIVE_SUMMARY_SYSTEM_PROMPT),
            ("user", ps.NARRATIVE_SUMMARY_USER_PROMPT),
        ]
    )
    narrative_chain = LLMChain(llm=llm_map[model], prompt=narrative_prompt)
    narrative = narrative_chain.invoke(
        dict(paper_title=paper_title, previous_notes=notes)
    )["text"]
    return narrative


def copywrite_summary(paper_title, narrative, model="GPT-3.5-Turbo"):
    """Copywrite a summary via LLMChain."""
    copywriting_prompt = ChatPromptTemplate.from_messages(
        [("system", ps.COPYWRITER_SYSTEM_PROMPT), ("user", ps.COPYWRITER_USER_PROMPT)]
    )
    copywriting_chain = LLMChain(llm=llm_map[model], prompt=copywriting_prompt)
    copywritten = copywriting_chain.invoke(
        dict(paper_title=paper_title, previous_summary=narrative)
    )["text"]
    return copywritten


def organize_notes(paper_title, notes, model="GPT-3.5-Turbo"):
    """Add header titles and organize notes via LLMChain."""
    organize_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", ps.FACTS_ORGANIZER_SYSTEM_PROMPT),
            ("user", ps.FACTS_ORGANIZER_USER_PROMPT),
        ]
    )
    organize_chain = LLMChain(llm=llm_map[model], prompt=organize_prompt)
    organized_sections = organize_chain.invoke(
        dict(paper_title=paper_title, previous_notes=notes)
    )["text"]
    return organized_sections


def convert_notes_to_markdown(paper_title, notes, model="GPT-3.5-Turbo"):
    """Convert notes to markdown via LLMChain."""
    markdown_prompt = ChatPromptTemplate.from_messages(
        [("system", ps.MARKDOWN_SYSTEM_PROMPT), ("user", ps.MARKDOWN_USER_PROMPT)]
    )
    markdown_chain = LLMChain(llm=llm_map[model], prompt=markdown_prompt)
    markdown = markdown_chain.invoke(
        dict(paper_title=paper_title, previous_notes=notes)
    )["text"]
    return markdown


def summarize_title_in_word(title, model="GPT-3.5-Turbo-HT"):
    """Summarize a title in a few words via LLMChain."""
    title_summarizer_prompt = ChatPromptTemplate.from_messages(
        [("system", ps.TITLE_SUMMARIZER_PROMPT)]
    )
    title_summarizer_chain = LLMChain(
        llm=llm_map[model], prompt=title_summarizer_prompt
    )
    keyword = title_summarizer_chain.invoke(dict(title=title))["text"].strip()
    return keyword


def rephrase_title(title, model="GPT-3.5-Turbo-HT"):
    """Summarize a title in a few words via LLMChain."""
    title_rephrase_prompt = ChatPromptTemplate.from_messages(
        [("system", ps.TITLE_REPHRASER_PROMPT)]
    )
    title_rephrase_chain = LLMChain(llm=llm_map[model], prompt=title_rephrase_prompt)
    phrase = title_rephrase_chain.invoke(dict(title=title))["text"].strip()
    return phrase


def generate_weekly_report(weekly_content_md: str, model="GPT-4-Turbo"):
    """Generate weekly report via LLMChain."""
    weekly_report_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", ps.WEEKLY_SYSTEM_PROMPT),
            ("user", ps.WEEKLY_USER_PROMPT),
            (
                "user",
                "Tip: Remember to add plenty of citations! Use the format (arxiv:1234.5678)."
            ),
        ]
    )
    weekly_report_chain = LLMChain(llm=llm_map[model], prompt=weekly_report_prompt)
    weekly_report = weekly_report_chain.invoke(dict(weekly_content=weekly_content_md))[
        "text"
    ]
    return weekly_report


def write_tweet(
    previous_tweets: str, tweet_style: str, tweet_facts: str, model="GPT-4-Turbo"
):
    """Write a tweet via LLMChain."""
    tweet_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", ps.TWEET_SYSTEM_PROMPT),
            ("user", ps.TWEET_USER_PROMPT),
        ]
    )
    tweet_chain = LLMChain(llm=llm_map[model], prompt=tweet_prompt)
    tweet = tweet_chain.invoke(
        dict(
            previous_tweets=previous_tweets,
            tweet_style=tweet_style,
            tweet_facts=tweet_facts,
        )
    )["text"]
    return tweet


def edit_tweet(tweet: str, model="GPT-4-Turbo"):
    """Edit a tweet via LLMChain."""
    tweet_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", ps.TWEET_EDIT_SYSTEM_PROMPT),
            ("user", ps.TWEET_EDIT_USER_PROMPT),
        ]
    )
    tweet_chain = LLMChain(llm=llm_map[model], prompt=tweet_prompt)
    edited_tweet = tweet_chain.invoke(dict(tweet=tweet))["text"]
    return edited_tweet


def experiment(content: str, model="GPT-4-Turbo"):
    """Run an experiment via LLMChain."""
    EXPERIMENT_SYSTEM_PROMPT = f"""
CONTENT
==========
{content} 

GUIDELINES
==========
Using at most 50 words (3 sentences), write one clear, step-by-step, layman-level paragraph explaining the most important technical finding presented in the notes above and how it is implemented. Include a numerical figure to support your report and be detailed in your descriptions. Do not focus on specific models, model comparisons or benchmarks in your report. Reply only with the summary paragraph.

SUMMARY
==========
"""
    experiment_prompt = ChatPromptTemplate.from_messages(
        [("user", EXPERIMENT_SYSTEM_PROMPT)]
    )
    experiment_chain = LLMChain(llm=llm_map[model], prompt=experiment_prompt)
    if "claude" not in model:
        result = experiment_chain.invoke(
            dict(experiment_content=content, stop=["\n\n"])
        )["text"]
    else:
        result = experiment_chain.invoke(dict(experiment_content=content))["text"]
    return result
