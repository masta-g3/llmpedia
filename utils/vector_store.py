import pandas as pd
import os

from langchain.text_splitter import RecursiveCharacterTextSplitter
from mlx_lm import generate
import tiktoken

import utils.db as db
import utils.prompts as ps
import utils.app_utils as au
from utils.instruct import run_instructor_query

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
    chunk_size=2000, chunk_overlap=50
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
                if verbose:
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
    summary = run_instructor_query(
        ps.SUMMARIZE_BY_PARTS_SYSTEM_PROMPT,
        ps.SUMMARIZE_BY_PARTS_USER_PROMPT.format(content=document),
        llm_model=model,
    )
    summary = summary.strip()
    if "<summary>" in summary:
        summary = summary.split("<summary>")[1].split("</summary>")[0]
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


def verify_llm_paper(paper_content: str, model="gpt-4o"):
    """Verify if a paper is about LLMs."""
    is_llm_paper = run_instructor_query(
        ps.LLM_VERIFIER_SYSTEM_PROMPT,
        ps.LLM_VERIFIER_USER_PROMPT.format(paper_content=paper_content),
        model=ps.LLMVerifier,
        llm_model=model,
    )
    is_llm_paper = is_llm_paper.dict()
    return is_llm_paper


def review_llm_paper(paper_content: str, model="gpt-4o"):
    """Review a paper."""
    review = run_instructor_query(
        ps.SUMMARIZER_SYSTEM_PROMPT,
        ps.SUMMARIZER_USER_PROMPT.format(paper_content=paper_content),
        model=ps.PaperReview,
        llm_model=model,
    )
    return review


def convert_notes_to_narrative(
    paper_title: str, notes: str, model: str = "gpt-4o"
) -> str:
    """Convert notes to narrative."""
    narrative = run_instructor_query(
        ps.NARRATIVE_SUMMARY_SYSTEM_PROMPT.format(paper_title=paper_title),
        ps.NARRATIVE_SUMMARY_USER_PROMPT.format(previous_notes=notes),
        llm_model=model,
    )
    if "<summary>" in narrative:
        narrative = narrative.split("<summary>")[1].split("</summary>")[0]
    return narrative


def convert_notes_to_bullets(
    paper_title: str, notes: str, model: str = "GPT-3.5-Turbo"
) -> str:
    """Convert notes to bullet point list."""
    bullet_list = run_instructor_query(
        ps.BULLET_LIST_SUMMARY_SYSTEM_PROMPT,
        ps.BULLET_LIST_SUMMARY_USER_PROMPT.format(
            paper_title=paper_title, previous_notes=notes
        ),
        llm_model=model,
    ).strip()
    if "<summary>" in bullet_list:
        bullet_list = bullet_list.split("<summary>")[1].split("</summary>")[0]
    return bullet_list


def copywrite_summary(paper_title, previous_notes, narrative, model="GPT-3.5-Turbo"):
    """Copywrite a summary."""
    copywritten = run_instructor_query(
        ps.COPYWRITER_SYSTEM_PROMPT,
        ps.COPYWRITER_USER_PROMPT.format(
            paper_title=paper_title,
            previous_notes=previous_notes,
            previous_summary=narrative,
        ),
        llm_model=model,
    )
    if "<improved_summary>" in copywritten:
        copywritten = copywritten.split("<improved_summary>")[1].split(
            "</improved_summary>"
        )[0]
    return copywritten


def organize_notes(paper_title, notes, model="GPT-3.5-Turbo"):
    """Add header titles and organize notes."""
    organized_sections = run_instructor_query(
        ps.FACTS_ORGANIZER_SYSTEM_PROMPT,
        ps.FACTS_ORGANIZER_USER_PROMPT.format(
            paper_title=paper_title, previous_notes=notes
        ),
        llm_model=model,
    )
    return organized_sections


def convert_notes_to_markdown(paper_title, notes, model="GPT-3.5-Turbo"):
    """Convert notes to markdown."""
    markdown = run_instructor_query(
        ps.MARKDOWN_SYSTEM_PROMPT.format(paper_title=paper_title),
        ps.MARKDOWN_USER_PROMPT.format(previous_notes=notes),
        llm_model=model,
    )
    return markdown


def rephrase_title(title, model="gpt-4o"):
    """Summarize a title as a short visual phrase."""
    phrase = run_instructor_query(
        ps.TITLE_REPHRASER_SYSTEM_PROMPT,
        ps.TITLE_REPHRASER_USER_PROMPT.format(title=title),
        llm_model=model,
        temperature=1.2,
    ).strip()
    return phrase


def generate_weekly_report(weekly_content_md: str, model="gpt-4o"):
    """Generate weekly report."""
    weekly_report = run_instructor_query(
        ps.WEEKLY_SYSTEM_PROMPT,
        ps.WEEKLY_USER_PROMPT.format(weekly_content=weekly_content_md),
        model=ps.WeeklyReview,
        llm_model=model,
        temperature=0.8,
    )
    return weekly_report


def generate_weekly_highlight(weekly_content_md: str, model="gpt-4o"):
    """Generate weekly highlight."""
    weekly_highlight = run_instructor_query(
        ps.WEEKLY_SYSTEM_PROMPT,
        ps.WEEKLY_HIGHLIGHT_USER_PROMPT.format(weekly_content=weekly_content_md),
        llm_model=model,
        temperature=0.5,
    )
    return weekly_highlight


def extract_document_repo(paper_content: str, model="gpt-4o"):
    """Extract weekly repos."""
    weekly_repos = run_instructor_query(
        ps.WEEKLY_SYSTEM_PROMPT,
        ps.WEEKLY_REPO_USER_PROMPT.format(content=paper_content),
        llm_model=model,
        model=ps.ExternalResources,
        temperature=0.0,
    )
    return weekly_repos


tweet_user_map = {
    # "review_v1": ps.TWEET_USER_PROMPT,
    "insight_v1": ps.TWEET_INSIGHT_USER_PROMPT,
    # "review_v2": ps.TWEET_REVIEW_USER_PROMPT,
}

tweet_edit_user_map = {
    # "review_v1": ps.TWEET_EDIT_USER_PROMPT,
    "insight_v1": ps.TWEET_INSIGHT_EDIT_USER_PROMPT,
}


def select_most_interesting_paper(arxiv_abstracts: str, model: str = "claude-haiku"):
    """Select the most interesting paper from a list of candidates."""
    response = run_instructor_query(
        ps.INTERESTING_SYSTEM_PROMPT,
        ps.INTERESTING_USER_PROMPT.format(abstracts=arxiv_abstracts),
        llm_model=model,
    )
    abstract_idx = int(
        response.split("<most_interesting_abstract>")[1].split(
            "</most_interesting_abstract>"
        )[0]
    )
    return abstract_idx


def write_tweet(
    previous_tweets: str,
    tweet_facts: str,
    tweet_type="new_review",
    model="gpt-4o",
    temperature: float = 0.8,
) -> str:
    """Write a tweet about an LLM paper."""
    system_prompt = ps.TWEET_SYSTEM_PROMPT
    user_prompt = tweet_user_map[tweet_type].format(
        previous_tweets=previous_tweets, tweet_facts=tweet_facts
    )
    tweet = run_instructor_query(
        system_prompt, user_prompt, llm_model=model, temperature=temperature
    )
    return tweet


def edit_tweet(
    tweet: str,
    tweet_facts: str,
    tweet_type="review",
    model="gpt-4o",
    temperature: float = 0.8,
) -> str:
    """Edit a tweet via run_instructor_query."""
    system_prompt = ps.TWEET_EDIT_SYSTEM_PROMPT
    user_prompt = tweet_edit_user_map[tweet_type].format(
        tweet=tweet, tweet_facts=tweet_facts
    )
    edited_tweet = run_instructor_query(
        system_prompt, user_prompt, llm_model=model, temperature=temperature
    )
    return edited_tweet
