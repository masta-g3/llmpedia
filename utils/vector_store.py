import pandas as pd
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken

import utils.db as db
import utils.prompts as ps
import utils.pydantic_objects as po
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
        process_id="summarize_doc_chunk",
    )
    summary = summary.strip()
    if "<summary>" in summary:
        summary = summary.split("<summary>")[1].split("</summary>")[0]
    return summary


def summarize_doc_chunk_mlx(paper_title: str, document: str, mlx_model, mlx_tokenizer):
    """Summarize a paper by segments with MLX models."""
    from mlx_lm import generate

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
        model=po.LLMVerifier,
        llm_model=model,
        process_id="verify_llm_paper",
    )
    is_llm_paper = is_llm_paper.dict()
    return is_llm_paper


def review_llm_paper(paper_content: str, model="gpt-4o"):
    """Review a paper."""
    review = run_instructor_query(
        ps.SUMMARIZER_SYSTEM_PROMPT,
        ps.SUMMARIZER_USER_PROMPT.format(paper_content=paper_content),
        model=po.PaperReview,
        llm_model=model,
        process_id="review_llm_paper",
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
        process_id="convert_notes_to_narrative",
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
        process_id="convert_notes_to_bullets",
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
        process_id="copywrite_summary",
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
        process_id="organize_notes",
    )
    return organized_sections


def convert_notes_to_markdown(paper_title, notes, model="GPT-3.5-Turbo"):
    """Convert notes to markdown."""
    markdown = run_instructor_query(
        ps.MARKDOWN_SYSTEM_PROMPT.format(paper_title=paper_title),
        ps.MARKDOWN_USER_PROMPT.format(previous_notes=notes),
        llm_model=model,
        process_id="convert_notes_to_markdown",
    )
    return markdown


def rephrase_title(title, model="gpt-4o"):
    """Summarize a title as a short visual phrase."""
    phrase = run_instructor_query(
        ps.TITLE_REPHRASER_SYSTEM_PROMPT,
        ps.TITLE_REPHRASER_USER_PROMPT.format(title=title),
        llm_model=model,
        temperature=1.0,
        process_id="rephrase_title",
    ).strip()
    return phrase


def generate_weekly_report(weekly_content_md: str, model="gpt-4o"):
    """Generate weekly report."""
    weekly_report = run_instructor_query(
        ps.WEEKLY_SYSTEM_PROMPT,
        ps.WEEKLY_USER_PROMPT.format(weekly_content=weekly_content_md),
        # model=po.WeeklyReview,
        llm_model=model,
        temperature=0.8,
        process_id="generate_weekly_report",
    )
    return weekly_report


def generate_weekly_highlight(weekly_content_md: str, model="gpt-4o"):
    """Generate weekly highlight."""
    weekly_highlight = run_instructor_query(
        ps.WEEKLY_SYSTEM_PROMPT,
        ps.WEEKLY_HIGHLIGHT_USER_PROMPT.format(weekly_content=weekly_content_md),
        llm_model=model,
        temperature=0.5,
        process_id="generate_weekly_highlight",
    )
    return weekly_highlight


def extract_document_repo(paper_content: str, model="gpt-4o"):
    """Extract weekly repos."""
    weekly_repos = run_instructor_query(
        ps.WEEKLY_SYSTEM_PROMPT,
        ps.WEEKLY_REPO_USER_PROMPT.format(content=paper_content),
        llm_model=model,
        model=po.ExternalResources,
        temperature=0.0,
        process_id="extract_document_repo",
    )
    return weekly_repos


tweet_user_map = {
    # "review_v1": ps.TWEET_USER_PROMPT,
    "insight_v1": ps.TWEET_INSIGHT_USER_PROMPT_V1,
    "insight_v2": ps.TWEET_INSIGHT_USER_PROMPT_V2,
    "insight_v3": ps.TWEET_INSIGHT_USER_PROMPT_V3,
    "insight_v4": ps.TWEET_INSIGHT_USER_PROMPT_V4,
    "insight_v5": ps.TWEET_INSIGHT_USER_PROMPT_V5,
    "insight_v6": ps.TWEET_INSIGHT_USER_PROMPT_V6,
    # "review_v2": ps.TWEET_REVIEW_USER_PROMPT,
}

tweet_edit_user_map = {
    "review": ps.TWEET_INSIGHT_EDIT_USER_PROMPT,
}


def select_most_interesting_paper(arxiv_abstracts: str, recent_llm_tweets_str: str, model: str = "gpt-4o-mini"):
    """Select the most interesting paper from a list of candidates."""
    response = run_instructor_query(
        ps.INTERESTING_SYSTEM_PROMPT,
        ps.INTERESTING_USER_PROMPT.format(abstracts=arxiv_abstracts, recent_llm_tweets=recent_llm_tweets_str),
        model=po.InterestingPaperSelection,
        llm_model=model,
        process_id="select_most_interesting_paper",
    )
    print(response)
    return response.selected_arxiv_code


def write_tweet(
    tweet_facts: str,
    tweet_type="new_review",
    model="gpt-4o",
    most_recent_tweets: str = None,
    recent_llm_tweets: str = None,
    temperature: float = 0.8,
) -> po.Tweet:
    """Write a tweet about an LLM paper."""
    system_prompt = ps.TWEET_SYSTEM_PROMPT
    user_prompt = tweet_user_map[tweet_type].format(
        tweet_facts=tweet_facts,
        most_recent_tweets=most_recent_tweets,
        recent_llm_tweets=recent_llm_tweets,

    )
    tweet = run_instructor_query(
        system_prompt,
        user_prompt,
        model=po.Tweet,
        llm_model=model,
        temperature=temperature,
        process_id="write_tweet",
    )
    return tweet


def edit_tweet(
    tweet: str,
    most_recent_tweets: str,
    tweet_type="review",
    model="gpt-4o",
    temperature: float = 0.8,
) -> str:
    """Edit a tweet via run_instructor_query."""
    system_prompt = ps.TWEET_SYSTEM_PROMPT
    user_prompt = tweet_edit_user_map[tweet_type].format(
        proposed_tweet=tweet, most_recent_tweets=most_recent_tweets
    )
    edited_tweet = run_instructor_query(
        system_prompt,
        user_prompt,
        llm_model=model,
        temperature=temperature,
        model=po.TweetEdit,
        process_id="edit_tweet",
    )
    return edited_tweet


def assess_tweet_ownership(
    paper_title: str,
    paper_authors: str,
    tweet_text: str,
    tweet_username: str,
    model: str = "gpt-4o",
):
    """Assess if a tweet is owned by LLMpedia."""
    system_prompt = ps.TWEET_OWNERSHIP_SYSTEM_PROMPT
    user_prompt = ps.TWEET_OWNERSHIP_USER_PROMPT.format(
        paper_title=paper_title,
        paper_authors=paper_authors,
        tweet_text=tweet_text,
        tweet_username=tweet_username,
    )
    tweet_ownership = run_instructor_query(
        system_prompt, user_prompt, llm_model=model, process_id="assess_tweet_ownership"
    )
    return tweet_ownership


def assess_llm_relevance(tweet_text: str, model: str = "gpt-4o") -> bool:
    """
    Assess if a tweet is related to LLMs or similar topics.
    
    Args:
        tweet_text (str): The text content of the tweet
        model (str): The model to use for assessment
        
    Returns:
        bool: True if the tweet is related to LLMs, False otherwise
    """
    system_prompt = ps.LLM_RELEVANCE_SYSTEM_PROMPT
    user_prompt = ps.LLM_RELEVANCE_USER_PROMPT.format(tweet_text=tweet_text)
    
    is_relevant = run_instructor_query(
        system_prompt,
        user_prompt,
        llm_model=model,
        process_id="assess_llm_relevance"
    )
    
    return bool(int(is_relevant))
