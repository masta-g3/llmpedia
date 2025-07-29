from pydantic import BaseModel
from typing import List, Tuple, Optional, Callable, Dict
import pandas as pd
import numpy as np
import datetime
import json
import os, re
import boto3
import random
from datetime import timedelta

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings.huggingface import HuggingFaceInferenceAPIEmbeddings
from langchain_cohere import CohereEmbeddings
from langchain.prompts.chat import ChatPromptTemplate
from langchain.chains import LLMChain

from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from utils.custom_langchain import NewCohereEmbeddings, NewPGVector
from utils.instruct import run_instructor_query
import utils.pydantic_objects as po
import utils.prompts as ps
from utils.db import db_utils, db

CONNECTION_STRING = (
    f"postgresql+psycopg2://{db_utils.db_params['user']}:{db_utils.db_params['password']}"
    f"@{db_utils.db_params['host']}:{db_utils.db_params['port']}/{db_utils.db_params['dbname']}"
)

VS_EMBEDDING_MODEL = "voyage"

report_sections_map = {
    "scratchpad": "Scratchpad",
    "new_developments_findings": "New Development & Findings",
    "highlight_of_the_week": "Highlight of the Week",
    "related_websites_libraries_repos": "Related Websites, Libraries and Repos",
}


def prepare_calendar_data(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Prepares data for the creation of a calendar heatmap."""
    df["published"] = pd.to_datetime(df["published"])
    df_year = df[df["published"].dt.year == int(year)].copy()
    ## publishes dates with zero 'Counts' with full year dates.
    df_year = (
        df_year.set_index("published")
        .reindex(pd.date_range(start=f"{year}-01-01", end=f"{year}-12-31", freq="D"))
        .fillna(0)
        .reset_index()
    )
    df_year.columns = ["published", "Count"]
    df_year["week"] = df_year["published"].dt.isocalendar().week - 1
    df_year["weekday"] = df_year["published"].dt.weekday
    return df_year


def get_weekly_summary(date_str: str):
    try:
        weekly_content = db.get_weekly_content(date_str, content_type="content")
        weekly_content = add_links_to_text_blob(weekly_content)
        weekly_highlight = db.get_weekly_content(date_str, content_type="highlight")
        weekly_highlight = add_links_to_text_blob(weekly_highlight)
        ## ToDo: Remove this.
        ## ---------------------
        if "\n" in weekly_highlight:
            weekly_highlight = "#### " + weekly_highlight.replace("###", "")
        ## ---------------------
        weekly_repos_df = db.get_weekly_repos(date_str)

        ## Process repo content.
        weekly_repos_df["repo_link"] = weekly_repos_df.apply(
            lambda row: f"[{row['title']}]({row['url']}): {row['description']}", axis=1
        )

        grouped_repos = (
            weekly_repos_df.groupby("topic")["repo_link"]
            .apply(lambda l: "\n".join(l))
            .reset_index()
        )
        grouped_repos["repo_count"] = (
            weekly_repos_df.groupby("topic")["repo_link"].count().values
        )
        grouped_repos.sort_values(by="repo_count", ascending=False, inplace=True)

        miscellaneous_row = grouped_repos[grouped_repos["topic"] == "Miscellaneous"]
        grouped_repos = grouped_repos[grouped_repos["topic"] != "Miscellaneous"]
        grouped_repos = pd.concat([grouped_repos, miscellaneous_row], ignore_index=True)

        repos_section = "## 💿 Repos & Libraries\n\n"
        repos_section += "Many web resources were shared this week. Below are some of them, grouped by topic.\n\n"
        for _, row in grouped_repos.iterrows():
            repos_section += f"#### {row['topic']}\n"
            repo_links = row["repo_link"].split("\n")
            for link in repo_links:
                repos_section += f"- {link}\n"
            repos_section += "\n"

    except:
        weekly_content = db.get_weekly_summary_old(date_str)
        weekly_highlight = ""
        repos_section = ""

    return weekly_content, weekly_highlight, repos_section


def parse_weekly_report(report_md: str):
    """Extract sections of the weekly report into dict."""
    sections = report_md.split("\n## ")
    parsed_report = {}
    for section in sections:
        if section.startswith("Scratchpad"):
            continue
        if section.strip():
            title, *content = section.split("\n", 1)
            clean_content = content[0].strip() if content else ""
            clean_content = add_links_to_text_blob(clean_content)
            parsed_report[title.strip()] = clean_content
    return parsed_report


def add_links_to_text_blob(response: str):
    """Add links to arxiv codes and Reddit posts in the response."""

    def arxiv_repl(match):
        return f"[arxiv:{match.group(1)}](https://llmpedia.ai/?arxiv_code={match.group(1)})"

    def reddit_repl(match):
        if match.group(1):  # New format: reddit:subreddit:post_id
            subreddit = match.group(1)
            reddit_id = match.group(2)
            permalink = f"https://www.reddit.com/r/{subreddit}/comments/{reddit_id}/"
            return f"[r/{subreddit}:{reddit_id}]({permalink})"
        else:  # Old format: reddit:post_id (fallback)
            reddit_id = match.group(2)
            return f"[reddit:{reddit_id}](https://www.reddit.com/search/?q={reddit_id})"

    # Apply both transformations
    response = re.sub(r"arxiv:(\d{4}\.\d{4,5})", arxiv_repl, response)
    response = re.sub(r"reddit:(?:([^:]+):)?([a-zA-Z0-9_/-]+)", reddit_repl, response)
    return response


def extract_arxiv_codes(text: str):
    """Extract unique arxiv codes from the text."""
    arxiv_codes = re.findall(r"arxiv:(\d{4}\.\d{4,5})", text)
    return list(set(arxiv_codes))


def extract_reddit_codes(text: str):
    """Extract unique Reddit post IDs from the text."""
    # Handle both new format (reddit:subreddit:post_id) and old format (reddit:post_id)
    matches = re.findall(r"reddit:(?:([^:]+):)?([a-zA-Z0-9_/-]+)", text)
    reddit_codes = []
    for subreddit, post_id in matches:
        if subreddit:  # New format with subreddit
            reddit_codes.append(f"{subreddit}:{post_id}")
        else:  # Old format, just post_id
            reddit_codes.append(post_id)
    return list(set(reddit_codes))


def extract_all_citations(text: str):
    """Extract all arxiv and reddit citations from text, returning them with prefixes."""
    arxiv_codes = extract_arxiv_codes(text)
    reddit_codes = extract_reddit_codes(text)

    ## Return with prefixes for consistency with citation format
    citations = []
    citations.extend([f"arxiv:{code}" for code in arxiv_codes])
    citations.extend([f"reddit:{code}" for code in reddit_codes])

    return citations


def get_img_link_for_blob(text_blob: str):
    """Identify `arxiv_code:XXXX.XXXXX` from a text blob, and generate a Markdown link to its img."""
    arxiv_code = re.findall(r"arxiv(?:_code)?:(\d{4}\.\d{4,5})", text_blob)
    if len(arxiv_code) == 0:
        return None
    arxiv_code = arxiv_code[0]
    return f"https://arxiv-art.s3.amazonaws.com/{arxiv_code}.png"


def numbered_to_bullet_list(list_str: str):
    """Convert a numbered liadd_links_to_text_blobst to a bullet list."""
    list_str = re.sub(r"^\d+\.", r"-", list_str, flags=re.MULTILINE).strip()
    list_str = list_str.replace("</|im_end|>", "").strip()
    ## Remove extra line breaks.
    list_str = re.sub(r"\n{3,}", "\n\n", list_str)
    return list_str


##################
## VECTOR STORE ##
##################


def initialize_retriever(collection_name):
    """Initialize retriever for GPT maestro."""
    if collection_name == "arxiv_vectors_cv3":
        embeddings = NewCohereEmbeddings(
            cohere_api_key=os.getenv("COHERE_API_KEY"), model="embed-english-v3.0"
        )
    elif collection_name == "arxiv_vectors":
        embeddings = HuggingFaceInferenceAPIEmbeddings(
            api_key=os.getenv("HUGGINGFACE_API_KEY"), model_name="thenlper/gte-large"
        )
    else:
        raise ValueError(f"Unknown collection name: {collection_name}")

    store = NewPGVector(
        collection_name=collection_name,
        connection_string=CONNECTION_STRING,
        embedding_function=embeddings,
    )
    retriever = store.as_retriever(search_type="similarity", search_kwargs={"k": 20})

    compressor = CohereRerank(
        top_n=10, cohere_api_key=os.getenv("COHERE_API_KEY"), user_agent="llmpedia"
    )
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )
    return compression_retriever


def create_rag_context(parent_docs: pd.DataFrame) -> str:
    """Create RAG context for LLM, including text excerpts, arxiv_codes,
    year of publication and citation counts."""
    rag_context = ""
    for subject, subject_df in parent_docs.groupby("subject"):
        rag_context += f"### {subject}\n\n"
        for idx, doc in subject_df.iterrows():
            arxiv_code = doc["arxiv_code"]
            title = doc["title"]
            year = doc["published"]
            citation_count = doc["citation_count"]
            text = "..." + doc["text"] + "..."
            rag_context += f"Tittle:'*{title}*', arxiv:{arxiv_code} ({year}, {citation_count} citations)\n\n{text}\n\n"
        rag_context += "---\n\n"

    return rag_context


######################
## VECTOR STORE NEW ##
######################


class RedditDiscussion(BaseModel):
    subreddit: str
    post_title: str
    post_content: str
    post_author: str
    post_score: int
    num_comments: int
    post_timestamp: datetime.datetime
    top_comments: List[str] = []
    comment_scores: List[int] = []


class RedditContent(BaseModel):
    reddit_id: str
    subreddit: str
    title: str
    content: str
    author: str
    score: int
    num_comments: int
    published_date: datetime.datetime
    content_type: str  # 'reddit_post' or 'reddit_comment'
    distance: float
    citations: int = 0  # Reddit posts don't have citations, default to 0

    @property
    def abstract(self) -> str:
        """Map content to abstract for reranker compatibility."""
        return self.content


class Document(BaseModel):
    arxiv_code: str
    title: str
    published_date: datetime.datetime
    citations: int
    abstract: str
    notes: str
    tokens: int
    distance: float
    reddit_discussions: Optional[List[RedditDiscussion]] = None
    reddit_metrics: Optional[Dict[str, int]] = None


query_config_json = """
{
  "title": "LOWER(a.title) LIKE LOWER('%%%s%%')",
  "min_publication_date": "a.published >= '%s'",
  "max_publication_date": "a.published <= '%s'",
  "topic_categories": "t.topic IN ('%s')",
  "min_citations": "s.citation_count > %d",
  "semantic_search_queries": "(%s)",
  "min_reddit_score": "EXISTS (SELECT 1 FROM reddit_posts r WHERE r.arxiv_code = a.arxiv_code AND r.score >= %d)",
  "subreddits": "EXISTS (SELECT 1 FROM reddit_posts r WHERE r.arxiv_code = a.arxiv_code AND r.subreddit IN ('%s'))",
  "has_reddit_discussion": "EXISTS (SELECT 1 FROM reddit_posts r WHERE r.arxiv_code = a.arxiv_code AND r.arxiv_code IS NOT NULL AND r.arxiv_code != '' AND r.arxiv_code != 'null')"
}
"""

query_config = json.loads(query_config_json)


def interrogate_paper(question: str, arxiv_code: str, model="gpt-4o") -> str:
    """Ask a question about a paper using full markdown content with fallback to extended notes."""
    ## Try to get full paper markdown content first (preferred)
    context, markdown_success = get_paper_markdown(arxiv_code)

    if not markdown_success:
        ## Fallback to extended notes if markdown unavailable
        context = db.get_extended_notes(arxiv_code, level=1)
        if not context or pd.isna(context):
            return "Paper content not available yet. Check back soon!"

    user_message = ps.create_interrogate_user_prompt(
        context=context, user_question=question
    )
    response = run_instructor_query(
        system_message=ps.INTERROGATE_PAPER_SYSTEM_PROMPT,
        user_message=user_message,
        model=None,
        llm_model=model,
        temperature=0.3,
        process_id="paper_qna",
    )
    response = response.replace("<response>", "").replace("</response>", "")
    return response


def decide_query_action(
    user_question: str, llm_model: str = "gpt-4o-mini"
) -> po.QueryDecision:
    """Decide the query action based on the user question."""
    system_message = "Please analyze the following user query and answer the question."
    user_message = ps.create_decision_user_prompt(user_question)
    response = run_instructor_query(
        system_message,
        user_message,
        po.QueryDecision,
        llm_model=llm_model,
        process_id="decide_query_action",
    )
    return response


def generate_query_object(user_question: str, llm_model: str):
    query_obj = run_instructor_query(
        ps.VS_QUERY_SYSTEM_PROMPT,
        ps.create_query_user_prompt(user_question),
        po.SearchCriteria,
        llm_model=llm_model,
        temperature=0.5,
        process_id="generate_query_object",
    )
    return query_obj


def decide_deep_search_step(
    original_question: str,
    scratchpad_content: str,
    previous_queries: list[str],
    llm_model: str,
) -> po.NextSearchStepDecision:
    """Decide whether to continue deep search based on scratchpad and determine the next query."""
    user_message = ps.create_deep_search_decision_user_prompt(
        original_question=original_question,
        scratchpad_content=scratchpad_content,
        previous_queries=previous_queries,
    )
    response = run_instructor_query(
        ps.DEEP_SEARCH_DECISION_SYSTEM_PROMPT,
        user_message,
        po.NextSearchStepDecision,
        llm_model=llm_model,
        temperature=0.3,  # Lower temp for more deterministic decision
        process_id="decide_deep_search_step",
    )
    return response


def analyze_and_update_scratchpad(
    original_question: str,
    scratchpad_content: str,
    current_query: str,
    reranked_docs: list[Document],
    llm_model: str,
) -> po.ScratchpadAnalysisResult:
    """Analyze reranked documents and update the research scratchpad."""
    user_message = ps.create_scratchpad_analysis_user_prompt(
        original_question=original_question,
        scratchpad_content=scratchpad_content,
        current_query=current_query,
        reranked_docs=reranked_docs,
    )
    analysis_result = run_instructor_query(
        ps.SCRATCHPAD_ANALYSIS_SYSTEM_PROMPT,
        user_message,
        po.ScratchpadAnalysisResult,
        llm_model=llm_model,
        temperature=0.5,
        process_id="analyze_scratchpad",
        thinking={"type": "enabled", "budget_tokens": 2048},
    )
    return analysis_result


def rerank_documents_new(
    user_question: str, documents: list, llm_model="gpt-4o", temperature=0.2
) -> po.RerankedDocuments:
    system_message = "You are an expert system that can identify and select relevant arxiv papers that can be used to answer a user query."
    rerank_msg = ps.create_rerank_user_prompt(user_question, documents)
    response = run_instructor_query(
        system_message,
        rerank_msg,
        po.RerankedDocuments,
        llm_model=llm_model,
        temperature=temperature,
        process_id="rerank_documents",
    )
    return response


def resolve_query(
    user_question: str,
    documents: list[Document],
    response_length: int,
    llm_model: str,
    custom_instructions: Optional[str] = None,
):
    system_message = "You are GPT Maestro, an AI expert focused on Large Language Models. Answer the user query leveraging the information provided in the context. Pay close attention to the provided guidelines."
    user_message = ps.create_resolve_user_prompt(
        user_question=user_question,
        documents=documents,
        response_length=response_length,
        custom_instructions=custom_instructions,
    )
    response = run_instructor_query(
        system_message=system_message,
        user_message=user_message,
        model=po.ResolveQuery,
        llm_model=llm_model,
        temperature=0.8,
        process_id="resolve_query",
    )
    return response


def resolve_from_scratchpad(
    original_question: str,
    scratchpad_content: str,
    response_length: int,  # May need different calculation based on scratchpad
    llm_model: str,
    custom_instructions: Optional[str] = None,
) -> str:  # Returning string directly for now
    """Generate final response based on the final scratchpad content."""

    user_message = ps.create_resolve_scratchpad_user_prompt(
        original_question=original_question,
        scratchpad_content=scratchpad_content,
        response_length=response_length,
        custom_instructions=custom_instructions,
    )

    response_obj = run_instructor_query(
        system_message=ps.RESOLVE_SCRATCHPAD_SYSTEM_PROMPT,
        user_message=user_message,
        model=po.ResolveScratchpadResponse,  # Use the new simple model
        llm_model=llm_model,
        temperature=0.7,  # Balance creativity and factuality for synthesis
        process_id="resolve_from_scratchpad",
    )

    return response_obj.response


def resolve_query_other(user_question: str) -> str:
    """Decide the query action based on the user question."""
    system_message = "You are the GPT Maestro, maintainer of the LLMpedia, a web-based Large Language Model encyclopedia and collection of research papers. You received the following unrelated comment from a user via our chat based system. Please respond to it in a friendly, yet serious and very concise (less than 20 words) manner."
    user_message = f"{user_question}"
    response = run_instructor_query(
        system_message,
        user_message,
        None,
        llm_model="gpt-4o-mini",
        process_id="resolve_query_other",
    )
    return response


def log_debug(msg: str, data: any = None, indent_level: int = 0):
    """Helper function to print debug information in a clean, structured way."""
    indent = "  " * indent_level
    print(f"{indent}🔍 {msg}")
    if data is not None:
        if isinstance(data, (list, dict)):
            import json

            print(f"{indent}   {json.dumps(data, indent=2, default=str)}")
        else:
            print(f"{indent}   {data}")


def get_similar_titles(
    title: str, df: pd.DataFrame, n: int = 5
) -> Tuple[List[str], str]:
    """Return similar titles based on topic cluster."""
    title = title.lower()
    if title in df["title"].str.lower().values:
        cluster = df[df["title"].str.lower() == title]["topic"].values[0]
        similar_df = df[df["topic"] == cluster]
        similar_df = similar_df[similar_df["title"].str.lower() != title]

        size = similar_df.shape[0]
        similar_df = similar_df.sample(min(n, size))

        similar_names = [
            f"{row['title']} (arxiv:{row['arxiv_code']})"
            for index, row in similar_df.iterrows()
        ]
        similar_names = [add_links_to_text_blob(title) for title in similar_names]

        return similar_names, cluster
    else:
        return [], ""


def get_similar_docs(
    arxiv_code: str, df: pd.DataFrame, n: int = 5
) -> Tuple[List[str], List[str], List[str]]:
    """Get most similar documents based on cosine similarity."""
    if arxiv_code in df.index:
        similar_docs = df.loc[arxiv_code]["similar_docs"]
        similar_docs = [d for d in similar_docs if d in df.index]

        if len(similar_docs) > n:
            similar_docs = np.random.choice(similar_docs, n, replace=False)

        similar_titles = [df.loc[doc]["title"] for doc in similar_docs]
        publish_dates = [df.loc[doc]["published"] for doc in similar_docs]

        return similar_docs, similar_titles, publish_dates
    else:
        return [], [], []


def get_paper_markdown(arxiv_code: str) -> Tuple[str, bool]:
    """Fetch and process paper markdown from S3."""
    s3 = boto3.client("s3")

    try:
        # First check if the paper directory exists in S3
        response = s3.list_objects_v2(Bucket="arxiv-md", Prefix=f"{arxiv_code}/")

        if "Contents" not in response:
            return "Paper content not available yet. Check back soon!", False

        # Get the paper.md content
        response = s3.get_object(Bucket="arxiv-md", Key=f"{arxiv_code}/paper.md")
        markdown_content = response.get("Body").read().decode("utf-8")

        # Process image references to point to S3
        # First, handle relative paths
        markdown_content = re.sub(
            r"!\[(.*?)\]\((?!http)(.*?)\)",
            lambda m: f"![{m.group(1)}](https://arxiv-md.s3.amazonaws.com/{arxiv_code}/{m.group(2)})",
            markdown_content,
        )

        # Then, handle paths that might start with the arxiv code
        markdown_content = re.sub(
            f"!\[(.*?)\]\({arxiv_code}/(.*?)\)",
            lambda m: f"![{m.group(1)}](https://arxiv-md.s3.amazonaws.com/{arxiv_code}/{m.group(2)})",
            markdown_content,
        )

        return markdown_content, True

    except Exception as e:
        return f"Error loading paper content: {str(e)}", False


def query_llmpedia_new(
    user_question: str,
    response_length: int = 4000,
    llm_model: str = "openai/gpt-4.1-nano",
    max_sources: int = 15,
    max_agents: int = 4,
    debug: bool = False,
    progress_callback: Optional[Callable[[str], None]] = None,
    show_only_sources: bool = False,
) -> Tuple[str, List[str], List[str]]:
    """Query LLMpedia using unified multi-agent deep research approach."""
    if progress_callback:
        progress_callback("🧠 Analyzing your question...")
    if debug:
        log_debug("~~Starting LLMpedia unified query pipeline~~")
        log_debug(
            "Input parameters:",
            {
                "question": user_question,
                "response_length": response_length,
                "max_sources": max_sources,
                "max_agents": max_agents,
                "show_only_sources": show_only_sources,
            },
        )

    action = decide_query_action(user_question, llm_model=llm_model)
    if debug:
        log_debug("Query action decision:", action.model_dump(), 1)

    if action.llm_query:
        if debug:
            log_debug("🚀 Initiating Multi-Agent Deep Research Mode", indent_level=1)

        ## Import here to avoid circular dependency
        from deep_research import deep_research_query

        if debug:
            log_debug(
                f"Deep research config: {max_agents} agents, {max_sources} sources each",
                indent_level=2,
            )

        ## Call unified multi-agent deep research implementation
        (
            final_answer_title,
            workflow_id,
            final_answer,
            referenced_codes_list,
            additional_relevant_codes,
        ) = deep_research_query(
            user_question=user_question,
            max_agents=max_agents,
            max_sources_per_agent=max_sources,
            response_length=response_length,
            llm_model=llm_model,
            progress_callback=progress_callback,
            verbose=debug,
        )

        if debug:
            log_debug("~~Finished LLMpedia unified query pipeline~~", indent_level=0)

        return (
            final_answer_title,
            final_answer,
            referenced_codes_list,
            additional_relevant_codes,
        )

    else:  # Non-LLM query
        if debug:
            log_debug(
                "Query classified as non-LLM related, generating simple response",
                indent_level=1,
            )
        if progress_callback:
            progress_callback("💬 Preparing a direct response...")
        answer = resolve_query_other(user_question)
        if debug:
            log_debug(
                "~~Finished LLMpedia query pipeline (non-LLM query)~~", indent_level=0
            )
        return answer, [], []


######################
## TEXT ANALYSIS    ##
######################


def get_domain_stopwords():
    """Returns domain-specific stopwords for LLM research."""
    domain_stopwords = [
        "paper",
        "model",
        "using",
        "used",
        "method",
        "approach",
        "result",
        "show",
        "propose",
        "proposed",
        "introduces",
        "introduce",
        "study",
        "work",
        "research",
        "experiment",
        "technique",
        "framework",
        "language",
        "large",
        "training",
        "test",
        "data",
        "dataset",
        "performance",
        "accuracy",
        "evaluation",
        "benchmark",
        "baseline",
    ]

    bigram_stopwords = [
        "neural network",
        "deep learning",
        "machine learning",
        "artificial intelligence",
        "pretrained model",
        "pre trained",
        "state art",
        "state of art",
        "evaluation result",
        "novel approach",
        "experimental result",
        "training data",
        "technical report",
        "empirical study",
        "recent work",
        "experimental evaluation",
        "language model",
        "large language",
        "large language model",
    ]

    stopwords = list(set(domain_stopwords + bigram_stopwords))

    return stopwords


def preprocess_text(text):
    """Basic text preprocessing with lemmatization."""
    try:
        lemmatizer = WordNetLemmatizer()
        lemmatizer.lemmatize("test")
    except:
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)
        lemmatizer = WordNetLemmatizer()

    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = text.replace("-", " ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    words = text.split()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    text = " ".join(lemmatized_words)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_trending_topics(documents, n=15, ngram_range=(2, 3), min_df=2, max_df=0.8):
    """Extract trending topics from a list of documents using TF-IDF vectorization."""

    try:
        nltk_stop = stopwords.words("english")
    except:
        nltk.download("stopwords", quiet=True)
        nltk_stop = stopwords.words("english")

    domain_stopwords = get_domain_stopwords()
    all_stopwords = list(set(nltk_stop + domain_stopwords))
    all_stopwords = [preprocess_text(word) for word in all_stopwords]
    all_stopwords = [
        word for word in all_stopwords if word and len(word) >= 2 and word.isalpha()
    ]
    all_stopwords = list(set(all_stopwords))

    # Create TF-IDF vectorizer for bi and trigrams with optimized parameters
    tfidf_vectorizer = TfidfVectorizer(
        stop_words=all_stopwords,  # Use the preprocessed list
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        max_features=50,
        token_pattern=r"(?u)\b[a-z][a-z]+\b",
    )

    tfidf_vectorizer.use_idf = True
    tfidf_vectorizer.smooth_idf = True
    tfidf_vectorizer.sublinear_tf = True

    if len(documents) > 1:
        tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
        feature_names = tfidf_vectorizer.get_feature_names_out()

        ## Sum TF-IDF scores for each term across all documents.
        tfidf_scores = tfidf_matrix.sum(axis=0).A1
        term_scores = {term: score for term, score in zip(feature_names, tfidf_scores)}

        ## Get all terms sorted by score
        all_sorted_terms = sorted(term_scores.items(), key=lambda x: x[1], reverse=True)

        ## Extract trigrams
        trigrams = {term for term, _ in all_sorted_terms if len(term.split()) == 3}

        ## Filter out bigrams that are subsets of trigrams
        filtered_terms = [
            (term, score)
            for term, score in all_sorted_terms
            if len(term.split()) != 2 or not any(term in tri for tri in trigrams)
        ]

        return filtered_terms[:n]
    else:
        return []


def get_trending_topics_from_papers(papers_df, time_window_days=7, n=15):
    """Extract trending topics from recent papers."""
    ## Get recent papers.
    today = pd.Timestamp.now().date()
    cutoff_date = today - pd.Timedelta(days=time_window_days)
    recent_papers = papers_df[papers_df["published"].dt.date >= cutoff_date]

    if len(recent_papers) == 0:
        return []

    ## Get titles and punchlines.
    titles = recent_papers["title"].fillna("").tolist()
    punchlines = recent_papers["punchline"].fillna("").tolist()

    ## Combine texts (giving more weight to titles by including them twice).
    ## Apply preprocess_text to each document.
    documents = (
        [preprocess_text(title) for title in titles]
        + [preprocess_text(title) for title in titles]
        + [preprocess_text(punchline) for punchline in punchlines]
    )

    # Extract trending topics
    return extract_trending_topics(documents, n=n)


def get_top_cited_papers(papers_df, n=5, time_window_days=None):
    """Get the top cited papers, optionally filtering by recency."""
    df = papers_df.copy()
    # Ensure 'published' column is in datetime format
    df["published"] = pd.to_datetime(df["published"], errors="coerce")

    if time_window_days is not None:
        today = pd.Timestamp.now().date()
        cutoff_date = today - pd.Timedelta(days=time_window_days)
        # Filter out rows where 'published' is NaT (Not a Time) after coercion
        df = df.dropna(subset=["published"])
        df = df[df["published"].dt.date >= cutoff_date]

    return df.sort_values("citation_count", ascending=False).head(n)


def process_trending_data(
    papers_df_fragment: pd.DataFrame, raw_trending_df: pd.DataFrame, top_n_display: int
) -> pd.DataFrame:
    """
    Processes raw trending data (typically from db.get_trending_papers)
    by joining with the current papers_df_fragment, calculating like counts,
    and returning the top N papers with tweet details.
    """
    if papers_df_fragment.empty or raw_trending_df.empty:
        return pd.DataFrame()

    # Create dictionaries for all the new data columns
    counts_dict = dict(
        zip(raw_trending_df["arxiv_code"], raw_trending_df["like_count"])
    )
    tweet_count_dict = dict(
        zip(raw_trending_df["arxiv_code"], raw_trending_df["tweet_count"])
    )
    tweets_dict = dict(zip(raw_trending_df["arxiv_code"], raw_trending_df["tweets"]))

    # Filter the main papers dataframe for codes present in the trending data
    # and create a copy to avoid SettingWithCopyWarning
    trending_papers_intermediate = papers_df_fragment[
        papers_df_fragment["arxiv_code"].isin(counts_dict.keys())
    ].copy()

    if trending_papers_intermediate.empty:
        return pd.DataFrame()

    # Map all the new data to the filtered papers
    trending_papers_intermediate["like_count"] = trending_papers_intermediate[
        "arxiv_code"
    ].map(counts_dict)
    trending_papers_intermediate["tweet_count"] = trending_papers_intermediate[
        "arxiv_code"
    ].map(tweet_count_dict)
    trending_papers_intermediate["tweets"] = trending_papers_intermediate[
        "arxiv_code"
    ].map(tweets_dict)

    # Ensure 'like_count' is numeric, coercing errors and filling NaNs with 0
    trending_papers_intermediate["like_count"] = (
        pd.to_numeric(trending_papers_intermediate["like_count"], errors="coerce")
        .fillna(0)
        .astype(int)
    )  # Convert to int after fillna

    # Sort by like_count in descending order
    trending_papers_intermediate.sort_values(
        "like_count", ascending=False, inplace=True
    )

    return trending_papers_intermediate.head(top_n_display)


def get_latest_weekly_highlight():
    """Get the arxiv code for the latest weekly highlight."""
    max_date = db_utils.get_max_table_date("weekly_content")
    date_report = max_date - pd.Timedelta(days=max_date.weekday())

    highlight_text = db.get_weekly_content(date_report, content_type="highlight")
    if highlight_text:
        match = re.search(r"arxiv:(\d{4}\.\d{4,5})", highlight_text)
        if match:
            return match.group(1)

    return None


######################
## REDDIT INTEGRATION ##
######################


def convert_reddit_data_to_discussions(
    reddit_df: pd.DataFrame,
) -> List[RedditDiscussion]:
    """Convert Reddit DataFrame to RedditDiscussion objects."""
    discussions = []

    if reddit_df.empty:
        return discussions

    ## Group by post to aggregate comments
    for arxiv_code, group in reddit_df.groupby("arxiv_code"):
        post_data = group.iloc[0]  # Get post data from first row

        ## Extract comments from the group
        comments = []
        comment_scores = []
        for _, row in group.iterrows():
            if pd.notna(row.get("comment_content")) and row["comment_content"].strip():
                comments.append(row["comment_content"])
                comment_scores.append(row.get("comment_score", 0))

        ## Sort comments by score and take top ones
        if comments:
            comment_data = list(zip(comments, comment_scores))
            comment_data.sort(key=lambda x: x[1], reverse=True)
            top_comments = [
                comment for comment, _ in comment_data[:5]
            ]  # Top 5 comments
            scores = [score for _, score in comment_data[:5]]
        else:
            top_comments = []
            scores = []

        discussion = RedditDiscussion(
            subreddit=post_data["subreddit"],
            post_title=post_data["post_title"],
            post_content=post_data.get("post_content", "") or "",
            post_author=post_data.get("post_author", "") or "",
            post_score=int(post_data.get("post_score", 0)),
            num_comments=int(post_data.get("num_comments", 0)),
            post_timestamp=pd.to_datetime(post_data["post_timestamp"]).to_pydatetime(),
            top_comments=top_comments,
            comment_scores=scores,
        )
        discussions.append(discussion)

    return discussions


def enhance_documents_with_reddit(documents: List[Document]) -> List[Document]:
    """Enhance Document objects with Reddit discussions and metrics."""
    if not documents:
        return documents

    ## Import here to avoid circular imports
    from utils.db import db

    ## Extract arXiv codes
    arxiv_codes = [doc.arxiv_code for doc in documents]

    ## Load Reddit data for these papers
    reddit_df = db.load_reddit_for_papers(arxiv_codes)
    reddit_metrics_df = db.get_reddit_discussions_summary(arxiv_codes)

    ## Create mapping of arXiv codes to Reddit data
    reddit_discussions_map = {}
    if not reddit_df.empty:
        for arxiv_code, group in reddit_df.groupby("arxiv_code"):
            reddit_discussions_map[arxiv_code] = convert_reddit_data_to_discussions(
                group
            )

    reddit_metrics_map = {}
    if not reddit_metrics_df.empty:
        for _, row in reddit_metrics_df.iterrows():
            reddit_metrics_map[row["arxiv_code"]] = {
                "post_count": int(row["post_count"]),
                "comment_count": int(row["comment_count"]),
                "total_post_score": int(row["total_post_score"]),
                "avg_post_score": float(row["avg_post_score"]),
                "subreddit_count": int(row["subreddit_count"]),
                "subreddits": row["subreddits"],
            }

    ## Enhance documents
    enhanced_documents = []
    for doc in documents:
        doc.reddit_discussions = reddit_discussions_map.get(doc.arxiv_code, [])
        doc.reddit_metrics = reddit_metrics_map.get(doc.arxiv_code, {})
        enhanced_documents.append(doc)

    return enhanced_documents
