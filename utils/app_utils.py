from pydantic import BaseModel
from typing import List, Tuple, Optional, Callable
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
    """Add links to arxiv codes in the response."""

    def repl(match):
        return f"[arxiv:{match.group(1)}](https://llmpedia.ai/?arxiv_code={match.group(1)})"

    return re.sub(r"arxiv:(\d{4}\.\d{4,5})", repl, response)


def extract_arxiv_codes(text: str):
    """Extract unique arxiv codes from the text."""
    arxiv_codes = re.findall(r"arxiv:(\d{4}\.\d{4,5})", text)
    return list(set(arxiv_codes))


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


class Document(BaseModel):
    arxiv_code: str
    title: str
    published_date: datetime.datetime
    citations: int
    abstract: str
    notes: str
    tokens: int
    distance: float


query_config_json = """
{
  "title": "LOWER(a.title) LIKE LOWER('%%%s%%')",
  "min_publication_date": "a.published >= '%s'",
  "max_publication_date": "a.published <= '%s'",
  "topic_categories": "t.topic IN ('%s')",
  "min_citations": "s.citation_count > %d",
  "semantic_search_queries": "(%s)"
}
"""

query_config = json.loads(query_config_json)


def interrogate_paper(question: str, arxiv_code: str, model="gpt-4o") -> str:
    """Ask a question about a paper."""
    context = db.get_extended_notes(arxiv_code, expected_tokens=2000)
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
    response_length: int, # May need different calculation based on scratchpad
    llm_model: str,
    custom_instructions: Optional[str] = None,
) -> str: # Returning string directly for now
    """Generate final response based on the final scratchpad content."""
    
    user_message = ps.create_resolve_scratchpad_user_prompt(
        original_question=original_question,
        scratchpad_content=scratchpad_content,
        response_length=response_length,
        custom_instructions=custom_instructions
    )
    
    response_obj = run_instructor_query(
        system_message=ps.RESOLVE_SCRATCHPAD_SYSTEM_PROMPT,
        user_message=user_message,
        model=po.ResolveScratchpadResponse, # Use the new simple model
        llm_model=llm_model,
        temperature=0.7, # Balance creativity and factuality for synthesis
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
    query_llm_model: str = "claude-3-7-sonnet-20250219",
    rerank_llm_model: str = "claude-3-7-sonnet-20250219",
    response_llm_model: str = "claude-3-7-sonnet-20250219",
    max_sources: int = 25,
    debug: bool = False,
    progress_callback: Optional[Callable[[str], None]] = None,
    custom_instructions: Optional[str] = None,
    show_only_sources: bool = False,
    deep_research: bool = False,
    deep_research_iterations: int = 3,
) -> Tuple[str, List[str], List[str]]:
    """Query LLMpedia with customized response parameters, including deep research."""
    if progress_callback:
        progress_callback("🧠 Analyzing your question...")
    if debug:
        log_debug("~~Starting LLMpedia query pipeline~~")
        log_debug(
            "Input parameters:",
            {
                "question": user_question,
                "response_length": response_length,
                "max_sources": max_sources,
                "show_only_sources": show_only_sources,
                "deep_research": deep_research,
                "deep_research_iterations": deep_research_iterations,
            },
        )

    action = decide_query_action(user_question, llm_model=query_llm_model)
    if debug:
        log_debug("Query action decision:", action.model_dump(), 1)

    if action.llm_query:
        if deep_research:
            if debug:
                log_debug("🚀 Initiating Sophisticated Deep Research Mode", indent_level=1)
            if progress_callback:
                progress_callback("🚀 Initiating Deep Research Mode (Scratchpad)..." )

            scratchpad = "" # Initialize empty scratchpad (string for now)
            executed_queries = []
            current_query = user_question
            MAX_ITERATIONS = deep_research_iterations
            DOCS_TO_ANALYZE_PER_ITERATION = 5 # How many top docs to feed into analysis
            final_answer = ""
            all_referenced_codes = set()
            all_relevant_codes = set()
            processed_arxiv_codes = set() # Keep track of codes already fetched from DB

            for i in range(MAX_ITERATIONS):
                iteration_num = i + 1
                if debug:
                    log_debug(f"Deep Research Iteration {iteration_num}/{MAX_ITERATIONS}", indent_level=2)
                    log_debug(f"Current search query: '{current_query}'", indent_level=3)
                    log_debug(f"Scratchpad before analysis:\n{scratchpad}", indent_level=3)
                if progress_callback:
                    progress_callback(f"🔄 Iteration {iteration_num}/{MAX_ITERATIONS}: Searching for '{current_query[:50]}...'" )

                executed_queries.append(current_query)

                ## 1. Generate search criteria & Fetch
                query_obj = generate_query_object(
                    user_question=current_query, llm_model=query_llm_model
                )
                if debug:
                    log_debug("Generated search criteria:", query_obj.model_dump(), 4)
                
                criteria_dict = query_obj.model_dump(exclude_none=True)
                # Fetch a decent number for reranking within the loop
                criteria_dict["limit"] = max_sources * 2 
                sql = db.generate_semantic_search_query(
                    criteria_dict, 
                    query_config, 
                    embedding_model=VS_EMBEDDING_MODEL,
                    exclude_arxiv_codes=processed_arxiv_codes # Pass codes to exclude
                )
                documents_df = db_utils.execute_read_query(sql)

                iteration_docs = []
                if not documents_df.empty:
                    # Add newly fetched codes to processed_arxiv_codes BEFORE filtering iteration_docs
                    # This ensures that if a document is fetched but later filtered out (e.g., by reranking),
                    # it's still marked as processed for future DB queries.
                    for arxiv_code in documents_df["arxiv_code"].tolist():
                        processed_arxiv_codes.add(arxiv_code)
                    
                    iteration_docs = [
                        Document(
                            arxiv_code=d["arxiv_code"],
                            title=d["title"],
                            published_date=d["published_date"].to_pydatetime(),
                            citations=int(d["citations"]),
                            abstract=d["abstract"],
                            notes=d["notes"], # Make sure notes are fetched if available
                            tokens=int(d["tokens"]),
                            distance=float(d["similarity_score"]),
                        )
                        for _, d in documents_df.iterrows()
                    ]
                    if debug:
                        log_debug(f"Fetched {len(iteration_docs)} documents for this iteration.", indent_level=4)
                elif debug:
                    log_debug("No documents found for this iteration's query.", indent_level=4)
                
                ## 2. Rerank documents found in THIS iteration
                reranked_iter_docs_obj = None
                top_docs_for_analysis = []
                if iteration_docs:
                    if progress_callback:
                        progress_callback(f"⚖️ Reranking {len(iteration_docs)} new documents...")
                    reranked_iter_docs_obj = rerank_documents_new(
                        user_question=user_question, # Rerank against original question
                        documents=iteration_docs,
                        llm_model=rerank_llm_model
                    )
                    
                    # Get top N documents based on reranking score for analysis
                    reranked_ids_scores = sorted(
                        [(int(d.document_id), d.selected) for d in reranked_iter_docs_obj.documents],
                        key=lambda x: x[1], reverse=True
                    )
                    top_ids = [id for id, score in reranked_ids_scores if score > 0][:DOCS_TO_ANALYZE_PER_ITERATION]
                    top_docs_for_analysis = [iteration_docs[id] for id in top_ids]
                    
                    if debug:
                        log_debug(f"Selected top {len(top_docs_for_analysis)} docs for analysis (score > 0):", [d.arxiv_code for d in top_docs_for_analysis], 4)
                        # Log all reranked docs for this iter
                        # log_debug("Reranking details for iteration:", reranked_iter_docs_obj.model_dump(), 5)
                else:
                     if progress_callback:
                        progress_callback("🤷 No new documents found to rerank.")

                ## 3. Analyze & Update Scratchpad
                if top_docs_for_analysis:
                    if progress_callback:
                        progress_callback(f"📝 Analyzing {len(top_docs_for_analysis)} documents and updating scratchpad...")
                    
                    analysis_result = analyze_and_update_scratchpad(
                        original_question=user_question,
                        scratchpad_content=scratchpad,
                        current_query=current_query,
                        reranked_docs=top_docs_for_analysis,
                        llm_model=query_llm_model # Use query model for analysis
                    )
                    scratchpad = analysis_result.updated_scratchpad
                    
                    # Keep track of all potentially relevant codes mentioned during analysis
                    for doc in top_docs_for_analysis:
                         all_relevant_codes.add(doc.arxiv_code)

                    if debug:
                        log_debug("Scratchpad analysis result:", analysis_result.model_dump(), 4)
                        log_debug(f"Scratchpad after analysis:\n{scratchpad}", 4)
                elif iteration_docs:
                     # Docs were found but none scored > 0 in rerank
                     if progress_callback:
                         progress_callback("😐 No highly relevant documents found in this batch for analysis.")
                     if debug:
                         log_debug("No documents passed reranking threshold for analysis.", 4)
                # else: (No docs found at all) - already logged

                ## 4. Decide Next Step (unless last iteration)
                if iteration_num == MAX_ITERATIONS:
                    if debug:
                        log_debug("Max iterations reached. Proceeding to final synthesis.", indent_level=3)
                    if progress_callback:
                         progress_callback("🏁 Reached max search iterations. Preparing final answer...")
                    break

                if progress_callback:
                    progress_callback("🤔 Deciding if more research is needed based on scratchpad...")

                decision = decide_deep_search_step(
                    original_question=user_question,
                    scratchpad_content=scratchpad,
                    previous_queries=executed_queries,
                    llm_model=query_llm_model,
                )
                if debug:
                    log_debug("Decision for next step:", decision.model_dump(), 4)
                    # log_debug(f"Reasoning: {decision.reasoning}", indent_level=5)
                
                if not decision.continue_search or not decision.next_query:
                    if debug:
                        log_debug("Decision is to stop searching. Proceeding to final synthesis.", indent_level=3)
                    if progress_callback:
                         progress_callback("✅ Sufficient information gathered. Preparing final answer...")
                    break
                else:
                    current_query = decision.next_query
                    if progress_callback:
                         progress_callback(f"🔍 Refining search based on notes. Next query: '{current_query[:50]}...'" )
            
            ## End of deep research loop

            if not scratchpad.strip() or scratchpad == "Scratchpad is currently empty. This is the first analysis.": # Check if scratchpad has meaningful content
                if debug:
                    log_debug("Scratchpad is empty after deep research, returning early", indent_level=2)
                return (
                    "I conducted a deep search but couldn't build a comprehensive understanding. Try rephrasing your question.",
                    [],
                    [],
                )

            ## 5. Synthesize Final Response from Scratchpad
            if progress_callback:
                progress_callback("✍️ Synthesizing final response from research notes...")
            if debug:
                log_debug("Generating final response from scratchpad content.", indent_level=2)
                log_debug(f"Final Scratchpad:\n{scratchpad}", indent_level=3)

            final_answer = resolve_from_scratchpad(
                 original_question=user_question,
                 scratchpad_content=scratchpad,
                 response_length=response_length, # May need adjustment
                 llm_model=response_llm_model,
                 custom_instructions=custom_instructions
            )
            
            # Extract referenced codes from the final answer (if resolve_from_scratchpad adds them)
            all_referenced_codes = set(extract_arxiv_codes(final_answer)) 
            # We tracked 'all_relevant_codes' during analysis
            additional_relevant_codes = list(all_relevant_codes - all_referenced_codes)
            referenced_codes_list = list(all_referenced_codes)

            if debug:
                log_debug(
                    "Final Response Statistics (Scratchpad Method):",
                    {
                        "final_answer_length_words": len(final_answer.split()),
                        "referenced_papers_in_answer": len(referenced_codes_list),
                        "additional_relevant_papers_found": len(additional_relevant_codes),
                        "total_relevant_papers_considered": len(all_relevant_codes),
                    },
                    2,
                )
                log_debug("~~Finished LLMpedia query pipeline (Sophisticated Deep Research)~~", indent_level=0)

            return final_answer, referenced_codes_list, additional_relevant_codes

        else:
            ## Standard (non-deep) search path
            if progress_callback:
                progress_callback("🎯 Understanding search intent...")
            query_obj = generate_query_object(
                user_question=user_question, llm_model=query_llm_model
            )
            if debug:
                log_debug("Generated search criteria (standard):", query_obj.model_dump(), 2)

            if progress_callback:
                progress_callback("📜 Formulating search strategy...")

            ## Fetch results
            criteria_dict = query_obj.model_dump(exclude_none=True)
            criteria_dict["limit"] = max_sources * 2 # Fetch more for reranking
            sql = db.generate_semantic_search_query(
                criteria_dict, query_config, embedding_model=VS_EMBEDDING_MODEL
            )

            if progress_callback:
                progress_callback("🔍 Searching the archive for relevant papers...")

            documents_df = db_utils.execute_read_query(sql)
            if documents_df.empty:
                 if debug:
                    log_debug("No documents found (standard), returning early", indent_level=2)
                 return (
                    "I don't know about that my friend. Try asking something else.",
                    [],
                    [],
                 )

            documents = [
                Document(
                    arxiv_code=d["arxiv_code"],
                    title=d["title"],
                    published_date=d["published_date"].to_pydatetime(),
                    citations=int(d["citations"]),
                    abstract=d["abstract"],
                    notes=d["notes"],
                    tokens=int(d["tokens"]),
                    distance=float(d["similarity_score"]),
                )
                for _, d in documents_df.iterrows()
            ]
            num_initial_docs = len(documents)
            if progress_callback:
                progress_callback(f"📄 Found {num_initial_docs} initial candidate papers.")

            if debug:
                log_debug(f"Retrieved {len(documents)} initial documents (standard)", indent_level=2)
                for doc in documents:
                    log_debug(
                        f"- {doc.title} (arxiv:{doc.arxiv_code}, citations: {doc.citations}, distance: {doc.distance})",
                        indent_level=3,
                    )

            ## Common processing starts after this block if deep_search is False
            
        ## ======================================
        ## COMMON PROCESSING (Standard Search Only)
        ## ======================================
        # This block now only runs if deep_research is FALSE
        if not deep_research:
            if not documents: 
                # Safeguard, should be caught earlier in standard path
                if debug:
                    log_debug("No documents available for reranking (standard path), returning early", indent_level=2)
                return (
                    "I couldn't find relevant information for your query.",
                    [],
                    [],
                )

            num_docs_for_rerank = len(documents)
            if progress_callback:
                progress_callback(f"⚖️ Evaluating relevance of {num_docs_for_rerank} candidates...")

            ## Rerank.
            reranked_documents = rerank_documents_new(
                user_question=user_question, documents=documents, llm_model=rerank_llm_model
            )

            if debug:
                log_debug("Reranking analysis:", indent_level=2)
                doc_map = {i: doc.arxiv_code for i, doc in enumerate(documents)}
                for doc_analysis in reranked_documents.documents:
                    relevance = (
                        "HIGH" if doc_analysis.selected == 1.0
                        else "MEDIUM" if doc_analysis.selected == 0.5 else "LOW"
                    )
                    arxiv_code = doc_map.get(int(doc_analysis.document_id), "Unknown")
                    log_debug(
                        f"- [{relevance}] Document {doc_analysis.document_id} (arxiv:{arxiv_code}):",
                        indent_level=3,
                    )
                    log_debug(f"Analysis: {doc_analysis.analysis}", indent_level=4)

            ## Filter documents based on reranking
            high_relevance_docs = [
                (i, d)
                for i, d in enumerate(documents)
                if i
                in [
                    int(da.document_id)
                    for da in reranked_documents.documents
                    if da.selected == 1.0
                ]
            ]
            high_relevance_docs.sort(key=lambda x: x[1].published_date, reverse=True)

            medium_relevance_docs = [
                (i, d)
                for i, d in enumerate(documents)
                if i
                in [
                    int(da.document_id)
                    for da in reranked_documents.documents
                    if da.selected == 0.5
                ]
            ]
            medium_relevance_docs.sort(key=lambda x: x[1].published_date, reverse=True)

            ## Combine, limit by max_sources, and preserve order
            combined_sorted_indices = [i for i, _ in high_relevance_docs] + [i for i, _ in medium_relevance_docs]
            final_document_indices = combined_sorted_indices[:max_sources]

            filtered_documents = [documents[idx] for idx in final_document_indices]

            if debug:
                log_debug(
                    f"Selected {len(filtered_documents)} documents after reranking:",
                    indent_level=2,
                )
                for doc in filtered_documents:
                    log_debug(
                        f"- {doc.title} (arxiv:{doc.arxiv_code}, citations: {doc.citations}, date: {doc.published_date})",
                        indent_level=3,
                    )

            num_filtered_docs = len(filtered_documents)
            if progress_callback:
                progress_callback(f"✅ Selected {num_filtered_docs} most relevant papers.")

            if not filtered_documents:
                if debug:
                    log_debug(
                        "No documents selected after reranking, returning early",
                        indent_level=2,
                    )
                return (
                    "I found some initial papers, but none seemed relevant enough to answer your question after review. Try rephrasing.",
                    [],
                    [],
                )

            ## Final Resolution (Standard Path)
            if progress_callback:
                progress_callback(f"✍️ Synthesizing response from {num_filtered_docs} papers...")

            answer_obj = resolve_query(
                user_question=user_question,
                documents=filtered_documents,
                response_length=response_length,
                llm_model=rerank_llm_model,
                custom_instructions=custom_instructions,
            )
            if debug:
                log_debug("Resolved query:", answer_obj.model_dump(), 2)

            answer = answer_obj.response
            answer_augment = add_links_to_text_blob(answer)
            referenced_arxiv_codes = extract_arxiv_codes(answer_augment)

            ## Identify relevant codes not directly referenced in the text
            filtered_arxiv_codes_set = {d.arxiv_code for d in filtered_documents}
            referenced_arxiv_codes_set = set(referenced_arxiv_codes)
            additional_relevant_codes = list(filtered_arxiv_codes_set - referenced_arxiv_codes_set)

            if debug:
                log_debug(
                    "Response statistics:",
                    {
                        "response_length_words": len(answer.split()),
                        "referenced_papers": len(referenced_arxiv_codes_set),
                        "additional_relevant_papers": len(additional_relevant_codes),
                        "total_sources_considered": len(filtered_arxiv_codes_set),
                    },
                    2,
                )
                log_debug("~~Finished LLMpedia query pipeline (Standard Search)~~", indent_level=0)
            
            return answer_augment, referenced_arxiv_codes, additional_relevant_codes

    else: # Non-LLM query
        if debug:
            log_debug(
                "Query classified as non-LLM related, generating simple response",
                indent_level=1,
            )
        if progress_callback:
             progress_callback("💬 Preparing a direct response...")
        answer = resolve_query_other(user_question)
        if debug:
            log_debug("~~Finished LLMpedia query pipeline (non-LLM query)~~", indent_level=0)
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
    all_stopwords = [word for word in all_stopwords if word and len(word) >= 2 and word.isalpha()]
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
    df["published"] = pd.to_datetime(df["published"], errors='coerce')

    if time_window_days is not None:
        today = pd.Timestamp.now().date()
        cutoff_date = today - pd.Timedelta(days=time_window_days)
        # Filter out rows where 'published' is NaT (Not a Time) after coercion
        df = df.dropna(subset=["published"])
        df = df[df["published"].dt.date >= cutoff_date]

    return df.sort_values("citation_count", ascending=False).head(n)


def process_trending_data(
    papers_df_fragment: pd.DataFrame,
    raw_trending_df: pd.DataFrame,
    top_n_display: int
) -> pd.DataFrame:
    """
    Processes raw trending data (typically from db.get_trending_papers)
    by joining with the current papers_df_fragment, calculating like counts,
    and returning the top N papers with tweet details.
    """
    if papers_df_fragment.empty or raw_trending_df.empty:
        return pd.DataFrame()

    # Create dictionaries for all the new data columns
    counts_dict = dict(zip(raw_trending_df["arxiv_code"], raw_trending_df["like_count"]))
    tweet_count_dict = dict(zip(raw_trending_df["arxiv_code"], raw_trending_df["tweet_count"]))
    tweets_dict = dict(zip(raw_trending_df["arxiv_code"], raw_trending_df["tweets"]))

    # Filter the main papers dataframe for codes present in the trending data
    # and create a copy to avoid SettingWithCopyWarning
    trending_papers_intermediate = papers_df_fragment[
        papers_df_fragment["arxiv_code"].isin(counts_dict.keys())
    ].copy()

    if trending_papers_intermediate.empty:
        return pd.DataFrame()

    # Map all the new data to the filtered papers
    trending_papers_intermediate["like_count"] = trending_papers_intermediate["arxiv_code"].map(counts_dict)
    trending_papers_intermediate["tweet_count"] = trending_papers_intermediate["arxiv_code"].map(tweet_count_dict)
    trending_papers_intermediate["tweets"] = trending_papers_intermediate["arxiv_code"].map(tweets_dict)

    # Ensure 'like_count' is numeric, coercing errors and filling NaNs with 0
    trending_papers_intermediate["like_count"] = pd.to_numeric(
        trending_papers_intermediate["like_count"], errors='coerce'
    ).fillna(0).astype(int) # Convert to int after fillna

    # Sort by like_count in descending order
    trending_papers_intermediate.sort_values("like_count", ascending=False, inplace=True)

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
