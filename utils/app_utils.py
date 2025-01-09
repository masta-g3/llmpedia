from pydantic import BaseModel
from typing import List, Tuple
import pandas as pd
import numpy as np
import datetime
import json
import os, re
import boto3

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings.huggingface import HuggingFaceInferenceAPIEmbeddings
from langchain_cohere import CohereEmbeddings
from langchain.prompts.chat import ChatPromptTemplate
from langchain.chains import LLMChain

from utils.custom_langchain import NewCohereEmbeddings, NewPGVector
from utils.models import llm_map
from utils.instruct import run_instructor_query
import utils.pydantic_objects as po
import utils.prompts as ps
import utils.db as db

CONNECTION_STRING = (
    f"postgresql+psycopg2://{db.db_params['user']}:{db.db_params['password']}"
    f"@{db.db_params['host']}:{db.db_params['port']}/{db.db_params['dbname']}"
)

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
        return f"[arxiv:{match.group(1)}](https://llmpedia.streamlit.app/?arxiv_code={match.group(1)})"

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


def question_to_query(question: str, model: str = "GPT-3.5-Turbo"):
    """Convert notes to narrative via LLMChain."""
    query_prompt = ChatPromptTemplate.from_messages(
        [("system", ps.QUESTION_TO_QUERY_PROMPT)]
    )
    query_chain = LLMChain(llm=llm_map[model], prompt=query_prompt)
    query = query_chain.invoke(dict(question=question))["text"]
    query = "[ " + query
    return query


def query_llmpedia(question: str, collection_name: str, model: str = "GPT-3.5-Turbo"):
    """Query LLMpedia via LLMChain."""
    rag_prompt_custom = ChatPromptTemplate.from_messages(
        [
            ("system", ps.VS_SYSTEM_TEMPLATE),
            ("human", "{question}"),
        ]
    )

    rag_llm_chain = LLMChain(
        llm=llm_map[model], prompt=rag_prompt_custom, verbose=False
    )
    compression_retriever = initialize_retriever(collection_name)
    queries = question_to_query(question, model=model)
    queries_list = json.loads(queries)
    all_parent_docs = []
    for query in queries_list:
        child_docs = compression_retriever.invoke(query)

        ## Map to parent chunk (for longer context).
        child_docs = [doc.metadata for doc in child_docs]
        child_ids = [(doc["arxiv_code"], doc["chunk_id"]) for doc in child_docs]
        parent_ids = db.get_arxiv_parent_chunk_ids(child_ids)
        parent_docs = db.get_arxiv_chunks(parent_ids, source="parent")
        if len(parent_docs) == 0:
            continue
        parent_docs["published"] = pd.to_datetime(parent_docs["published"]).dt.year
        parent_docs.sort_values(
            by=["published", "citation_count"], ascending=False, inplace=True
        )
        parent_docs.reset_index(drop=True, inplace=True)
        parent_docs["subject"] = query
        parent_docs = parent_docs.head(3)
        all_parent_docs.append(parent_docs)

    all_parent_docs = pd.concat(all_parent_docs, ignore_index=True)
    if len(all_parent_docs) == 0:
        return "No results found."

    ## Create custom prompt.
    all_parent_docs.drop_duplicates(subset=["text"], inplace=True)
    rag_context = create_rag_context(all_parent_docs)
    res = rag_llm_chain.invoke(dict(context=rag_context, question=question))["text"]
    if "Response" in res:
        res_response = res.split("Response\n")[1].split("###")[0].strip()
        content = add_links_to_text_blob(res_response)
    else:
        content = res[:]

    return content


######################
## VECTOR STORE NEW ##
######################


class Document(BaseModel):
    arxiv_code: str
    title: str
    published_date: datetime.datetime
    citations: int
    abstract: str
    distance: float
    notes: str = None


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


def decide_query_action(user_question: str) -> po.QueryDecision:
    """Decide the query action based on the user question."""
    system_message = "Please analyze the following user query and answer the question."
    user_message = ps.create_decision_user_prompt(user_question)
    response = run_instructor_query(
        system_message, user_message, po.QueryDecision, process_id="decide_query_action"
    )
    return response


def convert_query_to_vector(query: str, model_name: str):
    if "embed-english" in model_name:
        embeddings = CohereEmbeddings(
            cohere_api_key=os.getenv("COHERE_API_KEY"), model=model_name
        )
    else:
        embeddings = HuggingFaceEmbeddings(model_name=model_name)

    return embeddings.embed_query(query)


def generate_query_object(user_question: str, llm_model: str):
    system_message = ps.VS_QUERY_SYSTEM_PROMPT
    user_message = ps.create_query_user_prompt(user_question)
    query_obj = run_instructor_query(
        system_message,
        user_message,
        po.SearchCriteria,
        llm_model=llm_model,
        temperature=0.5,
        process_id="generate_query_object",
    )
    return query_obj


def format_query_condition(field_name: str, template: str, value: str):
    if isinstance(value, list) and "semantic_search_queries" in field_name:
        distance_scores = []
        for query in value:
            vector = convert_query_to_vector(query, "embed-english-v3.0")
            vector_str = ", ".join(map(str, vector))
            condition = f"l.embedding <-> ARRAY[{vector_str}]::vector "
            distance_scores.append(condition)
        if distance_scores:
            min_distance = f"LEAST({', '.join(distance_scores)})"
            return (
                f"({' OR '.join([c + ' < 1' for c in distance_scores])})",
                min_distance,
            )
        else:
            return "AND TRUE", "0 as min_distance"
    elif isinstance(value, list):
        value_str = "', '".join(value)
        return template % value_str, "0 as min_distance"
    else:
        return template % value, "0 as min_distance"


def generate_query(criteria: po.SearchCriteria, config: dict) -> str:
    query_parts = [
        "SELECT a.arxiv_code, a.title,  a.published, s.citation_count, a.summary AS abstract, ",
        "FROM arxiv_details a, semantic_details s, topics t, langchain_pg_embedding l ",
        "WHERE a.arxiv_code = s.arxiv_code "
        "AND a.arxiv_code = t.arxiv_code "
        "AND l.collection_id = '7bf1c691-da2b-4a2b-81b3-e7dd165bfa39' "
        "AND a.arxiv_code = l.cmetadata ->> 'arxiv_code' ",
    ]
    extra_selects = []

    for field, value in criteria.model_dump(exclude_none=True).items():
        if field in config:
            condition_str, max_similarity = format_query_condition(
                field, config[field], value
            )
            query_parts.append(f"AND {condition_str}")
            extra_selects.append(max_similarity)

    if len(extra_selects) > 1:
        extra_selects = list(filter(lambda x: x != "0 as min_distance", extra_selects))

    query_parts[0] += ", ".join(extra_selects)
    query_parts.append("ORDER BY 6 ASC ")

    return "\n".join(query_parts)


def rerank_documents_new(
    user_question: str, documents: list, llm_model="gpt-4o", temperature=0.2
) -> po.RerankedDocuments:
    system_message = "You are an expert system that can identify and select relevant arxiv papers that can be used to answer a user query."
    import streamlit as st

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
    user_question: str, documents: list[Document], response_length: str, llm_model: str
):
    system_message = "You are an AI academic focused on Large Language Models. Please answer the user query leveraging the information provided in the context."
    user_message = ps.create_resolve_user_prompt(
        user_question, documents, response_length
    )
    response = run_instructor_query(
        system_message=system_message,
        user_message=user_message,
        model=None,
        llm_model=llm_model,
        temperature=0.2,
        process_id="resolve_query",
    )
    return response


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


def query_llmpedia_new(
    user_question: str,
    response_length: str,
    query_llm_model: str,
    rerank_llm_model: str,
    response_llm_model: str,
) -> Tuple[str, List[str], List[str]]:
    """Extended RAG workflow to answer a user query with the LLMpedia."""
    action = decide_query_action(user_question)

    if action.llm_query:
        ## Create query.
        query_obj = generate_query_object(
            user_question=user_question, llm_model=query_llm_model
        )

        ## Fetch results.
        query_obj.topic_categories = None
        sql = generate_query(query_obj, query_config)
        documents = db.execute_query(sql, limit=20)
        if len(documents) == 0:
            return "Sorry, I don't know about that.", [], []
        documents = [
            Document(**dict(zip(Document.__fields__.keys(), d))) for d in documents
        ]

        ## Rerank.
        reranked_documents = rerank_documents_new(
            user_question, documents, llm_model=rerank_llm_model
        )
        filtered_document_ids = [
            int(d.document_id) for d in reranked_documents.documents if d.selected
        ]
        filtered_documents = [
            d for i, d in enumerate(documents) if i in filtered_document_ids
        ]
        if len(filtered_documents) == 0:
            return "Sorry, I don't know about that.", [], []

        ## Resolve.
        answer = resolve_query(
            user_question,
            filtered_documents,
            response_length,
            llm_model=response_llm_model,
        )
        answer_augment = add_links_to_text_blob(answer)
        referenced_arxiv_codes = extract_arxiv_codes(answer_augment)
        filtered_arxiv_codes = [d.arxiv_code for d in filtered_documents]
        filtered_arxiv_codes = [
            d for d in filtered_arxiv_codes if d not in referenced_arxiv_codes
        ]
        return answer_augment, referenced_arxiv_codes, filtered_arxiv_codes

    else:
        answer = resolve_query_other(user_question)
        return answer, [], []


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
    s3 = boto3.client('s3')
    
    try:
        # First check if the paper directory exists in S3
        response = s3.list_objects_v2(
            Bucket='arxiv-md',
            Prefix=f'{arxiv_code}/'
        )
        
        if 'Contents' not in response:
            return "Paper content not available yet. Check back soon!", False
            
        # Get the paper.md content
        response = s3.get_object(
            Bucket='arxiv-md',
            Key=f'{arxiv_code}/paper.md'
        )
        markdown_content = response.get('Body').read().decode('utf-8')
        
        # Process image references to point to S3
        # First, handle relative paths
        markdown_content = re.sub(
            r'!\[(.*?)\]\((?!http)(.*?)\)',
            lambda m: f'![{m.group(1)}](https://arxiv-md.s3.amazonaws.com/{arxiv_code}/{m.group(2)})',
            markdown_content
        )
        
        # Then, handle paths that might start with the arxiv code
        markdown_content = re.sub(
            f'!\[(.*?)\]\({arxiv_code}/(.*?)\)',
            lambda m: f'![{m.group(1)}](https://arxiv-md.s3.amazonaws.com/{arxiv_code}/{m.group(2)})',
            markdown_content
        )
        
        return markdown_content, True
        
    except Exception as e:
        return f"Error loading paper content: {str(e)}", False
