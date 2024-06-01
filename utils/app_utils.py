from pydantic import BaseModel
import pandas as pd
import datetime
import json
import os, re

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
    arxiv_code = re.findall(r"arxiv:(\d{4}\.\d{4,5})", text_blob)
    if len(arxiv_code) == 0:
        return None
    arxiv_code = arxiv_code[0]
    return f"https://llmpedia.s3.amazonaws.com/{arxiv_code}.png"


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


def interrogate_paper(question: str, arxiv_code: str) -> str:
    """Ask a question about a paper."""
    context = db.get_extended_notes(arxiv_code, expected_tokens=8000)
    system_message = "Read carefully the whitepaper, reason about the user question, and provide a comprehensive, git helpful and truthful response. Be direct and concise, using layman's language that is easy to understand. Avoid filler content, and reply with your answer in a single short sentence or paragraph and nothing else (no preambles, greetings, etc.)."
    user_message = ps.create_interrogate_user_prompt(question, context)
    response = run_instructor_query(system_message, user_message, None, llm_model="gpt-4o")
    response = response.replace("<response>", "").replace("</response>", "")
    return response


def decide_query_action(user_question: str) -> ps.QueryDecision:
    """Decide the query action based on the user question."""
    system_message = "Please analyze the following user query and answer the question."
    user_message = ps.create_decision_user_prompt(user_question)
    response = run_instructor_query(system_message, user_message, ps.QueryDecision)
    return response


def convert_query_to_vector(query: str, model_name: str):
    if "embed-english" in model_name:
        embeddings = CohereEmbeddings(
            cohere_api_key=os.getenv("COHERE_API_KEY"), model=model_name
        )
    else:
        embeddings = HuggingFaceEmbeddings(model_name=model_name)

    return embeddings.embed_query(query)


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


def generate_query(criteria: ps.SearchCriteria, config: dict) -> str:
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


def rerank_documents_new(user_question: str, documents: list) -> ps.RerankedDocuments:
    system_message = "You are an expert system that can identify and select relevant arxiv papers that can be used to answer a user query."
    import streamlit as st
    rerank_msg = ps.create_rerank_user_prompt(user_question, documents)
    response = run_instructor_query(system_message, rerank_msg, ps.RerankedDocuments)
    return response


def resolve_query(user_question: str, documents: list[Document], response_length: str):
    system_message = "You are an AI academic focused on Large Language Models. Please answer the user query leveraging the information provided in the context."
    user_message = ps.create_resolve_user_prompt(user_question, documents, response_length)
    response = run_instructor_query(system_message, user_message, None, llm_model="claude-3-sonnet-20240229")
    return response


def resolve_query_other(user_question: str) -> ps.QueryDecision:
    """Decide the query action based on the user question."""
    system_message = "You are the GPT Maestro, maintainer of the LLMpedia, a web-based Large Language Model encyclopedia. You received the following unrelated comment from a user via our chat based system. Please respond to it in a friendly, slightly-sarcastic, serious and very concise (less than 20 words) manner."
    user_message = f"{user_question}"
    response = run_instructor_query(system_message, user_message, None)
    return response


def query_llmpedia_new(user_question: str, response_length: str = "Normal") -> tuple:
    """Extended workflow to query LLMpedia."""
    ## Decide action.
    action = decide_query_action(user_question)

    if action.llm_query:
        ## Create query.
        query_obj = run_instructor_query(
            ps.VS_QUERY_SYSTEM_PROMPT,
            ps.create_query_user_prompt(user_question),
            ps.SearchCriteria,
        )
        print(query_obj)
        ## Fetch results.
        sql = generate_query(query_obj, query_config)
        documents = db.execute_query(sql, limit=20)
        if len(documents) == 0:
            return "Sorry, I don't know about that.", []
        documents = [
            Document(**dict(zip(Document.__fields__.keys(), d))) for d in documents
        ]
        ## Rerank.
        reranked_documents = rerank_documents_new(user_question, documents)
        print(reranked_documents)
        filtered_documents = {
            k: v for k, v in reranked_documents.documents.items() if v.selected
        }
        filtered_documents = [d for d in documents if d.title in filtered_documents]
        if len(filtered_documents) == 0:
            return "Sorry, I don't know about that.", []
        ## Resolve.
        answer = resolve_query(user_question, filtered_documents, response_length)
        answer_augment = add_links_to_text_blob(answer)
        arxiv_codes = extract_arxiv_codes(answer_augment)
        return answer_augment, arxiv_codes

    else:
        answer = resolve_query_other(user_question)
        return answer, []
