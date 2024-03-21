import pandas as pd
import json
import os, re

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
from langchain_community.embeddings.huggingface import HuggingFaceInferenceAPIEmbeddings
from langchain.prompts.chat import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

from utils.custom_langchain import NewCohereEmbeddings, NewPGVector
from utils.models import llm_map
import utils.prompts as ps
import utils.db as db

CONNECTION_STRING = (
    f"postgresql+psycopg2://{db.db_params['user']}:{db.db_params['password']}"
    f"@{db.db_params['host']}:{db.db_params['port']}/{db.db_params['dbname']}"
)


def parse_weekly_report(report_md: str):
    """Extract section of weekly report into dict."""
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


def get_img_link_for_blob(text_blob: str):
    """Identify `arxiv_code:XXXX.XXXXX` from a text blob, and generate a Markdown link to its img."""
    arxiv_code = re.findall(r"arxiv:(\d{4}\.\d{4,5})", text_blob)
    if len(arxiv_code) == 0:
        return None
    arxiv_code = arxiv_code[0]
    return f"https://llmpedia.s3.amazonaws.com/{arxiv_code}.png"


def numbered_to_bullet_list(list_str: str):
    """Convert a numbered list to a bullet list."""
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


def create_rag_context(parent_docs):
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


def question_to_query(question, model="GPT-3.5-Turbo"):
    """Convert notes to narrative via LLMChain."""
    query_prompt = ChatPromptTemplate.from_messages(
        [("system", ps.QUESTION_TO_QUERY_PROMPT)]
    )
    query_chain = LLMChain(llm=llm_map[model], prompt=query_prompt)
    query = query_chain.invoke(dict(question=question))["text"]
    query = "[ " + query
    return query


def query_llmpedia(question: str, collection_name, model="GPT-3.5-Turbo"):
    """Query LLMpedia via LLMChain."""
    rag_prompt_custom = ChatPromptTemplate.from_messages(
        [
            ("system", ps.VS_SYSYEM_TEMPLATE),
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
