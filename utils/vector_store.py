import pandas as pd
import os
import re

from langchain.embeddings.huggingface import HuggingFaceInferenceAPIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
from utils.custom_langchain import NewCohereEmbeddings, NewPGVector

import utils.db as db

CONNECTION_STRING = (
    f"postgresql+psycopg2://{db.db_params['user']}:{db.db_params['password']}"
    f"@{db.db_params['host']}:{db.db_params['port']}/{db.db_params['dbname']}"
)


def initialize_retriever(collection_name):
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

    # CustomCohereRerank.update_forward_refs()
    # co = cohere.Client(os.getenv("COHERE_API_KEY"))

    compressor = CohereRerank(
        top_n=10, cohere_api_key=os.getenv("COHERE_API_KEY"), user_agent="llmpedia"
    )
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )
    return compression_retriever


llm_map = {
    "GPT-3.5-Turbo": ChatOpenAI(model_name="gpt-3.5-turbo-1106", temperature=0.1),
    "GPT-4": ChatOpenAI(model_name="gpt-4", temperature=0.1),
    "GPT-4-Turbo": ChatOpenAI(model_name="gpt-4-1106-preview", temperature=0.1),
}


template = """You are the GPT maestro. Use the following pieces of documents to answer the user's question about Large Language Models (LLMs).

==========
{context}
==========

Question: `{question}`

- If the question is unrelated to LLMs reply without referencing the documents.
- Use up to three paragraphs to provide a complete, direct and useful answer. Break down concepts step by step and avoid using complex jargon.
- Be practical and reference any existing libraries or implementations mentioned on the documents.
- Add citations referencing the relevant arxiv_codes (e.g.: use the format `*reference content* (arxiv:1234.5678)`). 
- You do not need to quote or use all the documents presented. Prioritize most recent content and that with most citations.
- Use markdown to organize and structure your response.
- Reply with a subtle old-school mafia-style italo-american accent.

Helpful Answer:"""
rag_prompt_custom = PromptTemplate.from_template(template)


def add_links_to_response(response):
    """Add links to arxiv codes in the response."""

    def repl(match):
        return f"[arxiv:{match.group(1)}](https://llmpedia.streamlit.app/?arxiv_code={match.group(1)})"

    return re.sub(r"arxiv:(\d{4}\.\d{4,5})", repl, response)


def create_rag_context(parent_docs):
    """Create RAG context for LLM, including text excerpts, arxiv_codes,
    year of publication and citation counts."""
    rag_context = ""
    for idx, doc in parent_docs.iterrows():
        arxiv_code = doc["arxiv_code"]
        year = doc["published"]
        citation_count = doc["citation_count"]
        text = "..." + doc["text"] + "..."
        rag_context += (
            f"arxiv:{arxiv_code} ({year}, {citation_count} citations)\n\{text}\n\n"
        )

    return rag_context


def query_llmpedia(question: str, collection_name):
    """Sen API query call to GPT."""
    compression_retriever = initialize_retriever(collection_name)
    child_docs = compression_retriever.invoke(question)

    ## Map to parent chunk (for longer context).
    child_docs = [doc.metadata for doc in child_docs]
    child_ids = [(doc["arxiv_code"], doc["chunk_id"]) for doc in child_docs]
    parent_ids = db.get_arxiv_parent_chunk_ids(child_ids)
    parent_docs = db.get_arxiv_chunks(parent_ids, source="parent")
    parent_docs["published"] = pd.to_datetime(parent_docs["published"]).dt.year
    parent_docs.sort_values(
        by=["published", "citation_count"], ascending=False, inplace=True
    )
    parent_docs.reset_index(drop=True, inplace=True)
    parent_docs = parent_docs.head(5)

    ## Create custom prompt.
    rag_context = create_rag_context(parent_docs)

    llm_chain = LLMChain(
        llm=llm_map["GPT-3.5-Turbo"],
        prompt=rag_prompt_custom,
        verbose=False,
    )
    res = llm_chain.run(context=rag_context, question=question)
    content = add_links_to_response(res)

    return content
