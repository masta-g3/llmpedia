import json
import os
import streamlit as st
from pydantic import BaseModel
import cohere
import re

from langchain.embeddings.huggingface import HuggingFaceInferenceAPIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank

from utils.custom_langchain import NewCohereEmbeddings, NewPGVector

db_params = {**st.secrets["postgres"]}

CONNECTION_STRING = (
    f"postgresql+psycopg2://{db_params['user']}:{db_params['password']}"
    f"@{db_params['host']}:{db_params['port']}/{db_params['dbname']}"
)


def initialize_collection(collection_name):
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

    compressor = CohereRerank(top_n=5, cohere_api_key=os.getenv("COHERE_API_KEY"), user_agent="llmpedia")
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )
    return compression_retriever


llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.1)

template = """Use the following pieces of documents to answer the user's question. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use up to 500 words to provide a complete but concise answer. If possible break down concepts step by step.
Be practical and reference any existing libraries or implementations mentioned on the documents.
When providing your answer add citations referencing the relevant arxiv_codes (e.g.: *reference content* (arxiv:1234.5678)).
Use markdown format to organize and structure your response.
{context}
Question: {question}
Helpful Answer:"""
rag_prompt_custom = PromptTemplate.from_template(template)


def add_links_to_response(response):
    """ Add links to arxiv codes in the response."""
    def repl(match):
        return f"[arxiv:{match.group(1)}](https://llmpedia.streamlit.app/?arxiv_code={match.group(1)})"
    return re.sub(r"arxiv:(\d{4}\.\d{4,5})", repl, response)


def query_llmpedia(question: str, collection_name):
    """Sen API query call to GPT."""
    compression_retriever = initialize_collection(collection_name)
    rag_chain = (
        {"context": compression_retriever, "question": RunnablePassthrough()}
        | rag_prompt_custom
        | llm
    )
    res = rag_chain.invoke(question)
    content = json.loads(res.json())["content"]
    content = add_links_to_response(content)

    return content
