import json
import os
import streamlit as st
from pydantic import BaseModel
import cohere

from langchain.embeddings.huggingface import HuggingFaceInferenceAPIEmbeddings
from langchain.vectorstores.pgvector import PGVector
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank

class CustomCohereRerank(CohereRerank):
    class Config(BaseModel.Config):
        arbitrary_types_allowed = True


db_params = {**st.secrets["postgres"]}

CONNECTION_STRING = (
    f"postgresql+psycopg2://{db_params['user']}:{db_params['password']}"
    f"@{db_params['host']}:{db_params['port']}/{db_params['dbname']}"
)
COLLECTION_NAME = "arxiv_vectors"
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=HUGGINGFACE_API_KEY,
    model_name="thenlper/gte-large"
)

store = PGVector(
    collection_name=COLLECTION_NAME,
    connection_string=CONNECTION_STRING,
    embedding_function=embeddings,
)

retriever = store.as_retriever(search_type="mmr", search_kwargs={"k": 20})

CustomCohereRerank.update_forward_refs()
key = os.getenv("COHERE_API_KEY")
co = cohere.Client(key)

compressor = CustomCohereRerank(client=co, top_n=5)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.1)

template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use as many sentences as needed to provide a thorough, complete but concise answer. If possible break down concepts and tackle them step by step.
Be practical and reference any existing libraries or implementations mentioned on the documents if possible.
When providing your answer add citations referencing the relevant arxiv_codes (e.g.: *reference content* (arxiv:1234.5678)).
{context}
Question: {question}
Helpful Answer:"""
rag_prompt_custom = PromptTemplate.from_template(template)

rag_chain = (
    {"context": compression_retriever, "question": RunnablePassthrough()}
    | rag_prompt_custom
    | llm
)


def query_llmpedia(question: str):
    """ Sen API query call to GPT."""
    res = rag_chain.invoke(question)
    content = json.loads(res.json())["content"]
    return content