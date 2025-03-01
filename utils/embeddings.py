"""Functions for handling embeddings and vector operations."""
import os
from typing import List
from langchain_cohere import CohereEmbeddings
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
import voyageai
from sentence_transformers import SentenceTransformer


def convert_query_to_vector(query: str, model_name: str) -> List[float]:
    """Convert a text query into a vector using the specified embedding model."""
    if "embed-english" in model_name:
        embeddings = CohereEmbeddings(
            cohere_api_key=os.getenv("COHERE_API_KEY"), model=model_name
        )
    elif model_name == "voyage":
        client = voyageai.Client()
        return client.embed(
            [query], model="voyage-3-large", input_type="document"
        ).embeddings[0]
    elif model_name == "nvidia/NV-Embed-v2":
        model = SentenceTransformer(model_name, trust_remote_code=True)
        model.max_seq_length = 32768
        model.tokenizer.padding_side = "right"
        query_prefix = "Instruct: Identify the topic or theme of the following AI & Large Language Model document\nQuery: "
        return model.encode(
            query + model.tokenizer.eos_token,
            prompt=query_prefix,
            normalize_embeddings=True,
        ).tolist()
    else:
        embeddings = HuggingFaceEmbeddings(model_name=model_name)

    return embeddings.embed_query(query) 