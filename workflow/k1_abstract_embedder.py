from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import CohereEmbeddings
from langchain_community.vectorstores.pgvector import PGVector
from langchain_community.docstore.document import Document

from sqlalchemy.exc import IntegrityError, OperationalError
from tqdm import tqdm
import pandas as pd
import os, sys
import time
import json

from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.environ.get("PROJECT_PATH"))

import utils.paper_utils as pu
import utils.db as db

MAX_RETRIES = 3
RETRY_DELAY = 2
collection_name = "arxiv_abstracts"
model_name = "embed-english-v3.0"


def main():
    """Create embeddings for all arxiv chunks and upload them to DB."""
    CONNECTION_STRING = (
        f"postgresql+psycopg2://{pu.db_params['user']}:{pu.db_params['password']}"
        f"@{pu.db_params['host']}:{pu.db_params['port']}/{pu.db_params['dbname']}"
    )

    if "embed-english" in model_name:
        embeddings = CohereEmbeddings(
            cohere_api_key=os.getenv("COHERE_API_KEY"), model=model_name
        )
    else:
        embeddings = HuggingFaceEmbeddings(model_name=model_name)

    store = PGVector(
        collection_name=collection_name,
        connection_string=CONNECTION_STRING,
        embedding_function=embeddings,
    )

    arxiv_codes = db.get_arxiv_id_embeddings(collection_name)
    local_codes = db.get_arxiv_id_list(pu.db_params, "arxiv_details")
    processing_codes = list(set(local_codes) - set(arxiv_codes))
    processing_codes = sorted(processing_codes)[::-1]

    for arxiv_code in tqdm(processing_codes):
        summary = db.get_recursive_summary(arxiv_code)
        if summary is None:
            print(f"Could not find '{arxiv_code}' in the meta-database. Skipping...")
            continue
        metadata = {"arxiv_code": arxiv_code, "model": model_name}
        for attempt in range(MAX_RETRIES):
            try:
                store.add_documents(
                    [Document(page_content=summary, metadata=metadata)]
                )
                break
            except IntegrityError as e:
                continue
            except OperationalError as e:
                print(
                    f"Encountered error on paper {metadata['arxiv_code']}: {e}"
                )
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
            continue

    print("Process complete.")


if __name__ == "__main__":
    main()
