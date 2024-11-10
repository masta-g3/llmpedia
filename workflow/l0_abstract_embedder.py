from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.pgvector import PGVector
from langchain_community.docstore.document import Document
from langchain_cohere import CohereEmbeddings

from sqlalchemy.exc import IntegrityError, OperationalError
from tqdm import tqdm
import pandas as pd
import os, sys
import time
import json

from dotenv import load_dotenv

load_dotenv()
PROJECT_PATH = os.getenv('PROJECT_PATH', '/app')
sys.path.append(PROJECT_PATH)

import utils.paper_utils as pu
import utils.db as db
from utils.logging_utils import setup_logger

# Set up logging
logger = setup_logger(__name__, "l0_abstract_embedder.log")

MAX_RETRIES = 3
RETRY_DELAY = 2
collection_name = "arxiv_abstracts"
model_name = "embed-english-v3.0"

def main():
    """Create embeddings for all arxiv chunks and upload them to DB."""
    logger.info("Starting abstract embedding process")
    CONNECTION_STRING = (
        f"postgresql+psycopg2://{pu.db_params['user']}:{pu.db_params['password']}"
        f"@{pu.db_params['host']}:{pu.db_params['port']}/{pu.db_params['dbname']}"
    )

    if "embed-english" in model_name:
        logger.info(f"Using Cohere embeddings with model: {model_name}")
        embeddings = CohereEmbeddings(
            cohere_api_key=os.getenv("COHERE_API_KEY"), model=model_name
        )
    else:
        logger.info(f"Using HuggingFace embeddings with model: {model_name}")
        embeddings = HuggingFaceEmbeddings(model_name=model_name)

    store = PGVector(
        collection_name=collection_name,
        connection_string=CONNECTION_STRING,
        embedding_function=embeddings,
        use_jsonb=True,
    )

    arxiv_codes = db.get_arxiv_id_embeddings(collection_name)
    local_codes = db.get_arxiv_id_list(pu.db_params, "recursive_summaries")
    processing_codes = list(set(local_codes) - set(arxiv_codes))
    processing_codes = sorted(processing_codes)[::-1]

    logger.info(f"Found {len(processing_codes)} papers to process for embeddings")

    for arxiv_code in processing_codes:
        summary = db.get_recursive_summary(arxiv_code)
        if summary is None:
            logger.warning(f"Could not find '{arxiv_code}' in the meta-database. Skipping...")
            continue
        metadata = {"arxiv_code": arxiv_code, "model": model_name}
        for attempt in range(MAX_RETRIES):
            try:
                store.add_documents(
                    [Document(page_content=summary, metadata=metadata)]
                )
                logger.info(f"Successfully added embedding for {arxiv_code}")
                break
            except IntegrityError as e:
                logger.warning(f"IntegrityError for {arxiv_code}. Skipping...")
                break
            except OperationalError as e:
                logger.error(f"Encountered error on paper {metadata['arxiv_code']}: {e}")
                if attempt < MAX_RETRIES - 1:
                    logger.info(f"Retrying in {RETRY_DELAY} seconds...")
                    time.sleep(RETRY_DELAY)
                else:
                    logger.error(f"Failed to process {arxiv_code} after {MAX_RETRIES} attempts")
            continue

    logger.info("Abstract embedding process completed")

if __name__ == "__main__":
    main()
