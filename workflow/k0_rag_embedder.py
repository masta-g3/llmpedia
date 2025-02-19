from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.pgvector import PGVector
from langchain_community.docstore.document import Document
from langchain_cohere import CohereEmbeddings

from sqlalchemy.exc import IntegrityError, OperationalError
from tqdm import tqdm
from dotenv import load_dotenv
import pandas as pd
import os, sys
import time
import json

from dotenv import load_dotenv


load_dotenv()
PROJECT_PATH = os.getenv('PROJECT_PATH', '/app')
sys.path.append(PROJECT_PATH)

chunk_path = os.path.join(os.environ.get("PROJECT_PATH"), "data", "arxiv_chunks")

import utils.paper_utils as pu
import utils.db.db as db
from utils.logging_utils import setup_logger

logger = setup_logger(__name__, "k0_rag_embedder.log")

MAX_RETRIES = 3
RETRY_DELAY = 2

COLLECTION_NAMES = [
    "arxiv_vectors",
    "arxiv_vectors_cv3",
]

MODEL_NAME_MAP = {
    "arxiv_vectors": "thenlper/gte-large",
    "arxiv_vectors_cv3": "embed-english-v3.0",
}


def main():
    """Create embeddings for all arxiv chunks and upload them to DB."""
    CONNECTION_STRING = (
        f"postgresql+psycopg2://{pu.db_params['user']}:{pu.db_params['password']}"
        f"@{pu.db_params['host']}:{pu.db_params['port']}/{pu.db_params['dbname']}"
    )
    for COLLECTION_NAME in COLLECTION_NAMES:
        logger.info(f"Processing {COLLECTION_NAME}...")
        model_name = MODEL_NAME_MAP[COLLECTION_NAME]

        if "embed-english" in model_name:
            embeddings = CohereEmbeddings(
                cohere_api_key=os.getenv("COHERE_API_KEY"), model=model_name
            )
        else:
            embeddings = HuggingFaceEmbeddings(model_name=model_name)

        store = PGVector(
            collection_name=COLLECTION_NAME,
            connection_string=CONNECTION_STRING,
            embedding_function=embeddings,
            use_jsonb=True,
        )

        arxiv_codes = db.get_arxiv_id_embeddings(COLLECTION_NAME)
        local_codes = os.listdir(chunk_path)
        local_codes = [code.replace(".json", "") for code in local_codes]
        processing_codes = list(set(local_codes) - set(arxiv_codes))

        for idx, arxiv_code in enumerate(processing_codes):
            chunks_fname = os.path.join(chunk_path, f"{arxiv_code}.json")
            chunks_json = json.load(open(chunks_fname, "r"))
            chunks_df = pd.DataFrame(chunks_json)
            add_count = 0
            for idx, row in chunks_df.iterrows():
                chunk_text = row["text"]
                metadata = row.drop("text").to_dict()
                metadata["model"] = model_name
                for attempt in range(MAX_RETRIES):
                    try:
                        store.add_documents(
                            [Document(page_content=chunk_text, metadata=metadata)]
                        )
                        add_count += 1
                        break
                    except IntegrityError as e:
                        continue
                    except OperationalError as e:
                        logger.error(
                            f"Encountered error on paper {metadata['arxiv_code']}: {e}"
                        )
                        if attempt < MAX_RETRIES - 1:
                            time.sleep(RETRY_DELAY)
                    continue
            # if metadata:
            #     print(f"Added {add_count} vectors for {metadata['arxiv_code']}.")

        logger.info("Process complete.")


if __name__ == "__main__":
    main()
