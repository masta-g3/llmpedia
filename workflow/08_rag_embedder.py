from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores.pgvector import PGVector
from langchain.docstore.document import Document
from sqlalchemy.exc import IntegrityError, OperationalError
from tqdm import tqdm
from dotenv import load_dotenv
import pandas as pd
import os, sys
import time
import json

from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.environ.get("PROJECT_PATH"))
chunk_path = os.path.join(os.environ.get("PROJECT_PATH"), "data", "arxiv_chunks")

import utils.paper_utils as pu


def main():
    """Create embeddings for all arxiv chunks and upload them to DB."""
    CONNECTION_STRING = (
        f"postgresql+psycopg2://{pu.db_params['user']}:{pu.db_params['password']}"
        f"@{pu.db_params['host']}:{pu.db_params['port']}/{pu.db_params['dbname']}"
    )
    COLLECTION_NAME = "arxiv_vectors"

    ## Representation and storage.
    embeddings = HuggingFaceEmbeddings(model_name="thenlper/gte-large")
    store = PGVector(
        collection_name=COLLECTION_NAME,
        connection_string=CONNECTION_STRING,
        embedding_function=embeddings,
    )

    arxiv_codes = pu.get_arxiv_id_embeddings(pu.db_params)
    local_codes = os.listdir(chunk_path)
    local_codes = [code.replace(".json", "") for code in local_codes]
    processing_codes = list(set(local_codes) - set(arxiv_codes)) + ["2108.13349"]

    for arxiv_code in tqdm(processing_codes):
        chunks_fname = os.path.join(chunk_path, f"{arxiv_code}.json")
        chunks_json = json.load(open(chunks_fname, "r"))
        chunks_df = pd.DataFrame(chunks_json)
        add_count = 0
        metadata = None
        for idx, row in chunks_df.iterrows():
            chunk_text = row["text"]
            metadata = row.drop("text").to_dict()

            MAX_RETRIES = 3
            RETRY_DELAY = 2

            for attempt in range(MAX_RETRIES):
                try:
                    store.add_documents(
                        [Document(page_content=chunk_text, metadata=metadata)]
                    )
                    add_count += 1
                    break
                except IntegrityError:
                    continue
                except OperationalError as e:
                    print(f"Encountered error on paper {metadata['arxiv_code']}: {e}")
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(RETRY_DELAY)
                continue
        # if metadata:
        #     print(f"Added {add_count} vectors for {metadata['arxiv_code']}.")

    print("Process complete.")


if __name__ == "__main__":
    main()
