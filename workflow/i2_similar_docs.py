import sys, os
import pandas as pd
from dotenv import load_dotenv
import numpy as np

load_dotenv()
PROJECT_PATH = os.environ.get("PROJECT_PATH")
sys.path.append(PROJECT_PATH)
os.chdir(PROJECT_PATH)

import utils.paper_utils as pu
import utils.db as db
from utils.logging_utils import setup_logger
logger = setup_logger(__name__, "i2_similar_docs.log")

## Embedding configuration.
EMBEDDING_TYPE = "nv"
DOC_TYPE = "recursive_summary"

def find_most_similar_documents(arxiv_code: str, embeddings_map: dict, n: int = 5) -> list:
    """Get most similar documents based on cosine similarity of full embeddings."""
    target_embedding = np.array(embeddings_map[arxiv_code])
    target_embedding = target_embedding.reshape(1, -1)

    ## Process embeddings for similarity comparison.
    other_codes = [code for code in embeddings_map.keys() if code != arxiv_code]
    other_embeddings = np.array([embeddings_map[code] for code in other_codes])

    similarities = pu.euclidean_distances(target_embedding, other_embeddings)
    most_similar_indices = np.argsort(similarities[0])[:n]
    most_similar_docs = [other_codes[i] for i in most_similar_indices]
    return most_similar_docs

def main():
    """Main function."""
    logger.info("Starting similar document finding process")
    
    arxiv_df = db.load_arxiv()
    arxiv_codes = arxiv_df.index.tolist()
    logger.info(f"Found {len(arxiv_codes)} papers to process")

    embeddings_map = db.load_embeddings(
        arxiv_codes=arxiv_codes,
        doc_type=DOC_TYPE,
        embedding_type=EMBEDDING_TYPE,
    )
    logger.info(f"Loaded {len(embeddings_map)} embeddings")

    df = pd.DataFrame(index=list(embeddings_map.keys()))
    df["similar_docs"] = df.index.map(lambda x: find_most_similar_documents(x, embeddings_map, 10))
    df.reset_index(inplace=True)
    df.rename(columns={"index": "arxiv_code"}, inplace=True)
    df["similar_docs"] = df["similar_docs"].apply(db.list_to_pg_array)
    
    logger.info("Uploading similar documents to database")
    db.upload_df_to_db(
        df[["arxiv_code", "similar_docs"]],
        "similar_documents",
        pu.db_params,
        if_exists="replace",
    )
    logger.info("Similar document finding process completed")

if __name__ == "__main__":
    main()
