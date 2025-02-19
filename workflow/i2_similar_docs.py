import sys, os
import pandas as pd
from dotenv import load_dotenv
import numpy as np
from sklearn.neighbors import NearestNeighbors

load_dotenv()
PROJECT_PATH = os.environ.get("PROJECT_PATH")
sys.path.append(PROJECT_PATH)
os.chdir(PROJECT_PATH)

import utils.paper_utils as pu
import utils.db.db_utils as db_utils
import utils.db.paper_db as paper_db
import utils.db.embedding_db as embedding_db
from utils.logging_utils import setup_logger
logger = setup_logger(__name__, "i2_similar_docs.log")

## Embedding configuration.
EMBEDDING_TYPE = "nv"
DOC_TYPE = "recursive_summary"

def main():
    """Main function."""
    logger.info("Starting similar document finding process")
    
    arxiv_df = paper_db.load_arxiv()
    arxiv_codes = arxiv_df.index.tolist()
    logger.info(f"Found {len(arxiv_codes)} papers to process")

    embeddings_map = embedding_db.load_embeddings(
        arxiv_codes=arxiv_codes,
        doc_type=DOC_TYPE,
        embedding_type=EMBEDDING_TYPE,
    )
    logger.info(f"Loaded {len(embeddings_map)} embeddings")
    
    ## Build a list of document IDs and an embeddings matrix.
    doc_ids: list[str] = list(embeddings_map.keys())
    embeddings: np.ndarray = np.array([embeddings_map[doc_id] for doc_id in doc_ids])

    ## Use NearestNeighbors with one extra neighbor to skip the self-match.
    nbrs: NearestNeighbors = NearestNeighbors(n_neighbors=11, metric="euclidean")
    nbrs.fit(embeddings)
    distances, indices = nbrs.kneighbors(embeddings)

    ## Build list of similar document IDs for each paper.
    similar_docs_list: list[list[str]] = []
    for i, neighbor_indices in enumerate(indices):
        ## Skip first neighbor (it is self).
        similar_docs = [doc_ids[j] for j in neighbor_indices[1:]]
        similar_docs_list.append(similar_docs)

    df = pd.DataFrame({"arxiv_code": doc_ids, "similar_docs": similar_docs_list})
    
    df["similar_docs"] = df["similar_docs"].apply(db_utils.list_to_pg_array)
    
    logger.info("Uploading similar documents to database")
    db_utils.upload_dataframe(
        df[["arxiv_code", "similar_docs"]],
        "similar_documents",
        if_exists="replace",
    )
    logger.info("Similar document finding process completed")

if __name__ == "__main__":
    main()
