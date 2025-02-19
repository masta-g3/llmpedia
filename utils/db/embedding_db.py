"""Database operations for embedding-related functionality."""

from typing import List, Dict
import logging
from datetime import datetime

from .db_utils import execute_read_query, execute_write_query

## Constants for embedding dimensions
EMBEDDING_DIMENSIONS = {
    "gte": 1024,
    "nv": 4096,
    "voyage": 1024
}

def store_embeddings_batch(
    arxiv_codes: List[str],
    doc_type: str,
    embedding_type: str,
    embeddings: List[List[float]],
) -> bool:
    """Store multiple document embeddings in the appropriate arxiv_embeddings table based on dimension."""
    try:
        dimension = EMBEDDING_DIMENSIONS[embedding_type]
        query = f"""
            INSERT INTO arxiv_embeddings_{dimension} (arxiv_code, doc_type, embedding_type, embedding, tstp)
            VALUES (:arxiv_code, :doc_type, :embedding_type, :embedding, :tstp)
            ON CONFLICT (arxiv_code, doc_type, embedding_type) 
            DO UPDATE SET embedding = EXCLUDED.embedding, tstp = EXCLUDED.tstp
        """

        now = datetime.now()
        params = [
            {
                "arxiv_code": code,
                "doc_type": doc_type,
                "embedding_type": embedding_type,
                "embedding": str(emb),  # Convert embedding list to string
                "tstp": now,
            }
            for code, emb in zip(arxiv_codes, embeddings)
        ]
        
        return execute_write_query(query, params)
    except Exception as e:
        logging.error(f"Error storing embeddings batch: {str(e)}")
        return False

def load_embeddings(
    arxiv_codes: List[str],
    doc_type: str,
    embedding_type: str,
) -> Dict[str, List[float]]:
    """Load embeddings for specified documents from the database."""
    try:
        dimension = EMBEDDING_DIMENSIONS[embedding_type]
        query = f"""
            SELECT arxiv_code, embedding
            FROM arxiv_embeddings_{dimension}
            WHERE arxiv_code = ANY(:arxiv_codes)
            AND doc_type = :doc_type
            AND embedding_type = :embedding_type
            ORDER BY arxiv_code
        """
        
        params = {
            "arxiv_codes": arxiv_codes,
            "doc_type": doc_type,
            "embedding_type": embedding_type,
        }
        
        results = execute_read_query(query, params, as_dataframe=False)
        
        # Convert string embeddings back to float lists
        embeddings_dict = {}
        for code, emb_str in results:
            embeddings_dict[code] = [float(x) for x in emb_str.strip('[]').split(',')]
                
        return embeddings_dict
    except Exception as e:
        logging.error(f"Error loading embeddings: {str(e)}")
        return {}

def get_pending_embeddings(
    doc_type: str,
    embedding_type: str,
) -> List[str]:
    """Get list of arxiv codes that don't have embeddings yet for given doc_type and embedding model."""
    try:
        dimension = EMBEDDING_DIMENSIONS[embedding_type]
        query = f"""
            SELECT DISTINCT arxiv_code 
            FROM arxiv_embeddings_{dimension}
            WHERE doc_type = :doc_type
            AND embedding_type = :embedding_type
        """
        
        params = {
            "doc_type": doc_type,
            "embedding_type": embedding_type
        }
        
        df = execute_read_query(query, params)
        return df.arxiv_code.tolist()
    except Exception as e:
        logging.error(f"Error getting pending embeddings: {str(e)}")
        return []

def get_topic_embedding_dist() -> Dict[str, Dict[str, float]]:
    """Get mean and standard deviation for topic embeddings (dim1 & dim2)."""
    try:
        query = """
            SELECT AVG(dim1), STDDEV(dim1), AVG(dim2), STDDEV(dim2)
            FROM topics
        """
        
        results = execute_read_query(query, as_dataframe=False)
        if not results:
            return {}
            
        res = results[0]  # Get first row since we're aggregating
        return {
            "dim1": {"mean": res[0], "std": res[1]},
            "dim2": {"mean": res[2], "std": res[3]},
        }
    except Exception as e:
        logging.error(f"Error getting topic embedding distribution: {str(e)}")
        return {} 