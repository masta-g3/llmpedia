"""Database operations for embedding-related functionality."""

from typing import List, Dict
import logging
from datetime import datetime

from .db_utils import execute_read_query, execute_write_query
from utils.embeddings import convert_query_to_vector


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
    

def format_query_condition(field_name: str, template: str, value: str, embedding_model: str):
    """Format query conditions for semantic search, handling both vector and regular conditions."""
    if isinstance(value, list) and "semantic_search_queries" in field_name:
        distance_scores = []
        for query in value:
            vector = convert_query_to_vector(query, embedding_model)
            vector_str = ", ".join(map(str, vector))
            ## Using pgvector's cosine similarity operator <=> and converting to similarity score.
            condition = f"1 - (e.embedding <=> ARRAY[{vector_str}]::vector) "
            distance_scores.append(condition)
        if distance_scores:
            max_similarity = f"GREATEST({', '.join(distance_scores)})"
            return (
                f"({' OR '.join([c + ' > 0.6' for c in distance_scores])})",  # Threshold of 0.6 for similarity
                max_similarity,
            )
        else:
            return "AND TRUE", "0 as max_similarity"
    elif isinstance(value, list):
        value_str = "', '".join(value)
        return template % value_str, "0 as max_similarity"
    else:
        return template % value, "0 as max_similarity"


def generate_semantic_search_query(criteria: dict, config: dict, embedding_model: str = "embed-english-v3.0") -> str:
    """Generate SQL query for semantic search using pgvector."""
    query_parts = [
        ## Select basic paper info and notes.
        """SELECT 
            a.arxiv_code, 
            a.title, 
            a.published as published_date, 
            s.citation_count as citations, 
            a.summary AS abstract,
            n.notes""",
        ## From tables.
        """FROM arxiv_details a, 
             semantic_details s, 
             topics t, 
             arxiv_embeddings_1024 e,
             (SELECT DISTINCT ON (arxiv_code) arxiv_code, summary as notes, tokens 
              FROM summary_notes 
              ORDER BY arxiv_code, ABS(tokens - %d) ASC) n""" % (criteria.get('response_length', 1000) * 3),
        ## Join conditions.
        """WHERE a.arxiv_code = s.arxiv_code
        AND a.arxiv_code = t.arxiv_code 
        AND a.arxiv_code = n.arxiv_code 
        AND a.arxiv_code = e.arxiv_code 
        AND e.doc_type = 'abstract'
        AND e.embedding_type = '%s'""" % embedding_model,
    ]

    ## Add similarity conditions if present.
    similarity_scores = []
    for field, value in criteria.items():
        if value is not None and field in config and field not in ["response_length", "limit"]:
            condition_str, similarity_expr = format_query_condition(
                field, config[field], value, embedding_model
            )
            query_parts.append(f"AND {condition_str}")
            if similarity_expr != "0 as max_similarity":
                similarity_scores.append(similarity_expr)

    # Add similarity score to SELECT if we have any
    if similarity_scores:
        similarity_select = f"GREATEST({', '.join(similarity_scores)}) as similarity_score"
        query_parts[0] = query_parts[0].rstrip(",") + f", {similarity_select}"
        query_parts.append("ORDER BY similarity_score DESC")
    
    # Add LIMIT if specified in criteria
    if 'limit' in criteria and criteria['limit'] is not None:
        query_parts.append(f"LIMIT {criteria['limit']}")
    
    return "\n".join(query_parts)