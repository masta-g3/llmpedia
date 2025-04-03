"""Consolidated database operations for specific functionalities."""

from typing import Optional, Union, Dict, List, Tuple
from datetime import datetime
import pandas as pd
import logging

from .db_utils import (
    execute_read_query,
    simple_select_query,
    list_to_pg_array,  # Note: list_to_pg_array is imported but not used by the functions moved here
)
from utils.embeddings import convert_query_to_vector

###############
## PAPERS ##
###############


def load_arxiv(arxiv_code: Optional[str] = None, **kwargs) -> pd.DataFrame:
    """Load paper details from arxiv_details table."""
    return simple_select_query(
        table="arxiv_details",
        conditions={"arxiv_code": arxiv_code} if arxiv_code else None,
        **kwargs,
    )


def load_summaries() -> pd.DataFrame:
    """Load paper summaries from summaries table."""
    return simple_select_query(table="summaries", drop_cols=["tstp"])


def load_recursive_summaries(
    arxiv_code: Optional[str] = None, drop_tstp: bool = True
) -> pd.DataFrame:
    """Load narrated summaries from recursive_summaries table."""
    return simple_select_query(
        table="recursive_summaries",
        conditions={"arxiv_code": arxiv_code} if arxiv_code else None,
        drop_cols=["tstp"] if drop_tstp else None,
        rename_cols={"summary": "recursive_summary"},
    )


def load_bullet_list_summaries(
    arxiv_code: Optional[str] = None, drop_tstp: bool = True
) -> pd.DataFrame:
    """Load bullet list summaries from bullet_list_summaries table."""
    return simple_select_query(
        table="bullet_list_summaries",
        conditions={"arxiv_code": arxiv_code} if arxiv_code else None,
        drop_cols=["tstp"] if drop_tstp else None,
        rename_cols={"summary": "bullet_list_summary"},
    )


def load_summary_markdown(
    arxiv_code: Optional[str] = None, drop_tstp: bool = True
) -> pd.DataFrame:
    """Load summary markdown from summary_markdown table."""
    return simple_select_query(
        table="summary_markdown",
        conditions={"arxiv_code": arxiv_code} if arxiv_code else None,
        drop_cols=["tstp"] if drop_tstp else None,
        rename_cols={"summary": "markdown_notes"},
    )


def load_topics(arxiv_code: Optional[str] = None) -> pd.DataFrame:
    """Load paper topics from topics table."""
    return simple_select_query(
        table="topics", conditions={"arxiv_code": arxiv_code} if arxiv_code else None
    )


def load_similar_documents() -> pd.DataFrame:
    """Load similar documents from similar_documents table."""
    df = simple_select_query(table="similar_documents")
    if not df.empty:
        df["similar_docs"] = df["similar_docs"].apply(
            lambda x: x.strip("{}").split(",")
        )
    return df


def load_citations(arxiv_code: Optional[str] = None) -> pd.DataFrame:
    """Load citation details from semantic_details table."""
    return simple_select_query(
        table="semantic_details",
        conditions={"arxiv_code": arxiv_code} if arxiv_code else None,
        drop_cols=["paper_id"],
    )


def load_punchlines() -> pd.DataFrame:
    """Load paper punchlines from summary_punchlines table."""
    return simple_select_query(table="summary_punchlines", drop_cols=["tstp"])


def load_repositories(arxiv_code: Optional[str] = None) -> pd.DataFrame:
    """Load repository information from arxiv_repos table."""
    df = simple_select_query(
        table="arxiv_repos",
        conditions={"arxiv_code": arxiv_code} if arxiv_code else None,
        drop_cols=["tstp"],
        rename_cols={
            "title": "repo_title",
            "description": "repo_description",
            "url": "repo_url",
        },
    )
    return df if not df.empty and "repo_url" in df.columns else pd.DataFrame()


def get_arxiv_dashboard_script(arxiv_code: str, sel_col: str = "script_content") -> str:
    """Query DB to get script for the arxiv dashboard."""
    df = simple_select_query(
        table="arxiv_dashboards",
        conditions={"arxiv_code": arxiv_code},
        select_cols=[sel_col],
    )
    return df[sel_col].iloc[0] if not df.empty else None


def get_weekly_repos(date_str: str) -> pd.DataFrame:
    """Get weekly repos for a given date."""
    start_date = (
        pd.to_datetime(date_str).date()
        - pd.Timedelta(days=pd.to_datetime(date_str).weekday())
    ).strftime("%Y-%m-%d")
    end_date = (
        pd.to_datetime(date_str).date()
        + pd.Timedelta(days=6 - pd.to_datetime(date_str).weekday())
    ).strftime("%Y-%m-%d")

    query = """
        SELECT a.published, t.topic, r.url, r.title, r.description
        FROM arxiv_details a
        JOIN arxiv_repos r ON a.arxiv_code = r.arxiv_code
        JOIN topics t ON a.arxiv_code = t.arxiv_code
        WHERE a.published BETWEEN :start_date AND :end_date
        AND r.url IS NOT NULL
    """

    return execute_read_query(query, {"start_date": start_date, "end_date": end_date})


def get_weekly_content(
    date_str: str, content_type: Optional[str] = "content"
) -> Optional[str]:
    """Get weekly content for a given date."""
    df = simple_select_query(
        table="weekly_content",
        conditions={"date": date_str},
        select_cols=[content_type],
    )
    return df[content_type].iloc[0] if not df.empty else None


def get_weekly_summary_old(date_str: str) -> Optional[str]:
    """Get weekly summary for a given date (old approach)."""
    monday_date = (
        pd.to_datetime(date_str).date()
        - pd.Timedelta(days=pd.to_datetime(date_str).weekday())
    ).strftime("%Y-%m-%d")

    df = simple_select_query(
        table="weekly_reviews", conditions={"date": monday_date}, select_cols=["review"]
    )
    return df["review"].iloc[0] if not df.empty else None


def get_extended_notes(
    arxiv_code: str, level: Optional[int] = None, expected_tokens: Optional[int] = None
) -> Optional[str]:
    """Get extended summary for a given arxiv code."""
    if level is not None:
        df = simple_select_query(
            table="summary_notes",
            conditions={"arxiv_code": arxiv_code, "level": level},
            select_cols=["summary"],
        )
    elif expected_tokens is not None:
        query = """
            SELECT DISTINCT ON (arxiv_code) summary
            FROM summary_notes
            WHERE arxiv_code = :arxiv_code
            ORDER BY arxiv_code, ABS(tokens - :expected_tokens) ASC
        """
        df = execute_read_query(
            query, {"arxiv_code": arxiv_code, "expected_tokens": expected_tokens}
        )
    else:
        df = simple_select_query(
            table="summary_notes",
            conditions={"arxiv_code": arxiv_code},
            order_by="level DESC",
            select_cols=["summary"],
            limit=1,
        )

    summary = df["summary"].iloc[0] if not df.empty else None
    ##ToDo: Make use of this.
    if summary:
        summary = summary.replace("<original>", "").replace("</original>", "")
        summary = summary.replace("<new>", "").replace("</new>", "")
    return summary


###############
## EMBEDDINGS ##
###############

## Constants for embedding dimensions
EMBEDDING_DIMENSIONS = {"gte": 1024, "nv": 4096, "voyage": 1024}


def format_query_condition(
    field_name: str, template: str, value: str, embedding_model: str
):
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


def generate_semantic_search_query(
    criteria: dict, config: dict, embedding_model: str = "embed-english-v3.0"
) -> str:
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
              ORDER BY arxiv_code, ABS(tokens - %d) ASC) n"""
        % (criteria.get("response_length", 1000) * 3),
        ## Join conditions.
        """WHERE a.arxiv_code = s.arxiv_code
        AND a.arxiv_code = t.arxiv_code 
        AND a.arxiv_code = n.arxiv_code 
        AND a.arxiv_code = e.arxiv_code 
        AND e.doc_type = 'abstract'
        AND e.embedding_type = '%s'"""
        % embedding_model,
    ]

    ## Add similarity conditions if present.
    similarity_scores = []
    for field, value in criteria.items():
        if (
            value is not None
            and field in config
            and field not in ["response_length", "limit"]
        ):
            condition_str, similarity_expr = format_query_condition(
                field, config[field], value, embedding_model
            )
            query_parts.append(f"AND {condition_str}")
            if similarity_expr != "0 as max_similarity":
                similarity_scores.append(similarity_expr)

    # Add similarity score to SELECT if we have any
    if similarity_scores:
        similarity_select = (
            f"GREATEST({', '.join(similarity_scores)}) as similarity_score"
        )
        query_parts[0] = query_parts[0].rstrip(",") + f", {similarity_select}"
        query_parts.append("ORDER BY similarity_score DESC")

    # Add LIMIT if specified in criteria
    if "limit" in criteria and criteria["limit"] is not None:
        query_parts.append(f"LIMIT {criteria['limit']}")

    return "\n".join(query_parts)


###############
## TWEETS ##
###############


def load_tweet_insights(
    arxiv_code: Optional[str] = None, drop_rejected: bool = False
) -> pd.DataFrame:
    """Load tweet insights from the database."""
    conditions = {
        "tweet_type": [
            "insight_v1",
            "insight_v2",
            "insight_v3",
            "insight_v4",
            "insight_v5",
        ]
    }
    if arxiv_code:
        conditions["arxiv_code"] = arxiv_code
    if drop_rejected:
        conditions["rejected"] = False

    df = simple_select_query(
        table="tweet_reviews",
        conditions=conditions,
        order_by="tstp DESC",
        select_cols=["arxiv_code", "review", "tstp"],
    )

    if not df.empty:
        df.rename(columns={"review": "tweet_insight"}, inplace=True)

    return df
