"""Database operations for paper-related functionality."""

from typing import Optional, Union, Dict, List, Tuple
from datetime import datetime
import pandas as pd

from .db_utils import (
    execute_read_query,
    execute_write_query,
    simple_select_query,
    list_to_pg_array,
)

def load_arxiv(arxiv_code: Optional[str] = None, **kwargs) -> pd.DataFrame:
    """Load paper details from arxiv_details table."""
    return simple_select_query(
        table="arxiv_details",
        conditions={"arxiv_code": arxiv_code} if arxiv_code else None,
        **kwargs
    )

def load_summaries() -> pd.DataFrame:
    """Load paper summaries from summaries table."""
    return simple_select_query(
        table="summaries",
        drop_cols=["tstp"]
    )

def load_recursive_summaries(arxiv_code: Optional[str] = None, drop_tstp: bool = True) -> pd.DataFrame:
    """Load narrated summaries from recursive_summaries table."""
    return simple_select_query(
        table="recursive_summaries",
        conditions={"arxiv_code": arxiv_code} if arxiv_code else None,
        drop_cols=["tstp"] if drop_tstp else None,
        rename_cols={"summary": "recursive_summary"}
    )

def load_bullet_list_summaries() -> pd.DataFrame:
    """Load bullet list summaries from bullet_list_summaries table."""
    return simple_select_query(
        table="bullet_list_summaries",
        drop_cols=["tstp"],
        rename_cols={"summary": "bullet_list_summary"}
    )

def load_summary_notes() -> pd.DataFrame:
    """Load summary notes from summary_notes table."""
    return simple_select_query(
        table="summary_notes"
    )

def load_summary_markdown() -> pd.DataFrame:
    """Load summary markdown from summary_markdown table."""
    return simple_select_query(
        table="summary_markdown",
        drop_cols=["tstp"],
        rename_cols={"summary": "markdown_notes"}
    )

def load_topics() -> pd.DataFrame:
    """Load paper topics from topics table."""
    return simple_select_query(
        table="topics"
    )

def load_similar_documents() -> pd.DataFrame:
    """Load similar documents from similar_documents table."""
    df = simple_select_query(table="similar_documents")
    if not df.empty:
        df["similar_docs"] = df["similar_docs"].apply(lambda x: x.strip("{}").split(","))
    return df

def load_citations(arxiv_code: Optional[str] = None) -> pd.DataFrame:
    """Load citation details from semantic_details table."""
    return simple_select_query(
        table="semantic_details",
        conditions={"arxiv_code": arxiv_code} if arxiv_code else None,
        drop_cols=["paper_id"]
    )

def load_punchlines() -> pd.DataFrame:
    """Load paper punchlines from summary_punchlines table."""
    return simple_select_query(
        table="summary_punchlines",
        drop_cols=["tstp"]
    )

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
        }
    )
    return df if not df.empty and "repo_url" in df.columns else pd.DataFrame()


def get_arxiv_dashboard_script(arxiv_code: str, sel_col: str = "script_content") -> str:
    """Query DB to get script for the arxiv dashboard."""
    df = simple_select_query(
        table="arxiv_dashboards",
        conditions={"arxiv_code": arxiv_code},
        select_cols=[sel_col]
    )
    return df[sel_col].iloc[0] if not df.empty else None


def get_extended_content(arxiv_code: str) -> pd.DataFrame:
    """Get extended content for a given arxiv code."""
    query = """
        WITH max_level_notes AS (
            SELECT arxiv_code, MAX(level) as max_level
            FROM summary_notes
            GROUP BY arxiv_code
        )
        SELECT d.published, d.arxiv_code, d.title, d.authors, sd.citation_count, d.arxiv_comment,
               d.summary, s.contribution_content, s.takeaway_content, s.takeaway_example, 
               d.summary AS recursive_summary, sn.tokens, t.topic
        FROM summaries s
        JOIN arxiv_details d ON s.arxiv_code = d.arxiv_code
        LEFT JOIN semantic_details sd ON s.arxiv_code = sd.arxiv_code
        JOIN summary_notes sn ON s.arxiv_code = sn.arxiv_code
        JOIN topics t ON s.arxiv_code = t.arxiv_code
        JOIN max_level_notes mln ON sn.arxiv_code = mln.arxiv_code AND sn.level = mln.max_level
        WHERE d.arxiv_code = :arxiv_code
    """
    return execute_read_query(query, {"arxiv_code": arxiv_code})

def get_weekly_summary_inputs(date: str) -> pd.DataFrame:
    """Get weekly summaries for a given date (from last monday to next sunday)."""
    date_st = pd.to_datetime(date).date() - pd.Timedelta(days=pd.to_datetime(date).weekday())
    date_end = pd.to_datetime(date).date() + pd.Timedelta(days=6 - pd.to_datetime(date).weekday())
    
    query = """
        SELECT d.published, d.arxiv_code, d.title, d.authors, sd.citation_count, d.arxiv_comment,
               d.summary, s.contribution_content, s.takeaway_content, s.takeaway_example, 
               d.summary AS recursive_summary, sn.tokens, t.topic
        FROM summaries s
        JOIN arxiv_details d ON s.arxiv_code = d.arxiv_code
        LEFT JOIN semantic_details sd ON s.arxiv_code = sd.arxiv_code
        JOIN summary_notes sn ON s.arxiv_code = sn.arxiv_code
        JOIN topics t ON s.arxiv_code = t.arxiv_code
        WHERE d.published BETWEEN :date_st AND :date_end
        AND sn.level = (SELECT MAX(level) FROM summary_notes WHERE arxiv_code = s.arxiv_code)
    """
    
    df = execute_read_query(query, {"date_st": date_st, "date_end": date_end})
    if not df.empty and "citation_count" in df.columns:
        df["citation_count"] = df["citation_count"].fillna(0)
    return df

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

def get_papers_since(cutoff_time: datetime) -> pd.DataFrame:
    """Get papers added since a specific timestamp using bullet_list_summaries table."""
    query = """
        SELECT a.arxiv_code, a.title, a.authors, a.published, t.topic, b.tstp
        FROM bullet_list_summaries b
        JOIN arxiv_details a ON b.arxiv_code = a.arxiv_code
        LEFT JOIN topics t ON b.arxiv_code = t.arxiv_code
        WHERE b.tstp >= :cutoff_time
        ORDER BY b.tstp DESC
    """
    
    return execute_read_query(query, {"cutoff_time": cutoff_time.strftime("%Y-%m-%d %H:%M:%S")})

def get_weekly_content(date_str: str, content_type: Optional[str] = "content") -> Optional[str]:
    """Get weekly content for a given date."""
    df = simple_select_query(
        table="weekly_content",
        conditions={"date": date_str},
        select_cols=[content_type]
    )
    return df[content_type].iloc[0] if not df.empty else None

def get_weekly_summary_old(date_str: str) -> Optional[str]:
    """Get weekly summary for a given date (old approach)."""
    monday_date = (
        pd.to_datetime(date_str).date()
        - pd.Timedelta(days=pd.to_datetime(date_str).weekday())
    ).strftime("%Y-%m-%d")
    
    df = simple_select_query(
        table="weekly_reviews",
        conditions={"date": monday_date},
        select_cols=["review"]
    )
    return df["review"].iloc[0] if not df.empty else None

def check_weekly_summary_exists(date_str: str) -> bool:
    """Check if weekly summary exists for a given date."""
    df = simple_select_query(
        table="weekly_content",
        conditions={"date": date_str},
        select_cols=["COUNT(*) as count"]
    )
    return df["count"].iloc[0] > 0 if not df.empty else False

def get_extended_notes(
    arxiv_code: str, 
    level: Optional[int] = None, 
    expected_tokens: Optional[int] = None
) -> Optional[str]:
    """Get extended summary for a given arxiv code."""
    if level is not None:
        df = simple_select_query(
            table="summary_notes",
            conditions={"arxiv_code": arxiv_code, "level": level},
            select_cols=["summary"]
        )
    elif expected_tokens is not None:
        query = """
            SELECT DISTINCT ON (arxiv_code) summary
            FROM summary_notes
            WHERE arxiv_code = :arxiv_code
            ORDER BY arxiv_code, ABS(tokens - :expected_tokens) ASC
        """
        df = execute_read_query(query, {
            "arxiv_code": arxiv_code,
            "expected_tokens": expected_tokens
        })
    else:
        df = simple_select_query(
            table="summary_notes",
            conditions={"arxiv_code": arxiv_code},
            order_by="level DESC",
            select_cols=["summary"],
            limit=1
        )
    
    return df["summary"].iloc[0] if not df.empty else None

def get_arxiv_parent_chunk_ids(chunk_ids: List[Tuple[str, int]]) -> List[Tuple[str, int]]:
    """Get parent chunk IDs for a list of (arxiv_code, child_id) tuples."""
    conditions = " OR ".join(
        f"(arxiv_code = :code_{i} AND child_id = :child_{i})"
        for i, _ in enumerate(chunk_ids)
    )
    
    params = {}
    for i, (code, child_id) in enumerate(chunk_ids):
        params[f"code_{i}"] = code
        params[f"child_{i}"] = child_id
    
    query = f"""
        SELECT DISTINCT arxiv_code, parent_id
        FROM arxiv_chunk_map
        WHERE ({conditions})
        AND version = '10000_1000'
    """
    
    return execute_read_query(query, params, as_dataframe=False)

def get_arxiv_chunks(chunk_ids: List[Tuple[str, int]], source: str = "child") -> pd.DataFrame:
    """Get chunks with metadata for a list of (arxiv_code, chunk_id) tuples."""
    source_table = "arxiv_chunks" if source == "child" else "arxiv_parent_chunks"
    
    conditions = " OR ".join(
        f"(p.arxiv_code = :code_{i} AND p.chunk_id = :chunk_{i})"
        for i, _ in enumerate(chunk_ids)
    )
    
    params = {}
    for i, (code, chunk_id) in enumerate(chunk_ids):
        params[f"code_{i}"] = code
        params[f"chunk_{i}"] = chunk_id
    
    query = f"""
        SELECT d.arxiv_code, d.title, d.published, s.citation_count, p.text
        FROM {source_table} p
        JOIN arxiv_details d ON p.arxiv_code = d.arxiv_code
        JOIN semantic_details s ON p.arxiv_code = s.arxiv_code
        WHERE {conditions}
    """
    
    return execute_read_query(query, params)

def get_recursive_summary(
    arxiv_code: Optional[Union[str, List[str]]] = None
) -> Union[Dict[str, str], str, None]:
    """Get recursive summaries for papers."""
    codes = [arxiv_code] if isinstance(arxiv_code, str) else arxiv_code
    
    if len(codes) == 0:
        df = simple_select_query(
            table="recursive_summaries",
            select_cols=["arxiv_code", "summary"]
        )
    else:
        df = simple_select_query(
            table="recursive_summaries",
            conditions={"arxiv_code": codes},
            select_cols=["arxiv_code", "summary"]
        )
    
    results = df["summary"].to_dict() if not df.empty else {}
    
    return (
        results.get(arxiv_code) if isinstance(arxiv_code, str)
        else results
    )

def insert_recursive_summary(arxiv_code: str, summary: str) -> bool:
    """Insert data into recursive_summary table."""
    return execute_write_query(
        """
        INSERT INTO recursive_summaries (arxiv_code, summary, tstp)
        VALUES (:arxiv_code, :summary, :tstp)
        """,
        {
            "arxiv_code": arxiv_code,
            "summary": summary,
            "tstp": datetime.now()
        }
    )

def insert_bullet_list_summary(arxiv_code: str, summary: str) -> bool:
    """Insert data into bullet_list_summaries table."""
    return execute_write_query(
        """
        INSERT INTO bullet_list_summaries (arxiv_code, summary, tstp)
        VALUES (:arxiv_code, :summary, :tstp)
        """,
        {
            "arxiv_code": arxiv_code,
            "summary": summary,
            "tstp": datetime.now()
        }
    )