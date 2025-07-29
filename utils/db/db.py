"""Consolidated database operations for specific functionalities."""

from typing import Optional, Dict, List, Set
from datetime import datetime, timedelta
import pandas as pd

from .db_utils import execute_read_query, simple_select_query, query_db, get_arxiv_title_dict
from utils.embeddings import convert_query_to_vector

############
## PAPERS ##
############


def load_arxiv(arxiv_code: Optional[str] = None, drop_tstp: bool = True, **kwargs) -> pd.DataFrame:
    """Load paper details from arxiv_details table."""
    return simple_select_query(
        table="arxiv_details",
        conditions={"arxiv_code": arxiv_code} if arxiv_code else None,
        drop_cols=["tstp"] if drop_tstp else None,
        **kwargs,
    )


def load_summaries(drop_tstp: bool = True) -> pd.DataFrame:
    """Load paper summaries from summaries table."""
    return simple_select_query(
        table="summaries", 
        drop_cols=["tstp"] if drop_tstp else None
    )


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
    criteria: dict, 
    config: dict, 
    embedding_model: str = "embed-english-v3.0",
    exclude_arxiv_codes: Optional[Set[str]] = None
) -> str:
    """Generate SQL query for semantic search using pgvector, optionally excluding specific arxiv_codes."""
    query_parts = [
        ## Select basic paper info and notes.
        """SELECT 
            a.arxiv_code, 
            a.title, 
            a.published as published_date, 
            s.citation_count as citations, 
            a.summary AS abstract,
            n.notes,
            n.tokens""",
        ## From tables.
        """FROM arxiv_details a, 
             semantic_details s, 
             topics t, 
             arxiv_embeddings_1024 e,
             (SELECT DISTINCT ON (arxiv_code) arxiv_code, summary as notes, tokens 
              FROM summary_notes 
              ORDER BY arxiv_code, ABS(tokens - 3000) ASC) n""",
        ## Join conditions.
        """WHERE a.arxiv_code = s.arxiv_code
        AND a.arxiv_code = t.arxiv_code 
        AND a.arxiv_code = n.arxiv_code 
        AND a.arxiv_code = e.arxiv_code 
        AND e.doc_type = 'abstract'
        AND e.embedding_type = '%s'"""
        % embedding_model,
    ]

    # Add exclusion clause if exclude_arxiv_codes is provided
    if exclude_arxiv_codes and len(exclude_arxiv_codes) > 0:
        # Format codes for SQL IN clause: ('code1', 'code2')
        formatted_excluded_codes = ", ".join(f"'{code}'" for code in exclude_arxiv_codes)
        query_parts.append(f"AND a.arxiv_code NOT IN ({formatted_excluded_codes})")

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
        # Ensure comma is added if n.notes was the last thing
        if query_parts[0].endswith("n.notes"):
             query_parts[0] += f", {similarity_select}"
        else: # If it already had a comma (e.g. if you add other fields later)
             query_parts[0] = query_parts[0].rstrip(",") + f", {similarity_select}"
        
        # Only add ORDER BY if it's not already present from some other logic
        if not any("ORDER BY" in part for part in query_parts[1:]): # Check parts after SELECT
            query_parts.append("ORDER BY similarity_score DESC")

    # Add LIMIT if specified in criteria
    if "limit" in criteria and criteria["limit"] is not None:
        query_parts.append(f"LIMIT {criteria['limit']}")

    return "\n".join(query_parts)


###############
## TWEETS ##
###############


def load_tweet_insights(
    arxiv_code: Optional[str] = None,
    drop_rejected: bool = False,
    drop_tstp: bool = True,
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
        select_cols=(
            ["arxiv_code", "review", "tstp"]
            if not drop_tstp
            else ["arxiv_code", "review"]
        ),
    )

    if not df.empty:
        df.rename(columns={"review": "tweet_insight"}, inplace=True)

    return df


def get_random_interesting_facts(n=10, recency_days=7) -> List[Dict]:
    """Get random interesting facts only after cutoff date, with slight recency weighting."""
    ## Calculate the cutoff date for recent facts
    recent_cutoff = datetime.now() - timedelta(days=recency_days)
    recent_cutoff_str = recent_cutoff.strftime("%Y-%m-%d")

    ## Query only facts after the cutoff date
    query = f"""
        SELECT f.id, f.arxiv_code, f.fact, f.tstp,
               COALESCE(c.citation_count, 0) as citation_count
        FROM summary_interesting_facts f
        LEFT JOIN semantic_details c ON f.arxiv_code = c.arxiv_code
        WHERE f.tstp >= '{recent_cutoff_str}'
        ORDER BY RANDOM()
        LIMIT {n*5}
    """

    all_facts = query_db(query)

    if not all_facts:
        return []

    ## Find the most recent and oldest fact in the window for normalization
    timestamps = [pd.to_datetime(f["tstp"]) for f in all_facts]
    max_time = max(timestamps)
    min_time = min(timestamps)
    time_range = (max_time - min_time).total_seconds() or 1
    max_citations = max([f["citation_count"] for f in all_facts]) if all_facts else 1

    for fact in all_facts:
        ## Recency score: 1 for most recent, 0 for oldest, linear in between
        recency_score = (pd.to_datetime(fact["tstp"]) - min_time).total_seconds() / time_range
        ## Weight recency slightly (0.4) and citations more (0.6)
        citation_score = 0.6 * (fact["citation_count"] / max_citations) if max_citations > 0 else 0
        fact["score"] = 0.4 * recency_score + citation_score

    all_facts.sort(key=lambda x: x["score"], reverse=True)
    unique_facts = []
    seen_content = set()

    for fact in all_facts:
        if fact["fact"] not in seen_content and len(unique_facts) < n:
            seen_content.add(fact["fact"])
            unique_facts.append(fact)

    titles_dict = get_arxiv_title_dict()
    for fact in unique_facts:
        fact["paper_title"] = titles_dict.get(fact['arxiv_code'], "Unknown paper")

    return unique_facts


def read_last_n_tweet_analyses(n: int = 10) -> pd.DataFrame:
    """Read the last N tweet analyses from the database."""
    return simple_select_query(
        table="tweet_analysis",
        select_cols=["tstp", "thinking_process", "response"],
        order_by="tstp DESC",
        index_col=None,
        conditions={"LIMIT": n},
    )


def read_last_n_reddit_analyses(n: int = 10) -> pd.DataFrame:
    """Read the last N reddit analyses from the database (multi-subreddit only)."""
    return simple_select_query(
        table="reddit_analysis",
        select_cols=["tstp", "thinking_process", "response"],
        order_by="tstp DESC",
        index_col=None,
        conditions={"subreddit": "multi", "LIMIT": n},
    )


def load_reddit_posts(arxiv_code: Optional[str] = None, drop_tstp: bool = True) -> pd.DataFrame:
    """Load Reddit posts from reddit_posts table."""
    return simple_select_query(
        table="reddit_posts",
        conditions={"arxiv_code": arxiv_code} if arxiv_code else None,
        drop_cols=["tstp"] if drop_tstp else None,
        index_col=None,
    )


def load_reddit_comments(post_reddit_id: Optional[str] = None, drop_tstp: bool = True) -> pd.DataFrame:
    """Load Reddit comments from reddit_comments table."""
    return simple_select_query(
        table="reddit_comments",
        conditions={"post_reddit_id": post_reddit_id} if post_reddit_id else None,
        drop_cols=["tstp"] if drop_tstp else None,
        index_col=None,
    )


def get_top_reddit_comments(post_reddit_ids: List[str], max_comments: int = 5, min_score: int = 1) -> pd.DataFrame:
    """Get top-scoring comments for specific Reddit posts."""
    if not post_reddit_ids:
        return pd.DataFrame()
    
    query = """
        SELECT 
            reddit_id,
            post_reddit_id,
            author,
            body as content,
            score,
            depth,
            is_top_level,
            comment_timestamp as published_date
        FROM reddit_comments
        WHERE post_reddit_id IN :post_ids
          AND score >= :min_score
        ORDER BY post_reddit_id, score DESC
    """
    
    df = execute_read_query(query, {
        "post_ids": tuple(post_reddit_ids),
        "min_score": min_score
    })
    
    if df.empty:
        return df
    
    ## Limit to top comments per post
    top_comments = []
    for post_id in post_reddit_ids:
        post_comments = df[df['post_reddit_id'] == post_id].head(max_comments)
        top_comments.append(post_comments)
    
    return pd.concat(top_comments, ignore_index=True) if top_comments else pd.DataFrame()


def load_reddit_for_papers(arxiv_codes: List[str]) -> pd.DataFrame:
    """Load Reddit posts and comments for specific arXiv papers."""
    if not arxiv_codes:
        return pd.DataFrame()
    
    query = """
        SELECT 
            p.arxiv_code,
            p.reddit_id as post_reddit_id,
            p.subreddit,
            p.title as post_title,
            p.selftext as post_content,
            p.author as post_author,
            p.score as post_score,
            p.num_comments,
            p.post_timestamp,
            c.body as comment_content,
            c.author as comment_author,
            c.score as comment_score,
            c.depth as comment_depth,
            c.is_top_level
        FROM reddit_posts p
        LEFT JOIN reddit_comments c ON p.reddit_id = c.post_reddit_id
        WHERE p.arxiv_code IN :arxiv_codes
          AND p.arxiv_code IS NOT NULL
          AND p.arxiv_code != ''
          AND p.arxiv_code != 'null'
        ORDER BY p.post_score DESC, c.score DESC
    """
    
    return execute_read_query(query, {"arxiv_codes": tuple(arxiv_codes)})


def get_reddit_metrics(arxiv_code: str) -> Dict[str, int]:
    """Get Reddit engagement metrics for a specific arXiv paper."""
    df = simple_select_query(
        table="reddit_posts",
        conditions={"arxiv_code": arxiv_code},
        select_cols=["score", "num_comments"],
    )
    
    if df.empty:
        return {"total_posts": 0, "total_score": 0, "total_comments": 0, "avg_score": 0}
    
    return {
        "total_posts": len(df),
        "total_score": int(df["score"].sum()),
        "total_comments": int(df["num_comments"].sum()),
        "avg_score": int(df["score"].mean()) if len(df) > 0 else 0,
    }


def get_reddit_discussions_summary(arxiv_codes: List[str]) -> pd.DataFrame:
    """Get aggregated Reddit discussion metrics for multiple papers."""
    if not arxiv_codes:
        return pd.DataFrame()
    
    query = """
        SELECT 
            p.arxiv_code,
            COUNT(DISTINCT p.reddit_id) as post_count,
            COUNT(DISTINCT c.reddit_id) as comment_count,
            SUM(p.score) as total_post_score,
            AVG(p.score) as avg_post_score,
            COUNT(DISTINCT p.subreddit) as subreddit_count,
            ARRAY_AGG(DISTINCT p.subreddit) as subreddits,
            MAX(p.post_timestamp) as latest_discussion
        FROM reddit_posts p
        LEFT JOIN reddit_comments c ON p.reddit_id = c.post_reddit_id
        WHERE p.arxiv_code IN :arxiv_codes
          AND p.arxiv_code IS NOT NULL
          AND p.arxiv_code != ''
          AND p.arxiv_code != 'null'
        GROUP BY p.arxiv_code
        ORDER BY total_post_score DESC
    """
    
    return execute_read_query(query, {"arxiv_codes": tuple(arxiv_codes)})


def generate_reddit_semantic_search_query(
    criteria: dict,
    config: dict,
    embedding_model: str = "embed-english-v3.0",
) -> str:
    """Generate SQL query for semantic search over Reddit content using pgvector."""
    # Create Reddit-specific config mapping
    reddit_config = {
        "min_publication_date": "p.post_timestamp >= '%s'",
        "max_publication_date": "p.post_timestamp <= '%s'",
        "semantic_search_queries": "(%s)",
    }
    
    query_parts = [
        ## Select Reddit post/comment info
        """SELECT 
            p.reddit_id,
            p.subreddit,
            p.title,
            p.selftext as content,
            p.author,
            p.score,
            p.num_comments,
            p.post_timestamp as published_date,
            'reddit_post' as content_type""",
        ## From Reddit posts with embeddings - use 'e' alias to match format_query_condition expectations
        """FROM reddit_posts p, reddit_posts e""",
        ## Basic filters - ensure we're using the same record from both aliases
        """WHERE p.reddit_id = e.reddit_id
        AND p.title IS NOT NULL
        AND p.score >= 0""",
    ]

    ## Add similarity conditions if present - use reddit_config instead of passed config for date constraints
    similarity_scores = []
    for field, value in criteria.items():
        if (
            value is not None
            and field not in ["response_length", "limit"]
        ):
            # Use reddit_config for date constraints, fall back to passed config for other fields
            if field in reddit_config:
                condition_str, similarity_expr = format_query_condition(
                    field, reddit_config[field], value, embedding_model
                )
            elif field in config:
                condition_str, similarity_expr = format_query_condition(
                    field, config[field], value, embedding_model
                )
            else:
                continue
                
            query_parts.append(f"AND {condition_str}")
            if similarity_expr != "0 as max_similarity":
                similarity_scores.append(similarity_expr)

    # Add similarity score to SELECT if we have any
    if similarity_scores:
        similarity_select = (
            f"GREATEST({', '.join(similarity_scores)}) as similarity_score"
        )
        query_parts[0] += f", {similarity_select}"
        
        # Add ORDER BY for similarity
        if not any("ORDER BY" in part for part in query_parts[1:]):
            query_parts.append("ORDER BY similarity_score DESC")

    # Add LIMIT if specified in criteria
    if "limit" in criteria and criteria["limit"] is not None:
        query_parts.append(f"LIMIT {criteria['limit']}")

    return "\n".join(query_parts)


###############
## TRENDING PAPERS ##
###############


def get_trending_papers(n: int = 5, time_window_days: int = 7) -> pd.DataFrame:
    """Return papers with highest total like count on tweets within the given window, including individual tweet details."""
    cutoff_date = (datetime.now() - timedelta(days=time_window_days)).strftime("%Y-%m-%d")

    query = f"""
        SELECT 
            t.arxiv_code,
            SUM(t.like_count) AS like_count,
            COUNT(*) as tweet_count,
            ARRAY_AGG(
                json_build_object(
                    'text', t.text,
                    'author', t.author, 
                    'username', t.username,
                    'like_count', t.like_count,
                    'tweet_timestamp', t.tweet_timestamp,
                    'link', t.link,
                    'repost_count', t.repost_count,
                    'reply_count', t.reply_count
                ) ORDER BY t.like_count DESC
            ) AS tweets
        FROM llm_tweets t 
        WHERE t.arxiv_code IS NOT NULL 
          AND t.arxiv_code != 'null' 
          AND t.arxiv_code != '' 
          AND t.tstp >= '{cutoff_date}'
        GROUP BY t.arxiv_code
        ORDER BY like_count DESC
        LIMIT {n}
    """

    return execute_read_query(query)
