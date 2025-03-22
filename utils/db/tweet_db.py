"""Database operations for tweet-related functionality."""

from datetime import datetime
import pandas as pd
import logging
from typing import Optional, Union, List, Dict
from sqlalchemy.engine import Engine
import json

from .db_utils import (
    execute_read_query,
    execute_write_query,
    get_db_engine,
    simple_select_query,
)


def store_tweets(tweets: List[Dict], logger: logging.Logger, engine: Engine) -> bool:
    """Store tweets in the database."""
    try:
        for tweet in tweets:
            # Add collection timestamp
            tweet["tstp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Insert tweet with all available metrics
            query = """
                INSERT INTO llm_tweets (
                    text, author, username, link, tstp, tweet_timestamp,
                    reply_count, repost_count, like_count, view_count, bookmark_count,
                    has_media, is_verified, arxiv_code
                )
                VALUES (
                    :text, :author, :username, :link, :tstp, :tweet_timestamp,
                    :reply_count, :repost_count, :like_count, :view_count, :bookmark_count,
                    :has_media, :is_verified, :arxiv_code
                )
                ON CONFLICT (link) DO NOTHING;
            """

            # Ensure all fields have default values if not present
            tweet_data = {
                "text": tweet.get("text", ""),
                "author": tweet.get("author", ""),
                "username": tweet.get("username", ""),
                "link": tweet.get("link", ""),
                "tstp": tweet.get("tstp"),
                "tweet_timestamp": tweet.get("tweet_timestamp"),
                "reply_count": tweet.get("reply_count", 0),
                "repost_count": tweet.get("repost_count", 0),
                "like_count": tweet.get("like_count", 0),
                "view_count": tweet.get("view_count", 0),
                "bookmark_count": tweet.get("bookmark_count", 0),
                "has_media": tweet.get("has_media", False),
                "is_verified": tweet.get("is_verified", False),
                "arxiv_code": tweet.get("arxiv_code"),
            }

            execute_write_query(query, tweet_data)

        logger.info(f"Successfully stored {len(tweets)} tweets")
        return True
    except Exception as e:
        logger.error(f"Error storing tweets: {str(e)}")
        return False


def read_tweets(
    arxiv_code: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Read tweets from the database with optional filtering.

    Args:
        arxiv_code: Optional paper identifier
        start_date: Optional start date as string ('2024-01-26' or '2024-01-26 15:33:14')
        end_date: Optional end date as string ('2024-01-26' or '2024-01-26 15:33:14')
    """
    try:
        conditions = {}
        if arxiv_code:
            conditions["arxiv_code"] = arxiv_code

        if start_date:
            try:
                conditions["tstp >= "] = pd.to_datetime(start_date)
            except ValueError as e:
                logging.error(f"Invalid start_date format: {str(e)}")
                return pd.DataFrame()

        if end_date:
            try:
                conditions["tstp <= "] = pd.to_datetime(end_date)
            except ValueError as e:
                logging.error(f"Invalid end_date format: {str(e)}")
                return pd.DataFrame()

        return simple_select_query(
            table="llm_tweets",
            conditions=conditions if conditions else None,
            index_col=None,  # Don't set an index for tweets
        )
    except Exception as e:
        logging.error(f"Error reading tweets from database: {str(e)}")
        return pd.DataFrame()


def store_tweet_analysis(
    min_date: pd.Timestamp,
    max_date: pd.Timestamp,
    unique_count: int,
    thinking_process: str,
    response: str,
) -> bool:
    """Store tweet analysis results in the database."""
    try:
        query = """
            INSERT INTO tweet_analysis 
            (start_date, end_date, unique_tweets, thinking_process, response)
            VALUES 
            (:start_date, :end_date, :unique_tweets, :thinking_process, :response)
        """

        params = {
            "start_date": min_date,
            "end_date": max_date,
            "unique_tweets": unique_count,
            "thinking_process": thinking_process,
            "response": response,
        }

        success = execute_write_query(query, params)
        if success:
            logging.info(
                f"Successfully stored analysis results for {min_date.strftime('%Y-%m-%d %H:%M')} to {max_date.strftime('%Y-%m-%d %H:%M')}"
            )
        return success
    except Exception as e:
        logging.error(f"Error storing tweet analysis: {str(e)}")
        return False


def read_last_n_tweet_analyses(n: int = 10) -> pd.DataFrame:
    """Read the last N tweet analyses from the database."""
    return simple_select_query(
        table="tweet_analysis",
        select_cols=["tstp", "thinking_process", "response"],
        order_by="tstp DESC",
        index_col=None,
        conditions={"LIMIT": n},
    )


def store_tweet_reply(
    selected_tweet: str, response: str, meta_data: Optional[Dict] = None, 
    approval_status: str = "pending"
) -> bool:
    """Store tweet reply data in the database."""
    try:
        query = """
            INSERT INTO tweet_replies 
            (selected_tweet, response, meta_data, approval_status)
            VALUES 
            (:selected_tweet, :response, :meta_data, :approval_status)
        """

        # Convert meta_data dict to JSON string for PostgreSQL JSONB
        meta_data_json = json.dumps(meta_data) if meta_data is not None else None

        params = {
            "selected_tweet": selected_tweet,
            "response": response,
            "meta_data": meta_data_json,
            "approval_status": approval_status,
        }

        success = execute_write_query(query, params)
        if success:
            logging.info(
                f"Successfully stored tweet reply for tweet: {selected_tweet[:50]}..."
            )
        return success
    except Exception as e:
        logging.error(f"Error storing tweet reply: {str(e)}")
        return False


def read_tweet_replies(
    start_date: Optional[str] = None, end_date: Optional[str] = None
) -> pd.DataFrame:
    """Read tweet replies from the database with optional date range filtering."""
    try:
        conditions = {}
        if start_date:
            try:
                conditions["tstp >="] = pd.to_datetime(start_date)
            except ValueError as e:
                logging.error(f"Invalid start_date format: {str(e)}")
                return pd.DataFrame()

        if end_date:
            try:
                conditions["tstp <="] = pd.to_datetime(end_date)
            except ValueError as e:
                logging.error(f"Invalid end_date format: {str(e)}")
                return pd.DataFrame()

        return simple_select_query(
            table="tweet_replies",
            conditions=conditions if conditions else None,
            order_by="tstp DESC",
            index_col=None,
        )
    except Exception as e:
        logging.error(f"Error reading tweet replies from database: {str(e)}")
        return pd.DataFrame()


def insert_tweet_review(
    arxiv_code: str,
    review: str,
    tstp: datetime,
    tweet_type: str,
    rejected: bool = False,
) -> bool:
    """ Insert tweet review into the database. """
    try:
        query = """
            INSERT INTO tweet_reviews (arxiv_code, review, tstp, tweet_type, rejected)
            VALUES (:arxiv_code, :review, :tstp, :tweet_type, :rejected)
        """

        params = {
            "arxiv_code": arxiv_code,
            "review": review,
            "tstp": tstp,
            "tweet_type": tweet_type,
            "rejected": rejected,
        }

        success = execute_write_query(query, params)
        if success:
            logging.info(f"Successfully stored tweet review for {arxiv_code}.")
        return success
    except Exception as e:
        logging.error(f"Error inserting tweet review: {str(e)}")
        return False

def load_tweet_insights(arxiv_code: Optional[str] = None, drop_rejected: bool = False) -> pd.DataFrame:
    """Load tweet insights from the database."""
    conditions = {
        "tweet_type": [
            "insight_v1", "insight_v2", "insight_v3", 
            "insight_v4", "insight_v5"
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
        select_cols=["arxiv_code", "review", "tstp"]
    )
    
    if not df.empty:
        df.rename(columns={"review": "tweet_insight"}, inplace=True)
        
    return df

def update_tweet_reply_status(reply_id: int, approval_status: str) -> bool:
    """ Update the approval status of a tweet reply. """
    try:
        query = """
            UPDATE tweet_replies
            SET approval_status = :approval_status
            WHERE id = :reply_id
        """
        
        params = {
            "reply_id": reply_id,
            "approval_status": approval_status,
        }
        
        success = execute_write_query(query, params)
        if success:
            logging.info(f"Successfully updated approval status for tweet reply {reply_id} to {approval_status}.")
        return success
    except Exception as e:
        logging.error(f"Error updating tweet reply status: {str(e)}")
        return False


def get_pending_tweet_replies(limit: int = 10) -> pd.DataFrame:
    """ Get pending tweet replies from the database. """
    try:
        query = f"""
            SELECT id, tstp, selected_tweet, response, meta_data, approval_status
            FROM tweet_replies
            WHERE approval_status = 'pending'
            ORDER BY tstp DESC
            LIMIT {limit}
        """
        
        df = execute_read_query(query)
        return df
    except Exception as e:
        logging.error(f"Error getting pending tweet replies: {str(e)}")
        return pd.DataFrame()