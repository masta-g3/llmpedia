#!/usr/bin/env python3

import sys, os
from dotenv import load_dotenv
import pandas as pd
from typing import Optional
from sqlalchemy import create_engine, text
from dataclasses import dataclass
import argparse
from datetime import datetime

load_dotenv()
PROJECT_PATH = os.getenv('PROJECT_PATH', '/app')
sys.path.append(PROJECT_PATH)

from utils.logging_utils import setup_logger
import utils.db.tweet_db as tweet_db
import utils.vector_store as vs

# Set up logging
logger = setup_logger(__name__, "t1_analyze_tweets.log")

@dataclass
class TweetAnalysisResult:
    """Container for tweet analysis results."""
    min_date: pd.Timestamp
    max_date: pd.Timestamp
    unique_count: int
    thinking_process: str
    response: str

def format_tweets_for_analysis(tweets_df: pd.DataFrame) -> str:
    """Format tweets into a string for analysis."""
    tweets = []
    for _, row in tweets_df.iterrows():
        tweet_str = f"TWEET: {row['text']}\n"
        tweet_str += f"Author: {row['author']}\n"
        tweet_str += f"Metrics: {row['like_count']} likes, {row['reply_count']} replies\n"
        tweet_str += f"Time: {pd.to_datetime(row['tstp']).strftime('%Y-%m-%d %H:%M:%S')}\n"
        tweets.append(tweet_str)
    return "\n---\n".join(tweets)

def process_tweets(start_time: Optional[str] = None, time_span_hours: int = 6) -> Optional[TweetAnalysisResult]:
    """Process tweets from the last N hours and extract key metrics."""
    logger.info(f"Processing tweets from last {time_span_hours} hours")
    
    # Calculate time range
    end_date = pd.Timestamp.now()
    start_date = pd.Timestamp(start_time) if start_time else end_date - pd.Timedelta(hours=time_span_hours)
    
    # Get tweets
    tweets_df = tweet_db.read_tweets(
        start_date=start_date.strftime("%Y-%m-%d %H:%M:%S"),
        end_date=end_date.strftime("%Y-%m-%d %H:%M:%S")
    )
    
    if tweets_df.empty:
        logger.warning("No tweets found in the specified time range")
        return None
        
    # Remove duplicates and process timestamps
    tweets_df.drop_duplicates(subset="text", keep="first", inplace=True)
    tweets_df["tstp"] = pd.to_datetime(tweets_df["tstp"])
    
    # Get key metrics
    min_date = tweets_df["tstp"].min()
    max_date = tweets_df["tstp"].max()
    unique_count = tweets_df["text"].nunique()
    
    # Get previous analyses and format as diary entries
    previous_analyses = tweet_db.read_last_n_tweet_analyses(10)
    previous_entries = ""
    if not previous_analyses.empty:
        entries = []
        for _, row in previous_analyses.iterrows():
            timestamp = pd.to_datetime(row['tstp']).strftime("%Y-%m-%d %H:%M")
            entries.append(f"[{timestamp}] {row['response']}")
        previous_entries = "\n".join(entries)
    
    # Format and analyze tweets
    tweets_text = format_tweets_for_analysis(tweets_df)
    thinking_process, response = vs.analyze_tweet_patterns(
        tweets_text,
        previous_entries=previous_entries,
        start_date=min_date.strftime("%Y-%m-%d %H:%M:%S"),
        end_date=max_date.strftime("%Y-%m-%d %H:%M:%S")
    )
    
    logger.info(f"Found {unique_count} unique tweets between {min_date} and {max_date}")
    logger.info("Completed tweet pattern analysis")
    
    return TweetAnalysisResult(
        min_date=min_date,
        max_date=max_date,
        unique_count=unique_count,
        thinking_process=thinking_process,
        response=response
    )

def main():
    """Load tweets and analyze volume patterns."""
    parser = argparse.ArgumentParser(description="Analyze tweets from a specific time period")
    parser.add_argument("--start-time", type=str, help="Start time in YYYY-MM-DD HH:MM:SS format")
    args = parser.parse_args()

    logger.info("Starting tweet analysis process")
    
    try:
        # Process tweets from specified start time.
        result = process_tweets(start_time=args.start_time)
        if not result:
            logger.error("Failed to process tweets")
            sys.exit(1)
            
        # Store results
        tweet_db.store_tweet_analysis(
            min_date=result.min_date,
            max_date=result.max_date,
            unique_count=result.unique_count,
            thinking_process=result.thinking_process,
            response=result.response
        )
        logger.info("Tweet analysis completed successfully")
        
    except Exception as e:
        logger.error(f"An error occurred in the main process: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 