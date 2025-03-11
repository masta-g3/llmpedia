#!/usr/bin/env python3

import os
import sys
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Optional, List, Dict
import logging
import re
PROJECT_PATH = os.environ.get("PROJECT_PATH")
sys.path.append(PROJECT_PATH)

from utils.logging_utils import setup_logger
import utils.db.paper_db as paper_db
import utils.app_utils as au
import utils.tweet as tweet
from utils.tweet import TweetThread, boldify

logger = setup_logger(__name__, "a1_daily_update.log")


@dataclass
class DailyStats:
    """Container for daily paper statistics."""
    total_papers: int
    topic_distribution: Dict[str, int]
    top_cited_authors: List[str]
    trending_topics: List[str]
    time_window: timedelta
    top_cited_papers: List[Dict[str, str]]


def analyze_papers(papers_df: pd.DataFrame) -> DailyStats:
    """Analyze papers and generate interesting statistics."""
    if papers_df.empty:
        return DailyStats(
            total_papers=0,
            topic_distribution={},
            top_cited_authors=[],
            trending_topics=[],
            time_window=timedelta(hours=24),
            top_cited_papers=[]
        )
    
    ## Calculate basic stats
    total_papers = len(papers_df)
    
    ## Get topic distribution
    topic_dist = papers_df['topic'].value_counts().to_dict()
    
    ## Get top authors (split author strings and count unique)
    all_authors = []
    for authors in papers_df['authors']:
        all_authors.extend([a.strip() for a in authors.split(',')])
    top_authors = pd.Series(all_authors).value_counts().head(3).index.tolist()
    
    ## Find trending topics by looking at recent frequency
    trending = papers_df['topic'].value_counts().head(3).index.tolist()
    
    ## Get citation information for papers
    citations_df = paper_db.load_citations()
    papers_df = papers_df.merge(citations_df[['citation_count']], left_on='arxiv_code', right_index=True, how='left')
    papers_df['citation_count'] = papers_df['citation_count'].fillna(0)
    
    ## Get top cited papers with more than 10 citations
    top_cited = papers_df[papers_df['citation_count'] >= 10].nlargest(2, 'citation_count')
    top_cited_papers = [
        {'title': row['title'], 'citations': int(row['citation_count'])}
        for _, row in top_cited.iterrows()
    ]
    
    return DailyStats(
        total_papers=total_papers,
        topic_distribution=topic_dist,
        top_cited_authors=top_authors,
        trending_topics=trending,
        time_window=timedelta(hours=24),
        top_cited_papers=top_cited_papers
    )

def generate_tweet_content(stats: DailyStats) -> str:
    """Generate engaging tweet content from paper statistics."""
    if stats.total_papers == 0:
        return "LLMpedia update: No new papers added in the last 24 hours. Stay tuned for more updates! #AI #LLM"
    
    ## Format topic distribution
    top_topics = sorted(
        stats.topic_distribution.items(), 
        key=lambda x: x[1], 
        reverse=True
    )[:3]
    
    ## Create bullet points for topics using arrows
    topics_list = "\n".join([f"  → {topic} ({count})" for topic, count in top_topics])
    
    ## Create tweet content with emoji and better spacing
    tweet = boldify("📚 LLMpedia Nightly Digest: ")
    tweet += f"{stats.total_papers} new papers added in the last 24h\n"
    tweet += boldify(f"⭐️ Top research areas:")
    tweet += f"\n{topics_list}"
    
    ## Add top cited papers if any, with better formatting
    if stats.top_cited_papers:
        tweet += "\n\n✨ Highly cited papers:"
        for paper in stats.top_cited_papers:
            tweet += f"\n⭐ {paper['title']}\n   {paper['citations']} citations"
    
    return tweet

def create_daily_update_tweet(stats: DailyStats) -> TweetThread:
    """Create a TweetThread object for the daily update.
    
    Args:
        stats: Daily statistics object
        
    Returns:
        TweetThread object ready to be sent
    """
    content = generate_tweet_content(stats)
    
    # Create metadata with additional stats
    metadata = {
        "total_papers": stats.total_papers,
        "top_topics": stats.trending_topics,
        "time_window_hours": 24,
        "generated_at": datetime.now().isoformat()
    }
    
    # Create a simple tweet thread
    return TweetThread.create_simple_tweet(
        content=content,
        tweet_type="daily_update",
        metadata=metadata
    )

def main():
    """Generate and send daily update about new papers added to LLMpedia."""
    import time
    # time.sleep(60 * 60 * 1.33)  # Sleep for 1.33 hours
    logger.info("Starting daily update process")
    
    try:
        ## Get papers from last 24 hours
        cutoff_time = datetime.now() - timedelta(hours=24)
        papers_df = paper_db.get_papers_since(cutoff_time)
        logger.info(f"Found {len(papers_df)} papers in the last 24 hours")

        ## Skip if fewer than 4 papers
        if len(papers_df) < 4:
            logger.info("Fewer than 4 papers found, skipping daily update")
            return 2
        
        ## Analyze papers
        stats = analyze_papers(papers_df)
        logger.info(f"Analyzed papers: {stats}")
        
        ## Create tweet thread
        tweet_thread = create_daily_update_tweet(stats)
        logger.info(f"Generated tweet content: {tweet_thread.tweets[0].content}")
        
        ## Send tweet using new unified system
        # Try sending tweet with one retry
        for attempt in range(2):
            tweet_success = tweet.send_tweet2(
                tweet_content=tweet_thread,
                logger=logger, 
                verify=True,
                headless=False
            )
            if tweet_success:
                break
            elif attempt == 0:
                logger.warning("First tweet attempt failed, retrying after 30 seconds...")
                time.sleep(30)
        
        if tweet_success:
            logger.info("Successfully sent daily update tweet")
            return 0
        else:
            logger.error("Failed to send daily update tweet")
            return 1
        
    except Exception as e:
        logger.error(f"Error in daily update process: {str(e)}")
        raise

if __name__ == "__main__":
    sys.exit(main()) 