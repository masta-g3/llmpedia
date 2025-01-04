import os
import sys
import logging
from datetime import datetime

PROJECT_PATH = os.getenv("PROJECT_PATH", "/app")
sys.path.append(PROJECT_PATH)

from utils.logging_utils import setup_logger
from utils.tweet import collect_llm_tweets
from utils.db import store_tweets

def main():
    # Setup logging
    logger = setup_logger(__name__, "tweet_collector.log")
    logger.info("Starting tweet collection process")
    
    try:
        # Collect tweets
        tweets = collect_llm_tweets(logger, max_tweets=2000)
        logger.info(f"Collected {len(tweets)} relevant tweets")
        
        # Store tweets
        if tweets:
            store_tweets(tweets, logger)
        
        logger.info("Tweet collection process completed")
        
    except Exception as e:
        logger.error(f"Error in tweet collection process: {str(e)}")

if __name__ == "__main__":
    main() 