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
        # Collect and store tweets in batches
        total_stored = 0
        for tweet_batch in collect_llm_tweets(logger, max_tweets=2000, batch_size=100):
            if tweet_batch:
                store_tweets(tweet_batch, logger)
                total_stored += len(tweet_batch)
                logger.info(f"Total tweets stored so far: {total_stored}")
        
        logger.info(f"Tweet collection process completed. Total tweets stored: {total_stored}")
        
    except Exception as e:
        logger.error(f"Error in tweet collection process: {str(e)}")

if __name__ == "__main__":
    main() 