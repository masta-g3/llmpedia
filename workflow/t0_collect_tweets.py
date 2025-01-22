import os
import sys
import logging
from datetime import datetime

PROJECT_PATH = os.getenv("PROJECT_PATH", "/app")
sys.path.append(PROJECT_PATH)

from utils.logging_utils import setup_logger
from utils.tweet import collect_llm_tweets
from utils import db
from sqlalchemy import create_engine

def main():
    # Setup logging
    logger = setup_logger(__name__, "tweet_collector.log")
    logger.info("Starting tweet collection process")
    
    # Create engine once for reuse
    engine = create_engine(db.database_url)
    # try:
        # Collect and store tweets in batches
    total_stored = 0
    for tweet_batch in collect_llm_tweets(logger, max_tweets=500, batch_size=10):
        if tweet_batch:
            db.store_tweets(tweet_batch, logger, engine)
            total_stored += len(tweet_batch)
            logger.info(f"Total tweets stored so far: {total_stored}")
    
    logger.info(f"Tweet collection process completed. Total tweets stored: {total_stored}")

    # except Exception as e:
        # logger.error(f"Error in tweet collection process: {str(e)}")
    # finally:
        # engine.dispose()

if __name__ == "__main__":
    main() 