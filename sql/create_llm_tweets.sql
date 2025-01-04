-- Create table for storing LLM-related tweets
CREATE TABLE IF NOT EXISTS llm_tweets (
    id SERIAL PRIMARY KEY,
    text TEXT NOT NULL,
    author TEXT NOT NULL,
    username TEXT NOT NULL,
    link TEXT UNIQUE NOT NULL,  -- Tweet URL is unique
    tstp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,  -- When we collected the tweet
    tweet_timestamp TIMESTAMP,  -- When the tweet was posted
    reply_count INTEGER DEFAULT 0,
    repost_count INTEGER DEFAULT 0,
    like_count INTEGER DEFAULT 0,
    view_count INTEGER DEFAULT 0,
    bookmark_count INTEGER DEFAULT 0,
    has_media BOOLEAN DEFAULT FALSE,
    is_verified BOOLEAN DEFAULT FALSE,
    processed BOOLEAN DEFAULT FALSE,  -- Flag for processing status
    metadata JSONB  -- For storing any additional extracted metadata
);

-- Create indexes for common queries
CREATE INDEX IF NOT EXISTS llm_tweets_tstp_idx ON llm_tweets(tstp);
CREATE INDEX IF NOT EXISTS llm_tweets_tweet_timestamp_idx ON llm_tweets(tweet_timestamp);
CREATE INDEX IF NOT EXISTS llm_tweets_processed_idx ON llm_tweets(processed);
CREATE INDEX IF NOT EXISTS llm_tweets_username_idx ON llm_tweets(username);
CREATE INDEX IF NOT EXISTS llm_tweets_metrics_idx ON llm_tweets(view_count, like_count);

-- Grant necessary permissions
GRANT SELECT, INSERT, UPDATE ON llm_tweets TO public;
GRANT USAGE, SELECT ON SEQUENCE llm_tweets_id_seq TO public;