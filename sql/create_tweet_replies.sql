CREATE TABLE IF NOT EXISTS tweet_replies (
    id SERIAL PRIMARY KEY,
    tstp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    selected_tweet TEXT NOT NULL,
    response TEXT NOT NULL,
    meta_data JSONB
); 