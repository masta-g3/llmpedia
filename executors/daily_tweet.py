import pandas as pd
from langchain_community.callbacks import get_openai_callback
import os, sys
import json
from dotenv import load_dotenv

load_dotenv()

PROJECT_PATH = os.environ.get("PROJECT_PATH")
DATA_PATH = os.path.join(PROJECT_PATH, "data")
sys.path.append(PROJECT_PATH)

import utils.vector_store as vs
import utils.db as db


def main():
    """Generate a weekly review of highlights and takeaways from papers."""
    vs.validate_openai_env()

    ## Load previous tweets.
    n_tweets = 5
    with open(f"{DATA_PATH}/tweets.json", "r") as f:
        tweets = json.load(f)
    previous_tweets = "\n----------\n".join(
        [
            f"[{k}] {v}"
            for i, (k, v) in enumerate(tweets.items())
            if i > len(tweets) - n_tweets
        ]
    )
    ## ToDo: replace json with DF.
    # tweets_df = pd.DataFrame(tweets.items(), columns=["date", "tweet"])
    # tweets_df

    paper_summary = db.get_extended_notes(arxiv_code, expected_tokens=82000)
    title_map = db.get_arxiv_title_dict()
    paper_title = title_map[arxiv_code]

    tweet_style = "You are writing a post about *today's LLM paper review*. Below you can read a some summary notes on it; find some unexpected, interesting fact or finding to share and tweet about it. Use direct language and simple without mandy adjectives or modifiers."
    tweet_facts = (
        """```
    **Title: """
        + paper_title
        + """**
    """
        + paper_summary
        + "```"
    )

    with get_openai_callback() as cb:
        ## Run model.
        tweet = vs.write_tweet(
            previous_tweets=previous_tweets,
            tweet_style=tweet_style,
            tweet_facts=tweet_facts,
            model="claude-sonnet"
        )
        print(tweet)

        edited_tweet = vs.edit_tweet(tweet, model="claude-sonnet")
        print(cb)

    print("Original tweet: ")
    print(tweet)
    print("-="*20)
    print("Edited tweet: ")
    print(edited_tweet)


if __name__ == "__main__":
    arxiv_code = "2404.02418"
    print(f"Generating a tweet for {arxiv_code}...")
    main()
