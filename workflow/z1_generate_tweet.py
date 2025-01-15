#!/usr/bin/env python3

import datetime
import os, sys, re
import time
import random
import numpy as np
from dotenv import load_dotenv
import base64

load_dotenv()

PROJECT_PATH = os.environ.get("PROJECT_PATH")
DATA_PATH = os.path.join(PROJECT_PATH, "data")
IMG_PATH = os.path.join(DATA_PATH, "arxiv_art")
PAGE_PATH = os.path.join(DATA_PATH, "arxiv_first_page")
sys.path.append(PROJECT_PATH)

from utils.logging_utils import setup_logger
import utils.vector_store as vs
import utils.paper_utils as pu
import utils.notifications as em
import utils.db as db
import utils.tweet as tweet

logger = setup_logger(__name__, "z1_generate_tweet.log")


def bold(input_text, extra_str):
    chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    bold_chars = "𝗔𝗕𝗖𝗗𝗘𝗙𝗚𝗛𝗜𝗝𝗞𝗟𝗠𝗡𝗢𝗣𝗤𝗥𝗦𝗧𝗨𝗩𝗪𝗫𝗬𝗭𝗮𝗯𝗰𝗱𝗲𝗳𝗴𝗵𝗶𝗷𝗸𝗹𝗺𝗻𝗼𝗽𝗾𝗿𝘀𝘁𝘂𝘃𝘄𝘅𝘆𝘇𝟬𝟭𝟮𝟯𝟰𝟱𝟲𝟳𝟴𝟵"
    bold_italic_chars = "𝘼𝘽𝘾𝘿𝙀𝙁𝙂𝙃𝙄𝙅𝙆𝙇𝙈𝙉𝙊𝙋𝙌𝙍𝙎𝙏𝙐𝙑𝙒𝙓𝙔𝙕𝙖𝙗𝙘𝙙𝙚𝙛𝙜𝙝𝙞𝙟𝙠𝙡𝙢𝙣𝙤𝙥𝙦𝙧𝙨𝙩𝙪𝙫𝙬𝙭𝙮𝙯𝟬𝟭𝟮𝟯𝟰𝟱𝟲𝟳𝟴𝟵"

    ## Helper function to bold the characters within quotes
    def boldify(text):
        bolded_text = ""
        for character in text:
            if character in chars:
                bolded_text += bold_chars[chars.index(character)]
            else:
                bolded_text += character
        return bolded_text

    ## Helper function to bold and italicize the characters within asterisks
    def bold_italicize(text):
        bold_italic_text = ""
        for character in text:
            if character in chars:
                bold_italic_text += bold_italic_chars[chars.index(character)]
            else:
                bold_italic_text += character
        return bold_italic_text

    ## Regex to find text in double brackets and apply the boldify function to them.
    output = re.sub(
        r"\[\[([^]]*)\]\]",
        lambda m: "[[" + boldify(m.group(1)) + "]] (" + extra_str + ")",
        input_text,
    )
    output = output.replace("[[", "").replace("]]", "")
    
    ## Regex to find text in double asterisks and apply the bold_italicize function to them
    output = re.sub(r"\*\*([^*]*)\*\*", lambda m: bold_italicize(m.group(1)), output)
    
    ## Italicize "Moral:" but not the moral itself
    output = output.replace("Moral:", bold_italicize("Moral:"))

    return output.strip()


def main():
    """Generate a weekly review of highlights and takeaways from papers."""
    logger.info("Starting tweet generation process.")
    
    ## Randomly decide between regular tweet (90%) and fable (10%).
    is_fable = random.random() < 0.2
    tweet_type = "fable" if is_fable else "insight_v5"
    logger.info(f"Selected tweet type: {'fable' if is_fable else 'regular insight'}")

    ## Get list of papers not yet reviewed.
    arxiv_codes = pu.list_s3_files("arxiv-art", strip_extension=True)
    done_codes = db.get_arxiv_id_list(db.db_params, "tweet_reviews")
    arxiv_codes = list(set(arxiv_codes) - set(done_codes))
    arxiv_codes = sorted(arxiv_codes)[-250:]
    logger.info(f"Found {len(arxiv_codes)} recent papers")

    ## Select candidate papers based on citations.
    citations_df = db.load_citations()
    citations_df = citations_df[citations_df.index.isin(arxiv_codes)]
    citations_df["citation_count"] = citations_df["citation_count"].fillna(1) + 1
    citations_df["weight"] = (
        citations_df["citation_count"] / citations_df["citation_count"].sum()
    )
    citations_df["weight"] = citations_df["weight"] ** 0.5
    citations_df["weight"] = citations_df["weight"] / citations_df["weight"].sum()
    candidate_arxiv_codes = np.random.choice(
        citations_df.index,
        size=25,
        replace=False,
        p=citations_df["weight"] / citations_df["weight"].sum(),
    )
    logger.info(f"Selected {len(candidate_arxiv_codes)} candidate papers based on citations")

    ## Prepare abstracts for selection.
    candidate_abstracts = db.get_recursive_summary(arxiv_codes)
    abstracts_str = "\n".join(
        [
            f"<{code}>\n{abstract}\n</{code}>\n"
            for code, abstract in candidate_abstracts.items()
        ]
    )

    ## Select most interesting paper.
    logger.info("Selecting most interesting paper...")
    arxiv_code = vs.select_most_interesting_paper(
        abstracts_str, model="claude-3-5-sonnet-20241022"
    )
    logger.info(f"Selected paper: {arxiv_code}")

    ## Gather paper details.
    paper_summary = db.get_extended_notes(arxiv_code, expected_tokens=4500)
    paper_details = db.load_arxiv(arxiv_code)
    publish_date_full = paper_details["published"].iloc[0].strftime("%b %d, %Y")
    author = paper_details["authors"].iloc[0]
    title_map = db.get_arxiv_title_dict()
    paper_title = title_map[arxiv_code]

    ## Prepare tweet content.
    tweet_facts = (
        f"```**Title: {paper_title}**\n**Authors: {author}**\n{paper_summary}```"
    )
    post_tweet = f"arxiv link: https://arxiv.org/abs/{arxiv_code}\nllmpedia link: https://llmpedia.streamlit.app/?arxiv_code={arxiv_code}"

    ## Add repository link if available.
    repo_df = db.load_repositories(arxiv_code)
    if not repo_df.empty:
        repo_link = repo_df["repo_url"].values[0]
        if repo_link:
            post_tweet += f"\nrepo: {repo_link}"

    ## Generate content based on type.
    if is_fable:
        # For fables, include the paper's thumbnail image
        with open(f"{IMG_PATH}/{arxiv_code}.png", "rb") as img_file:
            b64_image = base64.b64encode(img_file.read()).decode("utf-8")
        
        tweet_content = vs.write_fable(
            tweet_facts=tweet_facts,
            image_data=b64_image,
            model="claude-3-5-sonnet-20241022",
            temperature=0.9,
        )
        edited_tweet = bold(tweet_content, publish_date_full)
    else:
        ## Get community context for regular tweets.
        recent_llm_tweets = []
        for batch in tweet.collect_llm_tweets(logger, max_tweets=10):
            recent_llm_tweets.extend(batch)
        recent_llm_tweets_str = "\n".join(
            [
                f"COMMUNITY TWEET {i+1}:\n{tweet['text']}"
                for i, tweet in enumerate(recent_llm_tweets)
            ]
        )
        
        most_recent_tweets = (
            db.load_tweet_insights(drop_rejected=True).head(7)["tweet_insight"].values
        )
        most_recent_tweets_str = "\n".join(
            [f"- {tweet.replace('Insight from ', 'From ')}" for tweet in most_recent_tweets]
        )
        
        tweet_obj = vs.write_tweet(
            tweet_facts=tweet_facts,
            tweet_type=tweet_type,
            most_recent_tweets=most_recent_tweets_str,
            recent_llm_tweets=recent_llm_tweets_str,
            model="claude-3-5-sonnet-20241022",
            temperature=0.9,
        )
        tweet_content = tweet_obj.edited_tweet
        edited_tweet = bold(tweet_content, publish_date_full)

    logger.info(f"Generated content: {edited_tweet}")

    ## Check for author's tweet.
    author_tweet = tweet.find_paper_author_tweet(arxiv_code, logger)
    if author_tweet:
        logger.info(f"Found author tweet: {author_tweet['text']}")

    ## Prepare images.
    tweet_image_path = f"{IMG_PATH}/{arxiv_code}.png"
    tweet_page_path = f"{PAGE_PATH}/{arxiv_code}.png"

    if not os.path.exists(tweet_image_path):
        pu.download_s3_file(
            arxiv_code, bucket_name="arxiv-art", prefix="data", format="png"
        )
    if not os.path.exists(tweet_page_path):
        pu.download_s3_file(
            arxiv_code, bucket_name="arxiv-first-page", prefix="data", format="png"
        )

    ## Analyze images for regular tweets only.
    analyzed_image_path = None
    if not is_fable:
        analyzed_image = vs.analyze_paper_images(arxiv_code, model="claude-3-5-sonnet-20241022")
        if analyzed_image:
            analyzed_image_path = os.path.join(DATA_PATH, "arxiv_md", arxiv_code, analyzed_image)
            if not os.path.exists(analyzed_image_path):
                logger.warning(f"Selected image {analyzed_image} not found in arxiv_md/{arxiv_code}")
                analyzed_image_path = None
    
    ## Send tweet.
    tweet_success = tweet.send_tweet(
        edited_tweet,
        tweet_image_path,
        post_tweet,
        logger,
        author_tweet=author_tweet,
        tweet_page_path=None if is_fable else tweet_page_path,
        analyzed_image_path=analyzed_image_path
    )

    ## Store results.
    if tweet_success:
        db.insert_tweet_review(
            arxiv_code,
            edited_tweet,
            datetime.datetime.now(),
            tweet_type,
            rejected=False,
        )
        em.send_email_alert(edited_tweet, arxiv_code)
        logger.info("Tweet stored in database and email alert sent.")
    else:
        logger.error("Failed to send tweet.")


if __name__ == "__main__":
    main()
