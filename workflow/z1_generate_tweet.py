#!/usr/bin/env python3

import datetime
import os, sys, re
import time
import random
import numpy as np
from dotenv import load_dotenv
import base64
from dataclasses import dataclass
from typing import Optional, Tuple, List
import logging

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
import utils.app_utils as au

logger = setup_logger(__name__, "z1_generate_tweet.log")

@dataclass
class TweetContent:
    """Container for generated tweet content and metadata."""
    content: str
    post_content: str
    tweet_type: str
    arxiv_code: str
    publish_date: str
    selected_image: Optional[str] = None
    selected_table: Optional[str] = None

@dataclass
class TweetImages:
    """Container for tweet image paths."""
    tweet_image: Optional[str] = None  # Art image
    tweet_page: Optional[str] = None   # First page or selected figure
    analyzed_image: Optional[str] = None  # Additional analyzed figure

def bold(input_text, extra_str):
    """Format text with bold and italic characters."""
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

def select_paper(logger: logging.Logger) -> Tuple[str, dict]:
    """Select paper and gather its details."""
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
    citations_df["weight"] = citations_df["citation_count"] / citations_df["citation_count"].sum()
    citations_df["weight"] = citations_df["weight"] ** 0.5
    citations_df["weight"] = citations_df["weight"] / citations_df["weight"].sum()
    
    candidate_arxiv_codes = np.random.choice(
        citations_df.index,
        size=25,
        replace=False,
        p=citations_df["weight"] / citations_df["weight"].sum(),
    )
    logger.info(f"Selected {len(candidate_arxiv_codes)} candidate papers based on citations")

    ## Prepare abstracts for selection (only for candidates).
    candidate_abstracts = db.get_recursive_summary(candidate_arxiv_codes)
    abstracts_str = "\n".join(
        [f"<{code}>\n{abstract}\n</{code}>\n"
        for code, abstract in candidate_abstracts.items()]
    )

    ## Select most interesting paper.
    logger.info("Selecting most interesting paper...")
    arxiv_code = vs.select_most_interesting_paper(
        abstracts_str, model="claude-3-5-sonnet-20241022"
    )
    
    ## Gather paper details.
    paper_details = db.load_arxiv(arxiv_code)
    
    return arxiv_code, paper_details

def prepare_tweet_facts(arxiv_code: str, paper_details: dict, tweet_type: str, logger: logging.Logger) -> Tuple[str, str, str]:
    """Prepare basic tweet information with content based on tweet type."""
    publish_date_full = paper_details["published"].iloc[0].strftime("%b %d, %Y")
    author = paper_details["authors"].iloc[0]
    title_map = db.get_arxiv_title_dict()
    paper_title = title_map[arxiv_code]
    
    ## Get content based on tweet type
    if tweet_type == "punchline":
        logger.info("Loading markdown content for punchline")
        markdown_content, success = au.get_paper_markdown(arxiv_code)
        if not success:
            raise Exception(f"Could not load markdown content for {arxiv_code}")
        content = markdown_content
    else:
        logger.info("Loading extended notes")
        content = db.get_extended_notes(arxiv_code, expected_tokens=4500)
    
    tweet_facts = f"```**Title: {paper_title}**\n**Authors: {author}**\n{content}```"
    post_tweet = f"arxiv link: https://arxiv.org/abs/{arxiv_code}\nllmpedia link: https://llmpedia.streamlit.app/?arxiv_code={arxiv_code}"

    ## Add repository link if available.
    repo_df = db.load_repositories(arxiv_code)
    if not repo_df.empty and repo_df["repo_url"].values[0]:
        repo_url = repo_df["repo_url"].values[0]
        post_tweet += f"\nrepo: {repo_url}"
        logger.info(f"Added repository link: {repo_url}")
        
    return tweet_facts, post_tweet, publish_date_full

def generate_tweet_content(tweet_type: str, tweet_facts: str, arxiv_code: str, publish_date: str, logger: logging.Logger) -> TweetContent:
    """Generate tweet content based on type."""
    if tweet_type == "fable":
        logger.info("Generating fable-style tweet")
        with open(f"{IMG_PATH}/{arxiv_code}.png", "rb") as img_file:
            b64_image = base64.b64encode(img_file.read()).decode("utf-8")
        
        content = vs.write_fable(
            tweet_facts=tweet_facts,
            image_data=b64_image,
            model="claude-3-5-sonnet-20241022",
            temperature=0.9,
        )
        return TweetContent(
            content=bold(content, publish_date),
            post_content=tweet_facts,
            tweet_type=tweet_type,
            arxiv_code=arxiv_code,
            publish_date=publish_date
        )
        
    elif tweet_type == "punchline":
        logger.info("Generating punchline-style tweet")
        paper_title = db.get_arxiv_title_dict()[arxiv_code]
        punchline_obj = vs.write_punchline_tweet(
            markdown_content=tweet_facts,  # Already contains markdown content
            paper_title=paper_title,
            model="claude-3-5-sonnet-20241022",
            temperature=0.9,
        )
        content = punchline_obj.line
        return TweetContent(
            content=bold(content, publish_date),
            post_content=tweet_facts,
            tweet_type=tweet_type,
            arxiv_code=arxiv_code,
            publish_date=publish_date,
            selected_image=punchline_obj.image,
            selected_table=punchline_obj.table
        )
        
    else:  # insight_v5
        logger.info("Generating insight-style tweet")
        recent_llm_tweets = []
        for batch in tweet.collect_llm_tweets(logger, max_tweets=10):
            recent_llm_tweets.extend(batch)
        recent_llm_tweets_str = "\n".join(
            [f"COMMUNITY TWEET {i+1}:\n{tweet['text']}"
             for i, tweet in enumerate(recent_llm_tweets)]
        )
        
        most_recent_tweets = db.load_tweet_insights(drop_rejected=True).head(7)["tweet_insight"].values
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
        content = tweet_obj.edited_tweet
    
        return TweetContent(
            content=bold(content, publish_date),
            post_content=tweet_facts,
            tweet_type=tweet_type,
            arxiv_code=arxiv_code,
            publish_date=publish_date
        )

def prepare_tweet_images(tweet_content: TweetContent, logger: logging.Logger) -> TweetImages:
    """Prepare images based on tweet type."""
    images = TweetImages()
    
    if tweet_content.tweet_type in ["insight_v5", "fable"]:
        ## Download art image if needed
        images.tweet_image = f"{IMG_PATH}/{tweet_content.arxiv_code}.png"
        if not os.path.exists(images.tweet_image):
            logger.info(f"Downloading art image for {tweet_content.arxiv_code}")
            pu.download_s3_file(
                tweet_content.arxiv_code,
                bucket_name="arxiv-art",
                prefix="data",
                format="png"
            )
    
    if tweet_content.tweet_type == "insight_v5":
        ## Download first page if needed
        images.tweet_page = f"{PAGE_PATH}/{tweet_content.arxiv_code}.png"
        if not os.path.exists(images.tweet_page):
            logger.info(f"Downloading first page for {tweet_content.arxiv_code}")
            pu.download_s3_file(
                tweet_content.arxiv_code,
                bucket_name="arxiv-first-page",
                prefix="data",
                format="png"
            )
            
        ## Get analyzed image if available
        analyzed_image = vs.analyze_paper_images(tweet_content.arxiv_code, model="claude-3-5-sonnet-20241022")
        if analyzed_image:
            images.analyzed_image = os.path.join(DATA_PATH, "arxiv_md", tweet_content.arxiv_code, analyzed_image)
            if not os.path.exists(images.analyzed_image):
                logger.warning(f"Selected image {analyzed_image} not found")
                images.analyzed_image = None
                
    elif tweet_content.tweet_type == "punchline" and tweet_content.selected_image:
        ## Use selected image for punchlines
        image_path = os.path.join(DATA_PATH, "arxiv_md", tweet_content.arxiv_code, 
                                tweet_content.selected_image.split("/")[-1])
        
        if os.path.exists(image_path):
            images.tweet_page = image_path
            logger.info(f"Using selected image: {image_path}")
        else:
            logger.warning(f"Selected image not found, falling back to first page")
            images.tweet_page = f"{PAGE_PATH}/{tweet_content.arxiv_code}.png"
            if not os.path.exists(images.tweet_page):
                pu.download_s3_file(
                    tweet_content.arxiv_code,
                    bucket_name="arxiv-first-page",
                    prefix="data",
                    format="png"
                )
    
    return images

def main():
    """Generate a weekly review of highlights and takeaways from papers."""
    logger.info("Starting tweet generation process.")
    
    ## Select tweet type
    rand_val = random.random()
    tweet_type = (
        "fable" if rand_val < 0.05
        else "punchline" if rand_val < 0.3
        else "insight_v5"
    )
    logger.info(f"Selected tweet type: {tweet_type}")

    ## Select paper and gather details
    arxiv_code, paper_details = select_paper(logger)
    logger.info(f"Selected paper: {arxiv_code}")

    ## Prepare basic tweet information with appropriate content
    tweet_facts, post_tweet, publish_date = prepare_tweet_facts(arxiv_code, paper_details, tweet_type, logger)

    ## Generate tweet content based on type
    tweet_content = generate_tweet_content(tweet_type, tweet_facts, arxiv_code, publish_date, logger)
    logger.info(f"Generated content: {tweet_content.content}")

    ## Prepare images based on tweet type
    images = prepare_tweet_images(tweet_content, logger)
    
    ## Log image paths
    logger.info("Sending tweet with the following images:")
    logger.info(f"- tweet_image_path: {images.tweet_image}")
    logger.info(f"- tweet_page_path: {images.tweet_page}")
    logger.info(f"- analyzed_image_path: {images.analyzed_image}")

    ## Check for author's tweet
    author_tweet = tweet.find_paper_author_tweet(arxiv_code, logger)
    if author_tweet:
        logger.info(f"Found author tweet: {author_tweet['text']}")

    ## Send tweet
    tweet_success = tweet.send_tweet(
        tweet_content=tweet_content.content,
        tweet_image_path=images.tweet_image,
        post_tweet=post_tweet,
        logger=logger,
        author_tweet=author_tweet,
        tweet_page_path=images.tweet_page,
        analyzed_image_path=images.analyzed_image
    )

    ## Store results
    if tweet_success:
        db.insert_tweet_review(
            arxiv_code,
            tweet_content.content,
            datetime.datetime.now(),
            tweet_type,
            rejected=False,
        )
        em.send_email_alert(tweet_content.content, arxiv_code)
        logger.info("Tweet stored in database and email alert sent.")
    else:
        logger.error("Failed to send tweet.")

if __name__ == "__main__":
    main()
