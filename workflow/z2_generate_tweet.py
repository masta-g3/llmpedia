#!/usr/bin/env python3

import datetime
import os
import sys
import random
import base64
from typing import Optional, List, Tuple
import logging
from pydantic import BaseModel, Field
import yaml
import time
import re
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium import webdriver

from dotenv import load_dotenv

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
import utils.db.db_utils as db_utils
import utils.db.paper_db as paper_db
import utils.db.tweet_db as tweet_db
import utils.tweet as tweet
from utils.tweet import *
import utils.app_utils as au

logger = setup_logger(__name__, "z2_generate_tweet.log")

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

####################
## DATA MODELS    ##
####################


class TweetImageConfig(BaseModel):
    """Configuration for a tweet's image source."""

    source_type: str = Field(
        ...,
        description="Type of image source: 'path' or 'function'",
        pattern="^(path|function)$",
    )
    source: str = Field(
        ..., description="Path template or function name that generates image path"
    )
    description: str = Field(
        default="", description="Description of what this image represents"
    )


class TweetContentConfig(BaseModel):
    """Configuration for a tweet's text content source."""

    content_type: str = Field(
        ...,
        description="Type of content: 'text' or 'function'",
        pattern="^(text|function)$",
    )
    content: str = Field(
        ..., description="Static text or function name that generates text"
    )
    description: str = Field(
        default="", description="Description of what this content represents"
    )


class TweetConfig(BaseModel):
    """Configuration for a single tweet in a thread."""

    content: TweetContentConfig = Field(
        ..., description="The tweet's text content configuration"
    )
    images: Optional[List[TweetImageConfig]] = Field(
        default=None, description="List of image configurations for this tweet"
    )
    position: int = Field(
        ..., description="Position of this tweet in the thread (0-based)", ge=0
    )


class TweetThreadConfig(BaseModel):
    """Configuration for a complete tweet thread type."""

    name: str = Field(..., description="Name of this tweet thread configuration")
    description: str = Field(
        ..., description="Description of what this tweet thread does"
    )
    tweets: List[TweetConfig] = Field(..., description="List of tweets in this thread")


class Tweet(BaseModel):
    """A single tweet with its actual content and media."""

    content: str = Field(..., description="The actual text content of the tweet")
    images: Optional[List[str]] = Field(
        default=None, description="List of actual image file paths"
    )
    position: int = Field(..., description="Position in the thread (0-based)", ge=0)


class TweetThread(BaseModel):
    """A complete tweet thread ready to be sent."""

    arxiv_code: str = Field(
        ..., description="Arxiv code of the paper being tweeted about"
    )
    tweet_type: str = Field(..., description="Type of tweet thread (e.g. 'insight_v5')")
    tweets: List[Tweet] = Field(..., description="List of tweets in the thread")
    created_at: datetime.datetime = Field(
        default_factory=datetime.datetime.now,
        description="When this tweet thread was generated",
    )


##########################
## CONTENT GENERATION   ##
##########################


def generate_insight_content(arxiv_code: str, paper_details: dict) -> str:
    """Generate main insight content for a paper."""
    tweet_facts = prepare_tweet_facts(arxiv_code, "insight_v5")
    tweet_obj = vs.write_tweet(
        tweet_facts=tweet_facts,
        tweet_type="insight_v5",
        model="claude-3-5-sonnet-20241022",
        temperature=0.9,
    )
    publish_date = paper_details["published"].strftime("%b %d, %Y")
    return bold(tweet_obj.edited_tweet, publish_date)


def generate_punchline_content(arxiv_code: str, paper_details: dict) -> str:
    """Generate punchline content for a paper."""
    markdown_content, success = au.get_paper_markdown(arxiv_code)
    if not success:
        raise Exception(f"Could not load markdown content for {arxiv_code}")

    punchline_obj = vs.write_punchline_tweet(
        markdown_content=markdown_content,
        paper_title=paper_details["title"],
        model="claude-3-5-sonnet-20241022",
        temperature=0.9,
    )
    publish_date = paper_details["published"].strftime("%b %d, %Y")
    return bold(punchline_obj.line, publish_date)


def generate_fable_content(arxiv_code: str, paper_details: dict) -> str:
    """Generate fable content for a paper."""
    tweet_facts = prepare_tweet_facts(arxiv_code, "fable")
    with open(f"{IMG_PATH}/{arxiv_code}.png", "rb") as img_file:
        b64_image = base64.b64encode(img_file.read()).decode("utf-8")

    content = vs.write_fable(
        tweet_facts=tweet_facts,
        image_data=b64_image,
        model="claude-3-5-sonnet-20241022",
        temperature=0.9,
    )
    publish_date = paper_details["published"].strftime("%b %d, %Y")
    return bold(content, publish_date)


def generate_links_content(arxiv_code: str, paper_details: dict) -> str:
    """Generate links content for a paper."""
    content = [
        f"arxiv link: https://arxiv.org/abs/{arxiv_code}",
        f"llmpedia link: https://llmpedia.streamlit.app/?arxiv_code={arxiv_code}",
    ]

    # Add repository link if available
    repo_df = paper_db.load_repositories(arxiv_code)
    if not repo_df.empty and repo_df["repo_url"].values[0]:
        content.append(f"repo: {repo_df['repo_url'].values[0]}")

    return "\n".join(content)


def generate_author_tweet(arxiv_code: str, paper_details: dict) -> str:
    """Generate author tweet content."""
    author_tweet = tweet.find_paper_author_tweet(arxiv_code)
    if author_tweet:
        return f"related discussion: {author_tweet['link']}"
    return ""


##########################
## IMAGE HANDLING       ##
##########################


def get_art_image(arxiv_code: str, paper_details: dict) -> str:
    """Get the art image path for a paper."""
    image_path = f"{IMG_PATH}/{arxiv_code}.png"
    if not os.path.exists(image_path):
        logger.info(f"Downloading art image for {arxiv_code}")
        pu.download_s3_file(
            arxiv_code, bucket_name="arxiv-art", prefix="data", format="png"
        )
    return image_path


def get_first_page(arxiv_code: str, paper_details: dict) -> str:
    """Get the first page image path for a paper."""
    image_path = f"{PAGE_PATH}/{arxiv_code}.png"
    if not os.path.exists(image_path):
        logger.info(f"Downloading first page for {arxiv_code}")
        pu.download_s3_file(
            arxiv_code, bucket_name="arxiv-first-page", prefix="data", format="png"
        )
    return image_path


def select_punchline_image(arxiv_code: str, paper_details: dict) -> str:
    """Select the most relevant image for a punchline tweet."""
    markdown_content, success = au.get_paper_markdown(arxiv_code)
    if not success:
        logger.warning(
            f"Could not load markdown content for {arxiv_code}, falling back to first page"
        )
        return get_first_page(arxiv_code, paper_details)

    punchline_obj = vs.write_punchline_tweet(
        markdown_content=markdown_content,
        paper_title=paper_details["title"],
        model="claude-3-5-sonnet-20241022",
        temperature=0.9,
    )

    if punchline_obj.image:
        image_path = os.path.join(
            DATA_PATH, "arxiv_md", arxiv_code, punchline_obj.image
        )
        if os.path.exists(image_path):
            return image_path
        logger.warning(
            f"Selected image {punchline_obj.image} not found, falling back to first page"
        )

    return get_first_page(arxiv_code, paper_details)


def get_analyzed_image(arxiv_code: str, paper_details: dict) -> Optional[str]:
    """Get an analyzed image from the paper if available."""
    analyzed_image = vs.analyze_paper_images(
        arxiv_code, model="claude-3-5-sonnet-20241022"
    )
    if analyzed_image:
        image_path = os.path.join(DATA_PATH, "arxiv_md", arxiv_code, analyzed_image)
        if os.path.exists(image_path):
            return image_path
        logger.warning(f"Selected analyzed image {analyzed_image} not found")
    return None


##########################
## HELPER FUNCTIONS     ##
##########################


def prepare_tweet_facts(arxiv_code: str, tweet_type: str) -> str:
    """Prepare tweet facts based on tweet type."""
    if tweet_type == "fable":
        markdown_content, success = au.get_paper_markdown(arxiv_code)
        if not success:
            raise Exception(f"Could not load markdown content for {arxiv_code}")
        return markdown_content
    else:
        paper_details = paper_db.load_arxiv(arxiv_code)
        title = paper_details["title"].iloc[0]
        author = paper_details["authors"].iloc[0]
        publish_date = paper_details["published"].iloc[0].strftime("%b %d, %Y")
        content = paper_db.get_extended_notes(arxiv_code, expected_tokens=4500)
        return f"**Title: {title}**\n**Authors: {author}**\n{content}"


def select_paper(logger: logging.Logger) -> str:
    """Select a paper to tweet about."""
    candidates = fetch_candidate_papers(logger)
    arxiv_code = choose_interesting_paper(candidates, logger)
    logger.info(f"Selected paper: {arxiv_code}")
    return arxiv_code


def fetch_candidate_papers(logger: logging.Logger) -> List[str]:
    """Fetch candidates from S3 excluding already reviewed papers."""
    arxiv_codes = pu.list_s3_files("arxiv-art", strip_extension=True)
    done_codes = db_utils.get_arxiv_id_list("tweet_reviews")
    candidates = list(set(arxiv_codes) - set(done_codes))
    candidates = sorted(candidates)[-250:]  # Limit to most recent 250
    logger.info(f"Found {len(candidates)} recent papers to consider")
    return candidates


def choose_interesting_paper(candidates: List[str], logger: logging.Logger) -> str:
    """Select the most interesting candidate paper based on citations."""
    citations_df = paper_db.load_citations()
    citations_df = citations_df[citations_df.index.isin(candidates)]
    citations_df["citation_count"] = citations_df["citation_count"].fillna(1) + 1
    citations_df["weight"] = (
        citations_df["citation_count"] / citations_df["citation_count"].sum()
    ) ** 0.5
    citations_df["weight"] = citations_df["weight"] / citations_df["weight"].sum()

    candidate_arxiv_codes = random.choices(
        citations_df.index, weights=citations_df["weight"], k=25
    )
    logger.info(
        f"Selected {len(candidate_arxiv_codes)} candidate papers based on citations"
    )

    candidate_abstracts = paper_db.get_recursive_summary(candidate_arxiv_codes)
    abstracts_str = "\n".join(
        [
            f"<{code}>\n{abstract}\n</{code}>\n"
            for code, abstract in candidate_abstracts.items()
        ]
    )
    logger.info("Selecting most interesting paper...")
    arxiv_code = vs.select_most_interesting_paper(
        abstracts_str, model="claude-3-5-sonnet-20241022"
    )
    logger.info(f"Candidate selected: {arxiv_code}")
    return arxiv_code


##########################
## TWEET GENERATOR      ##
##########################


class TweetGenerator:
    """Main class for generating tweets based on configurations."""

    def __init__(self, config_path: str):
        """Initialize with path to config YAML.

        Args:
            config_path: Path to the tweet types configuration YAML file
        """
        with open(config_path) as f:
            raw_configs = yaml.safe_load(f)["tweet_types"]

        # Convert raw configs to TweetThreadConfig objects
        self.configs = {
            tweet_type: TweetThreadConfig(**config)
            for tweet_type, config in raw_configs.items()
        }

        # Map of content generation functions
        self.content_generators = {
            "generate_insight_content": generate_insight_content,
            "generate_punchline_content": generate_punchline_content,
            "generate_fable_content": generate_fable_content,
            "generate_links_content": generate_links_content,
            "generate_author_tweet": generate_author_tweet,
        }

        # Map of image generation functions
        self.image_generators = {
            "get_art_image": get_art_image,
            "get_first_page": get_first_page,
            "select_punchline_image": select_punchline_image,
            "get_analyzed_image": get_analyzed_image,
        }

        # Validate configurations
        self._validate_configs()

    def _validate_configs(self):
        """Validate that all configured functions exist."""
        for tweet_type, config in self.configs.items():
            # Config is already validated as TweetThreadConfig in __init__
            for tweet in config.tweets:
                # Validate content generator
                if tweet.content.content_type == "function":
                    func_name = tweet.content.content
                    if func_name not in self.content_generators:
                        raise ValueError(
                            f"Unknown content generator function '{func_name}' in tweet type '{tweet_type}'"
                        )

                # Validate image generators
                if tweet.images:
                    for img in tweet.images:
                        if img.source_type == "function":
                            func_name = img.source
                            if func_name not in self.image_generators:
                                raise ValueError(
                                    f"Unknown image generator function '{func_name}' in tweet type '{tweet_type}'"
                                )

    def generate_tweet_thread(self, tweet_type: str, arxiv_code: str) -> TweetThread:
        """Generate a complete tweet thread based on configuration.

        Args:
            tweet_type: Type of tweet thread to generate (must exist in config)
            arxiv_code: Arxiv code of the paper to tweet about

        Returns:
            TweetThread object containing all generated tweets

        Raises:
            ValueError: If tweet type doesn't exist or configuration is invalid
        """
        if tweet_type not in self.configs:
            raise ValueError(f"Unknown tweet type: {tweet_type}")

        config = self.configs[tweet_type]  # Now a TweetThreadConfig object
        paper_details = paper_db.load_arxiv(arxiv_code).iloc[0].to_dict()
        tweets = []

        for tweet_config in config.tweets:  # Now accessing validated config objects
            # Generate content
            content_config = tweet_config.content
            if content_config.content_type == "function":
                func = self.content_generators[content_config.content]
                content = func(arxiv_code, paper_details)
            else:
                content = content_config.content

            # Skip empty tweets (like optional author tweets)
            if not content:
                continue

            # Generate images
            images = None
            if tweet_config.images:
                images = []
                for img_config in tweet_config.images:
                    if img_config.source_type == "function":
                        func = self.image_generators[img_config.source]
                        img_path = func(arxiv_code, paper_details)
                    else:
                        img_path = img_config.source.format(arxiv_code=arxiv_code)

                    if img_path:  # Only add if path exists
                        images.append(img_path)

                if not images:  # If no images were successfully generated
                    images = None

            tweets.append(
                Tweet(content=content, images=images, position=tweet_config.position)
            )

        return TweetThread(
            arxiv_code=arxiv_code,
            tweet_type=tweet_type,
            tweets=sorted(tweets, key=lambda t: t.position),  # Ensure correct order
        )


##########################
## TWEET VERIFICATION   ##
##########################


def verify_tweet_structure(
    thread: TweetThread, config: TweetThreadConfig
) -> Tuple[bool, str]:
    """Verify that the tweet thread structure matches its configuration."""
    try:
        # Check number of tweets
        if len(thread.tweets) != len(config.tweets):
            return (
                False,
                f"Tweet count mismatch: expected {len(config.tweets)}, got {len(thread.tweets)}",
            )

        # Check each tweet's position and structure
        for tweet, config_tweet in zip(thread.tweets, config.tweets):
            # Verify position
            if tweet.position != config_tweet.position:
                return (
                    False,
                    f"Position mismatch in tweet {tweet.position}: expected {config_tweet.position}",
                )

            # Verify content presence
            if not tweet.content:
                return False, f"Missing content in tweet {tweet.position}"

            # Verify image count matches config
            expected_images = len(config_tweet.images or [])
            actual_images = len(tweet.images or [])
            if expected_images != actual_images:
                return (
                    False,
                    f"Image count mismatch in tweet {tweet.position}: expected {expected_images}, got {actual_images}",
                )

        return True, "Thread structure verified successfully"

    except Exception as e:
        return False, f"Error verifying thread structure: {str(e)}"


def verify_tweet_content(tweet: Tweet, config: TweetConfig) -> Tuple[bool, str]:
    """Verify a single tweet's content against its configuration.

    Args:
        tweet: Single tweet to verify
        config: Configuration for this tweet

    Returns:
        Tuple of (success, error_message)
    """
    try:
        # Check for required URLs in link tweets
        if "generate_links_content" in config.content.content:
            required_urls = ["arxiv.org", "llmpedia.streamlit.app"]
            for url in required_urls:
                if url not in tweet.content:
                    return False, f"Missing required URL {url} in links tweet"

        # Verify image paths exist
        if tweet.images:
            for idx, image_path in enumerate(tweet.images):
                if not os.path.exists(image_path):
                    return False, f"Image {idx+1} not found: {image_path}"

        return True, "Tweet content verified successfully"

    except Exception as e:
        return False, f"Error verifying tweet content: {str(e)}"


def verify_tweet_ui_elements(
    driver: webdriver.Firefox,
    tweet_idx: int,
    expected_image_count: int,
    logger: Optional[logging.Logger] = None,
) -> Tuple[bool, str]:
    """Verify UI elements for a single tweet in the thread."""
    logger = logger or get_console_logger()

    try:
        # Verify tweet textarea
        textarea_selectors = [
            f"//div[@contenteditable='true' and @data-testid='tweetTextarea_{tweet_idx}']",
            "//div[@contenteditable='true' and @role='textbox']",
            "//div[@data-testid='tweetTextarea_0']",
        ]

        textarea_found = False
        for selector in textarea_selectors:
            try:
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.XPATH, selector))
                )
                textarea_found = True
                break
            except:
                continue

        if not textarea_found:
            return False, f"Could not find textarea for tweet {tweet_idx}"

        # Verify image upload indicators if images are expected
        if expected_image_count > 0:
            try:
                WebDriverWait(driver, 10).until(
                    lambda d: len(
                        d.find_elements(
                            By.XPATH, "//button[@aria-label='Remove media']"
                        )
                    )
                    == expected_image_count
                )
            except:
                return False, f"Image upload indicators not found for tweet {tweet_idx}"

        # Verify thread connection for non-first tweets
        if tweet_idx > 0:
            try:
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located(
                        (By.XPATH, "//div[@aria-label='Thread']")
                    )
                )
            except:
                return False, f"Thread connection not found for tweet {tweet_idx}"

        return True, "UI elements verified successfully"

    except Exception as e:
        return False, f"Error verifying UI elements: {str(e)}"


def verify_tweet_thread(
    thread: TweetThread,
    config: TweetThreadConfig,
    driver: webdriver.Firefox,
    logger: Optional[logging.Logger] = None,
    max_retries: int = 3,
    retry_delay: float = 1.0,
) -> Tuple[bool, str]:
    """Verify a complete tweet thread against its configuration and UI state."""
    logger = logger or get_console_logger()
    logger.info("Starting tweet thread verification")

    try:
        # Verify thread structure
        structure_ok, structure_msg = verify_tweet_structure(thread, config)
        if not structure_ok:
            return False, f"Thread structure verification failed: {structure_msg}"

        # Verify each tweet's content
        for tweet, tweet_config in zip(thread.tweets, config.tweets):
            content_ok, content_msg = verify_tweet_content(tweet, tweet_config)
            if not content_ok:
                return False, f"Tweet content verification failed: {content_msg}"

        # Verify UI elements with retry
        for tweet_idx, tweet in enumerate(thread.tweets):
            expected_images = len(tweet.images or [])
            retry_count = 0
            while retry_count < max_retries:
                ui_ok, ui_msg = verify_tweet_ui_elements(
                    driver, tweet_idx, expected_images, logger
                )
                if ui_ok:
                    break

                retry_count += 1
                if retry_count < max_retries:
                    wait_time = retry_delay * (2**retry_count)  # Exponential backoff
                    logger.warning(
                        f"UI verification retry {retry_count} for tweet {tweet_idx}. Waiting {wait_time}s..."
                    )
                    time.sleep(wait_time)
                else:
                    return (
                        False,
                        f"UI verification failed after {max_retries} retries: {ui_msg}",
                    )

        logger.info("Tweet thread verification completed successfully")
        return True, "Tweet thread verified successfully"

    except Exception as e:
        return False, f"Error during thread verification: {str(e)}"


##########################
## TWEET SENDING        ##
##########################


def send_tweet2(
    tweet_thread: TweetThread,
    logger: Optional[logging.Logger] = None,
    verify: bool = False,
) -> bool:
    """Send a complete tweet thread.

    Args:
        tweet_thread: TweetThread object containing all tweets and their media
        logger: Optional logger for tracking progress
        verify: Whether to verify tweet content before sending

    Returns:
        bool: Whether the complete thread was sent successfully
    """
    logger = logger or get_console_logger()
    logger.info(f"Starting to send tweet thread of type: {tweet_thread.tweet_type}")

    try:
        # Setup browser and login
        driver = setup_browser(logger, headless=False)
        login_twitter(driver, logger)

        # Verify thread if requested
        # if verify:
        #     logger.info("Verifying tweet thread before sending")
        #     generator = TweetGenerator("config/tweet_types.yaml")
        #     config = generator.configs[tweet_thread.tweet_type]
        #     success, msg = verify_tweet_thread(tweet_thread, config, driver, logger)
        #     if not success:
        #         logger.error(f"Tweet thread verification failed: {msg}")
        #         return False

        # Start new tweet
        logger.info("Starting new tweet thread")
        tweet_button = WebDriverWait(driver, 60).until(
            EC.element_to_be_clickable((By.XPATH, '//a[@aria-label="Post"]'))
        )
        tweet_button.click()
        time.sleep(2)  # Give UI time to stabilize

        # Process each tweet in the thread
        for i, tweet_data in enumerate(tweet_thread.tweets):
            logger.info(f"Processing tweet {i+1}/{len(tweet_thread.tweets)}")

            try:
                # If not first tweet, add new tweet to thread
                if i > 0:
                    logger.info("Adding reply to thread")
                    tweet_reply_btn = WebDriverWait(driver, 60).until(
                        EC.element_to_be_clickable(
                            (By.XPATH, "//button[@data-testid='addButton']")
                        )
                    )
                    tweet_reply_btn.click()
                    time.sleep(2)  # Give UI time to stabilize

                # Find and enter tweet content
                # More robust textarea selection using multiple possible selectors
                textarea_selectors = [
                    f"//div[@contenteditable='true' and @data-testid='tweetTextarea_{i}']",
                    "//div[@contenteditable='true' and @role='textbox']",
                    "//div[@data-testid='tweetTextarea_0']",  # Fallback for first tweet
                ]

                tweet_box = None
                for selector in textarea_selectors:
                    try:
                        tweet_box = WebDriverWait(driver, 10).until(
                            EC.element_to_be_clickable((By.XPATH, selector))
                        )
                        break
                    except:
                        continue

                if not tweet_box:
                    raise Exception("Could not find tweet textarea")

                tweet_box.send_keys(tweet_data.content)
                time.sleep(1)  # Let content settle

                # Handle images if present
                if tweet_data.images:
                    logger.info(
                        f"Processing {len(tweet_data.images)} images for tweet {i+1}"
                    )
                    try:
                        upload_input = driver.find_element(
                            By.XPATH,
                            '//input[@accept="image/jpeg,image/png,image/webp,image/gif,video/mp4,video/quicktime"]',
                        )

                        for idx, image_path in enumerate(tweet_data.images, 1):
                            if not os.path.exists(image_path):
                                logger.warning(f"Image not found: {image_path}")
                                continue

                            logger.info(
                                f"Uploading image {idx}/{len(tweet_data.images)}: {image_path}"
                            )
                            upload_input.send_keys(image_path)

                            # Wait for upload with timeout
                            try:
                                WebDriverWait(driver, 60).until(
                                    EC.presence_of_element_located(
                                        (
                                            By.XPATH,
                                            f"(//button[@aria-label='Remove media'])[{idx}]",
                                        )
                                    )
                                )
                            except Exception as e:
                                logger.error(f"Failed to upload image {idx}: {str(e)}")
                                return False

                    except Exception as e:
                        logger.error(f"Error handling images: {str(e)}")
                        return False

            except Exception as e:
                logger.error(f"Error processing tweet {i+1}: {str(e)}")
                return False

        # Send the complete thread
        logger.info("Preparing to send tweet thread")
        try:
            tweet_all_button = WebDriverWait(driver, 30).until(
                EC.element_to_be_clickable(
                    (
                        By.XPATH,
                        "//button[@data-testid='tweetButton']//span[contains(text(), 'Post all')]",
                    )
                )
            )

            # Visual confirmation and final send
            # driver.execute_script(
            #     """
            #     arguments[0].style.backgroundColor = '#ff0';
            #     arguments[0].style.border = '2px solid red';
            #     """,
            #     tweet_all_button,
            # )
            time.sleep(5)  # Final verification pause
            tweet_all_button.click()
            time.sleep(5)  # Wait for send to complete

            logger.info("Tweet thread sent successfully")
            return True

        except Exception as e:
            logger.error(f"Error sending tweet thread: {str(e)}")
            return False

    except Exception as e:
        logger.error(f"Unexpected error in send_tweet2: {str(e)}")
        return False

    finally:
        if "driver" in locals():
            driver.quit()
            logger.info("Browser closed")


def main():
    """Main function to generate and send tweets."""
    logger.info("Starting tweet generation process")

    # Select tweet type
    rand_val = random.random()
    tweet_type = (
        "fable" if rand_val < 0.0 else "punchline" if rand_val < 0.7 else "insight_v5"
    )
    logger.info(f"Selected tweet type: {tweet_type}")

    # Select paper
    arxiv_code = select_paper(logger)

    try:
        # Generate tweet thread
        generator = TweetGenerator("config/tweet_types.yaml")
        thread = generator.generate_tweet_thread(tweet_type, arxiv_code)

        # Send tweets
        success = send_tweet2(thread, logger)

        if success:
            tweet_db.insert_tweet_review(
                arxiv_code,
                thread.tweets[0].content,  # Store main tweet content
                datetime.datetime.now(),
                tweet_type,
                rejected=False,
            )
            em.send_email_alert(thread.tweets[0].content, arxiv_code)
            logger.info("Tweet thread stored in database and email alert sent")
        else:
            logger.error("Failed to send tweet thread")

    except Exception as e:
        logger.error(f"Error generating/sending tweet thread: {str(e)}")
        raise


if __name__ == "__main__":
    main()
