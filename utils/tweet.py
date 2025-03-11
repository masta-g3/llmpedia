import os, sys
import time
import random
import datetime
from typing import Tuple, List, Iterator, Optional, Union, Dict, Any
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.firefox.service import Service as FirefoxService
from dotenv import load_dotenv
from selenium.common.exceptions import TimeoutException
import logging
from urllib.parse import quote
from pydantic import BaseModel, Field
import utils.vector_store as vs
import utils.db.paper_db as paper_db
from utils.logging_utils import get_console_logger
import re

load_dotenv()

PROJECT_PATH = os.getenv("PROJECT_PATH", "/app")
sys.path.append(PROJECT_PATH)

url = "https://x.com/"
USERNAME = os.getenv("TWITTER_EMAIL")
PASSWORD = os.getenv("TWITTER_PASSWORD")
PHONE = os.getenv("TWITTER_PHONE")

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
    arxiv_code: Optional[str] = Field(
        default=None, description="Arxiv code of the paper being tweeted about"
    )
    tweet_type: str = Field(..., description="Type of tweet thread (e.g. 'insight_v5', 'daily_update')")
    tweets: List[Tweet] = Field(..., description="List of tweets in the thread")
    created_at: datetime.datetime = Field(
        default_factory=datetime.datetime.now,
        description="When this tweet thread was generated",
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional metadata for the tweet thread"
    )

    @classmethod
    def create_simple_tweet(cls, content: str, images: Optional[List[str]] = None, 
                           tweet_type: str = "simple", metadata: Optional[Dict[str, Any]] = None) -> "TweetThread":
        """Create a simple tweet thread with a single tweet.
        
        Args:
            content: The text content of the tweet
            images: Optional list of image paths
            tweet_type: Type identifier for the tweet
            metadata: Optional additional metadata
            
        Returns:
            A TweetThread object with a single tweet
        """
        tweet = Tweet(content=content, images=images, position=0)
        return cls(
            tweet_type=tweet_type,
            tweets=[tweet],
            metadata=metadata
        )

def boldify(text):
    chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    bold_chars = "𝗔𝗕𝗖𝗗𝗘𝗙𝗚𝗛𝗜𝗝𝗞𝗟𝗠𝗡𝗢𝗣𝗤𝗥𝗦𝗧𝗨𝗩𝗪𝗫𝗬𝗭𝗮𝗯𝗰𝗱𝗲𝗳𝗴𝗵𝗶𝗷𝗸𝗹𝗺𝗻𝗼𝗽𝗾𝗿𝘀𝘁𝘂𝘃𝘄𝘅𝘆𝘇𝟬𝟭𝟮𝟯𝟰𝟱𝟲𝟳𝟴𝟵"
    bolded_text = ""
    for character in text:
        if character in chars:
            bolded_text += bold_chars[chars.index(character)]
        else:
            bolded_text += character
    return bolded_text

def bold(input_text, extra_str):
    """Format text with bold and italic characters."""
    chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    bold_italic_chars = "𝘼𝘽𝘾𝘿𝙀𝙁𝙂𝙃𝙄𝙅𝙆𝙇𝙈𝙉𝙊𝙋𝙌𝙍𝙎𝙏𝙐𝙑𝙒𝙓𝙔𝙕𝙖𝙗𝙘𝙙𝙚𝙛𝙜𝙝𝙞𝙟𝙠𝙡𝙢𝙣𝙤𝙥𝙦𝙧𝙨𝙩𝙪𝙫𝙬𝙭𝙮𝙯𝟬𝟭𝟮𝟯𝟰𝟱𝟲𝟳𝟴𝟵"

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


def setup_browser(logger: logging.Logger, headless: bool = True):
    logger.info("Setting up browser")

    firefox_options = FirefoxOptions()

    # Set a proper user agent
    firefox_options.set_preference(
        "general.useragent.override",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    )

    # Additional preferences to make the browser appear more like a regular user
    firefox_options.set_preference("dom.webdriver.enabled", False)
    firefox_options.set_preference("useAutomationExtension", False)
    firefox_options.set_preference("privacy.trackingprotection.enabled", False)

    # Enable JavaScript and other important features
    firefox_options.set_preference("javascript.enabled", True)
    firefox_options.set_preference("dom.webnotifications.enabled", False)
    firefox_options.set_preference("network.http.connection-timeout", 60)
    firefox_options.set_preference("page.load.timeout", 60000)

    # Additional preferences to avoid detection
    firefox_options.set_preference("general.platform.override", "Win32")
    firefox_options.set_preference("general.appversion.override", "5.0 (Windows)")
    firefox_options.set_preference(
        "general.oscpu.override", "Windows NT 10.0; Win64; x64"
    )

    # Set proper window size
    firefox_options.add_argument("--width=1920")
    firefox_options.add_argument("--height=1080")
    if headless:
        firefox_options.add_argument("--headless")
    firefox_options.add_argument("--no-sandbox")
    firefox_options.add_argument("--disable-dev-shm-usage")

    try:
        service = FirefoxService()
        driver = webdriver.Firefox(options=firefox_options, service=service)
        driver.set_page_load_timeout(60)  # Increased timeout
        driver.set_script_timeout(60)  # Added script timeout
        driver.implicitly_wait(20)  # Added implicit wait
        driver.set_window_size(1920, 1080)

        # Execute JavaScript to modify navigator properties
        driver.execute_script(
            """
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined
            });
            Object.defineProperty(navigator, 'languages', {
                get: () => ['en-US', 'en']
            });
            Object.defineProperty(navigator, 'plugins', {
                get: () => [1, 2, 3, 4, 5]
            });
        """
        )

    except Exception as e:
        logger.error(f"Failed to create driver with default service: {str(e)}")
        try:
            geckodriver_path = os.getenv(
                "GECKODRIVER_PATH", "/usr/local/bin/geckodriver"
            )
            service = FirefoxService(executable_path=geckodriver_path)
            driver = webdriver.Firefox(options=firefox_options, service=service)
        except Exception as e:
            logger.error(
                f"Failed to create driver with explicit geckodriver path: {str(e)}"
            )
            raise

    logger.info("Browser setup complete")
    return driver


def login_twitter(driver: webdriver.Firefox, logger: logging.Logger):
    """Login to Twitter within any page of its domain."""
    logger.info("Attempting to log in to Twitter")

    max_retries = 3
    retry_count = 0

    while retry_count < max_retries:
        try:
            # Clear cookies and cache before attempting login
            driver.delete_all_cookies()

            driver.get(
                "https://twitter.com/login"
            )  # Changed to twitter.com instead of x.com
            logger.info("Navigation completed")

            # Wait longer for initial page load
            time.sleep(10)

            # Log debugging information
            # logger.info("Current URL after navigation: " + driver.current_url)
            logger.info("Page title: " + driver.title)

            # Try multiple selectors for the username field
            username_selectors = [
                (By.CSS_SELECTOR, 'input[autocomplete="username"]'),
                (By.NAME, "text"),
                (By.CSS_SELECTOR, 'input[name="text"]'),
                (By.CSS_SELECTOR, 'input[type="text"]'),
            ]

            username_field = None
            for selector in username_selectors:
                try:
                    username_field = WebDriverWait(driver, 20).until(
                        EC.presence_of_element_located(selector)
                    )
                    if username_field.is_displayed():
                        break
                except:
                    continue

            if not username_field:
                raise Exception("Could not find username field")

            # Clear field and enter username
            username_field.clear()
            username_field.send_keys(USERNAME)
            time.sleep(2)
            username_field.send_keys(Keys.RETURN)
            time.sleep(5)

            # Handle phone verification if needed
            try:
                identifier_field = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.NAME, "text"))
                )
                identifier_field.send_keys(PHONE)
                time.sleep(2)
                identifier_field.send_keys(Keys.RETURN)
                time.sleep(5)
            except:
                logger.info("Phone verification not required")

            # Enter password
            password_field = WebDriverWait(driver, 20).until(
                EC.presence_of_element_located((By.NAME, "password"))
            )
            password_field.send_keys(PASSWORD)
            time.sleep(2)
            password_field.send_keys(Keys.RETURN)

            # Wait for login to complete
            try:
                WebDriverWait(driver, 30).until(
                    EC.presence_of_element_located(
                        (By.XPATH, '//a[@aria-label="Post"]')
                    )
                )
                logger.info("Successfully logged in to X")
                return
            except:
                logger.warning("Could not find Post button after login")
                raise Exception("Login verification failed")

        except Exception as e:
            logger.error(f"Login attempt {retry_count + 1} failed: {str(e)}")
            retry_count += 1
            if retry_count >= max_retries:
                raise Exception(f"Failed to login after {max_retries} attempts")
            time.sleep(10)  # Wait before retrying

    raise Exception("Login failed after all retries")


def verify_tweet_elements(
    driver: webdriver.Firefox,
    expected_content: str,
    expected_image_count,
    logger: logging.Logger,
) -> Tuple[bool, str]:
    """Verify the presence of expected elements in a tweet composition."""
    logger.info("Verifying tweet elements")

    # Wait for main post element and scroll it into view
    main_post_btn = WebDriverWait(driver, 30).until(
        EC.presence_of_element_located((By.XPATH, '//div[@aria-label="Post text"]'))
    )

    # Scroll element into view and add a small delay
    driver.execute_script("arguments[0].scrollIntoView(true);", main_post_btn)
    time.sleep(1)

    # Try to click using JavaScript
    try:
        driver.execute_script("arguments[0].click();", main_post_btn)
    except Exception as e:
        logger.warning(f"JavaScript click failed: {str(e)}")
        # Fallback to regular click with actions
        from selenium.webdriver.common.action_chains import ActionChains

        actions = ActionChains(driver)
        actions.move_to_element(main_post_btn).click().perform()

    # Check for the correct number of uploaded images
    def correct_image_count(driver):
        remove_buttons = driver.find_elements(
            By.XPATH, "//button[@aria-label='Remove media']"
        )
        return len(remove_buttons) == expected_image_count

    WebDriverWait(driver, 30).until(correct_image_count)

    # Check for main tweet text
    main_tweet_elem = WebDriverWait(driver, 30).until(
        EC.presence_of_element_located(
            (By.XPATH, "//div[@data-testid='tweetTextarea_0']")
        )
    )
    main_tweet_text = main_tweet_elem.text.strip()
    if not main_tweet_text:
        logger.warning("Main tweet text is empty")
        return False, "Main tweet text is empty"
    elif main_tweet_text != expected_content.strip():
        logger.warning("Main tweet text does not match expected content")
        return False, "Main tweet text does not match expected content"

    # Check for post-tweet text
    post_tweet_elem = WebDriverWait(driver, 30).until(
        EC.presence_of_element_located(
            (By.XPATH, "//div[@data-testid='tweetTextarea_1']")
        )
    )
    post_tweet_text = post_tweet_elem.text.strip()
    if not post_tweet_text:
        logger.warning("Post-tweet text is empty")
        return False, "Post-tweet text is empty"

    verification_message = "All elements are present and correct"
    logger.info(f"Tweet element verification result: {verification_message}")
    return True, verification_message


def send_tweet(
    tweet_content: str,
    post_tweet: str | None = None,
    logger: logging.Logger | None = None,
    tweet_image_path: str | None = None,
    author_tweet: dict | None = None,
    tweet_page_path: str | None = None,
    analyzed_image_path: str | None = None,
    verify: bool = True,
    headless: bool = True,
) -> bool:
    """Send a tweet with content and images using Selenium."""

    logger = logger or get_console_logger()
    
    logger.info("Starting tweet sending process")
    driver = setup_browser(logger, headless=headless)
    login_twitter(driver, logger)

    logger.info("Composing tweet")
    # Click the "Post" button to start a new tweet
    tweet_button = WebDriverWait(driver, 60).until(
        EC.element_to_be_clickable((By.XPATH, '//a[@aria-label="Post"]'))
    )
    tweet_button.click()

    # Enter tweet content
    tweet_textarea = WebDriverWait(driver, 60).until(
        EC.presence_of_element_located((By.XPATH, '//div[@aria-label="Post text"]'))
    )
    tweet_textarea.send_keys(tweet_content)

    expected_image_count = 0
    upload_input = driver.find_element(
        By.XPATH,
        '//input[@accept="image/jpeg,image/png,image/webp,image/gif,video/mp4,video/quicktime"]',
    )
    ## ToDo: Replace below with single check at the end for all expected image counts (or trailing test).

    ## Upload first image if provided.
    if tweet_image_path:
        logger.info("Uploading first image")
        expected_image_count += 1
        upload_input.send_keys(tweet_image_path)

        ## Verify first image is uploaded.
        WebDriverWait(driver, 60).until(
            EC.presence_of_element_located(
                (By.XPATH, "(//button[@aria-label='Remove media'])[1]")
            )
        )

        ## Verify first image is uploaded.
        WebDriverWait(driver, 60).until(
            EC.presence_of_element_located(
                (By.XPATH, "(//button[@aria-label='Remove media'])[1]")
            )
        )

    ## Upload second image if provided.
    if tweet_page_path:
        logger.info("Uploading second image")
        upload_input.send_keys(tweet_page_path)
        expected_image_count += 1

        # Verify both images are uploaded
        def correct_image_count(driver):
            remove_buttons = driver.find_elements(
                By.XPATH, "//button[@aria-label='Remove media']"
            )
            return len(remove_buttons) == expected_image_count

        WebDriverWait(driver, 30).until(correct_image_count)

    ## Add image tweet if provided.
    if analyzed_image_path:
        time.sleep(10)
        logger.info("Adding image tweet")
        tweet_reply_btn = WebDriverWait(driver, 60).until(
            EC.element_to_be_clickable((By.XPATH, "//button[@data-testid='addButton']"))
        )
        tweet_reply_btn.click()

        ## Enter image tweet content.
        tweet_box = WebDriverWait(driver, 60).until(
            EC.element_to_be_clickable(
                (
                    By.XPATH,
                    "//div[@contenteditable='true' and @data-testid='tweetTextarea_1']",
                )
            )
        )
        tweet_box.send_keys("Key visualization from the paper 📊")

        ## Upload image.
        upload_input = driver.find_element(
            By.XPATH,
            '//input[@accept="image/jpeg,image/png,image/webp,image/gif,video/mp4,video/quicktime"]',
        )
        upload_input.send_keys(analyzed_image_path)

        # Verify image is uploaded
        WebDriverWait(driver, 60).until(
            EC.presence_of_element_located(
                (By.XPATH, "(//button[@aria-label='Remove media'])[1]")
            )
        )

    ## Add links tweet.
    if post_tweet:
        time.sleep(10)
        logger.info("Adding links tweet")
        tweet_reply_btn = WebDriverWait(driver, 60).until(
            EC.element_to_be_clickable((By.XPATH, "//button[@data-testid='addButton']"))
        )
        tweet_reply_btn.click()

        ## Enter links tweet content.
        tweet_box = WebDriverWait(driver, 60).until(
            EC.element_to_be_clickable(
                (
                    By.XPATH,
                    f"//div[@contenteditable='true' and @data-testid='tweetTextarea_{2 if analyzed_image_path else 1}']",
                )
            )
        )
        tweet_box.send_keys(post_tweet)

    # Add author tweet if provided
    if author_tweet:
        tweet_reply_btn = WebDriverWait(driver, 60).until(
            EC.element_to_be_clickable((By.XPATH, "//button[@data-testid='addButton']"))
        )
        tweet_reply_btn.click()

        tweet_box = WebDriverWait(driver, 60).until(
            EC.element_to_be_clickable(
                (
                    By.XPATH,
                    f"//div[@contenteditable='true' and @data-testid='tweetTextarea_{3 if analyzed_image_path else 2}']",
                )
            )
        )
        tweet_box.send_keys(f"related discussion: {author_tweet['link']}")

    # Verify tweet elements
    if verify:
        elements_verified, verification_message = verify_tweet_elements(
            driver, tweet_content, expected_image_count=expected_image_count, logger=logger
        )
        if not elements_verified:
            logger.error(f"Tweet verification failed: {verification_message}")
            return False

    # Send tweet
    time.sleep(5)
    logger.info("Attempting to send tweet")

    # Find and click the 'Post all' button
    tweet_all_button = WebDriverWait(driver, 30).until(
        EC.element_to_be_clickable(
            (
                By.XPATH,
                "//button[@data-testid='tweetButton']//span[contains(text(), 'Post all')]",
            )
        )
    )

    # Highlight for visual confirmation
    driver.execute_script(
        """
        arguments[0].style.backgroundColor = '#ff0';
        arguments[0].style.border = '2px solid red';
    """,
        tweet_all_button,
    )

    # Sleep before sending the tweet
    sleep_duration = 5
    logger.info(f"Sleeping for {sleep_duration} seconds before sending the tweet...")

    time.sleep(sleep_duration)
    tweet_all_button.click()
    time.sleep(5)
    logger.info("Tweet sent successfully")
    driver.quit()

    return True


def extract_author_tweet_data(tweet_elem, paper_title, paper_authors, logger: Optional[logging.Logger] = None) -> Optional[dict]:
    """Extract tweet data if it's from a paper author."""
    logger = logger or get_console_logger()
    try:
        # Extract tweet text
        tweet_text = tweet_elem.find_element(
            By.CSS_SELECTOR, '[data-testid="tweetText"]'
        ).text

        # Extract username
        username = tweet_elem.find_element(
            By.CSS_SELECTOR, '[data-testid="User-Name"]'
        ).text.split("@")[1].split("·")[0].strip()

        # Check if username matches any author
        if any(username.lower() in author.lower() for author in paper_authors):
            return {"text": tweet_text, "username": username}

    except Exception as e:
        logger.error(f"Error extracting tweet data: {str(e)}")
        return None

    return None

def extract_tweet_data(tweet_elem, logger: Optional[logging.Logger] = None) -> Optional[dict]:
    """Extract all relevant data from a tweet element."""
    logger = logger or get_console_logger()
    try:
        ## Get user info.
        user_name_elem = tweet_elem.find_element(
            By.CSS_SELECTOR, 'div[data-testid="User-Name"]'
        )
        user_name_parts = user_name_elem.text.split("\n")

        # Get tweet text
        tweet_text_elements = tweet_elem.find_elements(
            By.CSS_SELECTOR, 'div[data-testid="tweetText"]'
        )
        if not tweet_text_elements:
            logger.warning("No tweet text elements found")
            return None

        tweet_text = tweet_text_elements[0].text
        if not tweet_text.strip():
            logger.warning("Tweet text is empty")
            return None

        ## Build base tweet data.
        tweet_data = {
            "text": tweet_text,
            "author": user_name_parts[0],
            "username": user_name_parts[1],
            "link": tweet_elem.find_element(
                By.CSS_SELECTOR, 'a[href*="/status/"]'
            ).get_attribute("href"),
        }

        # Try to get timestamp, but don't fail if not found
        try:
            tweet_data["tweet_timestamp"] = tweet_elem.find_element(By.TAG_NAME, "time").get_attribute("datetime")
        except Exception as e:
            tweet_data["tweet_timestamp"] = None
            logger.warning("Error extracting tweet timestamp")

        # Get metrics
        metrics = {
            "reply_count": 0,
            "repost_count": 0,
            "like_count": 0,
            "view_count": 0,
            "bookmark_count": 0,
        }

        try:
            ## Extract metrics from aria-label of the metrics group.
            metrics_group = tweet_elem.find_element(By.CSS_SELECTOR, '[role="group"]')
            metrics_text = metrics_group.get_attribute("aria-label")

            ## Parse metrics from the aria-label text.
            if metrics_text:
                parts = metrics_text.lower().split(",")
                for part in parts:
                    part = part.strip()
                    if "repl" in part:
                        metrics["reply_count"] = int(part.split()[0])
                    elif "repost" in part:
                        metrics["repost_count"] = int(part.split()[0])
                    elif "like" in part:
                        metrics["like_count"] = int(part.split()[0])
                    elif "view" in part:
                        metrics["view_count"] = int(part.split()[0])
                    elif "bookmark" in part:
                        metrics["bookmark_count"] = int(part.split()[0])
        except Exception as e:
            logger.warning(f"Error extracting metrics: {str(e)}")

        ## Check for media.
        has_media = bool(
            tweet_elem.find_elements(
                By.CSS_SELECTOR,
                'div[data-testid="tweetPhoto"], div[data-testid="videoPlayer"]',
            )
        )

        ## Check for verification.
        is_verified = bool(
            user_name_elem.find_elements(
                By.CSS_SELECTOR, 'svg[aria-label="Verified account"]'
            )
        )

        return {
            **tweet_data,
            "has_media": has_media,
            "is_verified": is_verified,
            **metrics,
        }

    except Exception as e:
        logger.warning(f"Error extracting tweet details: {str(e)}")
        return None


def find_paper_author_tweet(arxiv_code: str, logger: Optional[logging.Logger] = None) -> Optional[dict]:
    """Find a tweet from a paper's author about their paper."""
    logger = logger or get_console_logger()
    logger.info(f"Searching for author tweet for {arxiv_code}")
    
    # Get paper details
    paper_details = paper_db.load_arxiv(arxiv_code)
    paper_title = paper_details["title"].iloc[0]
    paper_authors = paper_details["authors"].iloc[0].split(", ")
    
    # Initialize browser
    browser = setup_browser(logger, headless=False)
    login_twitter(browser, logger)
    if not browser:
        logger.error("Failed to initialize browser")
        return None
        
    # Search for paper title
    search_url = f"https://twitter.com/search?q={paper_title}&src=typed_query&f=live"
    browser.get(search_url)
    
    # Initialize variables
    tweets_checked = 0
    max_tweets_to_check = 100
    
    while tweets_checked < max_tweets_to_check:
        WebDriverWait(browser, 30).until(
            EC.presence_of_element_located(
                (By.CSS_SELECTOR, 'article[data-testid="tweet"]')
            )
        )
        tweet_elements = browser.find_elements(
            By.CSS_SELECTOR, 'article[data-testid="tweet"]'
        )

        # Process tweets
        for tweet_elem in tweet_elements:
            tweets_checked += 1
            if tweets_checked > max_tweets_to_check:
                break

            tweet_data = extract_author_tweet_data(
                tweet_elem, paper_title, paper_authors, logger
            )
            if tweet_data:
                logger.info(f"Found author tweet from: {tweet_data['username']}")
                browser.quit()
                return tweet_data

        # Scroll and check progress
        last_height = browser.execute_script(
            "return document.documentElement.scrollHeight"
        )
        browser.execute_script(
            "window.scrollTo(0, document.documentElement.scrollHeight);"
        )
        time.sleep(3)  # Wait for content to load

        new_height = browser.execute_script(
            "return document.documentElement.scrollHeight"
        )
        if new_height == last_height:
            logger.info("No new content loaded, breaking loop")
            break

    browser.quit()
    return None


def collect_llm_tweets(logger: Optional[logging.Logger] = None, max_tweets: int = 50, batch_size: int = 100) -> Iterator[List[dict]]:
    """Collect tweets about LLMs from the Twitter home feed in batches."""
    logger = logger or get_console_logger()
    logger.info("Starting collection of LLM-related tweets")

    browser = setup_browser(logger)
    current_batch = []
    tweets_checked = 0

    ## Login and navigate to home.
    login_twitter(browser, logger)
    browser.get("https://twitter.com/home")
    time.sleep(3) 

    while tweets_checked < max_tweets:
        ## Wait for and get tweets.
        WebDriverWait(browser, 15).until(
            EC.presence_of_element_located(
                (By.CSS_SELECTOR, 'article[data-testid="tweet"]')
            )
        )
        tweet_elements = browser.find_elements(
            By.CSS_SELECTOR, 'article[data-testid="tweet"]'
        )

        ## Process new tweets.
        for tweet_elem in tweet_elements:
            tweets_checked += 1
            if tweets_checked > max_tweets:
                break

            tweet_data = extract_tweet_data(tweet_elem, logger)
            if tweet_data:
                relevance_info = vs.assess_llm_relevance(
                    tweet_text=tweet_data["text"]
                )
                tweet_data["arxiv_code"] = relevance_info.arxiv_code
                if relevance_info.is_llm_related:
                    logger.info(
                        f"Found relevant tweet from: {tweet_data['username']} ({tweets_checked}/{max_tweets} tweets processed)"
                    )
                    current_batch.append(tweet_data)
                    
                    # Yield batch when it reaches the specified size
                    if len(current_batch) >= batch_size:
                        yield current_batch
                        current_batch = []

        ## Log progress every 10 tweets.
        if tweets_checked % 10 == 0:
            logger.info(
                f"Progress: {tweets_checked}/{max_tweets} tweets processed"
            )

        ## Scroll and check progress.
        last_height = browser.execute_script(
            "return document.documentElement.scrollHeight"
        )
        browser.execute_script(
            "window.scrollTo(0, document.documentElement.scrollHeight);"
        )
        time.sleep(5)

        new_height = browser.execute_script(
            "return document.documentElement.scrollHeight"
        )
        if new_height == last_height:
            logger.info("No new content loaded, breaking loop")
            break

        # Yield any remaining tweets in the final batch
        if current_batch:
            yield current_batch

        logger.info(
            f"Checked {tweets_checked} tweets"
        )

    browser.quit()


##########################
## TWEET VERIFICATION   ##
##########################


def verify_tweet_structure(
    thread: TweetThread, config: Optional[TweetThreadConfig] = None
) -> Tuple[bool, str]:
    """Verify that the tweet thread structure matches its configuration."""
    try:
        # Basic validation for all tweet threads
        if not thread.tweets:
            return False, "Tweet thread has no tweets"
            
        # Check for duplicate positions
        positions = [t.position for t in thread.tweets]
        if len(positions) != len(set(positions)):
            return False, "Tweet thread has duplicate positions"
            
        # If no config provided, just do basic validation
        if not config:
            return True, "Basic tweet structure verified successfully"
            
        # Check number of tweets against config
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


def verify_tweet_content(tweet: Tweet, config: Optional[TweetConfig] = None) -> Tuple[bool, str]:
    """Verify a single tweet's content against its configuration."""
    try:
        # Basic validation for all tweets
        if not tweet.content:
            return False, "Tweet has no content"
        
        # Verify image paths exist if any
        if tweet.images:
            for idx, image_path in enumerate(tweet.images):
                if not os.path.exists(image_path):
                    return False, f"Image {idx+1} not found: {image_path}"
        
        # If no config provided, just do basic validation
        if not config:
            return True, "Basic tweet content verified successfully"
            
        # Check for required URLs in link tweets if this is a links tweet
        if config.content.content_type == "function" and "generate_links_content" in config.content.content:
            required_urls = ["arxiv.org", "llmpedia.streamlit.app"]
            for url in required_urls:
                if url not in tweet.content:
                    return False, f"Missing required URL {url} in links tweet"

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
    config: Optional[TweetThreadConfig] = None,
    driver: Optional[webdriver.Firefox] = None,
    logger: Optional[logging.Logger] = None,
    max_retries: int = 3,
    retry_delay: float = 1.0,
) -> Tuple[bool, str]:
    """Verify a complete tweet thread against its configuration and UI state."""
    logger = logger or get_console_logger()
    try:
        # Verify thread structure
        structure_ok, structure_msg = verify_tweet_structure(thread, config)
        if not structure_ok:
            return False, f"Thread structure verification failed: {structure_msg}"

        # Verify each tweet's content
        for i, tweet in enumerate(thread.tweets):
            tweet_config = config.tweets[i] if config else None
            content_ok, content_msg = verify_tweet_content(tweet, tweet_config)
            if not content_ok:
                return False, f"Tweet content verification failed: {content_msg}"

        # Skip UI verification if no driver provided
        if not driver:
            logger.info("Skipping UI verification (no driver provided)")
            return True, "Tweet thread content verified successfully (no UI verification)"

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
    tweet_content: Union[TweetThread, str],
    logger: Optional[logging.Logger] = None,
    verify: bool = True,
    tweet_image_path: Optional[str] = None,
    tweet_page_path: Optional[str] = None,
    analyzed_image_path: Optional[str] = None,
    author_tweet: Optional[dict] = None,
    config: Optional[TweetThreadConfig] = None,
    headless: bool = True,
) -> bool:
    """Send a tweet or tweet thread.
    
    This function supports both TweetThread objects and simple string tweets.
    
    Args:
        tweet_content: Either a TweetThread object or a string with tweet content
        logger: Optional logger for tracking progress
        verify: Whether to verify tweet content before sending
        tweet_image_path: Optional image path (for simple string tweets only)
        tweet_page_path: Optional second image path (for simple string tweets only)
        analyzed_image_path: Optional analyzed image path (for simple string tweets only)
        author_tweet: Optional author tweet data (for simple string tweets only)
        config: Optional configuration for verification
        headless: Whether to run the browser in headless mode
        
    Returns:
        bool: Whether the tweet(s) were sent successfully
    """
    logger = logger or get_console_logger()
    
    # Convert string tweet to TweetThread if needed
    if isinstance(tweet_content, str):
        images = []
        if tweet_image_path:
            images.append(tweet_image_path)
        if tweet_page_path:
            images.append(tweet_page_path)
            
        # Create a simple tweet thread
        tweet_thread = TweetThread.create_simple_tweet(
            content=tweet_content,
            images=images if images else None,
            tweet_type="simple",
            metadata={
                "analyzed_image_path": analyzed_image_path,
                "author_tweet": author_tweet
            }
        )
    else:
        tweet_thread = tweet_content
    
    logger.info(f"Starting to send tweet thread of type: {tweet_thread.tweet_type}")
    
    try:
        # Setup browser and login
        driver = setup_browser(logger, headless=headless)
        login_twitter(driver, logger)
        
        # Verify thread if requested
        if verify:
            logger.info("Verifying tweet thread before sending")
            success, msg = verify_tweet_thread(tweet_thread, config, driver, logger)
            if not success:
                logger.error(f"Tweet thread verification failed: {msg}")
                return False
        
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
        
        # Handle analyzed image if provided (for simple tweets)
        if isinstance(tweet_content, str) and analyzed_image_path:
            time.sleep(2)
            logger.info("Adding image tweet")
            tweet_reply_btn = WebDriverWait(driver, 60).until(
                EC.element_to_be_clickable((By.XPATH, "//button[@data-testid='addButton']"))
            )
            tweet_reply_btn.click()
            
            # Enter image tweet content
            tweet_box = WebDriverWait(driver, 60).until(
                EC.element_to_be_clickable(
                    (
                        By.XPATH,
                        "//div[@contenteditable='true' and @data-testid='tweetTextarea_1']",
                    )
                )
            )
            tweet_box.send_keys("Key visualization from the paper 📊")
            
            # Upload image
            upload_input = driver.find_element(
                By.XPATH,
                '//input[@accept="image/jpeg,image/png,image/webp,image/gif,video/mp4,video/quicktime"]',
            )
            upload_input.send_keys(analyzed_image_path)
            
            # Verify image is uploaded
            WebDriverWait(driver, 60).until(
                EC.presence_of_element_located(
                    (By.XPATH, "(//button[@aria-label='Remove media'])[1]")
                )
            )
        
        # Handle author tweet if provided (for simple tweets)
        if isinstance(tweet_content, str) and author_tweet:
            tweet_idx = 2 if analyzed_image_path else 1
            tweet_reply_btn = WebDriverWait(driver, 60).until(
                EC.element_to_be_clickable((By.XPATH, "//button[@data-testid='addButton']"))
            )
            tweet_reply_btn.click()
            
            tweet_box = WebDriverWait(driver, 60).until(
                EC.element_to_be_clickable(
                    (
                        By.XPATH,
                        f"//div[@contenteditable='true' and @data-testid='tweetTextarea_{tweet_idx}']",
                    )
                )
            )
            tweet_box.send_keys(f"related discussion: {author_tweet['link']}")
        
        # Send the complete thread
        logger.info("Preparing to send tweet thread")
        try:
            tweet_all_button = WebDriverWait(driver, 30).until(
                EC.element_to_be_clickable(
                    (
                        By.XPATH,
                        "//button[@data-testid='tweetButton']//span[contains(text(), 'Post') or contains(text(), 'Post all')]",
                    )
                )
            )
            
            # Visual confirmation and final send
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


if __name__ == "__main__":
    from utils.logging_utils import setup_logger

    logger = setup_logger(__name__, "tweet_test.log")
    logger.info("Starting browser...")
    send_tweet(
        "𝗠𝗲𝗻𝘁𝗮𝗹𝗔𝗿𝗲𝗻𝗮: 𝗦𝗲𝗹𝗳-𝗽𝗹𝗮𝘆 𝗧𝗿𝗮𝗶𝗻𝗶𝗻𝗴 𝗼𝗳 𝗟𝗮𝗴𝘂𝗮𝗴𝗲 𝗠𝗼𝗱𝗲𝗹𝘀 𝗳𝗼𝗿 𝗗𝗶𝗮𝗴𝗻𝗼𝘀𝗶𝘀 𝗮𝗻𝗱 𝗧𝗿𝗲𝗮𝘁𝗺𝗲𝗻𝘁 𝗼𝗳 𝗠𝗲𝗻𝘁𝗮𝗹 𝗛𝗲𝗮𝗹𝘁𝗵 𝗗𝗶𝘀𝗼𝗿𝗱𝗲𝗿𝘀 (Oct 09, 2024): MentalArena is a self-play framework that trains language models to simulate both patient and therapist roles in mental health scenarios. Using GPT-3.5-turbo as a base, it outperformed the more advanced GPT-4o by 7.7% on mental health tasks. The framework generated 18,000 high-quality training samples, addressing data scarcity due to privacy concerns in mental health AI. MentalArena showed resilience against catastrophic forgetting, maintaining or improving performance on general benchmarks like BIG-Bench-Hard while excelling in specialized mental health applications. This demonstrates the potential of self-play training in generating domain-specific data and enhancing model performance in sensitive areas.",
        "/Users/manuelrueda/Documents/python/llmpedia/data/arxiv_art/1902.03545.png",
        "/Users/manuelrueda/Documents/python/llmpedia/data/arxiv_art/1902.03545.png",
        "XXX",
        logger,
    )
    logger.info("Browser started.")
