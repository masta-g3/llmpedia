import os, sys
import time
import random
from typing import Tuple, List, Iterator
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
import utils.vector_store as vs
import utils.db as db

load_dotenv()

PROJECT_PATH = os.getenv("PROJECT_PATH", "/app")
sys.path.append(PROJECT_PATH)

url = "https://x.com/"
USERNAME = os.getenv("TWITTER_EMAIL")
PASSWORD = os.getenv("TWITTER_PASSWORD")
PHONE = os.getenv("TWITTER_PHONE")


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
    post_tweet: str,
    logger: logging.Logger,
    tweet_image_path: str | None = None,
    author_tweet: dict | None = None,
    tweet_page_path: str | None = None,
    analyzed_image_path: str | None = None,
) -> bool:
    """Send a tweet with content and images using Selenium."""
    
    logger.info("Starting tweet sending process")
    driver = setup_browser(logger)
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


def extract_author_tweet_data(tweet_elem, paper_title, paper_authors, logger):
    """Extract tweet data if it's from a paper author."""
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

def extract_tweet_data(tweet_elem, logger: logging.Logger) -> dict:
    """Extract all relevant data from a tweet element."""
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
            return None

        tweet_text = tweet_text_elements[0].text
        if not tweet_text.strip():
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


def find_paper_author_tweet(arxiv_code: str, logger) -> dict:
    """Find a tweet from a paper's author about their paper."""
    logger.info(f"Searching for author tweet for {arxiv_code}")
    try:
        # Get paper details
        paper_details = db.load_arxiv(arxiv_code)
        paper_title = paper_details["title"].iloc[0]
        paper_authors = paper_details["authors"].iloc[0].split(", ")
        
        # Initialize browser
        browser = setup_browser(logger)
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
            try:
                # Wait for and get tweets
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

            except Exception as e:
                logger.error(f"Error during tweet collection: {str(e)}")
                break

        browser.quit()
        return None

    except Exception as e:
        logger.error(f"Error in find_paper_author_tweet: {str(e)}")
        if 'browser' in locals():
            browser.quit()
        return None


def collect_llm_tweets(logger: logging.Logger, max_tweets: int = 50, batch_size: int = 100) -> Iterator[List[dict]]:
    """Collect tweets about LLMs from the Twitter home feed in batches."""
    logger.info("Starting collection of LLM-related tweets")

    browser = setup_browser(logger)
    current_batch = []
    tweets_checked = 0

    # try:
    ## Login and navigate to home.
    login_twitter(browser, logger)
    browser.get("https://twitter.com/home")
    time.sleep(3) 

    while tweets_checked < max_tweets:
        # try:
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

            # except Exception as e:
                # logger.error(f"Error during tweet collection: {str(e)}")
                # break

        # Yield any remaining tweets in the final batch
        if current_batch:
            yield current_batch

        logger.info(
            f"Checked {tweets_checked} tweets"
        )

    # except Exception as e:
    #     logger.error(f"An error occurred while collecting LLM tweets: {str(e)}")
    #     if current_batch:
    #         yield current_batch

    # finally:
    #     browser.quit()


if __name__ == "__main__":
    from utils.logging_utils import setup_logger

    logger = setup_logger(__name__, "tweet_test.log")
    logger.info("Starting browser...")
    send_tweet(
        "𝗠𝗲𝗻𝘁𝗮𝗹𝗔𝗿𝗲𝗻𝗮: 𝗦𝗲𝗹𝗳-𝗽𝗹𝗮𝘆 𝗧𝗿𝗮𝗶𝗻𝗶𝗻𝗴 𝗼𝗳 𝗟𝗮𝗻𝗴𝘂𝗮𝗴𝗲 𝗠𝗼𝗱𝗲𝗹𝘀 𝗳𝗼𝗿 𝗗𝗶𝗮𝗴𝗻𝗼𝘀𝗶𝘀 𝗮𝗻𝗱 𝗧𝗿𝗲𝗮𝘁𝗺𝗲𝗻𝘁 𝗼𝗳 𝗠��𝗻𝘁𝗮𝗹 𝗛𝗲𝗮𝗹𝘁𝗵 𝗗𝗶𝘀𝗼𝗿𝗱𝗲𝗿𝘀 (Oct 09, 2024): MentalArena is a self-play framework that trains language models to simulate both patient and therapist roles in mental health scenarios. Using GPT-3.5-turbo as a base, it outperformed the more advanced GPT-4o by 7.7% on mental health tasks. The framework generated 18,000 high-quality training samples, addressing data scarcity due to privacy concerns in mental health AI. MentalArena showed resilience against catastrophic forgetting, maintaining or improving performance on general benchmarks like BIG-Bench-Hard while excelling in specialized mental health applications. This demonstrates the potential of self-play training in generating domain-specific data and enhancing model performance in sensitive areas.",
        "/Users/manuelrueda/Documents/python/llmpedia/data/arxiv_art/1902.03545.png",
        "/Users/manuelrueda/Documents/python/llmpedia/data/arxiv_art/1902.03545.png",
        "XXX",
        logger,
    )
    logger.info("Browser started.")
