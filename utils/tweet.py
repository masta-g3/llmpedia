import os, sys
import time
import random
from typing import Tuple
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

def setup_browser(logger: logging.Logger):
    logger.info("Setting up browser")
    firefox_options = FirefoxOptions()
    
    # Enable headless mode
    firefox_options.add_argument("--headless")
    firefox_options.add_argument("--no-sandbox")
    firefox_options.add_argument("--disable-dev-shm-usage")
    firefox_options.add_argument("--disable-gpu")
    firefox_options.add_argument("--window-size=1920,1080")
    firefox_options.add_argument("--remote-debugging-port=9222")
        
    # Set the MOZ_HEADLESS environment variable (optional)
    os.environ["MOZ_HEADLESS"] = "1"

    try:
        # Try to find geckodriver in PATH
        service = FirefoxService()
        driver = webdriver.Firefox(options=firefox_options, service=service)
    except Exception as e:
        logger.error(f"Failed to create driver with default service: {str(e)}")
        try:
            # If that fails, try with explicit geckodriver path
            geckodriver_path = os.getenv("GECKODRIVER_PATH", "/usr/local/bin/geckodriver")
            service = FirefoxService(executable_path=geckodriver_path)
            driver = webdriver.Firefox(options=firefox_options, service=service)
        except Exception as e:
            logger.error(f"Failed to create driver with explicit geckodriver path: {str(e)}")
            raise

    logger.info("Browser setup complete")
    return driver


def login_twitter(driver: webdriver.Firefox, logger: logging.Logger):
    """Login to Twitter within any page of its domain."""
    logger.info("Attempting to log in to Twitter")

    driver.get("https://x.com/i/flow/login")

    # Wait for the username field and enter the username
    try:
        username_field = WebDriverWait(driver, 15).until(
            EC.visibility_of_element_located((By.CSS_SELECTOR, 'input[autocomplete="username"]'))
        )
        username_field.send_keys(USERNAME)
        username_field.send_keys(Keys.RETURN)
    except TimeoutException:
        logger.error("Username field not found.")
        driver.quit()
        return

    # Check if additional identification is required (e.g., phone)
    try:
        identifier_field = WebDriverWait(driver, 5).until(
            EC.presence_of_element_located((By.NAME, "text"))
        )
        identifier_field.send_keys(PHONE)
        identifier_field.send_keys(Keys.RETURN)
    except TimeoutException:
        logger.info("No additional identifier required.")

    # Wait for the password field and enter the password
    try:
        password_field = WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.NAME, "password"))
        )
        password_field.send_keys(PASSWORD)
        password_field.send_keys(Keys.RETURN)
    except TimeoutException:
        logger.error("Password field not found.")
        driver.quit()
        return

    # Wait until the "Post" button is clickable to ensure login is complete
    try:
        WebDriverWait(driver, 15).until(
            EC.element_to_be_clickable((By.XPATH, '//a[@aria-label="Post"]'))
        )
        logger.info("Successfully logged in to Twitter")
    except TimeoutException:
        logger.error("Login failed or 'Post' button not found.")
        driver.quit()
        return


def navigate_to_profile(browser: webdriver.Firefox, profile_url: str, logger: logging.Logger):
    """Login to Twitter and navigate to a profile."""
    logger.info(f"Navigating to profile: {profile_url}")
    browser.get(profile_url)

def verify_tweet_elements(
  driver: webdriver.Firefox, expected_content: str, expected_image_count, logger: logging.Logger
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
    tweet_content: str, tweet_image_path: str, tweet_page_path: str, post_tweet: str, author_tweet: dict, logger: logging.Logger
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
        EC.presence_of_element_located(
            (By.XPATH, '//div[@aria-label="Post text"]')
        )
    )
    tweet_textarea.send_keys(tweet_content)

    # Upload first image
    logger.info("Uploading first image")
    upload_input = driver.find_element(
        By.XPATH,
        '//input[@accept="image/jpeg,image/png,image/webp,image/gif,video/mp4,video/quicktime"]',
    )
    upload_input.send_keys(tweet_image_path)

    # Verify first image is uploaded
    WebDriverWait(driver, 60).until(
        EC.presence_of_element_located(
            (By.XPATH, "(//button[@aria-label='Remove media'])[1]")
        )
    )

    # Upload second image
    logger.info("Uploading second image")
    upload_input.send_keys(tweet_page_path)

    # Verify both images are uploaded
    def correct_image_count(driver):
        remove_buttons = driver.find_elements(
            By.XPATH, "//button[@aria-label='Remove media']"
        )
        return len(remove_buttons) == 2

    WebDriverWait(driver, 30).until(correct_image_count)

    # Add follow-up tweet
    time.sleep(10)
    logger.info("Adding follow-up tweet section")
    tweet_reply_btn = WebDriverWait(driver, 60).until(
        EC.element_to_be_clickable((By.XPATH, "//button[@data-testid='addButton']"))
    )
    tweet_reply_btn.click()

    # Enter follow-up tweet content
    tweet_box = WebDriverWait(driver, 60).until(
        EC.element_to_be_clickable(
            (
                By.XPATH,
                "//div[@contenteditable='true' and @data-testid='tweetTextarea_1']",
            )
        )
    )
    tweet_box.send_keys(post_tweet)

    if author_tweet:
        # Add author tweet link as third tweet in thread
        tweet_reply_btn = WebDriverWait(driver, 60).until(
            EC.element_to_be_clickable((By.XPATH, "//button[@data-testid='addButton']"))
        )
        tweet_reply_btn.click()

        tweet_box = WebDriverWait(driver, 60).until(
            EC.element_to_be_clickable(
                (
                    By.XPATH,
                    "//div[@contenteditable='true' and @data-testid='tweetTextarea_2']",
                )
            )
        )
        tweet_box.send_keys(f"related discussion: {author_tweet['link']}")

    # Verify tweet elements
    elements_verified, verification_message = verify_tweet_elements(
        driver, tweet_content, expected_image_count=2, logger=logger
    )
    if not elements_verified:
        logger.error(f"Tweet verification failed: {verification_message}")
        return False

    # Send tweet
    time.sleep(5)
    logger.info("Attempting to send tweet")
    # No need to click tweet_box again

    # Find and click the 'Post all' button
    tweet_all_button = WebDriverWait(driver, 30).until(
        EC.element_to_be_clickable((
            By.XPATH,
            "//button[@data-testid='tweetButton']//span[contains(text(), 'Post all')]"
        ))
    )

    # Highlight for visual confirmation
    driver.execute_script("""
        arguments[0].style.backgroundColor = '#ff0';
        arguments[0].style.border = '2px solid red';
    """, tweet_all_button)

    # Optionally sleep before sending the tweet
    sleep_duration = 5  # Sleep between 10 seconds and 45 minutes
    logger.info(f"Sleeping for {sleep_duration} seconds before sending the tweet...")
    time.sleep(sleep_duration)
    tweet_all_button.click()
    time.sleep(5)
    logger.info("Tweet sent successfully")
    driver.quit()
    
    return True

def extract_author_tweet_data(tweet_elem, paper_title: str, paper_authors: str) -> dict:
    """Extract data from a single tweet element and assess if it's from an author."""
    try:
        user_name_elem = tweet_elem.find_element(
            By.CSS_SELECTOR, 'div[data-testid="User-Name"]'
        )
        user_name_parts = user_name_elem.text.split("\n")

        tweet_data = {
            "text": tweet_elem.find_element(
                By.CSS_SELECTOR, 'div[data-testid="tweetText"]'
            ).text,
            "timestamp": tweet_elem.find_element(By.TAG_NAME, "time").get_attribute(
                "datetime"
            ),
            "author": user_name_parts[0],
            "username": user_name_parts[1],
            "link": tweet_elem.find_element(
                By.CSS_SELECTOR, 'a[href*="/status/"]'
            ).get_attribute("href"),
        }

        if int(
            vs.assess_tweet_ownership(
                paper_title=paper_title,
                paper_authors=paper_authors,
                tweet_text=tweet_data["text"],
                tweet_username=tweet_data["username"],
                model="gpt-4o-mini",
            )
        ):
            return tweet_data
        return None

    except Exception as e:
        logger.warning(f"Error extracting tweet details: {str(e)}")
        return None

def find_paper_author_tweet(arxiv_code: str, logger: logging.Logger) -> dict:
    """
    Find a tweet from a paper's author discussing their paper.
    
    Args:
        arxiv_code (str): The arXiv code of the paper
        logger (logging.Logger): Logger instance
        
    Returns:
        dict: Tweet data if found, None if not found
    """
    logger.info(f"Searching for author tweet for paper {arxiv_code}")
    
    # Get paper details
    paper_details = db.load_arxiv(arxiv_code)
    paper_title = paper_details.loc[arxiv_code, "title"]
    paper_authors = paper_details.loc[arxiv_code, "authors"]

    browser = setup_browser(logger)
    # try:
    login_twitter(browser, logger)
    
    # Setup search
    search_url = f"https://twitter.com/search?q='{paper_title}'&src=typed_query"
    browser.get(search_url)
    time.sleep(5)

    tweets_checked = 0
    max_tweets_to_check = 20

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
                    tweet_elem, paper_title, paper_authors
                )
                if tweet_data:
                    logger.info(f"Found author tweet from: {tweet_data['username']}")
                    browser.quit()
                    return tweet_data

            # Scroll and check progress
            last_height = browser.execute_script("return document.body.scrollHeight")
            browser.execute_script("window.scrollBy(0, window.innerHeight);")
            time.sleep(random.uniform(2, 4))

            if browser.execute_script("return document.body.scrollHeight") == last_height:
                break

        except Exception as e:
            logger.error(f"Error during tweet collection: {str(e)}")
            break

    logger.info(f"Checked {tweets_checked} tweets, no author tweet found")
    browser.quit()
    return None
        
    # except Exception as e:
    #     logger.error(f"An error occurred while searching for author tweet: {str(e)}")
    #     return None
    # finally:
        # browser.quit()

if __name__ == "__main__":
    from utils.logging_utils import setup_logger
    logger = setup_logger(__name__, "tweet_test.log")
    logger.info("Starting browser...")
    send_tweet(
        "𝗠𝗲𝗻𝘁𝗮𝗹𝗔𝗿𝗲𝗻𝗮: 𝗦𝗲𝗹𝗳-𝗽𝗹𝗮𝘆 𝗧𝗿𝗮𝗶𝗻𝗶𝗻𝗴 𝗼𝗳 𝗟𝗮𝗻𝗴𝘂𝗮𝗴𝗲 𝗠𝗼𝗱𝗲𝗹𝘀 𝗳𝗼𝗿 𝗗𝗶𝗮𝗴𝗻𝗼𝘀𝗶𝘀 𝗮𝗻𝗱 𝗧𝗿𝗲𝗮𝘁𝗺𝗲𝗻𝘁 𝗼𝗳 𝗠𝗲𝗻𝘁𝗮𝗹 𝗛𝗲𝗮𝗹𝘁𝗵 𝗗𝗶𝘀𝗼𝗿𝗱𝗲𝗿𝘀 (Oct 09, 2024): MentalArena is a self-play framework that trains language models to simulate both patient and therapist roles in mental health scenarios. Using GPT-3.5-turbo as a base, it outperformed the more advanced GPT-4o by 7.7% on mental health tasks. The framework generated 18,000 high-quality training samples, addressing data scarcity due to privacy concerns in mental health AI. MentalArena showed resilience against catastrophic forgetting, maintaining or improving performance on general benchmarks like BIG-Bench-Hard while excelling in specialized mental health applications. This demonstrates the potential of self-play training in generating domain-specific data and enhancing model performance in sensitive areas.",
        "/Users/manuelrueda/Documents/python/llmpedia/data/arxiv_art/1902.03545.png",
        "/Users/manuelrueda/Documents/python/llmpedia/data/arxiv_art/1902.03545.png",
        "XXX",
        logger
    )
    logger.info("Browser started.")
