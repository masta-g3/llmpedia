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

load_dotenv()

PROJECT_PATH = os.getenv("PROJECT_PATH", "/app")
# GECKODRIVER_PATH = os.getenv("GECKODRIVER_PATH", "/usr/bin/geckodriver")
sys.path.append(PROJECT_PATH)
print(PROJECT_PATH)

from utils.logging_utils import setup_logger


# Set up logging
logger = setup_logger(__name__, "tweet_generation.log")

url = "https://x.com/"
USERNAME = os.getenv("TWITTER_EMAIL")
PASSWORD = os.getenv("TWITTER_PASSWORD")
PHONE = os.getenv("TWITTER_PHONE")


def setup_browser():
    logger.info("Setting up browser")
    firefox_options = FirefoxOptions()
    firefox_options.add_argument("--headless")
    firefox_options.add_argument("--no-sandbox")
    firefox_options.add_argument("--disable-dev-shm-usage")

    # Set the MOZ_HEADLESS environment variable
    os.environ["MOZ_HEADLESS"] = "1"

    try:
        # Try to find geckodriver in PATH
        service = FirefoxService()
        driver = webdriver.Firefox(options=firefox_options, service=service)
    except Exception as e:
        logger.error(f"Failed to create driver with default service: {str(e)}")
        try:
            # If that fails, try with explicit geckodriver path
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


def login_twitter(driver: webdriver.Firefox):
    """Login to Twitter within any page of its domain."""
    logger.info("Attempting to log in to Twitter")

    driver.get("https://twitter.com/login")

    # Wait for the username field and enter the username
    try:
        username_field = WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.NAME, "text"))
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


def navigate_to_profile(browser: webdriver.Firefox, profile_url: str):
    """Login to Twitter and navigate to a profile."""
    browser.get(profile_url)


def verify_tweet_elements(
    driver: webdriver.Firefox, expected_content: str, expected_image_count: int = 2
) -> Tuple[bool, str]:
    """Verify the presence of expected elements in a tweet composition."""
    logger.info("Verifying tweet elements")

    # Click on main post
    main_post_btn = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.XPATH, '//div[@aria-label="Post text"]'))
    )
    main_post_btn.click()

    # Check for the correct number of uploaded images
    def correct_image_count(driver):
        remove_buttons = driver.find_elements(
            By.XPATH, "//button[@aria-label='Remove media']"
        )
        return len(remove_buttons) == expected_image_count

    WebDriverWait(driver, 10).until(correct_image_count)

    # Check for main tweet text
    main_tweet_elem = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located(
            (By.XPATH, "//div[@data-testid='tweetTextarea_0']")
        )
    )
    main_tweet_text = main_tweet_elem.text.strip()
    if not main_tweet_text:
        return False, "Main tweet text is empty"
    elif main_tweet_text != expected_content.strip():
        return False, "Main tweet text does not match expected content"

    # Check for post-tweet text
    post_tweet_elem = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located(
            (By.XPATH, "//div[@data-testid='tweetTextarea_1']")
        )
    )
    post_tweet_text = post_tweet_elem.text.strip()
    if not post_tweet_text:
        return False, "Post-tweet text is empty"

    verification_message = "All elements are present and correct"
    logger.info(f"Tweet element verification result: {verification_message}")
    return True, verification_message


def send_tweet(
    tweet_content: str, tweet_image_path: str, tweet_page_path: str, post_tweet: str
) -> bool:
    """Send a tweet with content and images using Selenium."""
    logger.info("Starting tweet sending process")
    driver = webdriver.Firefox()
    login_twitter(driver)

    logger.info("Composing tweet")
    # Click the "Post" button to start a new tweet
    tweet_button = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.XPATH, '//a[@aria-label="Post"]'))
    )
    tweet_button.click()

    # Enter tweet content
    tweet_textarea = WebDriverWait(driver, 10).until(
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
    WebDriverWait(driver, 10).until(
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

    WebDriverWait(driver, 10).until(correct_image_count)

    # Add follow-up tweet
    logger.info("Adding follow-up tweet section")
    tweet_reply_btn = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.XPATH, "//button[@data-testid='addButton']"))
    )
    tweet_reply_btn.click()

    # Enter follow-up tweet content
    tweet_box = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable(
            (
                By.XPATH,
                "//div[@contenteditable='true' and @data-testid='tweetTextarea_1']",
            )
        )
    )
    tweet_box.send_keys(post_tweet)

    # Verify tweet elements
    elements_verified, verification_message = verify_tweet_elements(
        driver, tweet_content, expected_image_count=2
    )
    if not elements_verified:
        logger.error(f"Tweet verification failed: {verification_message}")
        return False

    # Optionally sleep before sending the tweet
    sleep_duration = random.randint(10, 2700)  # Sleep between 10 seconds and 45 minutes
    logger.info(f"Sleeping for {sleep_duration} seconds before sending the tweet...")
    # time.sleep(sleep_duration)

    # Send tweet
    logger.info("Attempting to send tweet")
    tweet_send_button = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable(
            (By.XPATH, "//button[@data-testid='tweetButton']")
        )
    )
    tweet_send_button.click()

    logger.info("Tweet sent successfully")
    return True


if __name__ == "__main__":
    print("Starting browser...")
    send_tweet(
        "𝗠𝗲𝗻𝘁𝗮𝗹𝗔𝗿𝗲𝗻𝗮: 𝗦𝗲𝗹𝗳-𝗽𝗹𝗮𝘆 𝗧𝗿𝗮𝗶𝗻𝗶𝗻𝗴 𝗼𝗳 𝗟𝗮𝗻𝗴𝘂𝗮𝗴𝗲 𝗠𝗼𝗱𝗲𝗹𝘀 𝗳𝗼𝗿 𝗗𝗶𝗮𝗴𝗻𝗼𝘀𝗶𝘀 𝗮𝗻𝗱 𝗧𝗿𝗲𝗮𝘁𝗺𝗲𝗻𝘁 𝗼𝗳 𝗠𝗲𝗻𝘁𝗮𝗹 𝗛𝗲𝗮𝗹𝘁𝗵 𝗗𝗶𝘀𝗼𝗿𝗱𝗲𝗿𝘀 (Oct 09, 2024): MentalArena is a self-play framework that trains language models to simulate both patient and therapist roles in mental health scenarios. Using GPT-3.5-turbo as a base, it outperformed the more advanced GPT-4o by 7.7% on mental health tasks. The framework generated 18,000 high-quality training samples, addressing data scarcity due to privacy concerns in mental health AI. MentalArena showed resilience against catastrophic forgetting, maintaining or improving performance on general benchmarks like BIG-Bench-Hard while excelling in specialized mental health applications. This demonstrates the potential of self-play training in generating domain-specific data and enhancing model performance in sensitive areas.",
        "/Users/manuelrueda/Documents/python/llmpedia/data/arxiv_art/1902.03545.png",
        "/Users/manuelrueda/Documents/python/llmpedia/data/arxiv_art/1902.03545.png",
        "XXX",
    )
    print("Browser started.")
