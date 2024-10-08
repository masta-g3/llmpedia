import os
import time
import random
from typing import Tuple

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.firefox.options import Options as FirefoxOptions

# Import environment variables
from dotenv import load_dotenv

load_dotenv()

url = "https://x.com/"
username = os.getenv("TWITTER_EMAIL")
userpass = os.getenv("TWITTER_PASSWORD")
phone = os.getenv("TWITTER_PHONE")


def setup_browser():
    firefox_options = FirefoxOptions()
    firefox_options.add_argument("--headless")

    service = webdriver.firefox.service.Service("/usr/bin/geckodriver")
    driver = webdriver.Firefox(options=firefox_options)
    return driver


def login_twitter(browser: webdriver.Firefox):
    """Login to Twitter within any page of its domain."""
    login = WebDriverWait(browser, 30).until(
        EC.presence_of_element_located(
            (By.CSS_SELECTOR, 'a[data-testid="loginButton"]')
        )
    )
    login.send_keys(Keys.ENTER)

    user = WebDriverWait(browser, 30).until(
        EC.presence_of_element_located(
            (By.XPATH, '//input[@name="text" and @autocomplete="username"]')
        )
    )
    user.send_keys(username)
    user.send_keys(Keys.ENTER)

    try:
        number = WebDriverWait(browser, 10).until(
            EC.presence_of_element_located(
                (By.CSS_SELECTOR, 'input[data-testid="ocfEnterTextTextInput"]')
            )
        )
        number.send_keys(phone)
        number.send_keys(Keys.ENTER)
    except:
        user = WebDriverWait(browser, 30).until(
            EC.presence_of_element_located(
                (By.XPATH, '//input[@name="text" and @autocomplete="username"]')
            )
        )
        user.send_keys(username)
        user.send_keys(Keys.ENTER)

        try:
            number = WebDriverWait(browser, 30).until(
                EC.presence_of_element_located(
                    (By.CSS_SELECTOR, 'input[data-testid="ocfEnterTextTextInput"]')
                )
            )
            number.send_keys(phone)
            number.send_keys(Keys.ENTER)
        except:
            raise Exception("Failed to login.")

    password = WebDriverWait(browser, 30).until(
        EC.presence_of_element_located((By.NAME, "password"))
    )
    password.send_keys(userpass)
    password.send_keys(Keys.ENTER)

    WebDriverWait(browser, 30).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, '[data-testid="tweet"]'))
    )


def verify_tweet_elements(browser: webdriver.Chrome, expected_image_count: int = 2) -> Tuple[bool, str]:
    """Verify the presence of expected elements in a tweet composition."""
    try:
        # Check for images
        def correct_image_count(driver):
            remove_buttons = driver.find_elements(
                By.XPATH, "//button[@aria-label='Remove media']"
            )
            return len(remove_buttons) == expected_image_count

        WebDriverWait(browser, 10).until(correct_image_count)

        # Check for main tweet text
        main_tweet_text = (
            WebDriverWait(browser, 10)
            .until(
                EC.presence_of_element_located(
                    (By.XPATH, "//div[@data-testid='tweetTextarea_0']")
                )
            )
            .text
        )
        if not main_tweet_text.strip():
            return False, "Main tweet text is empty"

        # Check for post-tweet text
        post_tweet_text = (
            WebDriverWait(browser, 10)
            .until(
                EC.presence_of_element_located(
                    (By.XPATH, "//div[@data-testid='tweetTextarea_1']")
                )
            )
            .text
        )
        if not post_tweet_text.strip():
            return False, "Post-tweet text is empty"

        return True, "All elements are present"
    except Exception as e:
        return False, f"Error verifying tweet elements: {str(e)}"


def send_tweet(
    tweet_content: str, tweet_image_path: str, tweet_page_path: str, post_tweet: str
) -> None:
    """Send a tweet with content and images using Selenium."""
    browser = setup_browser()
    browser.get(url)
    login_twitter(browser)

    # Compose tweet
    WebDriverWait(browser, 30).until(
        EC.visibility_of_element_located(
            (By.XPATH, "//div[@data-testid='tweetTextarea_0']")
        )
    )

    # Upload first image
    input_box = WebDriverWait(browser, 30).until(
        EC.presence_of_element_located((By.XPATH, "//input[@accept]"))
    )
    input_box.send_keys(tweet_image_path)

    # Wait for the first image to be loaded
    WebDriverWait(browser, 30).until(
        EC.presence_of_element_located(
            (By.XPATH, "//button[@aria-label='Remove media']")
        )
    )

    # Upload second image
    input_box = WebDriverWait(browser, 30).until(
        EC.presence_of_element_located((By.XPATH, "//input[@accept]"))
    )
    input_box.send_keys(tweet_page_path)

    # Wait for both images to be loaded
    def two_remove_buttons_present(driver):
        remove_buttons = driver.find_elements(
            By.XPATH, "//button[@aria-label='Remove media']"
        )
        return len(remove_buttons) == 2

    WebDriverWait(browser, 30).until(two_remove_buttons_present)

    # Add follow-up tweet section
    tweet_reply_btn = WebDriverWait(browser, 30).until(
        EC.element_to_be_clickable((By.XPATH, "//a[@data-testid='addButton']"))
    )
    browser.execute_script("arguments[0].click();", tweet_reply_btn)

    # Wait for the new tweet box to appear
    WebDriverWait(browser, 10).until(
        EC.presence_of_element_located(
            (By.XPATH, "//div[@data-testid='tweetTextarea_1']")
        )
    )

    # Add post-tweet
    tweet_box = WebDriverWait(browser, 30).until(
        EC.presence_of_element_located(
            (
                By.XPATH,
                "//div[@contenteditable='true' and @data-testid='tweetTextarea_1']",
            )
        )
    )
    tweet_box.send_keys(post_tweet.replace("\n", Keys.RETURN))

    # Add tweet
    tweet_box = WebDriverWait(browser, 30).until(
        EC.presence_of_element_located(
            (
                By.XPATH,
                "//div[@contenteditable='true' and @data-testid='tweetTextarea_0']",
            )
        )
    )
    tweet_box.send_keys(tweet_content.replace("\n", Keys.RETURN))

    # Verify tweet elements
    elements_verified, verification_message = verify_tweet_elements(browser)
    if not elements_verified:
        print(f"Tweet verification failed: {verification_message}")
        browser.quit()
        return False

    # Send tweet
    sleep_duration = random.randint(1, 45 * 60)
    print(f"Sleeping for {sleep_duration // 60} minutes before sending the tweet...")
    # time.sleep(sleep_duration)

    try:
        wait = WebDriverWait(browser, 10)
        button = wait.until(
            EC.element_to_be_clickable(
                (By.CSS_SELECTOR, 'button[data-testid="tweetButton"]')
            )
        )
        browser.execute_script("arguments[0].click();", button)
    except:
        button = WebDriverWait(browser, 30).until(
            EC.element_to_be_clickable(
                (By.CSS_SELECTOR, "div[data-testid='tweetButton']")
            )
        )
        browser.execute_script("arguments[0].click();", button)

    print("Tweet sent successfully.")
    time.sleep(10)
    browser.quit()
    return True

if __name__ == "__main__":
    print("Starting browser...")
    browser = setup_browser()
    print("Browser started.")
