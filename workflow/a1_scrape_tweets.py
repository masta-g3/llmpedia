import random
import sys, os
import re
from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.environ.get("PROJECT_PATH"))

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
from typing import List
import time
import csv
import os

import utils.paper_utils as pu


username = os.getenv("TWITTER_EMAIL")
userpass = os.getenv("TWITTER_PASSWORD")
phone = os.getenv("TWITTER_PHONE")

tweet_accounts = [
    "_reachsumit",
    "ADarmouni",
    "rohanpaul_ai",
    "iScienceLuvr",
    "arankomatsuzaki",
    "papers_anon",
    "fly51fly",
]


def login_twitter(browser: webdriver.Firefox):
    """Login to Twitter within any page of its domain."""
    login = WebDriverWait(browser, 30).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, 'a[data-testid="login"]'))
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


def navigate_to_profile(
    browser: webdriver.Firefox, profile_url: str, login: bool = True
):
    """Login to Twitter and navigate to a profile."""
    browser.get(profile_url)
    if login:
        login_twitter(browser)


def scroll_page(browser: webdriver.Firefox):
    browser.execute_script("window.scrollBy(0, window.innerHeight);")
    time.sleep(random.randrange(3, 7))


def extract_tweets(browser: webdriver.Firefox) -> List[dict]:
    """Extract tweets from the current page."""
    soup = BeautifulSoup(browser.page_source, "html.parser")
    tweets = soup.find_all("article", {"data-testid": "tweet"})
    extracted_tweets = []

    for tweet in tweets:
        try:
            pinned = tweet.find("div", {"data-testid": "socialContext"})
            if pinned and pinned.text == "Pinned":
                continue
            text = tweet.find("div", {"data-testid": "tweetText"}).text
            timestamp = tweet.find("time")["datetime"]
            extracted_tweets.append({"text": text, "timestamp": timestamp})
        except AttributeError:
            continue

    return extracted_tweets


def scrape_tweets(browser: webdriver.Firefox, max_tweets: int = 100) -> List[dict]:
    """Scrape tweets from a profile."""
    all_tweets = []
    last_height = browser.execute_script("return document.body.scrollHeight")

    while len(all_tweets) < max_tweets:
        new_tweets = extract_tweets(browser)

        all_tweets.extend([tweet for tweet in new_tweets if tweet not in all_tweets])

        scroll_page(browser)
        new_height = browser.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height
        # print(f"Scraped {len(all_tweets)} tweets so far...")
    return all_tweets


def extract_codes_from_tweets(tweets: List[str]) -> List[str]:
    """Extract arXiv codes from a list."""
    pattern = re.compile(r"\b\d{4}\.\d+\b")
    return [match.group(0) for tweet in tweets if (match := pattern.search(tweet))]


def save_tweets_to_csv(tweets: List[dict], filename: str):
    """Save tweets to a CSV file."""
    with open(filename, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["text", "timestamp"])
        writer.writeheader()
        for tweet in tweets:
            writer.writerow(tweet)


def main():
    all_tweets = []
    browser = webdriver.Firefox()
    login = True

    try:
        for account in tweet_accounts:
            profile_url = f"https://twitter.com/{account}"
            navigate_to_profile(browser, profile_url, login=login)
            login = False
            tweets = scrape_tweets(browser, max_tweets=30)
            all_tweets.extend(tweets)
            print(
                f"Successfully scraped and saved {len(tweets)} tweets from {account}."
            )

        new_codes = extract_codes_from_tweets([tweet["text"] for tweet in all_tweets])

        ## Remote paper list.
        gist_id = "1dd189493c1890df6e04aaea6d049643"
        gist_filename = "llm_queue.txt"
        paper_list = pu.fetch_queue_gist(gist_id, gist_filename)

        ## Update and upload arxiv codes.
        paper_list = list(set(paper_list + new_codes))
        done_codes = pu.get_local_arxiv_codes()
        nonllm_codes = pu.get_local_arxiv_codes("nonllm_arxiv_text")

        print(f"Total papers: {len(paper_list)}")
        paper_list = list(set(paper_list) - set(done_codes) - set(nonllm_codes))
        print(f"New papers: {len(paper_list)}")

        if len(paper_list) == 0:
            print("No new papers found. Exiting...")
            sys.exit(0)
        gist_url = pu.update_gist(
            os.environ["GITHUB_TOKEN"],
            gist_id,
            gist_filename,
            "Updated LLM queue.",
            "\n".join(paper_list),
        )
        time.sleep(20)

    except Exception as e:
        print(f"An error occurred: {str(e)}")

    finally:
        browser.quit()


if __name__ == "__main__":
    main()
