import datetime
import os, sys, re
import time
import random
import numpy as np
from dotenv import load_dotenv
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.firefox.options import Options as FirefoxOptions
import platform

load_dotenv()

PROJECT_PATH = os.environ.get("PROJECT_PATH")
DATA_PATH = os.path.join(PROJECT_PATH, "data")
IMG_PATH = os.path.join(DATA_PATH, "arxiv_art")
PAGE_PATH = os.path.join(DATA_PATH, "arxiv_first_page")
sys.path.append(PROJECT_PATH)

url = "https://x.com/"
username = os.getenv("TWITTER_EMAIL")
userpass = os.getenv("TWITTER_PASSWORD")
phone = os.getenv("TWITTER_PHONE")

import utils.vector_store as vs
import utils.paper_utils as pu
import utils.email as em
import utils.db as db


def verify_tweet_elements(browser, expected_image_count=2):
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


def bold(input_text, extra_str):
    chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    bold_chars = "𝗔𝗕𝗖𝗗𝗘𝗙𝗚𝗛𝗜𝗝𝗞𝗟𝗠𝗡𝗢𝗣𝗤𝗥𝗦𝗧𝗨𝗩𝗪𝗫𝗬𝗭𝗮𝗯𝗰𝗱𝗲𝗳𝗴𝗵𝗶𝗷𝗸𝗹𝗺𝗻𝗼𝗽𝗾𝗿𝘀𝘁𝘂𝘃𝘄𝘅𝘆𝘇𝟬𝟭𝟮𝟯𝟰𝟱𝟲𝟳𝟴𝟵"
    bold_italic_chars = "𝘼𝘽𝘾𝘿𝙀𝙁𝙂𝙃𝙄𝙅𝙆𝙇𝙈𝙉𝙊𝙋𝙌𝙍𝙎𝙏𝙐𝙑𝙒𝙓𝙔𝙕𝙖𝙗𝙘𝙙𝙚𝙛𝙜𝙝𝙞𝙟𝙠𝙡𝙢𝙣𝙤𝙥𝙦𝙧𝙨𝙩𝙪𝙫𝙬𝙭𝙮𝙯𝟬𝟭𝟮𝟯𝟰𝟱𝟲𝟳𝟴𝟵"

    # Helper function to bold the characters within quotes
    def boldify(text):
        bolded_text = ""
        for character in text:
            if character in chars:
                bolded_text += bold_chars[chars.index(character)]
            else:
                bolded_text += character
        return bolded_text

    # Helper function to bold and italicize the characters within asterisks
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
    # Regex to find text in double asterisks and apply the bold_italicize function to them
    output = re.sub(r"\*\*([^*]*)\*\*", lambda m: bold_italicize(m.group(1)), output)

    return output.strip()


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


def setup_browser():
    if os.path.exists("/.dockerenv"):
        firefox_options = FirefoxOptions()
        firefox_options.add_argument("--headless")
        firefox_options.add_argument("--no-sandbox")
        firefox_options.add_argument("--disable-dev-shm-usage")
        firefox_options.add_argument("--disable-gpu")
        
        service = webdriver.firefox.service.Service("/usr/bin/geckodriver")
        driver = webdriver.Firefox(service=service, options=firefox_options)
    else:
        if platform.system() == "Darwin":
            firefox_options = FirefoxOptions()
            firefox_options.add_argument("--headless")
            driver = webdriver.Firefox(options=firefox_options)
        else:
            raise Exception("Unsupported OS")

    return driver


def send_tweet(tweet_content, tweet_image_path, tweet_page_path, post_tweet):
    browser = setup_browser()
    browser.get(url)
    login_twitter(browser)

    ## Compose tweet.
    WebDriverWait(browser, 30).until(
        EC.visibility_of_element_located(
            (By.XPATH, "//div[@data-testid='tweetTextarea_0']")
        )
    )

    ## Upload first image.
    input_box = WebDriverWait(browser, 30).until(
        EC.presence_of_element_located((By.XPATH, "//input[@accept]"))
    )
    input_box.send_keys(tweet_image_path)

    ## Wait for the first image to be loaded.
    WebDriverWait(browser, 30).until(
        EC.presence_of_element_located(
            (By.XPATH, "//button[@aria-label='Remove media']")
        )
    )

    ## Upload second image.
    input_box = WebDriverWait(browser, 30).until(
        EC.presence_of_element_located((By.XPATH, "//input[@accept]"))
    )
    input_box.send_keys(tweet_page_path)

    ## Wait for both images to be loaded.
    def two_remove_buttons_present(driver):
        remove_buttons = driver.find_elements(
            By.XPATH, "//button[@aria-label='Remove media']"
        )
        return len(remove_buttons) == 2

    WebDriverWait(browser, 30).until(two_remove_buttons_present)

    ## Add follow-up tweet section.
    tweet_reply_btn = WebDriverWait(browser, 30).until(
        EC.element_to_be_clickable((By.XPATH, "//a[@data-testid='addButton']"))
    )
    # Use JavaScript to click the button
    browser.execute_script("arguments[0].click();", tweet_reply_btn)

    # Wait for the new tweet box to appear
    WebDriverWait(browser, 10).until(
        EC.presence_of_element_located(
            (By.XPATH, "//div[@data-testid='tweetTextarea_1']")
        )
    )

    ## Add post-tweet.
    tweet_box = WebDriverWait(browser, 30).until(
        EC.presence_of_element_located(
            (
                By.XPATH,
                "//div[@contenteditable='true' and @data-testid='tweetTextarea_1']",
            )
        )
    )
    tweet_box.send_keys(post_tweet.replace("\n", Keys.RETURN))

    ## Add tweet.
    tweet_box = WebDriverWait(browser, 30).until(
        EC.presence_of_element_located(
            (
                By.XPATH,
                "//div[@contenteditable='true' and @data-testid='tweetTextarea_0']",
            )
        )
    )
    tweet_box.send_keys(tweet_content.replace("\n", Keys.RETURN))

    ## Verify tweet elements.
    elements_verified, verification_message = verify_tweet_elements(browser)
    if not elements_verified:
        print(f"Tweet verification failed: {verification_message}")
        browser.quit()
        return False

    ## Send tweet.
    sleep_duration = random.randint(
        1, 45 * 60
    )
    print(
        f"Sleeping for {sleep_duration // 60} minutes before sending the tweet..."
    )
    time.sleep(sleep_duration)

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


def main():
    """Generate a weekly review of highlights and takeaways from papers."""
    vs.validate_openai_env()
    tweet_type = "review_v2"
    tweet_type = "fable_v1"
    tweet_type = "insight_v1"

    ## Define arxiv code.
    arxiv_codes = db.get_arxiv_id_list(db.db_params, "summary_notes")
    done_codes = db.get_arxiv_id_list(db.db_params, "tweet_reviews")
    arxiv_codes = list(set(arxiv_codes) - set(done_codes))
    arxiv_codes = sorted(arxiv_codes)[-100:]
    citations_df = db.load_citations()
    citations_df = citations_df[citations_df.index.isin(arxiv_codes)]

    ## Select randomly, with probability based on citation count.
    citations_df["citation_count"] = citations_df["citation_count"].fillna(1) + 1
    citations_df["weight"] = (
        citations_df["citation_count"] / citations_df["citation_count"].sum()
    )
    citations_df["weight"] = citations_df["weight"] ** 0.5
    citations_df["weight"] = citations_df["weight"] / citations_df["weight"].sum()
    candidate_arxiv_codes = np.random.choice(
        citations_df.index,
        size=50,
        replace=False,
        p=citations_df["weight"] / citations_df["weight"].sum(),
    )

    candidate_abstracts = [
        db.get_recursive_summary(arxiv_code) for arxiv_code in candidate_arxiv_codes
    ]

    abstracts_str = "\n".join(
        [
            f"<abstract{idx+1}>\n{abstract}\n</abstract{idx+1}>\n"
            for idx, abstract in enumerate(candidate_abstracts)
        ]
    )

    arxiv_code_idx = vs.select_most_interesting_paper(
        abstracts_str, model="gpt-4o-mini"
    )
    arxiv_code = candidate_arxiv_codes[arxiv_code_idx - 1]

    last_post = db.get_latest_tstp(
        db.db_params,
        "tweet_reviews",
        extra_condition="where tweet_type='review_v2' and rejected=false",
    )
    if datetime.datetime.date(last_post) == datetime.datetime.today().date():
        tweet_type = "insight_v1"

    ## Load previous tweets.
    previous_tweets_df = db.load_tweet_insights(drop_rejected=True).head(5)
    previous_tweets = previous_tweets_df["tweet_insight"].values
    previous_tweets_str = "\n".join(previous_tweets)

    paper_summary = db.get_extended_notes(arxiv_code, expected_tokens=3000)
    paper_details = db.load_arxiv(arxiv_code)
    publish_date = paper_details["published"][0].strftime("%B %Y")
    publish_date_full = paper_details["published"][0].strftime("%b %d, %Y")
    author = paper_details["authors"][0]
    title_map = db.get_arxiv_title_dict()
    paper_title = title_map[arxiv_code]

    tweet_facts = (
        f"```**Title: {paper_title}**\n**Authors: {author}**\n{paper_summary}```"
    )
    post_tweet = f"arxiv link: https://arxiv.org/abs/{arxiv_code}\nllmpedia link: https://llmpedia.streamlit.app/?arxiv_code={arxiv_code}"

    repo_df = db.load_repositories(arxiv_code)
    if not repo_df.empty:
        repo_link = repo_df["repo_url"].values[0]
        if repo_link:
            post_tweet += f"\nrepo: {repo_link}"

    ## Run model.
    tweet = vs.write_tweet(
        previous_tweets=previous_tweets_str,
        tweet_facts=tweet_facts,
        tweet_type=tweet_type,
        model="claude-3-5-sonnet-20240620",  # "gpt-4o-2024-08-06",
        temperature=0.8,
    )
    print("Generated tweet: ")
    print(tweet)

    if tweet_type != "review_v2":
        edited_tweet = vs.edit_tweet(
            tweet,
            tweet_facts=tweet_facts,
            tweet_type=tweet_type,
            model="claude-3-5-sonnet-20240620",  # "gpt-4o-2024-08-06",
            temperature=0.5,
        )
    else:
        edited_tweet = f'💭Review of "{paper_title}"\n\n{tweet}'
    edited_tweet = bold(edited_tweet, publish_date_full)

    print("Edited tweet: ")
    print(edited_tweet)

    ## Send tweet to API.
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

    send_tweet(edited_tweet, tweet_image_path, tweet_page_path, post_tweet)

    ## Store.
    db.insert_tweet_review(
        arxiv_code, edited_tweet, datetime.datetime.now(), tweet_type, rejected=False
    )
    em.send_email_alert(edited_tweet, arxiv_code)


if __name__ == "__main__":
    main()

