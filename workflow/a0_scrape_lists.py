import platform
import sys, os
import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from dateutil.parser import parse
from selenium import webdriver
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from dotenv import load_dotenv
import feedparser
import time

load_dotenv()
PROJECT_PATH = os.getenv('PROJECT_PATH', '/app')
sys.path.append(PROJECT_PATH)

import utils.paper_utils as pu

def scrape_ml_papers_of_the_week(start_date, end_date=None):
    if end_date is None:
        end_date = start_date

    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")
    year = start_date.year

    df = pd.DataFrame(columns=["arxiv_code", "title"])

    response = requests.get("https://github.com/dair-ai/ML-Papers-of-the-Week")
    soup = BeautifulSoup(response.content, "html.parser")

    for header in soup.find_all("h2"):
        date_range_text = header.get_text(strip=True)
        if "Top ML Papers of the Week" in date_range_text:
            date_range = extract_date_range(date_range_text, year)

            if overlaps_with_range(date_range, start_date, end_date):
                table = header.find_next("table")
                for row in table.find_all("tr")[1:]:
                    cols = row.find_all("td")
                    if len(cols) == 2:
                        title = cols[0].get_text(strip=True)
                        links = cols[1].find_all("a", href=True)
                        arxiv_link = next(
                            (link for link in links if "arxiv.org" in link["href"]),
                            None,
                        )
                        if arxiv_link:
                            arxiv_code = arxiv_link["href"].split("/")[-1]
                            df = df._append(
                                {"arxiv_code": arxiv_code, "title": title},
                                ignore_index=True,
                            )

    return df


def extract_date_range(header_text, year):
    date_part = header_text.split("(")[-1].split(")")[0]
    if " - " in date_part:
        start_date_str, end_date_str = date_part.split(" - ")
    else:
        start_date_str, end_date_str = date_part.split("-")

    if len(end_date_str.split()) == 1:
        end_date_str = start_date_str.split()[0] + " " + end_date_str

    start_date = parse(start_date_str.strip() + f", {year}")
    end_date = parse(end_date_str.strip() + f", {year}")
    return (start_date, end_date)


def overlaps_with_range(date_range, start_date, end_date):
    range_start, range_end = date_range
    return not (range_end < start_date or range_start > end_date)


def scrape_huggingface_papers(start_date, end_date=None):
    """Scrape arxiv codes and titles from huggingface.co/papers."""
    if end_date is None:
        end_date = start_date

    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")

    df = pd.DataFrame(columns=["arxiv_code", "title"])
    delta = timedelta(days=1)
    while start_date <= end_date:
        date_str = start_date.strftime("%Y-%m-%d")
        url = f"https://huggingface.co/papers?date={date_str}"
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")

        for link in soup.find_all("a", href=True, class_="cursor-pointer"):
            href = link["href"]
            if href.startswith("/papers/"):
                code = href.split("/")[-1]
                title = link.get_text(strip=True)
                if title:
                    df = df._append(
                        {"arxiv_code": code, "title": title}, ignore_index=True
                    )

        start_date += delta

    df.drop_duplicates(subset="arxiv_code", keep="first", inplace=True)
    return df


def setup_browser():
    chrome_options = ChromeOptions()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--remote-debugging-port=9222')

    if os.path.exists('/.dockerenv'):
        chrome_options.binary_location = '/usr/bin/chromium'
        service = ChromeService('/usr/bin/chromedriver')
        driver = webdriver.Chrome(service=service, options=chrome_options)
    else:
        if platform.system() == 'Darwin':
            driver = webdriver.Chrome(options=chrome_options)
        else:
            raise Exception("Unsupported OS")
        
    return driver


def scrape_rsrch_space_papers(start_date, end_date=None):
    if end_date is None:
        end_date = start_date

    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")

    df = pd.DataFrame(columns=["arxiv_code", "title"])

    driver = setup_browser()
    driver.get("http://rsrch.space")
    time.sleep(5)
    soup = BeautifulSoup(driver.page_source, "html.parser")
    driver.quit()

    entries = soup.find_all(
        "a", class_="flex justify-between text-secondary py-1 group text-md"
    )
    for entry in entries:
        date_str = entry.find("p", class_="font-berkeley").text.strip()
        entry_date = datetime.strptime(date_str, "%Y-%m-%d")

        if start_date <= entry_date <= end_date:
            href = entry["href"]
            arxiv_code = href.split("/")[-1]
            title = entry.find("strong").get_text(strip=True)
            df = df._append(
                {"arxiv_code": arxiv_code, "title": title}, ignore_index=True
            )
    return df


def scrape_ai_news_papers(start_date, end_date=None):
    if end_date is None:
        end_date = start_date

    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")
    df = pd.DataFrame(columns=["arxiv_code", "title"])

    response = requests.get("https://buttondown.email/ainews/archive/")
    soup = BeautifulSoup(response.content, "html.parser")
    mailinglist_entry = soup.find_all("div", class_="email-list")[0]
    ## Get all <a> elements under the div
    mailinglist_entry = mailinglist_entry.find_all("a", href=True)

    for entry in mailinglist_entry:
        date_str = entry.find("div", class_="email-metadata").text.strip()
        entry_date = datetime.strptime(date_str, "%B %d, %Y")

        if start_date <= entry_date <= end_date:
            time.sleep(2)
            href = entry["href"]
            deep_response = requests.get(href)
            deep_soup = BeautifulSoup(deep_response.content, "html.parser")

            ## Find all arxiv links.
            arxiv_links = deep_soup.find_all("a", href=True)
            for link in arxiv_links:
                if "arxiv.org/abs" in link["href"]:
                    arxiv_code = link["href"].split("/")[-1]
                    arxiv_code = arxiv_code[:10]
                    title = link.get_text(strip=True)
                    df = df._append(
                        {"arxiv_code": arxiv_code, "title": title}, ignore_index=True
                    )
    df.drop_duplicates(subset="arxiv_code", keep="first", inplace=True)
    return df


def scrape_llm_research_papers():
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    response = requests.get("https://www.llmsresearch.com", headers=headers)
    soup = BeautifulSoup(response.content, "html.parser")
    hyperlinks = soup.find_all("a", href=lambda href: href and href.startswith("/p/summary"))
    df = pd.DataFrame(columns=["arxiv_code", "title"])
    
    for link in hyperlinks:
        time.sleep(1)  # To avoid overwhelming the server
        paper_url = "https://www.llmsresearch.com" + link['href']
        paper_response = requests.get(paper_url, headers=headers)
        paper_soup = BeautifulSoup(paper_response.content, "html.parser")
        
        arxiv_links = paper_soup.find_all("a", class_="link", href=lambda href: href and "arxiv.org/abs" in href)
        
        for arxiv_link in arxiv_links:
            href = arxiv_link['href']
            arxiv_code = href.split("/")[-1].split("?")[0].split("v")[0]  # Extract code and remove version
            title = arxiv_link.get_text(strip=True)
            
            df = df._append(
                {"arxiv_code": arxiv_code, "title": title},
                ignore_index=True
            )
    
    df.drop_duplicates(subset="arxiv_code", keep="first", inplace=True)
    return df
    
    

def scrape_emergentmind_papers():
    url = "https://www.emergentmind.com/feeds/rss"
    feed = feedparser.parse(url)
    entries = [(entry.link.split("/")[-1].split("?")[0], entry.title) for entry in feed.entries]
    entries_df = pd.DataFrame(entries, columns=["arxiv_code", "title"])
    return entries_df


def main():
    """Scrape arxiv codes and titles from huggingface.co/papers."""
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        start_date = (datetime.now() - timedelta(days=14)).strftime("%Y-%m-%d")
        end_date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    else:
        start_date = sys.argv[1]
        end_date = sys.argv[2] if len(sys.argv) == 3 else None

    # Perform scraping.
    print("Scraping HuggingFace...")
    hf_df = scrape_huggingface_papers(start_date, end_date)
    print(f"Collected {hf_df.shape[0]} papers.")
    print("Scraping Research Space...")
    rsrch_df = scrape_rsrch_space_papers(start_date, end_date)
    print(f"Collected {rsrch_df.shape[0]} papers.")
    print("Scraping ML Papers of the Week...")
    dair_df = scrape_ml_papers_of_the_week(start_date, end_date)
    print(f"Collected {dair_df.shape[0]} papers.")
    print("Scraping AI News...")
    ai_news_df = scrape_ai_news_papers(start_date, end_date)
    print(f"Collected {ai_news_df.shape[0]} papers.")
    print("Scraping Emergent Mind...")
    em_df = scrape_emergentmind_papers()
    print(f"Collected {em_df.shape[0]} papers.")
    print("Scraping LLM Research ...")
    llmr_df = scrape_llm_research_papers()
    print(f"Collected {llmr_df.shape[0]} papers.")

    ## Combine and extract new codes.
    df = pd.concat([hf_df, rsrch_df, dair_df, ai_news_df, em_df, llmr_df], ignore_index=True)
    df.drop_duplicates(subset="arxiv_code", keep="first", inplace=True)
    ## Remove "vX" from arxiv codes if present.
    df["arxiv_code"] = df["arxiv_code"].str.replace(r"v\d+$", "", regex=True)
    new_codes = df["arxiv_code"].tolist()
    new_codes = [code for code in new_codes if pu.is_arxiv_code(code)]
    done_codes = pu.list_s3_files("arxiv-text")
    nonllm_codes = pu.list_s3_files("nonllm-arxiv-text")

    ## Remote paper list.
    gist_id = "1dd189493c1890df6e04aaea6d049643"
    gist_filename = "llm_queue.txt"
    paper_list = pu.fetch_queue_gist(gist_id, gist_filename)

    ## Update and upload arxiv codes.
    paper_list = list(set(paper_list + new_codes))
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


if __name__ == "__main__":
    main()
