import argparse
import os, sys
import pandas as pd
from dotenv import load_dotenv
import psycopg2

load_dotenv()
sys.path.append(os.environ.get("PROJECT_PATH"))
os.chdir(os.environ.get("PROJECT_PATH"))

import utils.db as db

table_names = [
    "arxiv_chunks",
    "arxiv_details",
    "arxiv_large_parent_chunks",
    "arxiv_parent_chunks",
    "arxiv_qna",
    "bullet_list_summary",
    "recursive_summaries",
    "semantic_details",
    "similar_documents",
    "summaries",
    "summaries_ext",
    "summary_markdown",
    "summary_notes",
    "summary_tweets",
    "topics",
    "tweet_reviews",
]


def delete_from_db(arxiv_code: str):
    with psycopg2.connect(**db.db_params) as conn:
        with conn.cursor() as cur:
            for table_name in table_names:
                cur.execute(
                    f"DELETE FROM {table_name} WHERE arxiv_code = %s", (arxiv_code,)
                )
                print(f"Deleted {arxiv_code} from {table_name}.")


def delete_paper(arxiv_code: str):
    """Delete paper from all data sources."""
    print("Cleaning up pickles...")
    ## Metadata.
    arxiv_df = pd.read_pickle("data/arxiv.pkl")
    if arxiv_code in arxiv_df.index:
        arxiv_df.drop(arxiv_code, inplace=True)
        arxiv_df.to_pickle("data/arxiv.pkl")

    ## GPT Reviews.
    reviews_df = pd.read_pickle("data/reviews.pkl")
    if arxiv_code in reviews_df.index:
        reviews_df.drop(arxiv_code, inplace=True)
        reviews_df.to_pickle("data/reviews.pkl")

    ## Cluster assignment.
    topics_df = pd.read_pickle("data/topics.pkl")
    if arxiv_code in topics_df.index:
        topics_df.drop(arxiv_code, inplace=True)
        topics_df.to_pickle("data/topics.pkl")

    print("Removing files...")
    ## Summaries.
    summary_file = f"data/summaries/{arxiv_code}.json"
    if os.path.exists(summary_file):
        os.remove(summary_file)
        print(f"Deleted {summary_file}.")

    ## Arxiv objects.
    arxiv_obj_file = f"data/arxiv_objects/{arxiv_code}.json"
    if os.path.exists(arxiv_obj_file):
        os.remove(arxiv_obj_file)
        print(f"Deleted {arxiv_obj_file}.")

    ## Citations.
    citations_file = f"data/semantic_meta/{arxiv_code}.json"
    if os.path.exists(citations_file):
        os.remove(citations_file)
        print(f"Deleted {citations_file}.")

    ## Arxiv text.
    arxiv_text_file = f"data/arxiv_text/{arxiv_code}.txt"
    if os.path.exists(arxiv_text_file):
        os.rename(arxiv_text_file, f"data/nonllm_arxiv_text/{arxiv_code}.txt")
        print(f"Moved {arxiv_text_file} raw file to non-LLM archive.")

    ## Arxiv chunks.
    arxiv_chunks_file = f"data/arxiv_chunks/{arxiv_code}.json"
    if os.path.exists(arxiv_chunks_file):
        os.remove(arxiv_chunks_file)
        print(f"Deleted {arxiv_chunks_file}.")

    ## Arxiv large parent chunks.
    arxiv_large_chunks_file = f"data/arxiv_large_parent_chunks/{arxiv_code}.json"
    if os.path.exists(arxiv_large_chunks_file):
        os.remove(arxiv_large_chunks_file)
        print(f"Deleted {arxiv_large_chunks_file}.")

    ## Arxiv QnA.
    qna_file = f"data/arxiv_qna/{arxiv_code}.json"
    if os.path.exists(qna_file):
        os.remove(qna_file)
        print(f"Deleted {qna_file}.")

    ## Images.
    img_file = f"img/{arxiv_code}.png"
    if os.path.exists(img_file):
        os.remove(img_file)
        print(f"Deleted {img_file}.")

    ## Delete from database.
    delete_from_db(arxiv_code)


def main(arxiv_code):
    """Delete paper identified by arXiv code."""
    if arxiv_code:
        delete_paper(arxiv_code)
    else:
        arxiv_codes = db.get_reported_non_llm_papers()
        for arxiv_code in arxiv_codes:
            delete_paper(arxiv_code)
            db.update_reported_status(arxiv_code, "non_llm")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Delete paper identified by arXiv code from all data sources."
    )
    parser.add_argument(
        "arxiv_code",
        help="arXiv code of the paper to be deleted.",
        type=str,
        nargs='?',
        default=None,
    )
    args = parser.parse_args()
    main(args.arxiv_code)
