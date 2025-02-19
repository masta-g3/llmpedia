import argparse
import os, sys
import pandas as pd
from dotenv import load_dotenv
import psycopg2
import boto3

load_dotenv()
PROJECT_PATH = os.getenv('PROJECT_PATH', '/app')
sys.path.append(PROJECT_PATH)
os.chdir(PROJECT_PATH)

import utils.db.db_utils as db_utils
import utils.db.logging_db as logging_db
table_names = [
    "arxiv_chunks",
    "arxiv_details",
    "arxiv_large_parent_chunks",
    "arxiv_parent_chunks",
    "arxiv_qna",
    "bullet_list_summaries",
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
    "arxiv_repos",
    "arxiv_dashboards",
    "arxiv_chunk_map",
    "issue_reports",
]

s3_buckets = {
    "arxiv-text": ["txt"],
    "arxiv-chunks": ["json"],
    "semantic-meta": ["json"],
    "weekly-content": ["json"],
    "arxiv-md": ["md", "png"],  # Markdown and its images
    "arxiv-art": ["png"],  # Thumbnails
    "arxiv-pdfs": ["pdf"],  # Original PDFs
    "nonllm-arxiv-text": ["txt"],  # For moving non-LLM papers
}


def delete_from_s3(arxiv_code: str):
    """Delete paper files from S3 buckets."""
    s3 = boto3.client('s3')
    for bucket, extensions in s3_buckets.items():
        for ext in extensions:
            try:
                # Special handling for markdown directory which contains multiple files
                if bucket == "arxiv-md":
                    try:
                        # List all objects with the prefix
                        response = s3.list_objects_v2(
                            Bucket=bucket,
                            Prefix=f"{arxiv_code}/"
                        )
                        # Delete each object in the directory
                        for obj in response.get('Contents', []):
                            s3.delete_object(Bucket=bucket, Key=obj['Key'])
                        print(f"Deleted markdown directory for {arxiv_code} from {bucket} bucket.")
                    except Exception as e:
                        print(f"Error deleting markdown directory for {arxiv_code} from {bucket}: {e}")
                else:
                    key = f"{arxiv_code}.{ext}"
                    s3.delete_object(Bucket=bucket, Key=key)
                    print(f"Deleted {key} from {bucket} bucket.")
            except Exception as e:
                print(f"Error deleting {arxiv_code}.{ext} from {bucket}: {e}")


def delete_from_vector_store(arxiv_code: str):
    """Delete paper embeddings from vector store."""
    try:
        with psycopg2.connect(**db_utils.db_params) as conn:
            with conn.cursor() as cur:
                ## Delete from langchain embeddings
                cur.execute("""
                    DELETE FROM langchain_pg_embedding 
                    WHERE cmetadata->>'arxiv_code' = %s
                """, (arxiv_code,))
                print(f"Deleted {arxiv_code} embeddings from vector store.")
    except Exception as e:
        print(f"Error deleting embeddings for {arxiv_code}: {e}")


def delete_from_db(arxiv_code: str):
    """Delete paper from all database tables."""
    with psycopg2.connect(**db_utils.db_params) as conn:
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
    try:
        arxiv_df = pd.read_pickle("data/arxiv.pkl")
        if arxiv_code in arxiv_df.index:
            arxiv_df.drop(arxiv_code, inplace=True)
            arxiv_df.to_pickle("data/arxiv.pkl")
            print(f"Removed {arxiv_code} from arxiv.pkl")
    except FileNotFoundError:
        print("arxiv.pkl not found, skipping...")

    ## GPT Reviews.
    try:
        reviews_df = pd.read_pickle("data/reviews.pkl")
        if arxiv_code in reviews_df.index:
            reviews_df.drop(arxiv_code, inplace=True)
            reviews_df.to_pickle("data/reviews.pkl")
            print(f"Removed {arxiv_code} from reviews.pkl")
    except FileNotFoundError:
        print("reviews.pkl not found, skipping...")

    ## Cluster assignment.
    try:
        topics_df = pd.read_pickle("data/topics.pkl")
        if arxiv_code in topics_df.index:
            topics_df.drop(arxiv_code, inplace=True)
            topics_df.to_pickle("data/topics.pkl")
            print(f"Removed {arxiv_code} from topics.pkl")
    except FileNotFoundError:
        print("topics.pkl not found, skipping...")

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
    img_file = f"data/arxiv_art/{arxiv_code}.png"
    if os.path.exists(img_file):
        os.remove(img_file)
        print(f"Deleted {img_file}.")

    ## Repository info.
    repo_file = f"data/arxiv_repos/{arxiv_code}.json"
    if os.path.exists(repo_file):
        os.remove(repo_file)
        print(f"Deleted {repo_file}.")

    ## Embeddings.
    embeddings_file = f"data/embeddings/{arxiv_code}.pkl"
    if os.path.exists(embeddings_file):
        os.remove(embeddings_file)
        print(f"Deleted {embeddings_file}.")

    ## PDF file.
    pdf_file = f"data/arxiv_pdfs/{arxiv_code}.pdf"
    if os.path.exists(pdf_file):
        os.remove(pdf_file)
        print(f"Deleted {pdf_file}.")

    ## Markdown directory.
    markdown_dir = f"data/arxiv_md/{arxiv_code}"
    if os.path.exists(markdown_dir):
        import shutil
        shutil.rmtree(markdown_dir)
        print(f"Deleted markdown directory {markdown_dir}.")

    ## Delete from S3.
    print("Cleaning up S3...")
    delete_from_s3(arxiv_code)

    ## Delete from vector store.
    print("Cleaning up vector store...")
    delete_from_vector_store(arxiv_code)

    ## Delete from database.
    print("Cleaning up database...")
    delete_from_db(arxiv_code)


def main(arxiv_code):
    """Delete paper identified by arXiv code."""
    if arxiv_code:
        delete_paper(arxiv_code)
    else:
        arxiv_codes = logging_db.get_reported_non_llm_papers()
        for arxiv_code in arxiv_codes:
            delete_paper(arxiv_code)
            logging_db.update_reported_status(arxiv_code, "non_llm")


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