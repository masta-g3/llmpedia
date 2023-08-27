import argparse
import os
import json
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
import psycopg2

load_dotenv()

db_params = {
    "dbname": os.environ["DB_NAME"],
    "user": os.environ["DB_USER"],
    "password": os.environ["DB_PASS"],
    "host": os.environ["DB_HOST"],
    "port": os.environ["DB_PORT"],
}
db_url = (
    f"postgresql+psycopg2://{db_params['user']}:{db_params['password']}"
    + "@{db_params['host']}:{db_params['port']}/{db_params['dbname']}"
)
engine = create_engine(db_url)


def delete_from_db(arxiv_code: str):
    with psycopg2.connect(**db_params) as conn:
        with conn.cursor() as cur:
            table_names = ["arxiv_details", "summaries", "topics"]
            for table_name in table_names:
                cur.execute(
                    f"DELETE FROM {table_name} WHERE arxiv_code = %s", (arxiv_code,)
                )
                print(f"Deleted {arxiv_code} from {table_name}.")


def delete_paper(arxiv_code: str):
    """Delete paper from all data sources."""
    ## Code map.
    try:
        with open("../arxiv_code_map.json", "r") as f:
            arxiv_code_map = json.load(f)
        del arxiv_code_map[arxiv_code]
        with open("../arxiv_code_map.json", "w") as f:
            json.dump(arxiv_code_map, f)
        print(f"Deleted {arxiv_code} from arxiv_code_map.json.")
    except:
        print("arXiv code not found in arxiv_code_map.json.")

    print("Cleaning up pickles...")

    ## Metadata.
    arxiv_df = pd.read_pickle("../data/arxiv.pkl")
    arxiv_df.drop(arxiv_code, inplace=True)
    arxiv_df.to_pickle("data/arxiv.pkl")

    ## GPT Reviews.
    reviews_df = pd.read_pickle("../data/reviews.pkl")
    reviews_df.drop(arxiv_code, inplace=True)
    reviews_df.to_pickle("data/reviews.pkl")

    ## Cluster assignment.
    topics_df = pd.read_pickle("../data/topics.pkl")
    topics_df.drop(arxiv_code, inplace=True)
    topics_df.to_pickle("data/topics.pkl")

    ## Summaries.
    summary_file = f"summaries/{arxiv_code}.json"
    if os.path.exists(summary_file):
        os.remove(summary_file)
        print(f"Deleted {summary_file}.")

    ## Arxiv objects.
    arxiv_obj_file = f"arxiv_objects/{arxiv_code}.json"
    if os.path.exists(arxiv_obj_file):
        os.remove(arxiv_obj_file)
        print(f"Deleted {arxiv_obj_file}.")

    ## Delete from database.
    delete_from_db(arxiv_code)


def main(arxiv_code):
    """Delete paper identified by arXiv code."""
    delete_paper(arxiv_code)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Delete paper identified by arXiv code from all data sources."
    )
    parser.add_argument(
        "arxiv_code", type=str, help="arXiv code of the paper to be deleted."
    )
    args = parser.parse_args()
    main(args.arxiv_code)
