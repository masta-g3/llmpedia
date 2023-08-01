import argparse
import os
import json
import pandas as pd

def delete_paper(arxiv_code: str):
    """Delete paper from all data sources."""
    ## Code map.
    with open("arxiv_code_map.json", "r") as f:
        arxiv_code_map = json.load(f)
    del arxiv_code_map[arxiv_code]
    with open("arxiv_code_map.json", "w") as f:
        json.dump(arxiv_code_map, f)

    ## Metadata.
    arxiv_df = pd.read_pickle("data/arxiv.pkl")
    arxiv_df.drop(arxiv_code, inplace=True)
    arxiv_df.to_pickle("data/arxiv.pkl")

    ## GPT Reviews.
    reviews_df = pd.read_pickle("data/reviews.pkl")
    reviews_df.drop(arxiv_code, inplace=True)
    reviews_df.to_pickle("data/reviews.pkl")

    ## Cluster assignment.
    topics_df = pd.read_pickle("data/topics.pkl")
    topics_df.drop(arxiv_code, inplace=True)
    topics_df.to_pickle("data/topics.pkl")

    ## Summaries.
    summary_file = f"summaries/{arxiv_code}.json"
    if os.path.exists(summary_file):
        os.remove(summary_file)

    ## Arxiv objects.
    arxiv_obj_file = f"arxiv_objects/{arxiv_code}.json"
    if os.path.exists(arxiv_obj_file):
        os.remove(arxiv_obj_file)

def main(arxiv_code):
    """Delete paper identified by arXiv code."""
    delete_paper(arxiv_code)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Delete paper identified by arXiv code from all data sources.')
    parser.add_argument('arxiv_code', type=str, help='arXiv code of the paper to be deleted.')
    args = parser.parse_args()
    main(args.arxiv_code)
