import os
import json
import pandas as pd


def load_or_create_reviews_df():
    """Load or create reviews DataFrame."""
    if os.path.exists("data/reviews.pkl"):
        reviews_df = pd.read_pickle("data/reviews.pkl")
    else:
        reviews_df = pd.DataFrame(
            columns=[
                "main_contribution",
                "headline",
                "description",
                "takeaways",
                "category",
                "novelty_analysis",
                "novelty_score",
                "technical_analysis",
                "technical_score",
                "enjoyable_analysis",
                "enjoyable_score",
            ]
        )
    return reviews_df


def update_reviews_df(df: pd.DataFrame, arxiv_code: str, data: dict) -> pd.DataFrame:
    """Update reviews DataFrame with new data."""
    review_data = {key: data.get(key, None) for key in df.columns}
    df.loc[arxiv_code] = review_data
    return df


def main():
    """Load arxiv code map and add missing summaries."""
    with open("arxiv_code_map.json", "r") as f:
        arxiv_code_map = json.load(f)

    reviews_df = load_or_create_reviews_df()

    items_added = 0
    for arxiv_code in arxiv_code_map.keys():
        if arxiv_code in reviews_df.index:
            continue
        fname = f"summaries/{arxiv_code}.json"
        if not os.path.exists(fname):
            print(f"ERROR: JSON file {fname} does not exist.")
            continue

        with open(fname, "r") as f:
            data = json.load(f)
            reviews_df = update_reviews_df(reviews_df, arxiv_code, data)
            items_added += 1

    if items_added > 0:
        reviews_df.to_pickle("data/reviews.pkl")

    print(f"Process complete. Added {items_added} items in total.")


if __name__ == "__main__":
    main()
