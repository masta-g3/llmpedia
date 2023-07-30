import pandas as pd
import json
import os

def main():
    """ Load all summaries and create a dataframe."""
    fnames = [f for f in os.listdir("summaries") if '.json' in f]
    result_dict = {}
    for fname in fnames:
        with open(f"summaries/{fname}", "r") as f:
            result_dict[fname] = json.load(f)

    result_df = pd.DataFrame(result_dict).T
    result_df["Published"] = pd.to_datetime(result_df["Published"])
    result_df = result_df.sort_values("Published", ascending=False)

    result_df.to_pickle("papers_df.pkl")
    print(f"Parsed {len(result_df)} papers.")


if __name__ == '__main__':
    main()