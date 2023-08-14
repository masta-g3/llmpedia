import os
import json
import pandas as pd


def get_author_list(authors):
    """Helper function to extract author names."""
    return [author["name"] for author in authors] if authors else []


def main():
    """Load summaries and add missing ones."""
    try:
        with open("../arxiv_code_map.json", "r") as f:
            staging_dict = json.load(f)
    except FileNotFoundError:
        print("File arxiv_code_map.json not found.")
        return

    try:
        df = pd.read_pickle("../data/arxiv.pkl")
    except FileNotFoundError:
        df = pd.DataFrame(
            columns=[
                "URL",
                "Title",
                "Published",
                "Updated",
                "Authors",
                "Summary",
                "Comment",
            ]
        )

    items_added = 0
    for arxiv_code, title in staging_dict.items():
        if arxiv_code not in df.index:
            try:
                with open(f"arxiv_objects/{arxiv_code}.json", "r") as f:
                    data = json.load(f)
            except FileNotFoundError:
                print(f"File {arxiv_code}.json not found.")
                continue

            authors = get_author_list(data.get("authors"))
            comment = data.get("arxiv_comment", "")

            df.loc[arxiv_code] = {
                "URL": data.get("id", ""),
                "Title": data.get("title", ""),
                "Published": data.get("published", ""),
                "Updated": data.get("updated", ""),
                "Authors": authors,
                "Summary": data.get("summary", ""),
                "Comment": comment,
            }
            items_added += 1

    df.to_pickle("data/arxiv.pkl")
    print(f"Process complete. Added {items_added} items in total.")


if __name__ == "__main__":
    main()
