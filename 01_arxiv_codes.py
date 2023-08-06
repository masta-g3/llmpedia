from arxiv_utils import tfidf_similarity, get_arxiv_info
import json
import shutil
import os, re


def rename_file(fname: str, arxiv_code: str):
    """ Rename file to Arxiv code. """
    if arxiv_code is None:
        return None
    old_path = os.path.join("summaries", fname)
    new_path = os.path.join("summaries", arxiv_code + ".json")
    if os.path.exists(old_path):
        shutil.move(old_path, new_path)
        return new_path
    return None


def main():
    """ Load summaries and add missing ones."""
    with open("arxiv_code_map.json", "r") as f:
        staging_dict = json.load(f)
    titles = list(staging_dict.values()) + ["XXXXXXXXXXX"]

    fnames = [f for f in os.listdir("summaries") if '.json' in f]
    items_added = 0
    errors = 0
    for fname in fnames:
        with open(f"summaries/{fname}", "r") as f:
            data = json.load(f)
            data_title = data.get("Title", None)
            if data_title is None:
                errors += 1
                print(f"ERROR: JSON file {fname} does not contain a title.")
                continue

            ## Check similarity against all titles
            title_sim = [tfidf_similarity(data_title, t) for t in titles]
            if max(title_sim) > 0.9:
                continue

            ## Extract arxiv code.
            arxiv_info = get_arxiv_info(data_title)
            if arxiv_info is None:
                errors += 1
                print(f"ERROR: Could not find {data_title} in Arxiv. Please verify.")
                continue
            arxiv_url = arxiv_info.entry_id
            arxiv_code = re.sub(r'v\d+$', '', arxiv_url.split("/")[-1])

            ## Store arxiv object.
            with open(f"arxiv_objects/{arxiv_code}.json", "w") as f:
                json.dump(arxiv_info._raw, f)

            ## Rename file if needed.
            if arxiv_code not in fname:
                rename_file(fname, arxiv_code)

            ## Add to staging dictionary.
            if arxiv_code not in staging_dict:
                staging_dict[arxiv_code] = data_title
                items_added += 1
                print(f"Added {data_title}.")

    if items_added > 0:
        with open("arxiv_code_map.json", "w") as f:
            json.dump(staging_dict, f)
    print(f"Process complete. Added {items_added} items in total.")
    if errors > 0:
        print(f"Encountered {errors} errors during processing.")

if __name__ == '__main__':
    main()
