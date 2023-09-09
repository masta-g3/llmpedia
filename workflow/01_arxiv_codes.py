import sys, os
import json
import shutil
import os, re
from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.environ.get("PROJECT_PATH"))
os.chdir(os.environ.get("PROJECT_PATH"))

import utils.paper_utils as pu

db_params = {
    "dbname": os.environ["DB_NAME"],
    "user": os.environ["DB_USER"],
    "password": os.environ["DB_PASS"],
    "host": os.environ["DB_HOST"],
    "port": os.environ["DB_PORT"],
}


import ast

def convert_string_to_dict(s):
    try:
        # Try to convert the string representation of a dictionary to an actual dictionary
        return ast.literal_eval(s)
    except (SyntaxError, ValueError):
        return s

def convert_innert_dict_strings_to_actual_dicts(data):
    if isinstance(data, str):
        return convert_string_to_dict(data)
    elif isinstance(data, dict):
        for key in data:
            data[key] = convert_innert_dict_strings_to_actual_dicts(data[key])
        return data
    elif isinstance(data, list):
        for i, item in enumerate(data):
            data[i] = convert_innert_dict_strings_to_actual_dicts(item)
        return data
    else:
        return data

def rename_file(fname: str, arxiv_code: str):
    """Rename file to Arxiv code."""
    if arxiv_code is None:
        return None
    old_path = os.path.join("summaries", fname)
    new_path = os.path.join("summaries", arxiv_code + ".json")
    if os.path.exists(old_path):
        shutil.move(old_path, new_path)
        return new_path
    return None


def main():
    """Load summaries and add missing ones."""
    titles = list(pu.get_arxiv_title_dict(db_params).values())

    fnames = [f for f in os.listdir("summaries") if ".json" in f]
    added_summaries = 0
    added_arxiv = 0
    errors = 0
    for fname in fnames:
        with open(f"summaries/{fname}", "r") as f:
            content = f.read().strip()
        data = json.loads(content)
        data = convert_innert_dict_strings_to_actual_dicts(data)

        data_title = data.get("Title", None)
        if data_title is None:
            errors += 1
            print(f"ERROR: JSON file {fname} does not contain a title.")
            continue

        ## Check similarity against all titles
        title_sim = [pu.tfidf_similarity(data_title, t) for t in titles]
        if max(title_sim) > 0.9:
            continue

        ## Get code.
        arxiv_info = pu.get_arxiv_info(data_title)
        if arxiv_info is None:
            print(f"ERROR: Could not find {data_title} in Arxiv. Please verify.")
            errors += 1
            continue
        arxiv_url = arxiv_info.entry_id
        arxiv_code = re.sub(r"v\d+$", "", arxiv_url.split("/")[-1])
        data["arxiv_code"] = arxiv_code

        ## Store summary.
        if not pu.check_in_db(arxiv_code, db_params, "summaries"):
            flat_entries = pu.transform_flat_dict(
                pu.flatten_dict(data), pu.summary_col_mapping
            )
            pu.upload_to_db(flat_entries, db_params, "summaries")
            print(f"Added {data_title} to summaries table.")
            added_summaries += 1

        ## Extract arxiv info.
        if not pu.check_in_db(arxiv_code, db_params, "arxiv_details"):
            ## Store in DB.
            processed_data = pu.process_arxiv_data(arxiv_info._raw)
            pu.store_local(arxiv_info._raw, arxiv_code, "arxiv_objects")
            pu.upload_to_db(processed_data, db_params, "arxiv_details")
            print(f"Added {data_title} to arxiv_details table.")

        ## Rename file if needed.
        if arxiv_code not in fname:
            rename_file(fname, arxiv_code)

    print(
        f"Process complete. Added {added_summaries} summaries and {added_arxiv} arxiv entries."
    )
    if errors > 0:
        print(f"Encountered {errors} errors during processing.")


if __name__ == "__main__":
    main()
