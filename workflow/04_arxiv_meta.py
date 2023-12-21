import sys, os
import json
import shutil
import os, re
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.environ.get("PROJECT_PATH"))
os.chdir(os.environ.get("PROJECT_PATH"))

import utils.paper_utils as pu
import utils.db as db


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
    titles = list(db.get_arxiv_title_dict(pu.db_params).values())
    codes = list(db.get_arxiv_title_dict(pu.db_params).keys())
    local_paper_path= os.path.join(
        os.environ.get("PROJECT_PATH"), "data", "summaries"
    )
    local_fnames = [f for f in os.listdir(local_paper_path) if ".json" in f]
    local_codes = [f.split(".json")[0] for f in local_fnames]

    pending_codes = list(set(local_codes) - set(codes))

    added_summaries = 0
    added_arxiv = 0
    errors = 0

    for arxiv_code in tqdm(pending_codes):
        # arxiv_code = fname.replace(".json", "")
        fname = arxiv_code + ".json"
        if arxiv_code in codes:
            # print(f"Skipping '{fname}' as it is already in the database.")
            continue

        with open(os.path.join(local_paper_path, fname), "r") as f:
            content = f.read().strip()
        data = json.loads(content)
        data = pu.convert_innert_dict_strings_to_actual_dicts(data)
        if "applied_example" in data["takeaways"]:
            data["takeaways"]["example"] = data["takeaways"]["applied_example"]
            del data["takeaways"]["applied_example"]

        data_title = data.get("Title", None)
        if data_title is None:
            errors += 1
            print(f"ERROR: JSON file {fname} does not contain a title.")
            continue

        ## Check similarity against all titles.
        pu.vectorizer.fit_transform([data_title] + titles)
        title_sim = pu.compute_optimized_similarity(data_title, titles)
        # title_sim = [pu.tfidf_similarity(data_title, t) for t in titles]
        if max(title_sim) > 0.95:
            print(f"ERROR: '{data_title}' is too similar to an existing title.")
            continue

        ## Get code.
        arxiv_info = pu.get_arxiv_info(arxiv_code, data_title)
        if arxiv_info is None:
            print(f"ERROR: Could not find '{data_title}' in Arxiv. Please verify.")
            errors += 1
            continue
        arxiv_url = arxiv_info.entry_id
        arxiv_code = re.sub(r"v\d+$", "", arxiv_url.split("/")[-1])
        data["arxiv_code"] = arxiv_code

        ## Store summary.
        if not db.check_in_db(arxiv_code, pu.db_params, "summaries"):
            flat_entries = pu.transform_flat_dict(
                pu.flatten_dict(data), pu.summary_col_mapping
            )
            db.upload_to_db(flat_entries, pu.db_params, "summaries")
            # print(f"Added '{data_title}' to summaries table.")
            added_summaries += 1

        ## Extract arxiv info.
        if not db.check_in_db(arxiv_code, pu.db_params, "arxiv_details"):
            ## Store in DB.
            processed_data = pu.process_arxiv_data(arxiv_info._raw)
            pu.store_local(arxiv_info._raw, arxiv_code, "arxiv_meta")
            db.upload_to_db(processed_data, pu.db_params, "arxiv_details")
            # print(f"Added '{data_title}' to arxiv_details table.")
            added_arxiv += 1

        ## Rename file if needed.
        # if arxiv_code not in fname:
        # rename_file(fname, arxiv_code)

    print(
        f"Process complete. Added {added_summaries} summaries and {added_arxiv} arxiv entries."
    )
    if errors > 0:
        print(f"Encountered {errors} errors during processing.")


if __name__ == "__main__":
    main()
