import sys, os
from dotenv import load_dotenv
from tqdm import tqdm
import random
import time

load_dotenv()
sys.path.append(os.environ.get("PROJECT_PATH"))

import utils.paper_utils as pu
import utils.db as db

semantic_map = {
    "paperId": "paper_id",
    "venue": "venue",
    "tldr_text": "tldr",
    "citationCount": "citation_count",
    "influentialCitationCount": "influential_citation_count",
}

OVERRIDE = False

def main():
    """ Load summaries and add missing ones."""
    arxiv_codes = db.get_arxiv_id_list(db.db_params, "summaries")
    done_codes = db.get_arxiv_id_list(db.db_params, "semantic_details")
    arxiv_codes = list(set(arxiv_codes) - set(done_codes))
    arxiv_codes = sorted(arxiv_codes)[::-1]

    items_added = 0
    errors = 0
    for arxiv_code in tqdm(arxiv_codes):
        if db.check_in_db(arxiv_code, db.db_params, "semantic_details"):
            if not OVERRIDE:
                continue
            else:
                db.remove_from_db(arxiv_code, db.db_params, "semantic_details")

        ## Get Semantic Scholar info.
        ss_info = pu.get_semantic_scholar_info(arxiv_code)
        if ss_info is None:
            print(f"\nERROR: Could not find {arxiv_code} in Semantic Scholar.")
            errors += 1
            continue

        ss_info = pu.transform_flat_dict(pu.flatten_dict(ss_info), semantic_map)
        ss_info["arxiv_code"] = arxiv_code
        pu.store_local(ss_info, arxiv_code, "semantic_meta")
        db.upload_to_db(ss_info, db.db_params, "semantic_details")
        items_added += 1
        # print(f"\nAdded {arxiv_code} to semantic_details.")
        time.sleep(0.005)

    print(f"Process complete. Added {items_added} items in total.")
    if errors > 0:
        print(f"Encountered {errors} errors during processing.")

if __name__ == '__main__':
    main()
