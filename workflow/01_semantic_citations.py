import utils.paper_utils as pu
import json
import os


db_params = {
    'dbname': os.environ['DB_NAME'],
    'user': os.environ['DB_USER'],
    'password': os.environ['DB_PASS'],
    'host': os.environ['DB_HOST'],
    'port': os.environ['DB_PORT']
}

semantic_map = {
    "paperId": "paper_id",
    "venue": "venue",
    "tldr_text": "tldr",
    "citationCount": "citation_count",
    "influentialCitationCount": "influential_citation_count",
}

def main():
    """ Load summaries and add missing ones."""
    with open("arxiv_code_map.json", "r") as f:
        staging_dict = json.load(f)

    items_added = 0
    errors = 0
    for arxiv_code, title in staging_dict.items():
        if not pu.check_in_db(arxiv_code, db_params, "semantic_details"):
            ## Get Semantic Scholar info.
            ss_info = pu.get_semantic_scholar_info(arxiv_code)
            if ss_info is None:
                errors += 1
                print(f"ERROR: Could not find {arxiv_code} in Semantic Scholar.")
                continue
        ss_info = pu.transform_flat_dict(pu.flatten_dict(ss_info), semantic_map)
        ss_info["arxiv_code"] = arxiv_code
        pu.store_local(ss_info, arxiv_code, "semantic_objects")
        pu.upload_to_db(ss_info, db_params, "semantic_details")
        items_added += 1
        print(f"Added {arxiv_code}.")

    print(f"Process complete. Added {items_added} items in total.")
    if errors > 0:
        print(f"Encountered {errors} errors during processing.")

if __name__ == '__main__':
    main()
