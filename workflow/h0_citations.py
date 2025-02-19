import sys, os
from dotenv import load_dotenv
from tqdm import tqdm
import random
import time
import pandas as pd

load_dotenv()
PROJECT_PATH = os.getenv('PROJECT_PATH', '/app')
sys.path.append(PROJECT_PATH)

import utils.paper_utils as pu
import utils.db.db_utils as db_utils
from utils.logging_utils import setup_logger

# Set up logging
logger = setup_logger(__name__, "h0_citations.log")

semantic_map = {
    "paperId": "paper_id",
    "venue": "venue",
    "tldr_text": "tldr",
    "citationCount": "citation_count",
    "influentialCitationCount": "influential_citation_count",
}

OVERRIDE = True

def main():
    """Load summaries and add missing ones."""
    logger.info("Starting citation fetching process.")
    arxiv_codes = db_utils.get_arxiv_id_list("summaries")
    existing_codes = set(db_utils.get_arxiv_id_list("semantic_details"))
    if not OVERRIDE:
        arxiv_codes = list(set(arxiv_codes) - existing_codes)
    arxiv_codes = sorted(arxiv_codes)[::-1][:100]

    total_papers = len(arxiv_codes)
    logger.info(f"Found {total_papers} papers to process for citations.")

    items_added = 0
    errors = 0
    title_dict = db_utils.get_arxiv_title_dict()
    
    for idx, arxiv_code in enumerate(arxiv_codes, 1):
        paper_title = title_dict.get(arxiv_code, "Unknown Title")
        clear_previous = False
        already_exists = arxiv_code in existing_codes
        if not OVERRIDE:
            if already_exists:
                logger.info(f"[{idx}/{total_papers}] Skipping: {arxiv_code} - '{paper_title}' (already exists)")
                continue
        else:
            clear_previous = True

        ss_info = pu.get_semantic_scholar_info(arxiv_code)
        if ss_info is None:
            logger.warning(f"[{idx}/{total_papers}] Failed to fetch citations: {arxiv_code} - '{paper_title}'")
            errors += 1
            continue

        if already_exists and clear_previous:
            logger.info(f"[{idx}/{total_papers}] Updating: {arxiv_code} - '{paper_title}'")
            db_utils.remove_by_arxiv_code(arxiv_code, "semantic_details")

        ss_info = pu.transform_flat_dict(pu.flatten_dict(ss_info), semantic_map)
        ss_info["arxiv_code"] = arxiv_code
        df = pd.DataFrame([ss_info])
        db_utils.upload_dataframe(df, "semantic_details")
        logger.info(f"[{idx}/{total_papers}] Added citations: {arxiv_code} - '{paper_title}'")
        items_added += 1
        time.sleep(random.uniform(0.12, 0.5))

    logger.info(f"Process complete. Added {items_added} items in total. Encountered {errors} errors.")

if __name__ == "__main__":
    main()
