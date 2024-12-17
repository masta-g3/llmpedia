import sys, os
from dotenv import load_dotenv

load_dotenv()

PROJECT_PATH = os.getenv('PROJECT_PATH', '/app')
sys.path.append(PROJECT_PATH)

os.chdir(PROJECT_PATH)

import pandas as pd
from tqdm import tqdm

import utils.paper_utils as pu
import utils.vector_store as vs
import utils.db as db
from utils.logging_utils import setup_logger

# Set up logging
logger = setup_logger(__name__, "f0_review.log")

LOCAL_PAPER_PATH = os.path.join(os.environ.get("PROJECT_PATH"), "data", "summaries")
RETRIES = 1


def main():
    logger.info("Starting paper review process")
    ## Health check.
    vs.validate_openai_env()

    ## Get paper list.
    arxiv_codes = db.get_arxiv_id_list(pu.db_params, "summary_notes")
    existing_papers = db.get_arxiv_id_list(pu.db_params, "summaries")
    arxiv_codes = list(set(arxiv_codes) - set(existing_papers))

    arxiv_codes = sorted(arxiv_codes)[::-1]
    logger.info(f"Found {len(arxiv_codes)} papers to review")

    for arxiv_code in arxiv_codes:
        new_content = db.get_extended_notes(arxiv_code, expected_tokens=1200)

        ## Try to run LLM process up to 3 times.
        success = False
        for i in range(RETRIES):
            try:
                summary = vs.review_llm_paper(new_content, model="claude-3-5-sonnet-20241022")
                success = True
                break
            except Exception as e:
                logger.error(f"Failed to run LLM for '{arxiv_code}'. Attempt {i+1}/{RETRIES}.")
                logger.error(str(e))
                continue
        if not success:
            logger.warning(f"Failed to run LLM for '{arxiv_code}' after {RETRIES} attempts. Skipping...")
            continue

        ## Extract and combine results.
        result_dict = summary.model_dump_json()

        ## Store on DB.
        data = pu.convert_innert_dict_strings_to_actual_dicts(result_dict)
        ## ToDo: Legacy, remove.
        if "applied_example" in data["takeaways"]:
            data["takeaways"]["example"] = data["takeaways"]["applied_example"]
            del data["takeaways"]["applied_example"]

        flat_entries = pu.transform_flat_dict(
            pu.flatten_dict(data), pu.summary_col_mapping
        )
        flat_entries["arxiv_code"] = arxiv_code
        flat_entries["tstp"] = pd.Timestamp.now()
        logger.info(f"Uploading review for {arxiv_code} to database")
        db.upload_to_db(flat_entries, pu.db_params, "summaries")

    logger.info("Paper review process completed")


if __name__ == "__main__":
    main()
