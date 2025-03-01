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
import utils.db.db_utils as db_utils
import utils.db.paper_db as paper_db
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
    arxiv_codes = db_utils.get_arxiv_id_list("summary_notes")
    existing_papers = db_utils.get_arxiv_id_list("summaries")
    arxiv_codes = list(set(arxiv_codes) - set(existing_papers))

    arxiv_codes = sorted(arxiv_codes)[::-1]
    total_papers = len(arxiv_codes)
    logger.info(f"Found {total_papers} papers to review")
    
    title_map = db_utils.get_arxiv_title_dict()

    for idx, arxiv_code in enumerate(arxiv_codes, 1):
        paper_title = title_map.get(arxiv_code, "Unknown Title")
        new_content = paper_db.get_extended_notes(arxiv_code, expected_tokens=1200)

        ## Try to run LLM process up to 3 times.
        success = False
        for i in range(RETRIES):
            try:
                logger.info(f"[{idx}/{total_papers}] Reviewing: {arxiv_code} - '{paper_title}' (attempt {i+1}/{RETRIES})")
                summary = vs.review_llm_paper(new_content, model="claude-3-5-sonnet-20241022")
                success = True
                break
            except Exception as e:
                logger.error(f"[{idx}/{total_papers}] Failed review: {arxiv_code} - '{paper_title}' (attempt {i+1}/{RETRIES})")
                logger.error(str(e))
                continue
        if not success:
            logger.warning(f"[{idx}/{total_papers}] Skipping: {arxiv_code} - '{paper_title}' (failed after {RETRIES} attempts)")
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
        df = pd.DataFrame([flat_entries])
        db_utils.upload_dataframe(df, "summaries")

    logger.info("Paper review process completed")


if __name__ == "__main__":
    main()
