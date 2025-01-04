import sys, os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from langchain_community.callbacks import get_openai_callback
from tqdm import tqdm

PROJECT_PATH = os.getenv('PROJECT_PATH', '/app')
sys.path.append(PROJECT_PATH)

os.chdir(PROJECT_PATH)

import utils.vector_store as vs
import utils.db as db
from utils.logging_utils import setup_logger

# Set up logging
logger = setup_logger(__name__, "e2_narrate_punchline.log")

def main():
    logger.info("Starting punchline generation process")
    vs.validate_openai_env()

    arxiv_codes = db.get_arxiv_id_list(db.db_params, "summary_notes")
    title_map = db.get_arxiv_title_dict(db.db_params)
    done_codes = db.get_arxiv_id_list(db.db_params, "punchline_summaries")
    arxiv_codes = list(set(arxiv_codes) - set(done_codes))
    arxiv_codes = sorted(arxiv_codes)[::-1]

    logger.info(f"Found {len(arxiv_codes)} papers to process for punchline summaries")

    for arxiv_code in arxiv_codes:
        paper_notes = db.get_extended_notes(arxiv_code, expected_tokens=500)
        paper_title = title_map[arxiv_code]

        logger.info(f"Generating punchline for: {arxiv_code} - '{paper_title}'")
        punchline = vs.generate_paper_punchline(
            paper_title, paper_notes, model="claude-3-5-sonnet-20241022"
        )
        
        db.insert_punchline_summary(arxiv_code, punchline)

    logger.info("Punchline generation process completed")

if __name__ == "__main__":
    main() 