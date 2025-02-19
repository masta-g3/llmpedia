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
import utils.db.db_utils as db_utils
import utils.db.paper_db as paper_db
from utils.logging_utils import setup_logger

# Set up logging
logger = setup_logger(__name__, "e0_narrate.log")

def main():
    logger.info("Starting paper summary narration process.")
    vs.validate_openai_env()

    arxiv_codes = db_utils.get_arxiv_id_list("summary_notes")
    title_map = db_utils.get_arxiv_title_dict()
    done_codes = db_utils.get_arxiv_id_list("recursive_summaries")
    arxiv_codes = list(set(arxiv_codes) - set(done_codes))
    arxiv_codes = sorted(arxiv_codes)[::-1]
    
    total_papers = len(arxiv_codes)
    logger.info(f"Found {total_papers} papers to narrate.")

    for idx, arxiv_code in enumerate(arxiv_codes, 1):
        paper_notes = paper_db.get_extended_notes(arxiv_code, expected_tokens=1200)
        paper_title = title_map[arxiv_code]

        logger.info(f"[{idx}/{total_papers}] Narrating: {arxiv_code} - '{paper_title}'")
        narrative = vs.convert_notes_to_narrative(
            paper_title, paper_notes, model="claude-3-5-sonnet-20241022"
        )
        
        copywritten = vs.copywrite_summary(
            paper_title, paper_notes, narrative, model="claude-3-5-sonnet-20241022"
        )
        
        paper_db.insert_recursive_summary(arxiv_code, copywritten)

    logger.info("Paper narration process completed.")

if __name__ == "__main__":
    main()
