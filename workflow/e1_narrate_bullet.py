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
logger = setup_logger(__name__, "e1_narrate_bullet.log")

def main():
    logger.info("Starting bullet list narration process")
    vs.validate_openai_env()

    arxiv_codes = db.get_arxiv_id_list(db.db_params, "summary_notes")
    title_map = db.get_arxiv_title_dict(db.db_params)
    done_codes = db.get_arxiv_id_list(db.db_params, "bullet_list_summaries")
    arxiv_codes = list(set(arxiv_codes) - set(done_codes))
    arxiv_codes = sorted(arxiv_codes)[::-1]

    logger.info(f"Found {len(arxiv_codes)} papers to process for bullet list summaries")

    for arxiv_code in arxiv_codes:
        paper_notes = db.get_extended_notes(arxiv_code, expected_tokens=500)
        paper_title = title_map[arxiv_code]

        logger.info(f"Generating bullet list for: {arxiv_code} - '{paper_title}'")
        bullet_list = vs.convert_notes_to_bullets(
            paper_title, paper_notes, model="claude-3-5-sonnet-20241022"
        )
        bullet_list = bullet_list.replace("\n\n", "\n")
        
        db.insert_bullet_list_summary(arxiv_code, bullet_list)

    logger.info("Bullet list narration process completed")

if __name__ == "__main__":
    main()
