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
logger = setup_logger(__name__, "e1_narrate_bullet.log")

def main():
    logger.info("Starting bullet list narration process")
    vs.validate_openai_env()

    arxiv_codes = db_utils.get_arxiv_id_list("summary_notes")
    title_map = db_utils.get_arxiv_title_dict()
    done_codes = db_utils.get_arxiv_id_list("bullet_list_summaries")
    arxiv_codes = list(set(arxiv_codes) - set(done_codes))
    arxiv_codes = sorted(arxiv_codes)[::-1]

    total_papers = len(arxiv_codes)
    logger.info(f"Found {total_papers} papers to process for bullet list summaries")

    for idx, arxiv_code in enumerate(arxiv_codes, 1):
        paper_notes = paper_db.get_extended_notes(arxiv_code, expected_tokens=500)
        paper_title = title_map[arxiv_code]

        bullet_list = vs.convert_notes_to_bullets(
            paper_title, paper_notes, model="claude-3-5-sonnet-20241022"
        )
        bullet_list = bullet_list.replace("\n\n", "\n")
        
        paper_db.insert_bullet_list_summary(arxiv_code, bullet_list)
        logger.info(f"[{idx}/{total_papers}] Stored bullet list: {arxiv_code} - '{paper_title}'")

    logger.info("Bullet list narration process completed")

if __name__ == "__main__":
    main()
