import sys, os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from langchain_community.callbacks import get_openai_callback
from tqdm import tqdm

PROJECT_PATH = os.getenv("PROJECT_PATH", "/app")
sys.path.append(PROJECT_PATH)

os.chdir(PROJECT_PATH)

import utils.vector_store as vs
import utils.db.db_utils as db_utils
from utils.logging_utils import setup_logger

# Set up logging
logger = setup_logger(__name__, "e2_narrate_punchline.log")


def main():
    logger.info("Starting punchline generation process")
    vs.validate_openai_env()

    arxiv_codes = db_utils.get_arxiv_id_list("summary_notes")
    title_map = db_utils.get_arxiv_title_dict()
    done_codes = db_utils.get_arxiv_id_list("summary_punchlines")
    arxiv_codes = list(set(arxiv_codes) - set(done_codes))
    arxiv_codes = sorted(arxiv_codes)[::-1]

    logger.info(f"Found {len(arxiv_codes)} papers to process for punchline summaries")

    for arxiv_code in arxiv_codes:
        paper_notes = db_utils.get_extended_notes(arxiv_code, expected_tokens=500)
        paper_title = title_map[arxiv_code]

        logger.info(f"Generating punchline for: {arxiv_code} - '{paper_title}'")
        punchline = vs.generate_paper_punchline(
            paper_title, paper_notes, model="claude-3-5-sonnet-20241022"
        )

        data = {
            "arxiv_code": arxiv_code,
            "punchline": punchline,
            "tstp": pd.Timestamp.now(),
        }
        df = pd.DataFrame([data])
        db_utils.upload_dataframe(df, "summary_punchlines")

    logger.info("Punchline generation process completed")


if __name__ == "__main__":
    main()
