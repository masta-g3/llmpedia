import sys, os
from dotenv import load_dotenv

load_dotenv()

PROJECT_PATH = os.getenv('PROJECT_PATH', '/app')
sys.path.append(PROJECT_PATH)

os.chdir(PROJECT_PATH)

from tqdm import tqdm
import pandas as pd
import utils.paper_utils as pu
import utils.db.db_utils as db_utils
from utils.logging_utils import setup_logger

# Set up logging
logger = setup_logger(__name__, "c0_fetch_meta.log")

def main():
    logger.info("Starting metadata fetching process.")
    arxiv_codes = pu.list_s3_files("arxiv-text", strip_extension=True)
    done_codes = db_utils.get_arxiv_id_list("arxiv_details")
    arxiv_codes = list(set(arxiv_codes) - set(done_codes))
    arxiv_codes = sorted(arxiv_codes)[::-1]
    
    total_papers = len(arxiv_codes)
    logger.info(f"Found {total_papers} papers with missing meta-data.")

    for idx, arxiv_code in enumerate(arxiv_codes, 1):
        arxiv_info = pu.get_arxiv_info(arxiv_code)
        if arxiv_info is None:
            logger.warning(f"[{idx}/{total_papers}] Failed to fetch metadata: {arxiv_code}")
            continue
        processed_meta = pu.process_arxiv_data(arxiv_info._raw)
        df = pd.DataFrame([processed_meta])
        db_utils.upload_dataframe(df, "arxiv_details")
        logger.info(f"[{idx}/{total_papers}] Stored metadata: {arxiv_code} - '{processed_meta.get('title', 'Unknown Title')}'")

    logger.info("Metadata fetching process completed.")

if __name__ == "__main__":
    main()
