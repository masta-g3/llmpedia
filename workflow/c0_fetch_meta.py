import sys, os
from dotenv import load_dotenv

load_dotenv()

PROJECT_PATH = os.getenv('PROJECT_PATH', '/app')
sys.path.append(PROJECT_PATH)

os.chdir(PROJECT_PATH)

from tqdm import tqdm
import utils.paper_utils as pu
import utils.db as db
from utils.logging_utils import setup_logger

# Set up logging
logger = setup_logger(__name__, "c0_fetch_meta.log")

def main():
    logger.info("Starting metadata fetching process")
    arxiv_codes = pu.list_s3_files("arxiv-text")
    done_codes = db.get_arxiv_id_list(db.db_params, "arxiv_details")
    arxiv_codes = list(set(arxiv_codes) - set(done_codes))
    arxiv_codes = sorted(arxiv_codes)[::-1]
    
    logger.info(f"Found {len(arxiv_codes)} new papers to fetch metadata for")

    for arxiv_code in arxiv_codes:
        logger.info(f"Fetching metadata for {arxiv_code}")
        arxiv_info = pu.get_arxiv_info(arxiv_code)
        if arxiv_info is None:
            logger.warning(f"Could not find '{arxiv_code}' in Arxiv meta-data. Skipping...")
            continue
        processed_meta = pu.process_arxiv_data(arxiv_info._raw)
        db.upload_to_db(processed_meta, pu.db_params, "arxiv_details")

    logger.info("Metadata fetching process completed.")

if __name__ == "__main__":
    main()
