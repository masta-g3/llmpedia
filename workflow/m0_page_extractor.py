import sys
import os
import requests
from tqdm import tqdm
from dotenv import load_dotenv
from pdf2image import convert_from_bytes
import gc

load_dotenv()
PROJECT_PATH = os.environ.get("PROJECT_PATH")
sys.path.append(PROJECT_PATH)
os.chdir(PROJECT_PATH)

import utils.paper_utils as pu
import utils.db.db_utils as db_utils
from utils.logging_utils import setup_logger

# Set up logging
logger = setup_logger(__name__, "m0_page_extractor.log")


def process_arxiv_code(arxiv_code: str, page_dir: str, idx: int, total: int, title: str) -> None:
    """Process a single arxiv code to extract its first page."""
    pdf_url = f"https://arxiv.org/pdf/{arxiv_code}.pdf"
    try:
        response = requests.get(pdf_url, timeout=30)
        response.raise_for_status()

        pdf_data = response.content
        images = convert_from_bytes(pdf_data, first_page=1, last_page=1)

        if images:
            first_page = images[0]
            png_path = os.path.join(page_dir, f"{arxiv_code}.png")

            width, height = first_page.size
            new_width = 800
            new_height = int(height * new_width / width)
            first_page = first_page.resize((new_width, new_height))
            first_page.save(png_path, "PNG")

            pu.upload_s3_file(
                arxiv_code, "arxiv-first-page", prefix="data", format="png"
            )
            logger.info(f"[{idx}/{total}] Extracted first page: {arxiv_code} - '{title}'")
        else:
            logger.warning(
                f"[{idx}/{total}] Failed to extract page: {arxiv_code} - '{title}'"
            )
    except Exception as e:
        logger.error(f"[{idx}/{total}] Error processing: {arxiv_code} - '{title}' - {str(e)}")
    finally:
        gc.collect()


def main():
    logger.info("Starting page extraction process.")
    page_dir = os.path.join(PROJECT_PATH, "data", "arxiv_first_page/")

    # Get arxiv codes from the S3 "arxiv-text" bucket
    arxiv_codes = pu.list_s3_files("arxiv-text", strip_extension=True)
    done_codes = pu.list_s3_files("arxiv-first-page", strip_extension=True)

    # Find the difference between all arxiv codes and the done codes
    arxiv_codes = list(set(arxiv_codes) - set(done_codes))
    arxiv_codes = sorted(arxiv_codes)[::-1]
    
    total_papers = len(arxiv_codes)
    logger.info(f"Found {total_papers} papers to process for page extraction.")
    
    title_map = db_utils.get_arxiv_title_dict()

    for idx, arxiv_code in enumerate(arxiv_codes, 1):
        paper_title = title_map.get(arxiv_code, "Unknown Title")
        process_arxiv_code(arxiv_code, page_dir, idx, total_papers, paper_title)
        gc.collect()  # Force garbage collection after each iteration

    logger.info("Page extraction process completed.")


if __name__ == "__main__":
    main()
