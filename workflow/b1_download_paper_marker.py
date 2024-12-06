import sys, os
from dotenv import load_dotenv
import time
from pathlib import Path

load_dotenv()

PROJECT_PATH = os.getenv('PROJECT_PATH', '/app')
sys.path.append(PROJECT_PATH)

os.chdir(PROJECT_PATH)

import utils.paper_utils as pu
import utils.db as db
from utils.logging_utils import setup_logger

# Set up logging
logger = setup_logger(__name__, "b1_download_paper_marker.log")

def main():
    logger.info("Starting paper download and conversion process")

    # Get list of papers we need to process
    arxiv_codes = pu.list_s3_files("arxiv-text", strip_extension=True)
    title_dict = db.get_arxiv_title_dict(pu.db_params)
    
    # Check which papers are already processed as markdown
    done_markdowns = pu.list_s3_files("arxiv-markdown", strip_extension=True)
    
    # Get papers that need to be processed
    arxiv_codes = list(set(arxiv_codes) - set(done_markdowns))
    arxiv_codes = sorted(arxiv_codes)[::-1]  # Process newest papers first

    logger.info(f"Found {len(arxiv_codes)} papers to process")

    ## Iterate through papers
    for arxiv_code in arxiv_codes:
        time.sleep(3)
        title = title_dict.get(arxiv_code, "Unknown Title")
        
        # Ensure PDF exists locally and in S3
        pdf_path = os.path.join(PROJECT_PATH, "data/arxiv_pdf", f"{arxiv_code}.pdf")
        if not pu.ensure_pdf_exists(arxiv_code, pdf_path, logger):
            continue

        # Convert to markdown
        try:
            markdown_text = pu.convert_pdf_to_markdown(pdf_path)
            markdown_path = os.path.join(PROJECT_PATH, "data/arxiv_markdown", f"{arxiv_code}.md")
            with open(markdown_path, 'w', encoding='utf-8') as f:
                f.write(markdown_text)
            logger.info(f"Converted {arxiv_code} to markdown")

            # Upload markdown to S3
            pu.upload_s3_file(arxiv_code, "arxiv-markdown", prefix="data", format="md")
            logger.info(f"'{arxiv_code}' - '{title}' uploaded to S3")

        except Exception as e:
            logger.error(f"Failed to convert {arxiv_code} to markdown: {str(e)}")
            continue

    logger.info("Completed paper download and conversion process")

if __name__ == "__main__":
    main() 