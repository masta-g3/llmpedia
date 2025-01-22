import sys, os
from dotenv import load_dotenv
import time
from pathlib import Path
import pypdfium2
import psycopg2

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
    done_markdowns = pu.list_s3_directories("arxiv-md")
    
    # Get papers that need to be processed
    arxiv_codes = list(set(arxiv_codes) - set(done_markdowns))
    arxiv_codes = sorted(arxiv_codes)[::-1][:50]
    
    logger.info(f"Found {len(arxiv_codes)} papers to process")

    ## Iterate through papers
    for arxiv_code in arxiv_codes:
        
        # Ensure PDF exists locally and in S3
        pdf_path = os.path.join(PROJECT_PATH, "data/arxiv_pdfs", f"{arxiv_code}.pdf")
        if not pu.ensure_pdf_exists(arxiv_code, pdf_path, logger):
            continue

        # Convert to markdown
        # try:
        markdown_text, images = pu.convert_pdf_to_markdown(pdf_path)

        if markdown_text is None:
            logger.error(f"Failed to convert {arxiv_code} to markdown")
            continue
        
        # Create paper directory if it doesn't exist (both locally and on S3)
        paper_dir = os.path.join(PROJECT_PATH, "data/arxiv_md", arxiv_code)
        os.makedirs(paper_dir, exist_ok=True)
        
        # Save markdown file in paper directory
        markdown_path = os.path.join(paper_dir, "paper.md")
        with open(markdown_path, 'w', encoding='utf-8') as f:
            f.write(markdown_text)
        
        # Save each image in paper directory
        for img_name, img in images.items():
            # Save locally in paper directory
            local_path = os.path.join(paper_dir, img_name)
            img.save(local_path, "PNG")
        
        # Upload entire paper directory to S3
        pu.upload_s3_file(
            local_path=paper_dir,
            bucket_name="arxiv-md",
            key=arxiv_code,
            recursive=True
        )
        logger.info(f"Uploaded paper directory for {arxiv_code} to S3")
        logger.info(f"Successfully processed '{arxiv_code}'")

        # except pypdfium2._helpers.misc.PdfiumError as e:
        #     logger.error(f"Failed to convert {arxiv_code} to markdown: {str(e)}")
        #     continue

    logger.info("Completed paper download and conversion process.")

if __name__ == "__main__":
    main() 