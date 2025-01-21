import os
import sys
import argparse
import boto3
from dotenv import load_dotenv

load_dotenv()
PROJECT_PATH = os.environ.get("PROJECT_PATH")
sys.path.append(PROJECT_PATH)

import utils.paper_utils as pu
from utils.logging_utils import setup_logger

# Set up logging
logger = setup_logger(__name__, "check_corrupt_pdfs.log")


def check_corrupt_pdfs(
    directory_path: str, move_corrupt: bool = True, delete_from_s3: bool = True
) -> tuple[list[str], list[str]]:
    """Scan directory for corrupt PDFs and optionally move them and delete from S3."""

    corrupt_dir = None
    if move_corrupt:
        corrupt_dir = os.path.join(directory_path, "corrupt_pdfs")
        os.makedirs(corrupt_dir, exist_ok=True)

    corrupt_arxiv_codes = []
    valid_arxiv_codes = []

    # Get all PDF files in directory
    pdf_files = [f for f in os.listdir(directory_path) if f.endswith(".pdf")]

    print(f"Found {len(pdf_files)} PDF files to check...")

    for pdf_file in pdf_files:
        arxiv_code = pdf_file.replace(".pdf", "")
        pdf_path = os.path.join(directory_path, pdf_file)

        try:
            with open(pdf_path, "rb") as file:
                try:
                    import PyPDF2

                    PyPDF2.PdfReader(file)
                    valid_arxiv_codes.append(arxiv_code)
                    # print(f"✓ Valid: {pdf_file}")

                except Exception as e:
                    corrupt_arxiv_codes.append(arxiv_code)
                    print(f"✗ Corrupt: {pdf_file}")
                    print(f"  Error: {str(e)}")

                    if move_corrupt:
                        corrupt_file_path = os.path.join(corrupt_dir, pdf_file)
                        import shutil

                        shutil.move(pdf_path, corrupt_file_path)

                    if delete_from_s3:
                        try:
                            s3 = boto3.client("s3")
                            s3.delete_object(
                                Bucket="arxiv-pdfs", Key=f"{arxiv_code}.pdf"
                            )
                            print(f"  Deleted from S3: {arxiv_code}.pdf")
                        except Exception as s3_err:
                            print(f"  Failed to delete from S3: {str(s3_err)}")

        except Exception as e:
            print(f"! Cannot access file {pdf_file}: {str(e)}")
            corrupt_arxiv_codes.append(arxiv_code)

    print("\nScan Complete!")
    print(f"Total PDFs checked: {len(pdf_files)}")
    print(f"Valid PDFs: {len(valid_arxiv_codes)}")
    print(f"Corrupt PDFs: {len(corrupt_arxiv_codes)}")

    if move_corrupt and corrupt_arxiv_codes:
        print(f"\nCorrupt files have been moved to: {corrupt_dir}")

    return corrupt_arxiv_codes, valid_arxiv_codes


def main():
    """Check for corrupt PDFs in the arxiv_pdfs directory."""
    parser = argparse.ArgumentParser(description="Check PDF files for corruption")
    parser.add_argument(
        "--directory",
        help="Directory containing PDF files to check (default: data/arxiv_pdfs)",
    )
    parser.add_argument(
        "--no-move",
        action="store_true",
        help="Don't move corrupt files to corrupt_pdfs directory",
    )
    parser.add_argument(
        "--no-s3-delete", action="store_true", help="Don't delete corrupt files from S3"
    )

    args = parser.parse_args()

    directory = args.directory or os.path.join(PROJECT_PATH, "data", "arxiv_pdfs")
    move_corrupt = not args.no_move
    delete_from_s3 = not args.no_s3_delete

    logger.info(f"Starting PDF corruption check in {directory}")
    logger.info(f"Move corrupt files: {move_corrupt}")
    logger.info(f"Delete from S3: {delete_from_s3}")

    corrupt_codes, valid_codes = pu.check_corrupt_pdfs(
        directory_path=directory,
        move_corrupt=move_corrupt,
        delete_from_s3=delete_from_s3,
    )

    if corrupt_codes:
        logger.info("Corrupt arXiv codes:")
        for code in corrupt_codes:
            logger.info(f"  {code}")

    logger.info("PDF check complete")


if __name__ == "__main__":
    main()
