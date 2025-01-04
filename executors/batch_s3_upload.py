import os
import sys
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()
PROJECT_PATH = os.environ.get("PROJECT_PATH")
sys.path.append(PROJECT_PATH)

import utils.paper_utils as pu

def upload_files_to_s3(local_dir, bucket_name, file_extension, override=False):
    # Get existing files
    all_s3_files = pu.list_s3_files(bucket_name, strip_extension=False)

    # Get local files
    local_files = os.listdir(local_dir)
    local_files = [f for f in local_files if f.endswith(file_extension)]

    if override:
        pending_files = local_files
    else:
        pending_files = [f for f in local_files if f not in all_s3_files]

    for filename in tqdm(pending_files):
        arxiv_code = os.path.splitext(filename)[0]
        format = file_extension[1:]  # Remove the dot
        pu.upload_s3_file(arxiv_code, bucket_name, prefix="data", format=format)

def upload_arxiv_images():
    print("Uploading arxiv images...")
    upload_files_to_s3(os.path.join(PROJECT_PATH, "data", "arxiv_art"), "arxiv-art", ".png")

def upload_arxiv_text():
    print("Uploading arxiv text...")
    upload_files_to_s3(os.path.join(PROJECT_PATH, "data", "arxiv_text"), "arxiv-text", ".txt")

def upload_nonllm_arxiv_text():
    print("Uploading nonllm arxiv text...")
    upload_files_to_s3(os.path.join(PROJECT_PATH, "data", "nonllm_arxiv_text"), "nonllm-arxiv-text", ".txt")

def upload_arxiv_first_page():
    print("Uploading arxiv first page...")
    upload_files_to_s3(os.path.join(PROJECT_PATH, "data", "arxiv_first_page"), "arxiv-first-page", ".png")

def upload_arxiv_chunks():
    print("Uploading arxiv chunks...")
    upload_files_to_s3(os.path.join(PROJECT_PATH, "data", "arxiv_chunks"), "arxiv-chunks", ".json")

def upload_arxiv_large_chunks():
    print("Uploading arxiv large chunks...")
    upload_files_to_s3(os.path.join(PROJECT_PATH, "data", "arxiv_large_chunks"), "arxiv-large-chunks", ".json")

if __name__ == "__main__":
    upload_arxiv_images()
    upload_arxiv_text()
    upload_nonllm_arxiv_text()
    upload_arxiv_first_page()
    upload_arxiv_chunks()
    upload_arxiv_large_chunks()
