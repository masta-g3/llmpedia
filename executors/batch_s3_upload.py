import os
import sys
from dotenv import load_dotenv
from tqdm import tqdm
import boto3

load_dotenv()
PROJECT_PATH = os.environ.get("PROJECT_PATH")
sys.path.append(PROJECT_PATH)

s3 = boto3.client("s3")
paginator = s3.get_paginator('list_objects_v2')

def upload_files_to_s3(local_dir, bucket_name, file_extension, override=False):
    s3_iterator = paginator.paginate(Bucket=bucket_name)
    all_s3_files = []
    for s3_files in s3_iterator:
        if 'Contents' in s3_files:
            s3_files = [f["Key"] for f in s3_files["Contents"]]
            all_s3_files.extend(s3_files)

    local_files = os.listdir(local_dir)
    local_files = [f for f in local_files if f.endswith(file_extension)]

    if override:
        pending_files = local_files
    else:
        pending_files = [f for f in local_files if f not in all_s3_files]

    for filename in tqdm(pending_files):
        s3.upload_file(os.path.join(local_dir, filename), bucket_name, filename)
        # print(f"Uploaded {filename} to S3 bucket {bucket_name}")

def upload_arxiv_images():
    print("Uploading arxiv images...")
    img_dir = os.path.join(PROJECT_PATH, "imgs")
    upload_files_to_s3(img_dir, "llmpedia", ".png")

def upload_arxiv_text():
    print("Uploading arxiv text...")
    text_dir = os.path.join(PROJECT_PATH, "data", "arxiv_text")
    upload_files_to_s3(text_dir, "arxiv-text", ".txt")

def upload_nonllm_arxiv_text():
    print("Uploading nonllm arxiv text...")
    text_dir = os.path.join(PROJECT_PATH, "data", "nonllm_arxiv_text")
    upload_files_to_s3(text_dir, "nonllm-arxiv-text", ".txt")

def upload_arxiv_first_page():
    print("Uploading arxiv first page...")
    page_dir = os.path.join(PROJECT_PATH, "arxiv_first_page")
    upload_files_to_s3(page_dir, "arxiv-first-page", ".png")

def upload_arxiv_chunks():
    print("Uploading arxiv chunks...")
    chunk_dir = os.path.join(PROJECT_PATH, "data", "arxiv_chunks")
    upload_files_to_s3(chunk_dir, "arxiv-chunks", ".json")

def upload_arxiv_large_chunks():
    print("Uploading arxiv large chunks...")
    chunk_dir = os.path.join(PROJECT_PATH, "data", "arxiv_large_chunks")
    upload_files_to_s3(chunk_dir, "arxiv-large-chunks", ".json")

if __name__ == "__main__":
    upload_arxiv_images()
    upload_arxiv_text()
    upload_nonllm_arxiv_text()
    upload_arxiv_first_page()
    upload_arxiv_chunks()
    upload_arxiv_large_chunks()
