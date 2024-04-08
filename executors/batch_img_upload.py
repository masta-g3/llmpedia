import os, sys
from dotenv import load_dotenv
from tqdm import tqdm
import boto3

load_dotenv()
PROJECT_PATH = os.environ.get("PROJECT_PATH")
sys.path.append(PROJECT_PATH)
img_dir = os.path.join(PROJECT_PATH, "imgs")

s3 = boto3.client("s3")
paginator = s3.get_paginator('list_objects_v2')
bucket_name = "llmpedia"

override = False

s3_iterator = paginator.paginate(Bucket=bucket_name)
all_s3_files = []
for s3_files in s3_iterator:
    s3_files = [f["Key"] for f in s3_files["Contents"]]
    all_s3_files.extend(s3_files)
local_files = os.listdir(img_dir)
local_files = [f for f in local_files if f.endswith(".png")]
if override:
    pending_files = local_files
else:
    pending_files = [f for f in local_files if f not in all_s3_files]

for filename in tqdm(pending_files):
    s3.upload_file(os.path.join(img_dir, filename), bucket_name, filename)
    print(f"Uploaded {filename} to S3")