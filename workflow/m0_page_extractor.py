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

def process_arxiv_code(arxiv_code, page_dir):
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

            pu.upload_s3_file(arxiv_code=arxiv_code, bucket_name="arxiv-first-page", prefix="data", format="png")
            # print(f"Processed and uploaded {arxiv_code}")
        else:
            print(f"\nCould not extract the first page of '{arxiv_code}'. Skipping...")
    except Exception as e:
        print(f"\nError processing '{arxiv_code}': {str(e)}. Skipping...")
    finally:
        gc.collect()  # Force garbage collection

def main():
    page_dir = os.path.join(PROJECT_PATH, "data", "arxiv_first_page/")
    
    # Get arxiv codes from the S3 "arxiv-text" bucket
    arxiv_codes = pu.list_s3_files("arxiv-text", strip_extension=True)
    done_codes = pu.list_s3_files("arxiv-first-page", strip_extension=True)
    
    # Find the difference between all arxiv codes and the done codes
    arxiv_codes = list(set(arxiv_codes) - set(done_codes))
    arxiv_codes = sorted(arxiv_codes)[::-1]

    for arxiv_code in tqdm(arxiv_codes):
        process_arxiv_code(arxiv_code, page_dir)
        gc.collect()  # Force garbage collection after each iteration

    print("Done.")

if __name__ == "__main__":
    main()