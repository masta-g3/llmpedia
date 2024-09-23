import sys
import os
import requests
from tqdm import tqdm
from dotenv import load_dotenv
from pdf2image import convert_from_bytes

load_dotenv()
PROJECT_PATH = os.environ.get("PROJECT_PATH")
sys.path.append(PROJECT_PATH)
os.chdir(PROJECT_PATH)

import utils.paper_utils as pu

def main():
    page_dir = os.path.join(PROJECT_PATH, "data", "arxiv_first_page/")
    
    # Get arxiv codes from the S3 "arxiv-text" bucket
    arxiv_codes = pu.list_s3_files("arxiv-text", strip_extension=True)
    done_codes = pu.list_s3_files("arxiv-first-page", strip_extension=True)
    
    # Find the difference between all arxiv codes and the done codes
    arxiv_codes = list(set(arxiv_codes) - set(done_codes))
    arxiv_codes = sorted(arxiv_codes)[::-1]

    for arxiv_code in tqdm(arxiv_codes):
        pdf_url = f"https://arxiv.org/pdf/{arxiv_code}.pdf"
        response = requests.get(pdf_url)

        if response.status_code == 200:
            pdf_data = response.content
            images = convert_from_bytes(pdf_data, first_page=1)
            if len(images) > 0:
                first_page = images[0]
                png_path = os.path.join(page_dir, f"{arxiv_code}.png")

                ## Downscale image.
                width, height = first_page.size
                new_width = 800
                new_height = int(height * new_width / width)
                first_page = first_page.resize((new_width, new_height))
                first_page.save(png_path, "PNG")

                # Upload to S3 "arxiv-first-page" bucket
                pu.upload_s3_file(arxiv_code=arxiv_code, bucket="arxiv-first-page", prefix="data", extension="png")
                print(f"Processed and uploaded {arxiv_code}")
            else:
                print(f"\nCould not extract the first page of '{arxiv_code}'. Skipping...")
        else:
            print(f"\nCould not retrieve the PDF for '{arxiv_code}'. Skipping...")

    print("Done.")

if __name__ == "__main__":
    main()