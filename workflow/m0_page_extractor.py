import sys
import os
from dotenv import load_dotenv
import requests
from tqdm import tqdm
from pdf2image import convert_from_bytes

load_dotenv()
PROJECT_PATH = os.environ.get("PROJECT_PATH")
sys.path.append(PROJECT_PATH)
os.chdir(PROJECT_PATH)

import utils.paper_utils as pu
import utils.db as db

def main():
    page_dir = os.path.join(PROJECT_PATH, "front_page/")
    arxiv_codes = pu.get_local_arxiv_codes()
    done_codes = [f.replace(".png", "") for f in os.listdir(page_dir)]
    arxiv_codes = list(set(arxiv_codes) - set(done_codes))
    arxiv_codes = sorted(arxiv_codes)[::-1][:20]

    for arxiv_code in tqdm(arxiv_codes):
        pdf_url = f"https://arxiv.org/pdf/{arxiv_code}.pdf"
        response = requests.get(pdf_url)

        if response.status_code == 200:
            pdf_data = response.content
            images = convert_from_bytes(pdf_data, first_page=1, last_page=1)
            if len(images) > 0:
                first_page = images[0]
                png_path = os.path.join(page_dir, f"{arxiv_code}.png")

                ## Downscale image.
                width, height = first_page.size
                new_width = 800
                new_height = int(height * new_width / width)
                first_page = first_page.resize((new_width, new_height))
                first_page.save(png_path, "PNG")
            else:
                print(f"\nCould not extract the first page of '{arxiv_code}'. Skipping...")
        else:
            print(f"\nCould not retrieve the PDF for '{arxiv_code}'. Skipping...")

    print("Done.")

if __name__ == "__main__":
    main()