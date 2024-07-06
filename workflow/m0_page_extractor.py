import sys
import os
import requests
from tqdm import tqdm
from dotenv import load_dotenv
from pdf2image import convert_from_bytes
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

load_dotenv()
PROJECT_PATH = os.environ.get("PROJECT_PATH")
sys.path.append(PROJECT_PATH)
os.chdir(PROJECT_PATH)

import utils.paper_utils as pu


def create_grid_image(pages, cols=5, max_pages=20):
    pages = pages[:max_pages]  # Ensure no more than max_pages are used
    rows = (len(pages) + cols - 1) // cols  # Calculate needed rows

    # Check and adjust dimensions of each image to match the first one (for uniformity)
    first_page_width, first_page_height = pages[0].size
    resized_pages = [page.resize((first_page_width, first_page_height), Image.LANCZOS) for page in pages]

    # Create a new image for the grid
    grid_width = cols * first_page_width
    grid_height = rows * first_page_height
    grid_image = Image.new('RGB', (grid_width, grid_height), 'white')

    font = ImageFont.load_default()

    # Place each page in the grid
    for index, page in enumerate(resized_pages):
        x = index % cols * first_page_width
        y = index // cols * first_page_height

        # Create a draw object and draw the page number
        draw = ImageDraw.Draw(page)
        text = str(index + 1)
        textwidth, textheight = draw.textsize(text, font=font)
        draw.text((first_page_width - textwidth - 10, 10), text, font=font, fill="red")

        grid_image.paste(page, (x, y))

    return grid_image


def main():
    page_dir = os.path.join(PROJECT_PATH, "front_page/")
    grid_dir = os.path.join(PROJECT_PATH, "paper_grid/")
    arxiv_codes = pu.get_local_arxiv_codes()
    done_codes = [f.replace(".png", "") for f in os.listdir(page_dir)]
    arxiv_codes = list(set(arxiv_codes) - set(done_codes))
    arxiv_codes = sorted(arxiv_codes)[::-1][:30]

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

                # grid_image = create_grid_image(images)
                # grid_path = os.path.join(grid_dir, f"{arxiv_code}_grid.png")
                # grid_image.save(grid_path, "PNG")
            else:
                print(f"\nCould not extract the first page of '{arxiv_code}'. Skipping...")
        else:
            print(f"\nCould not retrieve the PDF for '{arxiv_code}'. Skipping...")

    print("Done.")

if __name__ == "__main__":
    main()