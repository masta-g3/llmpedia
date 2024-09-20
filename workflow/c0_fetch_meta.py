import sys, os
from dotenv import load_dotenv

load_dotenv()

PROJECT_PATH = os.getenv('PROJECT_PATH', '/app')
sys.path.append(PROJECT_PATH)

os.chdir(PROJECT_PATH)

from tqdm import tqdm
import utils.paper_utils as pu
import utils.db as db


def main():
    arxiv_codes = pu.list_s3_files("arxiv-text")
    done_codes = db.get_arxiv_id_list(db.db_params, "arxiv_details")
    arxiv_codes = list(set(arxiv_codes) - set(done_codes))
    arxiv_codes = sorted(arxiv_codes)[::-1]

    for arxiv_code in tqdm(arxiv_codes):
        arxiv_info = pu.get_arxiv_info(arxiv_code)
        if arxiv_info is None:
            print(f"\nCould not find '{arxiv_code}' in Arxiv meta-data. Skipping...")
            continue
        processed_meta = pu.process_arxiv_data(arxiv_info._raw)

        ## Store.
        db.upload_to_db(processed_meta, pu.db_params, "arxiv_details")

    print("Done.")


if __name__ == "__main__":
    main()
