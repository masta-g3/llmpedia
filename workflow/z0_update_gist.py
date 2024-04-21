import sys, os
from datetime import datetime
from dotenv import load_dotenv
import json
import os

load_dotenv()
PROJECT_PATH = os.environ.get("PROJECT_PATH")
sys.path.append(PROJECT_PATH)

import utils.paper_utils as pu
import utils.db as db

def main():
    ## Check key is on env.
    if "GITHUB_TOKEN" not in os.environ:
        raise ValueError("Please set GITHUB_TOKEN in .env file.")

    ## Params.
    arxiv_codes = db.get_arxiv_id_list(pu.db_params, "summaries")
    title_map = db.get_arxiv_title_dict(pu.db_params)
    title_map = {k: v for k, v in title_map.items() if k in arxiv_codes}
    titles = list(title_map.values())

    token = os.environ["GITHUB_TOKEN"]
    gist_id = "8f7227397b1053b42e727bbd6abf1d2e"
    gist_filename = "llm_papers.txt"
    gist_description = f"Updated {datetime.now().strftime('%Y-%m-%d')}"
    gist_content = "\n".join(titles)

    ## Write to disk.
    gist_path = os.path.join(PROJECT_PATH, "data", gist_filename)
    with open(gist_path, "w") as f:
        f.write(gist_content)

    ## Execute.
    gist_url = pu.update_gist(token, gist_id, gist_filename, gist_description, gist_content)
    print(f"Done! Gist URL: {gist_url}")


if __name__ == "__main__":
    main()