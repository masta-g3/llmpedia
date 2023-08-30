from datetime import datetime
from dotenv import load_dotenv
import json
import os

from utils.paper_utils import update_gist


def main():
    ## Check key is on env.
    load_dotenv()
    if "GITHUB_TOKEN" not in os.environ:
        raise ValueError("Please set GITHUB_TOKEN in .env file.")
    with open("arxiv_code_map.json") as f:
        data = json.load(f)

    ## Params.
    codes = list(data.keys())
    titles = list(data.values())
    token = os.environ["GITHUB_TOKEN"]
    gist_id = "8f7227397b1053b42e727bbd6abf1d2e"
    gist_filename = "llm_papers.txt"
    gist_description = f"Updated {datetime.now().strftime('%Y-%m-%d')}"
    gist_content = "\n".join(titles)

    ## Write to disk.
    with open("llm_papers.txt", "w") as f:
        f.write(gist_content)

    ## Execute.
    gist_url = update_gist(token, gist_id, gist_filename, gist_description, gist_content)
    print(f"Done! Gist URL: {gist_url}")


if __name__ == "__main__":
    main()