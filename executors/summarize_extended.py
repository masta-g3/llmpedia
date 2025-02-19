import sys, os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.callbacks import get_openai_callback
from tqdm import tqdm

PROJECT_PATH = os.getenv('PROJECT_PATH', '/app')
sys.path.append(PROJECT_PATH)

os.chdir(PROJECT_PATH)

import utils.vector_store as vs
import utils.paper_utils as pu
import utils.db.db_utils as db_utils
import utils.db.paper_db as paper_db

def main():
    vs.validate_openai_env()
    title_map = db_utils.get_arxiv_title_dict()
    arxiv_codes = db_utils.get_arxiv_id_list("summary_notes")
    done_codes = db_utils.get_arxiv_id_list("summary_markdown")
    arxiv_codes = list(set(arxiv_codes) - set(done_codes))
    arxiv_codes = sorted(arxiv_codes)[::-1]
    # arxiv_codes = ["2404.05961"]

    for arxiv_code in tqdm(arxiv_codes):
        paper_notes = paper_db.get_extended_notes(
            arxiv_code=arxiv_code, expected_tokens=3000
        )
        paper_title = title_map[arxiv_code]

        ## Convert notes to Markdown format and store.
        # markdown_notes = vs.convert_notes_to_markdown(paper_title, paper_notes, model="GPT-4-Turbo")
        notes_org = vs.organize_notes(
            paper_title, paper_notes, model="claude-sonnet"
        )

        markdown_notes = vs.convert_notes_to_markdown(
            paper_title, notes_org, model="claude-sonnet"
        )
        markdown_df = pd.DataFrame(
            {
                "arxiv_code": [arxiv_code],
                "summary": [markdown_notes],
                "tstp": [pd.Timestamp.now()],
            }
        )
        db_utils.upload_df_to_db(markdown_df, "summary_markdown")

    print("Done!")


if __name__ == "__main__":
    main()
