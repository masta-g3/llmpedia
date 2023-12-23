import sys, os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.callbacks import get_openai_callback
from tqdm import tqdm

sys.path.append(os.environ.get("PROJECT_PATH"))
os.chdir(os.environ.get("PROJECT_PATH"))

import utils.vector_store as vs
import utils.paper_utils as pu
import utils.db as db


def main():
    title_map = db.get_arxiv_title_dict(db.db_params)
    arxiv_codes = db.get_arxiv_id_list(db.db_params, "summary_notes")
    # done_codes = db.get_arxiv_id_list(db.db_params, "summary_markdown")
    done_codes = []
    arxiv_codes = list(set(arxiv_codes) - set(done_codes))
    arxiv_codes = sorted(arxiv_codes)[::-1]

    ## Create DF to store results.
    df = pd.DataFrame(columns=["arxiv_code", "markdown_notes"])

    for arxiv_code in tqdm(arxiv_codes):
        paper_notes = db.get_extended_notes(arxiv_code=arxiv_code, expected_tokens=4000)
        paper_title = title_map[arxiv_code]

        with get_openai_callback() as cb:
            ## Convert notes to Markdown format.
            markdown_notes = vs.convert_notes_to_markdown(paper_title, paper_notes, model="GPT-4-Turbo")
            print(markdown_notes)
            df = df.append({"arxiv_code": arxiv_code, "markdown_notes": markdown_notes}, ignore_index=True)
            print(cb)

    ## Store results locally.
    df.to_pickle("summary_markdown.pkl")
    print("Done!")


if __name__ == "__main__":
    main()
