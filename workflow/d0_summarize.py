import sys, os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from langchain_community.callbacks import get_openai_callback
from tqdm import tqdm

PROJECT_PATH = os.getenv('PROJECT_PATH', '/app')
sys.path.append(PROJECT_PATH)

os.chdir(PROJECT_PATH)

# from utils.models import get_mlx_model
import utils.vector_store as vs
import utils.paper_utils as pu
import utils.db as db


def shorten_list(list_str: str):
    """Shorten a bullet point list by taking the top 10 and bottom elements."""
    split_list = list_str.split("\n")
    if len(split_list) > 20:
        start_list_str = "\n".join(split_list[:5])
        end_list_str = "\n".join(split_list[-10:])
        list_str = f"{start_list_str}\n\n[...]\n{end_list_str}"
    return list_str


def main():
    arxiv_codes = pu.list_s3_files("arxiv-text")
    done_codes = db.get_arxiv_id_list(db.db_params, "summary_notes")
    arxiv_codes = list(set(arxiv_codes) - set(done_codes))
    arxiv_codes = sorted(arxiv_codes)[::-1]

    for arxiv_code in tqdm(arxiv_codes):
        paper_content = pu.load_local(arxiv_code, "arxiv_text", format="txt", s3_bucket="arxiv-text")
        paper_content = pu.preprocess_arxiv_doc(paper_content)
        title_dict = db.get_arxiv_title_dict()
        paper_title = title_dict.get(arxiv_code, None)
        if paper_title is None:
            print(f"Could not find '{arxiv_code}' in the meta-database. Skipping...")
            continue

        summaries_dict, token_dict = vs.recursive_summarize_by_parts(
            paper_title,
            paper_content,
            max_tokens=500,
            model="gpt-4o-mini",
            verbose=False,
        )

        summary_notes = pd.DataFrame(
            summaries_dict.items(), columns=["level", "summary"]
        )
        summary_notes["tokens"] = summary_notes.level.map(token_dict)
        summary_notes["arxiv_code"] = arxiv_code
        summary_notes["tstp"] = pd.Timestamp.now()
        db.upload_df_to_db(summary_notes, "summary_notes", db.db_params)

    print("Done!")


if __name__ == "__main__":
    main()
