import sys, os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from langchain.callbacks import get_openai_callback
from tqdm import tqdm

sys.path.append(os.environ.get("PROJECT_PATH"))
os.chdir(os.environ.get("PROJECT_PATH"))

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
    arxiv_codes = pu.get_local_arxiv_codes()
    done_codes = db.get_arxiv_id_list(db.db_params, "summary_notes")
    arxiv_codes = list(set(arxiv_codes) - set(done_codes))
    arxiv_codes = sorted(arxiv_codes)[::-1]

    for arxiv_code in tqdm(arxiv_codes):
        paper_content = pu.load_local(arxiv_code, "arxiv_text", format="txt")
        paper_content = pu.preprocess_arxiv_doc(paper_content)
        title_dict = db.get_arxiv_title_dict(db.db_params)
        paper_title = title_dict.get(arxiv_code, None)
        if paper_title is None:
            print(f"Could not find '{arxiv_code}' in the meta-database. Skipping...")
            continue

        with get_openai_callback() as cb:
            i = 1
            ori_token_count = len(vs.token_encoder.encode(paper_content))
            token_count = ori_token_count + 0
            print(f"Starting tokens: {ori_token_count}")
            summaries_dict = {}
            token_dict = {}

            while token_count > 400:
                print("------------------------")
                print(f"Summarization iteration {i}...")
                paper_content = vs.summarize_by_segments(paper_title, paper_content)

                token_diff = token_count - len(vs.token_encoder.encode(paper_content))
                token_count = len(vs.token_encoder.encode(paper_content))
                frac = len(vs.token_encoder.encode(paper_content)) / ori_token_count
                summaries_dict[i] = paper_content
                token_dict[i] = token_count
                i += 1
                print(f"Total tokens: {token_count}")
                print(f"Compression: {frac:.2f}")

                if token_diff < 50:
                    print("Cannot compress further. Stopping.")
                    break

        ## Insert notes as code, level, summary & tokens,
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
