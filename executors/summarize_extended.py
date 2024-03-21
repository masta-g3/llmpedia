import sys, os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.callbacks import get_openai_callback
from tqdm import tqdm

sys.path.append(os.environ.get("PROJECT_PATH"))
os.chdir(os.environ.get("PROJECT_PATH"))

import utils.vector_store as vs
import utils.paper_utils as pu
import utils.db as db


def main():
    vs.validate_openai_env()
    title_map = db.get_arxiv_title_dict(db.db_params)
    arxiv_codes = db.get_arxiv_id_list(db.db_params, "summary_notes")
    done_codes = db.get_arxiv_id_list(db.db_params, "summary_markdown")
    arxiv_codes = list(set(arxiv_codes) - set(done_codes))
    arxiv_codes = sorted(arxiv_codes)[::-1]
    arxiv_codes = ["2403.08540"]

    with get_openai_callback() as cb:
        for arxiv_code in tqdm(arxiv_codes):
            paper_notes = db.get_extended_notes(
                arxiv_code=arxiv_code, expected_tokens=7000
            )
            paper_title = title_map[arxiv_code]

            ## Convert notes to Markdown format and store.
            # markdown_notes = vs.convert_notes_to_markdown(paper_title, paper_notes, model="GPT-4-Turbo")🔍 Today's LLM paper review When is Tree Search Useful for LLM Planning? It Depends on the Discriminator (Feb 2024) explores the efficiency of tree search (TS) and iterative correction (IC) versus re-ranking (RR) for multi-step problems. The study finds that TS and IC require a discriminator accuracy of at least 90% to significantly outperform RR. However, most current LLM discriminators don't meet this mark, making TS and IC less practical due to their high computational costs and minimal performance benefits.
            #
            # 📊 Experiments on text-to-SQL parsing and mathematical reasoning show TS's limited real-world utility, being 10-20 times slower and only slightly more effective than simpler methods. This underscores the importance of highly accurate discriminators to make the efficiency trade-off worthwhile, challenging the notion that complex planning methods automatically yield better results.
            #
            # 🌐 The research presents a unified agent framework to evaluate different planning strategies on LLM performance, suggesting observation-enhanced discriminators as a way to improve accuracy. This approach, incorporating executability checks and execution results, could enhance LLM planning capabilities, indicating a path forward in optimizing planning methods for LLMs.
            notes_org = vs.organize_notes(paper_title, paper_notes, model="GPT-4-Turbo")
            markdown_notes = vs.convert_notes_to_markdown(
                paper_title, notes_org, model="GPT-4-Turbo"
            )
            markdown_df = pd.DataFrame(
                {
                    "arxiv_code": [arxiv_code],
                    "summary": [markdown_notes],
                    "tstp": [pd.Timestamp.now()],
                }
            )
            db.upload_df_to_db(markdown_df, "summary_markdown", db.db_params)
            print(cb)

    print("Done!")


if __name__ == "__main__":
    main()
