import sys, os
import json
import shutil
import os, re
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

from langchain.text_splitter import RecursiveCharacterTextSplitter

sys.path.append(os.environ.get("PROJECT_PATH"))
os.chdir(os.environ.get("PROJECT_PATH"))
import utils.paper_utils as pu

data_path = os.path.join(os.environ.get("PROJECT_PATH"), "data", "arxiv_text")
meta_path = os.path.join(os.environ.get("PROJECT_PATH"), "data", "arxiv_meta")
chunk_path = os.path.join(os.environ.get("PROJECT_PATH"), "data", "arxiv_chunks")

## Helper splitter.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,
)


def main():
    """Chunk arxiv docs into smaller blocks."""
    ## Get raw paper list.
    fnames = os.listdir(data_path)
    arxiv_local = [fname.replace(".txt", "")
                   for fname in fnames if fname.endswith(".txt")]
    arxiv_done = pu.get_arxiv_id_list(pu.db_params, "arxiv_chunks")
    arxiv_pending = list(set(arxiv_local) - set(arxiv_done))
    print(f"Found {len(arxiv_pending)} papers pending.")

    for arxiv_code in tqdm(arxiv_pending):
        ## Open doc and meta_data.
        doc_txt = pu.load_local(arxiv_code, data_path, False, "txt")
        doc_meta = pu.load_local(arxiv_code, meta_path, False, "json")
        authors_str = f"{doc_meta['authors'][0]['name']}, et al."
        year_str = doc_meta["published"][:4]
        doc_texts = text_splitter.split_text(doc_txt)
        # doc_chunks = doc_texts[1:-1]
        doc_chunks = [doc.replace("\n", " ") for doc in doc_texts]

        # print("\n", len(doc_chunks))
        # print(f"(Authors: {authors_str}, Year: {year_str})")

        ## Store document chunks in DB.
        doc_chunks_df = pd.DataFrame.from_dict(doc_chunks)
        doc_chunks_df["arxiv_code"] = arxiv_code
        doc_chunks_df["chunk_id"] = doc_chunks_df.index
        doc_chunks_df.columns = ["text", "arxiv_code", "chunk_id"]
        pu.upload_df_to_db(doc_chunks_df, "arxiv_chunks", pu.db_params)

        ## Store document chunks in JSON.
        doc_chunks_list = doc_chunks_df.to_dict(orient="records")
        pu.store_local(doc_chunks_list, arxiv_code, chunk_path, relative=False)
        # print(f"Stored {len(doc_chunks_list)} chunks for {arxiv_code}.")

if __name__ == "__main__":
    main()
