import sys, os
import pandas as pd
from dotenv import load_dotenv
import numpy as np

load_dotenv()
PROJECT_PATH = os.environ.get("PROJECT_PATH")
sys.path.append(PROJECT_PATH)
os.chdir(PROJECT_PATH)

import utils.paper_utils as pu
import utils.db as db

db_params = pu.db_params


def find_most_similar_documents(arxiv_code: str, df: pd.DataFrame, n: int = 5) -> list:
    """ Get most similar documents based on cosine similarity. """
    target_df = df.loc[arxiv_code]
    target_embedding = target_df.loc[["dim1", "dim2"]].to_numpy()
    target_embedding = target_embedding.reshape(1, -1)

    other_df = df.drop(arxiv_code)
    other_embeddings = other_df[["dim1", "dim2"]].to_numpy()

    similarities = pu.euclidean_distances(target_embedding, other_embeddings)
    most_similar_indices = np.argsort(similarities[0])[:n]
    most_similar_docs = other_df.iloc[most_similar_indices].index.tolist()
    return most_similar_docs


def main():
    """ Main function. """
    df = db.load_topics()
    df["similar_docs"] = df.index.map(lambda x: find_most_similar_documents(x, df, 10))
    df.reset_index(inplace=True)
    df['similar_docs'].apply(db.list_to_pg_array)
    db.upload_df_to_db(
        df[["arxiv_code", "similar_docs"]],
        "similar_documents",
        pu.db_params,
        if_exists="replace",
    )
    print("Done!")


if __name__ == "__main__":
    main()
