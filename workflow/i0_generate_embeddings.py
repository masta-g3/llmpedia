import json
import sys, os
import pandas as pd
import warnings
from dotenv import load_dotenv
import voyageai

load_dotenv()
PROJECT_PATH = os.environ.get("PROJECT_PATH")
sys.path.append(PROJECT_PATH)
os.chdir(PROJECT_PATH)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

from sentence_transformers import SentenceTransformer
import utils.paper_utils as pu
import utils.db.embedding_db as embedding_db
import utils.db.paper_db as paper_db
import utils.db.db_utils as db_utils
from utils.logging_utils import setup_logger

logger = setup_logger(__name__, "i0_generate_embeddings.log")

REFIT = False
BATCH_SIZE = 100
EMBEDDING_TYPES = ["voyage", "nv"]
CONTENT_COLS = [["recursive_summary"], ["abstract"], ["recursive_summary", "abstract"]]


def initialize_embedding_model(
    embedding_type: str,
) -> SentenceTransformer | voyageai.Client:
    """Initialize and configure an embedding model based on the specified type."""
    if embedding_type == "gte":
        model = SentenceTransformer("barisaydin/gte-large")
    elif embedding_type == "nv":
        model = SentenceTransformer("nvidia/NV-Embed-v2", trust_remote_code=True)
        model.max_seq_length = 32768
        model.tokenizer.padding_side = "right"
    elif embedding_type == "voyage":
        model = voyageai.Client()
    else:
        raise ValueError(
            f"Unknown embedding type: {embedding_type}. Must be one of: 'gte', 'nv', 'voyage'"
        )
    return model


def process_and_store_batch(
    batch_df: pd.DataFrame,
    embedding_model: SentenceTransformer | voyageai.Client,
    all_content: list,
    content_cols: list[str],
    embedding_type: str,
) -> None:
    """Process a batch of documents and store their embeddings."""
    doc_type = "_".join(content_cols)
    # logger.info(f"Processing batch of {len(batch_df)} documents for {doc_type} using {embedding_type}")

    batch_content = (
        batch_df[content_cols]
        .apply(lambda x: "\n".join(x.astype(str)), axis=1)
        .tolist()
    )

    if embedding_type == "nv":
        batch_content = [
            text + embedding_model.tokenizer.eos_token for text in batch_content
        ]
        query_prefix = "Instruct: Identify the topic or theme of the following AI & Large Language Model document\nQuery: "
        batch_embeddings = embedding_model.encode(
            batch_content,
            prompt=query_prefix,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
    elif embedding_type == "voyage":
        batch_embeddings = embedding_model.embed(
            batch_content,
            model="voyage-3-large",
            input_type="document",
        ).embeddings
    else:
        batch_embeddings = embedding_model.encode(batch_content)

    ## Convert embeddings to list format.
    embeddings_list = [
        emb.tolist() if hasattr(emb, "tolist") else emb for emb in batch_embeddings
    ]

    embedding_db.store_embeddings_batch(
        arxiv_codes=list(batch_df.index),
        doc_type=doc_type,
        embedding_type=embedding_type,
        embeddings=embeddings_list,
    )

    all_content.extend(batch_content)


def load_content_data(content_cols: list[str]) -> pd.DataFrame:
    """Load and prepare data based on required content columns."""
    df = paper_db.load_arxiv().reset_index()
    df.rename(columns={"summary": "abstract"}, inplace=True)

    if "recursive_summary" in content_cols:
        df = df.merge(paper_db.load_recursive_summaries(), on="arxiv_code", how="inner")
    elif "abstract" in content_cols:
        df = df[df["abstract"].notna()]

    if len(df) == 0:
        return pd.DataFrame()

    return df.set_index("arxiv_code")


def main():
    """Process documents and create embeddings."""
    logger.info("Starting embedding generation process")

    for embedding_type in EMBEDDING_TYPES:
        logger.info(
            f"\n{'='*50}\nProcessing embeddings for model: {embedding_type}\n{'='*50}"
        )

        # Initialize model-specific resources
        embedding_model = initialize_embedding_model(embedding_type)
        content_by_type = {}  # Store content separately for each doc_type

        # Process each content type for this embedding model
        for cols in CONTENT_COLS:
            doc_type = "_".join(cols)
            logger.info(f"\n{'-'*40}\nProcessing {doc_type} embeddings\n{'-'*40}")
            content_by_type[doc_type] = (
                []
            )  # Initialize content list for this doc_type

            # Load data for current content type
            df = load_content_data(cols)
            if len(df) == 0:
                logger.info(f"No documents with required content for {doc_type}")
                continue

            # Get pending documents
            if not REFIT:
                existing_codes = embedding_db.get_pending_embeddings(
                    doc_type,
                    embedding_type,
                )
                df_to_process = df[~df.index.isin(existing_codes)]

                if len(df_to_process) == 0:
                    logger.info(f"No new documents to process for {doc_type}")
                    continue
                logger.info(
                    f"Found {len(df_to_process)} pending documents for {doc_type}"
                )
            else:
                df_to_process = df

            # Process in batches
            for i in range(0, len(df_to_process), BATCH_SIZE):
                batch_df = df_to_process.iloc[i : i + BATCH_SIZE]
                batch_num = i // BATCH_SIZE + 1
                total_batches = (len(df_to_process) + BATCH_SIZE - 1) // BATCH_SIZE
                logger.info(f"Processing batch {batch_num}/{total_batches}")

                try:
                    process_and_store_batch(
                        batch_df,
                        embedding_model,
                        content_by_type[doc_type],
                        cols,
                        embedding_type,
                    )
                except Exception as e:
                    logger.error(
                        f"Error processing batch {batch_num} for {cols} with {embedding_type}: {str(e)}"
                    )
                    raise

            if content_by_type[doc_type]:
                content_path = os.path.join(
                    PROJECT_PATH,
                    "data",
                    "bertopic",
                    f"content_{embedding_type}_{doc_type}.json",
                )
                with open(content_path, "w") as f:
                    json.dump(content_by_type[doc_type], f)
                logger.info(
                    f"Stored {doc_type} content for {embedding_type} at {content_path}"
                )

        logger.info(f"Completed processing for {embedding_type}")
        del embedding_model

    logger.info("Successfully processed all documents")


if __name__ == "__main__":
    main()
