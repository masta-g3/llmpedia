import json
import sys, os
import pandas as pd
import re
import openai
import warnings
from dotenv import load_dotenv
import numpy as np

load_dotenv()
PROJECT_PATH = os.environ.get("PROJECT_PATH")
sys.path.append(PROJECT_PATH)
os.chdir(PROJECT_PATH)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

from umap import UMAP
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.representation import OpenAI, MaximalMarginalRelevance
from hdbscan import HDBSCAN
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk

import utils.paper_utils as pu
import utils.db as db
from utils.logging_utils import setup_logger

# Set up logging
logger = setup_logger(__name__, "i0_topic_model.log")

db_params = pu.db_params

## Download necessary NLTK data.
nltk.download("wordnet", quiet=True)
nltk.download("stopwords", quiet=True)

REFIT = False
embedding_type = "nv"

## Create a lemmatizer and list of stop words.
LEMMATIZER = WordNetLemmatizer()
STOP_WORDS = list(set(stopwords.words("english")))

PROMPT = """I have done a clustering analysis on a set of Large Language Model related whitepapers. One of the clusters contains the following documents:
[DOCUMENTS]
The cluster top keywords: [KEYWORDS]

Based on this information, extract a short but highly descriptive and specific cluster  label using a few words. Consider that there will be other Large Language Model clusters, to be sure to identify what makes this one unique and give it a specific label. Do not use "Large Language Model", "Innovations" or "Advances" in your description. Make sure it is in the following format:
topic: <topic label>
"""


def process_text(text: str) -> str:
    """Preprocess text."""
    text = text.lower()
    text = re.sub(r"\W", " ", text)
    text = " ".join(
        [LEMMATIZER.lemmatize(word) for word in text.split() if word not in STOP_WORDS]
    )
    return text


def load_and_process_data(title_map: dict) -> pd.DataFrame:
    """Load and process data from json files, return DataFrame."""
    logger.info("Loading and processing paper content data.")
    df = pd.DataFrame(columns=["title", "summary"])
    for arxiv_code, title in title_map.items():
        fpath = os.path.join(PROJECT_PATH, "data", "summaries", f"{arxiv_code}.json")
        fpath_meta = os.path.join(
            PROJECT_PATH, "data", "arxiv_meta", f"{arxiv_code}.json"
        )
        with open(fpath) as f:
            summary = json.load(f)
            summary = pu.convert_innert_dict_strings_to_actual_dicts(summary)
        with open(fpath_meta) as f:
            meta = json.load(f)
        summary["Summary"] = meta["summary"]

        df.loc[str(arxiv_code)] = [
            title,
            summary["Summary"],
        ]
    logger.info(f"Loaded and processed data for {len(df)} papers.")
    return df


def create_embeddings(df: pd.DataFrame, embedding_model: SentenceTransformer = None, embedding_type: str = "gte") -> tuple:
    """Create embeddings for documents using either GTE or NV-Embed-v2 model, returning content, model, and embeddings."""
    logger.info(f"Creating embeddings using {embedding_type} model...")
    content_cols = ["summary"]
    df_dict = (
        df[content_cols].apply(lambda x: "\n".join(x.astype(str)), axis=1).to_dict()
    )
    all_content = list(df_dict.values())
    
    if embedding_model is None:
        if embedding_type == "gte":
            embedding_model = SentenceTransformer("barisaydin/gte-large")
        elif embedding_type == "nv":
            embedding_model = SentenceTransformer('nvidia/NV-Embed-v2', trust_remote_code=True)
            embedding_model.max_seq_length = 32768
            embedding_model.tokenizer.padding_side = "right"
        else:
            raise ValueError(f"Unknown embedding type: {embedding_type}. Must be either 'gte' or 'nv'")
    
    if embedding_type == "nv":
        # Add EOS token to each input
        all_content = [text + embedding_model.tokenizer.eos_token for text in all_content]
        
        # Use query prefix for topic identification
        query_prefix = "Instruct: Identify the topic or theme of the following academic documents\nQuery: "
        embeddings = embedding_model.encode(
            all_content, 
            batch_size=1, 
            prompt=query_prefix, 
            normalize_embeddings=True,
            show_progress_bar=True
        )
    else:
        embeddings = embedding_model.encode(all_content)
    
    logger.info("Embeddings created successfully.")
    return all_content, embedding_model, embeddings


def create_topic_model(prompt: str) -> BERTopic:
    """Create topic model."""
    logger.info("Creating topic model...")
    load_dotenv()
    umap_model = UMAP(
        n_neighbors=15, n_components=10, min_dist=0.0, metric="cosine", random_state=42
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=20,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    vectorizer_model = CountVectorizer(
        stop_words=STOP_WORDS,
        ngram_range=(2, 3),
        min_df=3,
        max_df=0.8,
        preprocessor=process_text,
    )
    mmr_model = MaximalMarginalRelevance(diversity=0.3)
    openai.api_key = os.getenv("OPENAI_API_KEY")
    openai_model = OpenAI(
        client=openai.OpenAI(),
        model="gpt-4o",
        exponential_backoff=True,
        chat=True,
        prompt=prompt,
        nr_docs=15,
    )
    topic_model = BERTopic(
        embedding_model=None,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        representation_model=[mmr_model, openai_model],
        top_n_words=10,
        verbose=True,
    )
    return topic_model


def extract_topics_and_embeddings(
    all_content: list,
    embeddings: list,
    topic_model: BERTopic,
    reduced_model: UMAP,
    refit=False,
) -> tuple:
    """Extract topics and embeddings."""
    if refit:
        topics, _ = topic_model.fit_transform(all_content, embeddings)
        reduced_embeddings = reduced_model.fit_transform(embeddings)
    else:
        topics, _ = topic_model.transform(all_content, embeddings)
        reduced_embeddings = reduced_model.transform(embeddings)

    topic_dists = db.get_topic_embedding_dist()
    reduced_embeddings[:, 0] = (
        reduced_embeddings[:, 0] - topic_dists["dim1"]["mean"]
    ) / topic_dists["dim1"]["std"]
    reduced_embeddings[:, 1] = (
        reduced_embeddings[:, 1] - topic_dists["dim2"]["mean"]
    ) / topic_dists["dim2"]["std"]

    return topics, reduced_embeddings, reduced_model


def store_topics_and_embeddings(
    df: pd.DataFrame,
    all_content: list,
    topics: list,
    reduced_embeddings: list,
    topic_model: BERTopic,
    reduced_model: UMAP,
    refit=False,
):
    """Store topics and embeddings."""
    if refit:
        ## Avoid lock issue.
        topic_model.representation_model = None
        topic_model.save(
            os.path.join(PROJECT_PATH, "data", "bertopic", "topic_model.pkl"),
            save_ctfidf=True,
            save_embedding_model=True,
            serialization="pickle",
        )
        pd.to_pickle(
            reduced_model, os.path.join(PROJECT_PATH, "data", "reduced_model.pkl")
        )
        with open(
            os.path.join(PROJECT_PATH, "data", "bertopic", "all_content.json"), "w"
        ) as f:
            json.dump(all_content, f)

    topic_names = topic_model.get_topic_info().set_index("Topic")["Name"]
    topic_names[-1] = "Miscellaneous"
    clean_topic_names = [
        topic_names[t].split("_")[-1].replace('"', "").strip() for t in topics
    ]
    df["topic"] = clean_topic_names
    df["dim1"] = reduced_embeddings[:, 0]
    df["dim2"] = reduced_embeddings[:, 1]
    df.index.name = "arxiv_code"
    df.reset_index(inplace=True)
    if_exists_policy = "replace" if refit else "append"
    db.upload_df_to_db(
        df[["arxiv_code", "topic", "dim1", "dim2"]],
        "topics",
        pu.db_params,
        if_exists=if_exists_policy,
    )


def create_embeddings_in_batches(df: pd.DataFrame, batch_size: int = 50) -> tuple[list, list]:
    """Process document embeddings in batches to manage memory usage, returning combined content and embeddings."""
    logger.info(f"Creating embeddings for {len(df)} documents in batches of {batch_size}")
    
    # Initialize embedding model once
    if embedding_type == "gte":
        embedding_model = SentenceTransformer("barisaydin/gte-large")
    elif embedding_type == "nv":
        embedding_model = SentenceTransformer('nvidia/NV-Embed-v2', trust_remote_code=True)
        embedding_model.max_seq_length = 32768
        embedding_model.tokenizer.padding_side = "right"
    else:
        raise ValueError(f"Unknown embedding type: {embedding_type}")
    
    all_content = []
    all_embeddings = []
    
    # Process in batches
    for i in range(0, len(df), batch_size):
        batch_df = df.iloc[i:i + batch_size]
        batch_num = i//batch_size + 1
        total_batches = (len(df) + batch_size - 1) // batch_size
        logger.info(f"Processing embedding batch {batch_num}/{total_batches}")
        
        try:
            batch_content, _, batch_embeddings = create_embeddings(
                batch_df, 
                embedding_model=embedding_model,
                embedding_type=embedding_type
            )
            all_content.extend(batch_content)
            all_embeddings.append(batch_embeddings)
            logger.info(f"Successfully created embeddings for batch {batch_num}")
        except Exception as e:
            logger.error(f"Error processing batch {batch_num}: {str(e)}")
            raise
    
    # Combine all embeddings
    combined_embeddings = np.vstack(all_embeddings)
    logger.info(f"Successfully created embeddings for all {len(df)} documents")
    return all_content, combined_embeddings


def create_and_fit_topic_model(all_content: list, embeddings: list, refit: bool = False) -> tuple:
    """Create and fit topic model on the complete set of document embeddings, returning topics, embeddings, and models."""
    logger.info("Creating and fitting topic model...")
    
    if refit:
        # Initialize new models for refit
        topic_model = create_topic_model(PROMPT)
        reduced_model = UMAP(
            n_neighbors=15,
            n_components=2,
            min_dist=0.0,
            metric="cosine",
            random_state=42,
        )
    else:
        # Load existing models for transform
        logger.info("Loading existing models...")
        topic_model_path = os.path.join(PROJECT_PATH, "data", "bertopic", "topic_model.pkl")
        reduced_model_path = os.path.join(PROJECT_PATH, "data", "reduced_model.pkl")
        
        if not os.path.exists(topic_model_path) or not os.path.exists(reduced_model_path):
            raise FileNotFoundError(
                "Cannot run in non-REFIT mode: Missing model files. "
                "Please run with REFIT=True first."
            )
        
        topic_model = BERTopic.load(path=topic_model_path)
        reduced_model = pd.read_pickle(reduced_model_path)
        
        # Verify model has topics
        if not topic_model.get_topic(0):
            raise ValueError(
                "Loaded topic model has no topics. "
                "Please run with REFIT=True to initialize the model."
            )
    
    # Extract topics and embeddings
    try:
        topics, reduced_embeddings, reduced_model = extract_topics_and_embeddings(
            all_content, embeddings, topic_model, reduced_model, refit=refit
        )
    except Exception as e:
        if not refit:
            logger.error(
                "Error transforming new documents. The model might be incompatible. "
                "Consider running with REFIT=True."
            )
        raise
    
    return topics, reduced_embeddings, topic_model, reduced_model


def main():
    """Process documents, create embeddings, and generate topic models either in full refit or incremental mode."""
    arxiv_codes = db.get_arxiv_id_list(db_params, "summaries")
    title_map = db.get_arxiv_title_dict(db_params)
    title_map = {k: v for k, v in title_map.items() if k in arxiv_codes}
    df = db.load_arxiv().reset_index()

    if REFIT:
        df = df[df["arxiv_code"].isin(arxiv_codes)]
        logger.info("Refit mode: Processing documents in batches")
        df.set_index("arxiv_code", inplace=True)
        all_content, embeddings = create_embeddings_in_batches(df, batch_size=50)
    else:
        # For non-refit, process only pending documents
        done_codes = db.get_arxiv_id_list(db_params, "topics")
        working_codes = list(set(arxiv_codes) - set(done_codes))
        df = df[df["arxiv_code"].isin(working_codes)]
        if len(df) == 0:
            logger.info("No new documents to process.")
            return
        logger.info(f"Non-refit mode: Processing {len(df)} pending documents")
        df.set_index("arxiv_code", inplace=True)
        all_content, _, embeddings = create_embeddings(df, embedding_type=embedding_type)
    
    # Create and fit topic model using all embeddings
    topics, reduced_embeddings, topic_model, reduced_model = create_and_fit_topic_model(
        all_content, embeddings, refit=REFIT
    )
    
    # Store results
    store_topics_and_embeddings(
        df, all_content, topics, reduced_embeddings, topic_model, reduced_model, refit=REFIT
    )


if __name__ == "__main__":
    main()
