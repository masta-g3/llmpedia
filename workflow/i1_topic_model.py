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
logger = setup_logger(__name__, "i1_topic_model.log")

db_params = pu.db_params
REFIT = False
embedding_type = "nv"
doc_type = "recursive_summary"

## Download necessary NLTK data.
nltk.download("wordnet", quiet=True)
nltk.download("stopwords", quiet=True)

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
        nr_docs=20,
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

def create_and_fit_topic_model(
    all_content: list, embeddings: np.array, refit: bool = False
) -> tuple:
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
        topic_model_path = os.path.join(
            PROJECT_PATH, "data", "bertopic", "topic_model.pkl"
        )
        reduced_model_path = os.path.join(PROJECT_PATH, "data", "reduced_model.pkl")

        if not os.path.exists(topic_model_path) or not os.path.exists(
            reduced_model_path
        ):
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

    ## Extract topics and embeddings.
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
    """Load embeddings and generate topic models."""
    logger.info("Starting topic model generation")
    
    ## Load data with recursive summaries.
    df = db.load_arxiv().reset_index()
    df.rename(columns={"summary": "abstract"}, inplace=True)
    df = df.merge(db.load_recursive_summaries(), on="arxiv_code", how="inner")
    
    ## For non-refit, process only pending documents.
    if not REFIT:
        done_codes = db.get_arxiv_id_list(db_params, "topics")
        working_codes = list(set(df.arxiv_code) - set(done_codes))
        df = df[df.arxiv_code.isin(working_codes)]
        if len(df) == 0:
            logger.info("No new documents to process")
            return
    
    logger.info(f"Processing {len(df)} documents")
    df.set_index("arxiv_code", inplace=True)
    
    ## Load embeddings and generate content.
    embeddings_map = db.load_embeddings(
        arxiv_codes=list(df.index),
        doc_type=doc_type,
        embedding_type=embedding_type,
    )

    all_content = df[doc_type].to_dict()

    ## align content and embeddings.
    arxiv_codes = list(embeddings_map.keys())
    all_content = [all_content[code] for code in arxiv_codes]
    embeddings = np.array([np.array(embeddings_map[code])for code in arxiv_codes])
    
    logger.info(f"Loaded {len(embeddings_map)} embeddings and corresponding content")

    topics, reduced_embeddings, topic_model, reduced_model = create_and_fit_topic_model(
        all_content, embeddings, refit=REFIT
    )

    store_topics_and_embeddings(
        df,
        all_content,
        topics,
        reduced_embeddings,
        topic_model,
        reduced_model,
        refit=REFIT,
    )
    
    logger.info("Successfully processed all documents")

if __name__ == "__main__":
    main()