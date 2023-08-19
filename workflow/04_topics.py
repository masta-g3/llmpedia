import json
import os
import pandas as pd
from typing import Callable
from umap import UMAP
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.representation import OpenAI, MaximalMarginalRelevance
from hdbscan import HDBSCAN
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
import re
import openai
from dotenv import load_dotenv
import warnings

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

## Download necessary NLTK data.
nltk.download("wordnet")
nltk.download("stopwords")

## Create a lemmatizer and list of stop words.
LEMMATIZER = WordNetLemmatizer()
STOP_WORDS = list(set(stopwords.words("english")))

PROMPT = """
I have a topic that contains the following Large Language Model related documents:
[DOCUMENTS]
The topic is described by the following keywords: [KEYWORDS]

Based on the information above, extract a short but highly descriptive topic label of at most 5 words. Do not use "Large Language Model" in your description. Make sure it is in the following format:
topic: <topic label>
"""


def _process_text(text: str, lemmatizer: WordNetLemmatizer, stop_words: list) -> str:
    """Preprocess text."""
    text = text.lower()
    text = re.sub(r"\W", " ", text)
    text = " ".join(
        [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
    )
    return text


def load_title_map() -> dict:
    """Load title map from json file."""
    with open("arxiv_code_map.json") as f:
        title_map = json.load(f)
    return title_map


def load_and_process_data(title_map: dict) -> pd.DataFrame:
    """Load and process data from json files, return DataFrame."""
    df = pd.DataFrame(columns=["title", "summary", "main_contribution", "takeaways"])
    for arxiv_code, title in title_map.items():
        with open(f"summaries/{arxiv_code}.json") as f:
            summary = json.load(f)
        df.loc[arxiv_code] = [
            title,
            summary["Summary"],
            summary["main_contribution"],
            summary["takeaways"],
        ]
    return df


def create_embeddings(df: pd.DataFrame) -> tuple:
    """Create embeddings."""
    content_cols = ["summary", "main_contribution", "takeaways"]
    df_dict = (
        df[content_cols].apply(lambda x: "\n".join(x.astype(str)), axis=1).to_dict()
    )
    all_content = list(df_dict.values())
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedding_model.encode(all_content, show_progress_bar=True)
    return all_content, embedding_model, embeddings


def create_topic_model(
    embeddings: list, prompt: str, lemmatizer: WordNetLemmatizer, stop_words: list
) -> BERTopic:
    """Create topic model."""
    load_dotenv()
    umap_model = UMAP(
        n_neighbors=15, n_components=10, min_dist=0.0, metric="cosine", random_state=200
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=4,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=False,
    )
    vectorizer_model = CountVectorizer(
        stop_words=stop_words,
        ngram_range=(2, 3),
        min_df=1,
        max_df=0.8,
        preprocessor=lambda text: _process_text(text, lemmatizer, stop_words),
    )
    mmr_model = MaximalMarginalRelevance(diversity=0.5)
    openai.api_key = os.getenv("OPENAI_API_KEY")
    openai_model = OpenAI(
        model="gpt-4", exponential_backoff=True, chat=True, prompt=prompt, nr_docs=8
    )
    topic_model = BERTopic(
        embedding_model=embeddings,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        representation_model=[mmr_model, openai_model],
        top_n_words=10,
        verbose=True,
    )
    return topic_model


def extract_topics_and_embeddings(
    all_content: list, embeddings: list, topic_model: BERTopic
) -> tuple:
    """Extract topics and embeddings."""
    topics, _ = topic_model.fit_transform(all_content, embeddings)
    reduced_embeddings = UMAP(
        n_neighbors=10, n_components=2, min_dist=0.0, metric="cosine", random_state=42
    ).fit_transform(embeddings)
    reduced_embeddings = (
        reduced_embeddings - reduced_embeddings.mean(axis=0)
    ) / reduced_embeddings.std(axis=0)
    return topics, reduced_embeddings


def store_topics_and_embeddings(
    df: pd.DataFrame, topics: list, reduced_embeddings: list, topic_model: BERTopic
):
    """Store topics and embeddings."""
    topic_names = topic_model.get_topic_info().set_index("Topic")["Name"]
    topic_names[-1] = "Miscellaneous"
    clean_topic_names = [
        topic_names[t].split("_")[-1].replace('"', "").strip() for t in topics
    ]
    df["topic"] = clean_topic_names
    df["dim1"] = reduced_embeddings[:, 0]
    df["dim2"] = reduced_embeddings[:, 1]
    df[["topic", "dim1", "dim2"]].to_pickle("data/topics.pkl")


def main():
    """Main function."""
    title_map = load_title_map()
    df = load_and_process_data(title_map)
    all_content, embedding_model, embeddings = create_embeddings(df)
    topic_model = create_topic_model(embedding_model, PROMPT, LEMMATIZER, STOP_WORDS)
    topics, reduced_embeddings = extract_topics_and_embeddings(
        all_content, embeddings, topic_model
    )
    store_topics_and_embeddings(df, topics, reduced_embeddings, topic_model)
    print("Done!")


if __name__ == "__main__":
    main()
