import os
import arxiv
import shutil
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def preprocess(text):
    """ Clean and simplify text string. """
    text = ''.join(c.lower() if c.isalnum() else ' ' for c in text)
    return text


def tfidf_similarity(title1, title2):
    """ Compute cosine similarity of TF-IDF representation between 2 strings. """
    title1 = preprocess(title1)
    title2 = preprocess(title2)
    vectorizer = TfidfVectorizer().fit_transform([title1, title2])
    vectors = vectorizer.toarray()
    return cosine_similarity(vectors[0:1], vectors[1:2])[0][0]


def get_arxiv_info(title):
    """ Search article in Arxiv by name and retrieve meta-data. """
    search = arxiv.Search(
        query=preprocess(title),
        max_results=40,
        sort_by=arxiv.SortCriterion.Relevance
    )
    res = list(search.results())
    if len(res) > 0:
        ## Sort by title similarity.
        res = sorted(res, key=lambda x: tfidf_similarity(title, x.title), reverse=True)
        new_title = res[0].title
        title_sim = tfidf_similarity(title, new_title)
        if title_sim > 0.7:
            return res[0]
        else:
            return None
    return None


