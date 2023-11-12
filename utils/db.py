from sqlalchemy import create_engine, text
import streamlit as st
import pandas as pd
import uuid
import os

db_params = {**st.secrets["postgres"]}


database_url = f"postgresql+psycopg2://{db_params['user']}:{db_params['password']}@{db_params['host']}:{db_params['port']}/{db_params['dbname']}"


def log_error_db(error):
    """Log error in DB along with streamlit app state."""
    engine = create_engine(database_url)
    with engine.begin() as conn:
        error_id = str(uuid.uuid4())
        tstp = pd.to_datetime("now").strftime("%Y-%m-%d %H:%M:%S")
        query = text(
            """
            INSERT INTO error_logs (error_id, tstp, error)
            VALUES (:error_id, :tstp, :error);
        """
        )
        conn.execute(
            query,
            {
                "error_id": str(error_id),
                "tstp": tstp,
                "error": str(error),
            },
        )


def log_qna_db(user_question, response):
    """Log Q&A in DB along with streamlit app state."""
    engine = create_engine(database_url)
    with engine.begin() as conn:
        qna_id = str(uuid.uuid4())
        tstp = pd.to_datetime("now").strftime("%Y-%m-%d %H:%M:%S")
        query = text(
            """
            INSERT INTO qna_logs (qna_id, tstp, user_question, response)
            VALUES (:qna_id, :tstp, :user_question, :response);
        """
        )
        conn.execute(
            query,
            {
                "qna_id": str(qna_id),
                "tstp": tstp,
                "user_question": str(user_question),
                "response": str(response),
            },
        )


def load_arxiv():
    query = "SELECT * FROM arxiv_details;"
    conn = create_engine(database_url)
    arxiv_df = pd.read_sql(query, conn)
    arxiv_df.set_index("arxiv_code", inplace=True)
    return arxiv_df


def load_reviews():
    query = "SELECT * FROM summaries;"
    conn = create_engine(database_url)
    summaries_df = pd.read_sql(query, conn)
    summaries_df.set_index("arxiv_code", inplace=True)
    return summaries_df


def load_topics():
    query = "SELECT * FROM topics;"
    conn = create_engine(database_url)
    topics_df = pd.read_sql(query, conn)
    topics_df.set_index("arxiv_code", inplace=True)
    return topics_df


def load_citations():
    query = "SELECT * FROM semantic_details;"
    conn = create_engine(database_url)
    citations_df = pd.read_sql(query, conn)
    citations_df.set_index("arxiv_code", inplace=True)
    citations_df.drop(columns=["paper_id"], inplace=True)
    return citations_df


def get_arxiv_parent_chunk_ids(chunk_ids: list):
    """Get (arxiv_code, parent_id) for a list of (arxiv_code, child_id) tuples."""
    ## ToDo: Improve version param.
    engine = create_engine(database_url)
    with engine.begin() as conn:
        # Prepare a list of conditions for matching pairs of arxiv_code and child_id
        conditions = " OR ".join([
            f"(arxiv_code = '{arxiv_code}' AND child_id = {child_id})"
            for arxiv_code, child_id in chunk_ids
        ])
        query = text(
            f"""
            SELECT DISTINCT arxiv_code, parent_id
            FROM arxiv_chunk_map
            WHERE ({conditions})
            AND version = '10000_1000';
--             AND version = '5000_500';
            """
        )
        result = conn.execute(query)
        parent_ids = result.fetchall()
    return parent_ids


def get_arxiv_chunks(chunk_ids: list, source="child"):
    """Get chunks with metadata for a list of (arxiv_code, chunk_id) tuples."""
    engine = create_engine(database_url)
    source_table = "arxiv_chunks" if source == "child" else "arxiv_parent_chunks"
    with engine.begin() as conn:
        # Prepare a list of conditions for matching pairs of arxiv_code and chunk_id
        conditions = " OR ".join([
            f"(p.arxiv_code = '{arxiv_code}' AND p.chunk_id = {chunk_id})"
            for arxiv_code, chunk_id in chunk_ids
        ])
        query = text(
            f"""
            SELECT d.arxiv_code, d.published, s.citation_count, p.text
            FROM {source_table} p , arxiv_details d, semantic_details s
            WHERE p.arxiv_code = d.arxiv_code
            AND p.arxiv_code = s.arxiv_code
            AND ({conditions});
            """
        )
        result = conn.execute(query)
        chunks = result.fetchall()
        chunks_df = pd.DataFrame(chunks)
    return chunks_df
