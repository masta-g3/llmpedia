from sqlalchemy import create_engine, text
import pandas as pd
import uuid
import os

db_params = {
    "dbname": os.environ["DB_NAME"],
    "user": os.environ["DB_USER"],
    "password": os.environ["DB_PASS"],
    "host": os.environ["DB_HOST"],
    "port": os.environ["DB_PORT"],
}

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
