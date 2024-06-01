from sqlalchemy import create_engine, text
from datetime import datetime
import streamlit as st
import pandas as pd
import psycopg2
import uuid
import os

try:
    db_params = {
        "dbname": os.environ["DB_NAME"],
        "user": os.environ["DB_USER"],
        "password": os.environ["DB_PASS"],
        "host": os.environ["DB_HOST"],
        "port": os.environ["DB_PORT"],
    }
except:
    db_params = {**st.secrets["postgres"]}


database_url = f"postgresql+psycopg2://{db_params['user']}:{db_params['password']}@{db_params['host']}:{db_params['port']}/{db_params['dbname']}"


def list_to_pg_array(lst):
    return "{" + ",".join(lst) + "}"


def pg_array_to_list(array_str):
    return array_str.strip("{}").split(",")


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
    return True


def log_qna_db(user_question, response):
    """Log Q&A in DB along with streamlit app state."""
    try:
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
    except Exception as e:
        print(f"Error in logging Q&A: {e}")
    return True


def log_visit(entrypoint: str):
    """Log user visit in DB."""
    try:
        engine = create_engine(database_url)
        with engine.begin() as conn:
            visit_id = str(uuid.uuid4())
            tstp = pd.to_datetime("now").strftime("%Y-%m-%d %H:%M:%S")
            query = text(
                """
                INSERT INTO visit_logs (visit_id, tstp, entrypoint)
                VALUES (:visit_id, :tstp, :entrypoint);
            """
            )
            conn.execute(
                query,
                {
                    "visit_id": str(visit_id),
                    "tstp": tstp,
                    "entrypoint": str(entrypoint),
                },
            )
    except Exception as e:
        print(f"Error in logging visit: {e}")
    return True


def report_issue(arxiv_code, issue_type):
    """Report an issue in DB."""
    engine = create_engine(database_url)
    with engine.begin() as conn:
        issue_id = str(uuid.uuid4())
        tstp = pd.to_datetime("now").strftime("%Y-%m-%d %H:%M:%S")
        query = text(
            """
            INSERT INTO issue_reports (issue_id, tstp, arxiv_code, issue_type, resolved)
            VALUES (:issue_id, :tstp, :arxiv_code, :issue_type, :resolved);
        """
        )
        conn.execute(
            query,
            {
                "issue_id": str(issue_id),
                "tstp": tstp,
                "arxiv_code": str(arxiv_code),
                "issue_type": str(issue_type),
                "resolved": False,
            },
        )
    return True


def get_reported_non_llm_papers():
    """Get a list of non-LLM papers reported by users."""
    engine = create_engine(database_url)
    with engine.begin() as conn:
        query = text(
            """
            SELECT arxiv_code
            FROM issue_reports
            WHERE issue_type = 'non_llm'
            AND resolved = False;
            """
        )
        result = conn.execute(query)
        reported_papers = result.fetchall()
    reported_papers = [paper[0] for paper in reported_papers]
    return reported_papers


def update_reported_status(arxiv_code, issue_type, resolved=True):
    """Update user-reported issue status in DB (resolved or not)."""
    engine = create_engine(database_url)
    with engine.begin() as conn:
        query = text(
            """
            UPDATE issue_reports
            SET resolved = :resolved
            WHERE arxiv_code = :arxiv_code
            AND issue_type = :issue_type;
            """
        )
        conn.execute(
            query,
            {"resolved": resolved, "arxiv_code": arxiv_code, "issue_type": issue_type},
        )
    return True


def insert_recursive_summary(arxiv_code, summary):
    """Insert data into recursive_summary table in DB."""
    engine = create_engine(database_url)
    with engine.begin() as conn:
        query = text(
            """
            INSERT INTO recursive_summaries (arxiv_code, summary, tstp)
            VALUES (:arxiv_code, :summary, :tstp);
            """
        )
        conn.execute(
            query,
            {"arxiv_code": arxiv_code, "summary": summary, "tstp": datetime.now()},
        )
    return True


def insert_bullet_list_summary(arxiv_code, summary):
    """Insert data into bullet_list_summaries table in DB."""
    engine = create_engine(database_url)
    with engine.begin() as conn:
        query = text(
            """
            INSERT INTO bullet_list_summaries (arxiv_code, summary, tstp)
            VALUES (:arxiv_code, :summary, :tstp);
            """
        )
        conn.execute(
            query,
            {"arxiv_code": arxiv_code, "summary": summary, "tstp": datetime.now()},
        )
    return True


def load_arxiv(arxiv_code: str = None):
    query = "SELECT * FROM arxiv_details"
    if arxiv_code:
        query += f" WHERE arxiv_code = '{arxiv_code}'"
    conn = create_engine(database_url)
    arxiv_df = pd.read_sql(query, conn)
    arxiv_df.set_index("arxiv_code", inplace=True)
    return arxiv_df


def load_summaries():
    query = "SELECT * FROM summaries;"
    conn = create_engine(database_url)
    summaries_df = pd.read_sql(query, conn)
    summaries_df.set_index("arxiv_code", inplace=True)
    summaries_df.drop(columns=["tstp"], inplace=True)
    return summaries_df


def load_recursive_summaries():
    query = "SELECT * FROM recursive_summaries;"
    conn = create_engine(database_url)
    recursive_summaries_df = pd.read_sql(query, conn)
    recursive_summaries_df.set_index("arxiv_code", inplace=True)
    recursive_summaries_df.rename(
        columns={"summary": "recursive_summary"}, inplace=True
    )
    recursive_summaries_df.drop(columns=["tstp"], inplace=True)
    return recursive_summaries_df


def load_bullet_list_summaries():
    query = "SELECT * FROM bullet_list_summaries;"
    conn = create_engine(database_url)
    bullet_list_summaries_df = pd.read_sql(query, conn)
    bullet_list_summaries_df.set_index("arxiv_code", inplace=True)
    bullet_list_summaries_df.rename(columns={"summary": "bullet_list_summary"}, inplace=True)
    bullet_list_summaries_df.drop(columns=["tstp"], inplace=True)
    return bullet_list_summaries_df


def load_summary_notes():
    query = "SELECT * FROM summary_notes;"
    conn = create_engine(database_url)
    extended_summaries_df = pd.read_sql(query, conn)
    extended_summaries_df.set_index("arxiv_code", inplace=True)
    return extended_summaries_df


def load_summary_markdown():
    query = "SELECT * FROM summary_markdown;"
    conn = create_engine(database_url)
    markdown_summaries_df = pd.read_sql(query, conn)
    markdown_summaries_df.set_index("arxiv_code", inplace=True)
    markdown_summaries_df.rename(columns={"summary": "markdown_notes"}, inplace=True)
    markdown_summaries_df.drop(columns=["tstp"], inplace=True)
    return markdown_summaries_df


def load_topics():
    query = "SELECT * FROM topics;"
    conn = create_engine(database_url)
    topics_df = pd.read_sql(query, conn)
    topics_df.set_index("arxiv_code", inplace=True)
    return topics_df


def load_similar_documents():
    query = "SELECT * FROM similar_documents;"
    conn = create_engine(database_url)
    similar_docs_df = pd.read_sql(query, conn)
    similar_docs_df.set_index("arxiv_code", inplace=True)
    similar_docs_df["similar_docs"] = similar_docs_df["similar_docs"].apply(
        pg_array_to_list
    )
    return similar_docs_df


def load_citations(arxiv_code=None):
    query = "SELECT * FROM semantic_details"
    if arxiv_code:
        query += f" WHERE arxiv_code = '{arxiv_code}';"
    conn = create_engine(database_url)
    citations_df = pd.read_sql(query, conn)
    citations_df.set_index("arxiv_code", inplace=True)
    citations_df.drop(columns=["paper_id"], inplace=True)
    return citations_df


def load_tweet_insights(arxiv_code=None):
    query = "SELECT * FROM tweet_reviews where tweet_type = 'insight_v1'"
    if arxiv_code:
        query += f" WHERE arxiv_code = '{arxiv_code}';"
    conn = create_engine(database_url)
    tweet_reviews_df = pd.read_sql(query, conn)
    tweet_reviews_df.set_index("arxiv_code", inplace=True)
    tweet_reviews_df.drop(columns=["tstp", "rejected", "tweet_type"], inplace=True)
    tweet_reviews_df.rename(columns={"review": "tweet_insight"}, inplace=True)
    return tweet_reviews_df


def get_arxiv_parent_chunk_ids(chunk_ids: list):
    """Get (arxiv_code, parent_id) for a list of (arxiv_code, child_id) tuples."""
    ## ToDo: Improve version param.
    engine = create_engine(database_url)
    with engine.begin() as conn:
        # Prepare a list of conditions for matching pairs of arxiv_code and child_id
        conditions = " OR ".join(
            [
                f"(arxiv_code = '{arxiv_code}' AND child_id = {child_id})"
                for arxiv_code, child_id in chunk_ids
            ]
        )
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
        conditions = " OR ".join(
            [
                f"(p.arxiv_code = '{arxiv_code}' AND p.chunk_id = {chunk_id})"
                for arxiv_code, chunk_id in chunk_ids
            ]
        )
        query = text(
            f"""
            SELECT d.arxiv_code, d.title, d.published, s.citation_count, p.text
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


def execute_query(query, db_params=db_params, limit=None):
    """Upload a dictionary to a database."""
    if limit and "LIMIT" not in query:
        query = query.strip().rstrip(";") + f" LIMIT {limit};"
    with psycopg2.connect(**db_params) as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            return cur.fetchall()


def check_in_db(arxiv_code, db_params, table_name):
    """Check if an arxiv code is in the database."""
    with psycopg2.connect(**db_params) as conn:
        with conn.cursor() as cur:
            cur.execute(f"SELECT * FROM {table_name} WHERE arxiv_code = '{arxiv_code}'")
            return bool(cur.rowcount)


def upload_to_db(data, db_params, table_name):
    """Upload a dictionary to a database."""
    with psycopg2.connect(**db_params) as conn:
        with conn.cursor() as cur:
            columns = ", ".join(data.keys())
            placeholders = ", ".join(["%s"] * len(data))
            cur.execute(
                f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})",
                list(data.values()),
            )


def remove_from_db(arxiv_code, db_params, table_name):
    """Remove an entry from the database."""
    with psycopg2.connect(**db_params) as conn:
        with conn.cursor() as cur:
            cur.execute(f"DELETE FROM {table_name} WHERE arxiv_code = '{arxiv_code}'")


def upload_df_to_db(
    df: pd.DataFrame, table_name: str, params: dict, if_exists: str = "append"
):
    """Upload a dataframe to a database."""
    db_url = (
        f"postgresql+psycopg2://{params['user']}:{params['password']}"
        f"@{params['host']}:{params['port']}/{params['dbname']}"
    )
    engine = create_engine(db_url)
    df.to_sql(
        table_name,
        engine,
        if_exists=if_exists,
        index=False,
        method="multi",
        chunksize=10,
    )

    ## Commit.
    with psycopg2.connect(**params) as conn:
        with conn.cursor() as cur:
            cur.execute("COMMIT")

    ## Close.
    engine.dispose()

    return True


def get_arxiv_id_list(db_params=db_params, table_name="arxiv_details"):
    """Get a list of all arxiv codes in the database."""
    with psycopg2.connect(**db_params) as conn:
        with conn.cursor() as cur:
            cur.execute(f"SELECT DISTINCT arxiv_code FROM {table_name}")
            return [row[0] for row in cur.fetchall()]


def get_latest_tstp(
    db_params=db_params, table_name="arxiv_details", extra_condition=""
):
    """Get the latest timestamp in the database."""
    with psycopg2.connect(**db_params) as conn:
        with conn.cursor() as cur:
            cur.execute(f"SELECT MAX(tstp) FROM {table_name} {extra_condition};")
            return cur.fetchone()[0]


def get_max_table_date(db_params, table_name, date_col="date"):
    """Get the max date in a table."""
    with psycopg2.connect(**db_params) as conn:
        with conn.cursor() as cur:
            cur.execute(f"SELECT MAX({date_col}) FROM {table_name};")
            return cur.fetchone()[0]


def get_arxiv_id_embeddings(collection_name, db_params=db_params):
    with psycopg2.connect(**db_params) as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT DISTINCT a.cmetadata->>'arxiv_code' AS arxiv_code
                FROM langchain_pg_embedding a, langchain_pg_collection b
                WHERE a.collection_id = b.uuid
                AND b.name = '{collection_name}'
                AND a.cmetadata->>'arxiv_code' IS NOT NULL;"""
            )
            return [row[0] for row in cur.fetchall()]


def get_arxiv_title_dict(db_params=db_params):
    """Get a list of all arxiv titles in the database."""
    with psycopg2.connect(**db_params) as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
            SELECT a.arxiv_code, a.title 
            FROM arxiv_details a
            WHERE a.title IS NOT NULL
            """
            )
            title_map = {row[0]: row[1] for row in cur.fetchall()}
            return title_map


def get_topic_embedding_dist(db_params=db_params):
    """Get mean and stdDev for topic embeddings (dim1 & dim2)."""
    with psycopg2.connect(**db_params) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
            SELECT AVG(dim1), STDDEV(dim1), AVG(dim2), STDDEV(dim2)
            FROM topics
            """
            )
            res = cur.fetchone()
            res = {
                "dim1": {"mean": res[0], "std": res[1]},
                "dim2": {"mean": res[2], "std": res[3]},
            }
            return res


def get_weekly_summary_inputs(date: str):
    """Get weekly summaries for a given date (from last monday to next sunday)."""
    engine = create_engine(database_url)
    ## Find last monday if not monday.
    date_st = pd.to_datetime(date).date() - pd.Timedelta(
        days=pd.to_datetime(date).weekday()
    )
    ## Find next sunday if not sunday.
    date_end = pd.to_datetime(date).date() + pd.Timedelta(
        days=6 - pd.to_datetime(date).weekday()
    )
    with engine.begin() as conn:
        query = text(
            f"""
                SELECT d.published, d.arxiv_code, d.title, d.authors, sd.citation_count, d.arxiv_comment,
                       d.summary, s.contribution_content, s.takeaway_content, s.takeaway_example, 
                       d.summary AS recursive_summary, sn.tokens
                FROM summaries s
                JOIN arxiv_details d ON s.arxiv_code = d.arxiv_code
                LEFT JOIN semantic_details sd ON s.arxiv_code = sd.arxiv_code
                JOIN summary_notes sn ON s.arxiv_code = sn.arxiv_code
                WHERE d.published BETWEEN '{date_st}' AND '{date_end}'
                AND sn.level = (SELECT MAX(level) FROM summary_notes WHERE arxiv_code = s.arxiv_code)
            """
        )
        result = conn.execute(query)
        summaries = result.fetchall()
        summaries_df = pd.DataFrame(summaries)
        summaries_df["citation_count"] = summaries_df["citation_count"].fillna(0)
    return summaries_df


def check_weekly_summary_exists(date_str: str):
    """Check if weekly summary exists for a given date."""
    engine = create_engine(database_url)
    with engine.begin() as conn:
        query = text(
            f"""
            SELECT COUNT(*)
            FROM weekly_reviews
            WHERE date = '{date_str}'
            """
        )
        result = conn.execute(query)
        count = result.fetchone()[0]

    engine.dispose()
    return count > 0


def get_weekly_summary(date_str: str):
    """Get weekly summary for a given date."""
    engine = create_engine(database_url)
    date_str = (
        pd.to_datetime(date_str).date()
        - pd.Timedelta(days=pd.to_datetime(date_str).weekday())
    ).strftime("%Y-%m-%d")
    with engine.begin() as conn:
        query = text(
            f"""
            SELECT review
            FROM weekly_reviews
            WHERE date = '{date_str}'
            """
        )
        result = conn.execute(query)
        review = result.fetchone()[0]

    engine.dispose()
    return review


def get_extended_notes(arxiv_code: str, level=None, expected_tokens=None):
    """Get extended summary for a given arxiv code."""
    engine = create_engine(database_url)
    with engine.begin() as conn:
        if level:
            query = text(
                f"""
                SELECT arxiv_code, level, summary
                FROM summary_notes
                WHERE arxiv_code = '{arxiv_code}'
                AND level = '{level}';
                """
            )
        elif expected_tokens:
            query = text(
                f"""
                SELECT DISTINCT ON (arxiv_code) arxiv_code, level, summary, tokens
                FROM summary_notes
                WHERE arxiv_code = '{arxiv_code}'
                ORDER BY arxiv_code, ABS(tokens - {expected_tokens}) ASC;
                """
            )
        else:
            query = text(
                f"""
                SELECT DISTINCT ON (arxiv_code) arxiv_code, level, summary
                FROM summary_notes
                WHERE arxiv_code = '{arxiv_code}'
                ORDER BY arxiv_code, level DESC;
                """
            )
        result = conn.execute(query)
        summary = result.fetchone()
    engine.dispose()
    return summary[2]


def get_recursive_summary(arxiv_code: str) -> str:
    """Get recursive summary for a given arxiv code."""
    engine = create_engine(database_url)
    with engine.begin() as conn:
        query = text(
            f"""
            SELECT arxiv_code, summary
            FROM recursive_summaries
            WHERE arxiv_code = '{arxiv_code}';
            """
        )
        result = conn.execute(query)
        summary = result.fetchone()
    engine.dispose()
    result = summary[1] if summary else None
    return result


def insert_tweet_review(arxiv_code, review, tstp, tweet_type, rejected=False):
    """Insert tweet review into the database."""
    engine = create_engine(database_url)
    with engine.begin() as conn:
        query = text(
            """
            INSERT INTO tweet_reviews (arxiv_code, review, tstp, tweet_type, rejected)
            VALUES (:arxiv_code, :review, :tstp, :tweet_type, :rejected);
            """
        )

        conn.execute(
            query,
            {
                "arxiv_code": arxiv_code,
                "review": review,
                "tstp": tstp,
                "tweet_type": tweet_type,
                "rejected": rejected,
            },
        )
    return True
