"""Core database utilities for managing connections and executing queries (READ-ONLY version)."""

from typing import Any, Union, Optional, Dict, List, Generator
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from contextlib import contextmanager
from datetime import datetime, timedelta
import pandas as pd
import os
import streamlit as st
import psycopg2
import logging
import random

## Get database parameters from environment or streamlit secrets
try:
    db_params = {
        "dbname": os.environ["DB_NAME"],
        "user": os.environ["DB_USER"],
        "password": os.environ["DB_PASSWORD"],
        "host": os.environ["DB_HOST"],
        "port": os.environ["DB_PORT"],
    }
except:
    db_params = {**st.secrets["postgres"]}

database_url = f"postgresql+psycopg2://{db_params['user']}:{db_params['password']}@{db_params['host']}:{db_params['port']}/{db_params['dbname']}"


@contextmanager
def get_db_engine() -> Generator[Engine, None, None]:
    """Context manager for database engine to ensure proper disposal."""
    engine = create_engine(database_url)
    try:
        yield engine
    finally:
        engine.dispose()


def execute_read_query(
    query_string: str, params: Optional[dict] = None, as_dataframe: bool = True
) -> Union[pd.DataFrame, Any]:
    """Execute a read query and return results as DataFrame or raw data."""
    # Safety check to prevent write operations
    query_lower = query_string.lower().strip()
    if any(
        query_lower.startswith(op)
        for op in ["insert", "update", "delete", "drop", "alter", "create"]
    ):
        logging.error("Attempted write operation in read-only mode: %s", query_string)
        raise PermissionError(
            "This application has read-only access to the database. Write operations are not permitted."
        )

    with get_db_engine() as engine:
        with engine.begin() as conn:
            if as_dataframe:
                return pd.read_sql(text(query_string), conn, params=params)
            else:
                result = conn.execute(text(query_string), params or {})
                return result.fetchall()


def execute_write_query(query_string: str, params: Optional[dict] = None) -> bool:
    """Execute a write query - DISABLED in read-only mode."""
    logging.error("Attempted write operation in read-only mode: %s", query_string)
    raise PermissionError(
        "This application has read-only access to the database. Write operations are not permitted."
    )


def batch_list(lst: list, batch_size: int = 1000) -> list[list]:
    """Split a list into batches of specified size."""
    return [lst[i : i + batch_size] for i in range(0, len(lst), batch_size)]


def build_where_clause(fields_and_values: dict) -> str:
    """Build a WHERE clause from a dictionary of field-value pairs."""
    conditions = []
    for field, value in fields_and_values.items():
        if isinstance(value, (list, tuple)):
            value_str = "', '".join(str(v) for v in value)
            conditions.append(f"{field} IN ('{value_str}')")
        else:
            conditions.append(f"{field} = '{value}'")
    return " AND ".join(conditions)


def simple_select_query(
    table: str,
    conditions: Optional[Dict] = None,
    order_by: Optional[str] = None,
    index_col: Optional[str] = "arxiv_code",
    drop_cols: Optional[List[str]] = None,
    rename_cols: Optional[Dict[str, str]] = None,
    select_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Execute a simple SELECT query with optional conditions and DataFrame processing."""
    try:
        # Build base query
        query = f"SELECT {', '.join(select_cols) if select_cols else '*'} FROM {table}"

        # Add WHERE clause if conditions exist
        params = {}
        if conditions:
            where_clauses = []
            for key, value in conditions.items():
                if key == "LIMIT":
                    continue  # Handle LIMIT separately

                # Convert numpy arrays to regular Python lists
                if hasattr(value, "dtype") and hasattr(
                    value, "tolist"
                ):  # Check if it's a numpy array
                    value = value.tolist()

                if isinstance(value, (list, tuple)):
                    # Handle list values with IN clause
                    param_key = key.replace(" ", "_")
                    where_clauses.append(f"{key} IN :{param_key}")
                    params[param_key] = tuple(
                        value
                    )  # SQLAlchemy requires tuple for IN clause
                elif " " in key:  # For operators like >=, <=
                    param_key = (
                        key.replace(" ", "_")
                        .replace(">=", "gte")
                        .replace("<=", "lte")
                        .replace(">", "gt")
                        .replace("<", "lt")
                    )
                    where_clauses.append(f"{key} :{param_key}")
                    params[param_key] = value
                else:
                    where_clauses.append(f"{key} = :{key}")
                    params[key] = value

            if where_clauses:
                query += " WHERE " + " AND ".join(where_clauses)

        # Add ORDER BY if specified
        if order_by:
            query += f" ORDER BY {order_by}"

        # Add LIMIT if specified in conditions
        if conditions and "LIMIT" in conditions:
            query += f" LIMIT {conditions['LIMIT']}"

        # Execute query
        with get_db_engine() as engine:
            df = pd.read_sql(text(query), engine, params=params)

        # Post-process DataFrame
        if not df.empty:
            if index_col and index_col in df.columns:
                df.set_index(index_col, inplace=True)
            if drop_cols:
                df.drop(
                    columns=[col for col in drop_cols if col in df.columns],
                    inplace=True,
                )
            if rename_cols:
                df.rename(
                    columns={k: v for k, v in rename_cols.items() if k in df.columns},
                    inplace=True,
                )

        return df
    except Exception as e:
        logging.error("Error in simple_select_query: %s", str(e))
        raise e


def get_arxiv_id_list(table_name: str = "arxiv_details") -> List[str]:
    """Get a list of all arxiv codes in the specified table."""
    try:
        with psycopg2.connect(**db_params) as conn:
            with conn.cursor() as cur:
                cur.execute(f"SELECT DISTINCT arxiv_code FROM {table_name}")
                return [row[0] for row in cur.fetchall()]
    except Exception as e:
        logging.error("Error in get_arxiv_id_list: %s", str(e))
        raise e


def get_arxiv_title_dict() -> Dict[str, str]:
    """Get a mapping of arxiv codes to their titles."""
    df = simple_select_query(
        table="arxiv_details",
        select_cols=["arxiv_code", "title"],
        conditions={"title IS NOT": None},
    )
    return df["title"].to_dict() if not df.empty else {}


def get_max_table_date(
    table_name: str, date_col: str = "date"
) -> Optional[pd.Timestamp]:
    """Get the max date in a table."""
    df = simple_select_query(
        table=table_name, select_cols=[f"MAX({date_col}) as max_date"]
    )
    return df["max_date"].iloc[0] if not df.empty else None


def list_to_pg_array(lst):
    """Convert a list to a PostgreSQL array string format."""
    lst = [str(x).replace("arxiv_code:", "") for x in lst]
    lst = [x.replace("arxiv:", "") for x in lst]
    return "{" + ",".join(lst) + "}"


def query_db(query_string: str, params: Optional[dict] = None) -> List[Dict]:
    """Execute a query and return results as a list of dictionaries."""
    df = execute_read_query(query_string, params)
    return df.to_dict(orient="records") if not df.empty else []


def get_random_interesting_facts(n=10, recency_days=7) -> List[Dict]:
    """Get random interesting facts with bias toward recent ones."""
    # Calculate the cutoff date for recent facts
    recent_cutoff = datetime.now() - timedelta(days=recency_days)
    recent_cutoff_str = recent_cutoff.strftime("%Y-%m-%d")

    # Get both recent and older facts in a single query
    # UNION ensures no duplicates of exact same rows
    query = f"""
        (SELECT id, arxiv_code, fact, tstp, 
               1 as is_recent
        FROM summary_interesting_facts
        WHERE tstp >= '{recent_cutoff_str}'
        ORDER BY RANDOM()
        LIMIT {min(n*2, 50)})
        
        UNION ALL
        
        (SELECT id, arxiv_code, fact, tstp,
               0 as is_recent
        FROM summary_interesting_facts
        WHERE tstp < '{recent_cutoff_str}'
        ORDER BY RANDOM()
        LIMIT {n*2})
    """
    
    # Get facts and deduplicate based on content
    all_facts = query_db(query)
    unique_facts = []
    seen_content = set()
    
    # First prioritize recent facts (they come first in the results)
    for fact in all_facts:
        if fact['fact'] not in seen_content and len(unique_facts) < n:
            seen_content.add(fact['fact'])
            unique_facts.append(fact)
    
    # Ensure at least 70% recent facts if possible (recency bias)
    if len(unique_facts) > n:
        recent_facts = [f for f in unique_facts if f['is_recent'] == 1]
        older_facts = [f for f in unique_facts if f['is_recent'] == 0]
        
        # Try to keep at least 70% recent facts if we have enough
        recent_target = min(int(n * 0.7), len(recent_facts))
        older_target = n - recent_target
        
        # Combine with preference for recent facts
        final_facts = (
            sorted(recent_facts, key=lambda x: x['tstp'], reverse=True)[:recent_target] + 
            sorted(older_facts, key=lambda x: x['tstp'], reverse=True)[:older_target]
        )
        unique_facts = final_facts
    
    # Enhance facts with paper title for context
    for fact in unique_facts:
        title_query = f"""
            SELECT title 
            FROM arxiv_details 
            WHERE arxiv_code = '{fact['arxiv_code']}'
        """
        title_result = query_db(title_query)
        if title_result:
            fact["paper_title"] = title_result[0]["title"]
        else:
            fact["paper_title"] = "Unknown paper"

    return unique_facts
