"""Core database utilities for managing connections and executing queries."""

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from contextlib import contextmanager
import pandas as pd
import os
import streamlit as st
from typing import Any, Union, Optional, Dict, List, Generator
import psycopg2

## Get database parameters from environment or streamlit secrets
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

@contextmanager
def get_db_engine() -> Generator[Engine, None, None]:
    """Context manager for database engine to ensure proper disposal."""
    engine = create_engine(database_url)
    try:
        yield engine
    finally:
        engine.dispose()

def execute_read_query(query_string: str, params: Optional[dict] = None, as_dataframe: bool = True) -> Union[pd.DataFrame, Any]:
    """Execute a read query and return results as DataFrame or raw data."""
    with get_db_engine() as engine:
        with engine.begin() as conn:
            if as_dataframe:
                return pd.read_sql(text(query_string), conn, params=params)
            else:
                result = conn.execute(text(query_string), params or {})
                return result.fetchall()

def execute_write_query(query_string: str, params: Optional[dict] = None) -> bool:
    """Execute a write query (INSERT/UPDATE/DELETE) and return success status."""
    try:
        with get_db_engine() as engine:
            with engine.begin() as conn:
                conn.execute(text(query_string), params or {})
        return True
    except Exception as e:
        raise e

def batch_list(lst: list, batch_size: int = 1000) -> list[list]:
    """Split a list into batches of specified size."""
    return [lst[i:i + batch_size] for i in range(0, len(lst), batch_size)]

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
    select_cols: Optional[List[str]] = None
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
                if hasattr(value, 'dtype') and hasattr(value, 'tolist'):  # Check if it's a numpy array
                    value = value.tolist()
                
                if isinstance(value, (list, tuple)):
                    # Handle list values with IN clause
                    param_key = key.replace(" ", "_")
                    where_clauses.append(f"{key} IN :{param_key}")
                    params[param_key] = tuple(value)  # SQLAlchemy requires tuple for IN clause
                elif " " in key:  # For operators like >=, <=
                    param_key = key.replace(" ", "_").replace(">=", "gte").replace("<=", "lte").replace(">", "gt").replace("<", "lt")
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
                df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)
            if rename_cols:
                df.rename(columns={k: v for k, v in rename_cols.items() if k in df.columns}, inplace=True)
        
        return df
    except Exception as e:
        raise e

def get_arxiv_id_list(table_name: str = "arxiv_details") -> List[str]:
    """Get a list of all arxiv codes in the specified table."""
    try:
        with psycopg2.connect(**db_params) as conn:
            with conn.cursor() as cur:
                cur.execute(f"SELECT DISTINCT arxiv_code FROM {table_name}")
                return [row[0] for row in cur.fetchall()]
    except Exception as e:
        raise e
    

def get_arxiv_title_dict() -> Dict[str, str]:
    """Get a mapping of arxiv codes to their titles."""
    df = simple_select_query(
        table="arxiv_details",
        select_cols=["arxiv_code", "title"],
        conditions={"title IS NOT": None}
    )
    return df["title"].to_dict() if not df.empty else {} 

def remove_by_arxiv_code(arxiv_code: str, table: str) -> bool:
    """Delete entries from a table based on arxiv_code."""
    return execute_write_query(
        f"DELETE FROM {table} WHERE arxiv_code = :arxiv_code",
        {"arxiv_code": arxiv_code}
    )


def get_max_table_date(table_name: str, date_col: str = "date") -> Optional[pd.Timestamp]:
    """Get the max date in a table."""
    df = simple_select_query(
        table=table_name,
        select_cols=[f"MAX({date_col}) as max_date"]
    )
    return df["max_date"].iloc[0] if not df.empty else None


def upload_dataframe(
    df: pd.DataFrame,
    table: str,
    if_exists: str = "append",
    index: bool = False,
    chunk_size: Optional[int] = None
) -> bool:
    """ Upload a pandas DataFrame to the specified database table. """
    try:
        with get_db_engine() as engine:
            df.to_sql(
                name=table,
                con=engine,
                if_exists=if_exists,
                index=index,
                chunksize=chunk_size
            )
        return True
    except Exception as e:
        raise e


def list_to_pg_array(lst):
    lst = [str(x).replace("arxiv_code:", "") for x in lst]
    lst = [x.replace("arxiv:", "") for x in lst]
    return "{" + ",".join(lst) + "}"