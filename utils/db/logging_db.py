"""Database operations for logging-related functionality."""

import uuid
from datetime import datetime
from typing import Optional

from .db_utils import execute_write_query, execute_read_query

def log_instructor_query(
    model_name: str,
    process_id: str,
    prompt_tokens: int,
    completion_tokens: int,
    prompt_cost: float,
    completion_cost: float,
) -> bool:
    """Log token usage in DB."""
    try:
        query = """
            INSERT INTO token_usage_logs (id, tstp, model_name, process_id, prompt_tokens, completion_tokens, prompt_cost, completion_cost)
            VALUES (:id, :tstp, :model_name, :process_id, :prompt_tokens, :completion_tokens, :prompt_cost, :completion_cost)
        """
        
        params = {
            "id": str(uuid.uuid4()),
            "tstp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model_name": model_name,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "process_id": process_id,
            "prompt_cost": prompt_cost,
            "completion_cost": completion_cost,
        }
        
        return execute_write_query(query, params)
    except Exception as e:
        raise e

def log_error_db(error: str) -> bool:
    """Log error in DB along with streamlit app state."""
    try:
        query = """
            INSERT INTO error_logs (error_id, tstp, error)
            VALUES (:error_id, :tstp, :error)
        """
        
        params = {
            "error_id": str(uuid.uuid4()),
            "tstp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "error": str(error),
        }
        
        return execute_write_query(query, params)
    except Exception as e:
        raise e

def log_qna_db(user_question: str, response: str) -> bool:
    """Log Q&A in DB along with streamlit app state."""
    try:
        query = """
            INSERT INTO qna_logs (qna_id, tstp, user_question, response)
            VALUES (:qna_id, :tstp, :user_question, :response)
        """
        
        params = {
            "qna_id": str(uuid.uuid4()),
            "tstp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "user_question": str(user_question),
            "response": str(response),
        }
        
        return execute_write_query(query, params)
    except Exception as e:
        raise e

def log_visit(entrypoint: str) -> bool:
    """Log user visit in DB."""
    try:
        query = """
            INSERT INTO visit_logs (visit_id, tstp, entrypoint)
            VALUES (:visit_id, :tstp, :entrypoint)
        """
        
        params = {
            "visit_id": str(uuid.uuid4()),
            "tstp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "entrypoint": str(entrypoint),
        }
        
        return execute_write_query(query, params)
    except Exception as e:
        raise e

def report_issue(arxiv_code: str, issue_type: str) -> bool:
    """Report an issue in DB."""
    try:
        query = """
            INSERT INTO issue_reports (issue_id, tstp, arxiv_code, issue_type, resolved)
            VALUES (:issue_id, :tstp, :arxiv_code, :issue_type, :resolved)
        """
        
        params = {
            "issue_id": str(uuid.uuid4()),
            "tstp": datetime.now(),
            "arxiv_code": arxiv_code,
            "issue_type": issue_type,
            "resolved": False,
        }
        
        return execute_write_query(query, params)
    except Exception as e:
        raise e

def get_active_users(lookback_hours: int = 12) -> int:
    """Get the count of unique visitors in the last `lookback_hours` hours."""
    query = f"""
        SELECT COUNT(DISTINCT visit_id) 
        FROM visit_logs
        WHERE tstp > NOW() - INTERVAL '{lookback_hours} hours'
    """
    result = execute_read_query(query)
    return result['count'].iloc[0]

def log_feature_poll_vote(feature_name: str, is_custom_suggestion: bool, session_id: Optional[str] = None) -> bool:
    """Log a feature poll vote in the database."""
    try:
        query = """
            INSERT INTO feature_poll_votes (tstp, feature_name, is_custom_suggestion, session_id)
            VALUES (:tstp, :feature_name, :is_custom_suggestion, :session_id)
        """
        
        params = {
            "tstp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "feature_name": feature_name,
            "is_custom_suggestion": is_custom_suggestion,
            "session_id": session_id,
        }
        
        return execute_write_query(query, params)
    except Exception as e:
        raise e
