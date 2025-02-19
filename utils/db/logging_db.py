"""Database operations for logging-related functionality."""

import uuid
from datetime import datetime
from typing import Optional

from .db_utils import execute_write_query

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

def log_workflow_error(step_name: str, script_path: str, error_message: str) -> bool:
    """Log workflow execution errors to the database."""
    try:
        query = """
            INSERT INTO workflow_errors 
            (tstp, step_name, script_path, error_message)
            VALUES (:tstp, :step_name, :script_path, :error_message)
        """
        
        params = {
            "tstp": datetime.now(),
            "step_name": step_name,
            "script_path": script_path,
            "error_message": error_message,
        }
        
        return execute_write_query(query, params)
    except Exception as e:
        raise e

def log_workflow_run(
    step_name: str, 
    script_path: str, 
    status: str, 
    error_message: Optional[str] = None
) -> bool:
    """Log workflow execution status to the database."""
    try:
        query = """
            INSERT INTO workflow_runs 
            (tstp, step_name, script_path, status, error_message)
            VALUES (:tstp, :step_name, :script_path, :status, :error_message)
        """
        
        params = {
            "tstp": datetime.now(),
            "step_name": step_name,
            "script_path": script_path,
            "status": status,
            "error_message": error_message,
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
    

def update_reported_status(arxiv_code: str, issue_type: str, resolved: bool = True) -> bool:
    """Update user-reported issue status in DB (resolved or not)."""
    try:
        query = """
            UPDATE issue_reports
            SET resolved = :resolved
            WHERE arxiv_code = :arxiv_code
            AND issue_type = :issue_type
        """
        
        params = {
            "resolved": resolved,
            "arxiv_code": arxiv_code,
            "issue_type": issue_type,
        }
        
        return execute_write_query(query, params)
    except Exception as e:
        raise e
    
def get_reported_non_llm_papers() -> list[str]:
    """Get a list of non-LLM papers reported by users."""
    try:
        query = """
            SELECT arxiv_code
            FROM issue_reports 
            WHERE issue_type = 'non_llm'
            AND resolved = False
        """
        
        result = execute_write_query(query, {}, fetch=True)
        return [paper[0] for paper in result]
    except Exception as e:
        raise e
