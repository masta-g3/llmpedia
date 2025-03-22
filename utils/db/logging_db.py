"""Database operations for logging-related functionality (READ-ONLY version)."""

import uuid
import logging
from datetime import datetime
from typing import Optional
import requests
import json
import os

# Import custom functions
from .db_utils import execute_read_query

# Setup logging
logger = logging.getLogger("llmpedia_app")

# Define an API endpoint for logging (can be configured in environment variables)
API_ENDPOINT = os.environ.get("LOGGING_API_ENDPOINT", "")
API_KEY = os.environ.get("LOGGING_API_KEY", "")

def _send_log_to_api(data: dict) -> bool:
    """Send log data to remote API if configured."""
    if not API_ENDPOINT:
        logger.info(f"Log event: {data}")
        return True  # No API configured, just log locally
        
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}" if API_KEY else ""
        }
        response = requests.post(API_ENDPOINT, json=data, headers=headers)
        if response.status_code in (200, 201):
            return True
        else:
            logger.error(f"Failed to send log to API: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        logger.error(f"Error sending log to API: {str(e)}")
        return False

def log_error_db(error: str) -> bool:
    """Log error (through API instead of direct DB access)."""
    data = {
        "event_type": "error",
        "error_id": str(uuid.uuid4()),
        "timestamp": datetime.now().isoformat(),
        "error": str(error)
    }
    return _send_log_to_api(data)

def log_qna_db(user_question: str, response: str) -> bool:
    """Log Q&A (through API instead of direct DB access)."""
    data = {
        "event_type": "qna",
        "qna_id": str(uuid.uuid4()),
        "timestamp": datetime.now().isoformat(),
        "user_question": str(user_question),
        "response": str(response)
    }
    return _send_log_to_api(data)

def log_visit(entrypoint: str) -> bool:
    """Log user visit (through API instead of direct DB access)."""
    data = {
        "event_type": "visit",
        "visit_id": str(uuid.uuid4()),
        "timestamp": datetime.now().isoformat(),
        "entrypoint": str(entrypoint)
    }
    return _send_log_to_api(data)

def report_issue(arxiv_code: str, issue_type: str) -> bool:
    """Report an issue (through API instead of direct DB access)."""
    data = {
        "event_type": "issue_report",
        "issue_id": str(uuid.uuid4()),
        "timestamp": datetime.now().isoformat(),
        "arxiv_code": arxiv_code,
        "issue_type": issue_type,
        "resolved": False
    }
    return _send_log_to_api(data)