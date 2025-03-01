import os
import logging
import utils.db.paper_db as paper_db
import utils.db.db_utils as db_utils

def main():
    # Get list of arxiv IDs from database
    arxiv_ids = db_utils.get_arxiv_id_list(table_name="llm_tweets")
    
    # Get title dictionary
    title_dict = paper_db.get_arxiv_title_dict()
    
    # ... rest of existing code ... 