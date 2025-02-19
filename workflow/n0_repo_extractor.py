import sys, os
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
import re

load_dotenv()

PROJECT_PATH = os.getenv('PROJECT_PATH', '/app')
sys.path.append(PROJECT_PATH)

os.chdir(PROJECT_PATH)

import utils.vector_store as vs
import utils.db.db_utils as db_utils
import utils.db.paper_db as paper_db
import utils.paper_utils as pu
from utils.logging_utils import setup_logger

logger = setup_logger(__name__, "n0_repo_extractor.log")

def main():
    vs.validate_openai_env()

    logger.info("Starting repo extraction process.")

    arxiv_codes = db_utils.get_arxiv_id_list("arxiv_details")
    done_codes = db_utils.get_arxiv_id_list("arxiv_repos")
    pending_arxiv_codes = list(set(arxiv_codes) - set(done_codes))
    pending_arxiv_codes = sorted(pending_arxiv_codes)[::-1]
    
    total_papers = len(pending_arxiv_codes)
    logger.info(f"Found {total_papers} papers to process for repo extraction")
    
    title_map = db_utils.get_arxiv_title_dict()
    external_resources = []
    
    # Define URL pattern to match any web links
    url_pattern = r'https?://[^\s<>\[\]]+|www\.[^\s<>\[\]]+'
    
    for idx, arxiv_code in enumerate(pending_arxiv_codes, 1):
        paper_title = title_map.get(arxiv_code, "Unknown Title")
        content_df = paper_db.get_extended_content(arxiv_code)
        if len(content_df) == 0:
            logger.warning(f"[{idx}/{total_papers}] No content found: {arxiv_code} - '{paper_title}'")
            continue
                    
        row = content_df.iloc[0]
        paper_markdown = pu.format_paper_summary(row)
        
        has_urls = bool(re.search(url_pattern, paper_markdown))
        
        if has_urls:
            tmp_resources = vs.extract_document_repo(paper_markdown, model="claude-3-5-sonnet-20241022")
            for r in tmp_resources.resources:
                r.arxiv_code = arxiv_code
            if tmp_resources.resources:
                tmp_resources_dicts = [e.model_dump() for e in tmp_resources.resources]
                external_resources.extend(tmp_resources_dicts)
                logger.info(f"[{idx}/{total_papers}] Found {len(tmp_resources.resources)} repos: {arxiv_code} - '{paper_title}'")
            else:
                external_resources.append(
                    {
                        "arxiv_code": arxiv_code,
                        "url": None,
                        "title": None,
                        "description": None,
                    }
                )
                logger.info(f"[{idx}/{total_papers}] No repos found: {arxiv_code} - '{paper_title}'")
        else:
            external_resources.append(
                {
                    "arxiv_code": arxiv_code,
                    "url": None,
                    "title": None,
                    "description": None,
                }
            )
            logger.info(f"[{idx}/{total_papers}] No repos found: {arxiv_code} - '{paper_title}'")
            
        weekly_repos_df = pd.DataFrame(external_resources)
        weekly_repos_df["tstp"] = pd.Timestamp.now()
        db_utils.upload_dataframe(weekly_repos_df, "arxiv_repos")
        external_resources.clear()
        
    logger.info("Repo extraction process completed.")


if __name__ == "__main__":
    main()
