import sys, os
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

PROJECT_PATH = os.getenv('PROJECT_PATH', '/app')
sys.path.append(PROJECT_PATH)

os.chdir(PROJECT_PATH)

import utils.vector_store as vs
import utils.db as db
import utils.paper_utils as pu
from utils.logging_utils import setup_logger

logger = setup_logger(__name__, "n0_repo_extractor.log")

def main():
    vs.validate_openai_env()

    logger.info("Starting repo extraction process.")

    arxiv_codes = db.get_arxiv_id_list(db.db_params, "arxiv_details")
    done_codes = db.get_arxiv_id_list(db.db_params, "arxiv_repos")
    pending_arxiv_codes = list(set(arxiv_codes) - set(done_codes))
    pending_arxiv_codes = sorted(pending_arxiv_codes)[::-1]

    external_resources = []
    for idx, arxiv_code in enumerate(pending_arxiv_codes):
        content_df = db.get_extended_content(arxiv_code)
        if len(content_df) == 0:
            logger.warning(f" [{idx}/{len(pending_arxiv_codes)}] No content found for {arxiv_code}. Skipping.")
            continue
        row = content_df.iloc[0]
        paper_markdown = pu.format_paper_summary(row)
        if "http" in paper_markdown:
            tmp_resources = vs.extract_document_repo(paper_markdown, model="claude-3-5-sonnet-20241022")
            for r in tmp_resources.resources:
                r.arxiv_code = arxiv_code
            if tmp_resources.resources:
                tmp_resources_dicts = [e.model_dump() for e in tmp_resources.resources]
                external_resources.extend(tmp_resources_dicts)
            else:
                external_resources.append(
                    {
                        "arxiv_code": arxiv_code,
                        "url": None,
                        "title": None,
                        "description": None,
                    }
                )
        else:
            external_resources.append(
                {
                    "arxiv_code": arxiv_code,
                    "url": None,
                    "title": None,
                    "description": None,
                }
            )
        weekly_repos_df = pd.DataFrame(external_resources)
        weekly_repos_df["tstp"] = pd.Timestamp.now()
        try:
            db.upload_df_to_db(weekly_repos_df, "arxiv_repos", pu.db_params)
            logger.info(f" [{idx}/{len(pending_arxiv_codes)}] Uploaded external resources for {arxiv_code}.")
        except Exception as e:
            logger.error(f" [{idx}/{len(pending_arxiv_codes)}] Error uploading external resources for {arxiv_code}: {e}")
        external_resources.clear()
    logger.info("Done!")


if __name__ == "__main__":
    main()
