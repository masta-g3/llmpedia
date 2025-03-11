import sys, os
import pandas as pd
from dotenv import load_dotenv
import time

load_dotenv()

PROJECT_PATH = os.getenv("PROJECT_PATH", "/app")
sys.path.append(PROJECT_PATH)

os.chdir(PROJECT_PATH)

import utils.paper_utils as pu
import utils.vector_store as vs
import utils.db.paper_db as paper_db
import utils.db.db_utils as db_utils
from utils.logging_utils import setup_logger

# Set up logging
logger = setup_logger(__name__, "weekly_review.log")

summaries_path = os.path.join(os.environ.get("PROJECT_PATH"), "data", "summaries")
meta_path = os.path.join(os.environ.get("PROJECT_PATH"), "data", "arxiv_meta")
review_path = os.path.join(os.environ.get("PROJECT_PATH"), "data", "weekly_reviews")


def main(date_str: str):
    """Generate a weekly review of highlights and takeaways from papers."""
    ## Convert date_str to datetime and shift to previous Monday if needed.
    date_str_dt = pd.to_datetime(date_str)
    days_since_monday = date_str_dt.weekday()
    if days_since_monday > 0:
        date_str_dt = date_str_dt - pd.Timedelta(days=days_since_monday)
        date_str = date_str_dt.strftime("%Y-%m-%d")
    logger.info(f"Starting weekly review generation for week of {date_str}")

    ## Check if we have the summary.
    vs.validate_openai_env()
    if paper_db.check_weekly_summary_exists(date_str):
        logger.info(f"Weekly summary already exists. Skipping...")
        return

    ## Get data to generate summary.
    date_st = pd.to_datetime(date_str)
    weekly_content_df = paper_db.get_weekly_summary_inputs(date_str)
    logger.info(f"Found {len(weekly_content_df)} papers")

    ## Get weekly total counts for the last 8 weeks.
    prev_mondays = pd.date_range(
        date_st - pd.Timedelta(days=7 * 8), date_st, freq="W-MON"
    )
    prev_mondays = [date.strftime("%Y-%m-%d") for date in prev_mondays]
    weekly_counts = {
        date_str: len(paper_db.get_weekly_summary_inputs(date_str))
        for date_str in prev_mondays
    }
    logger.info("Retrieved weekly counts for the past 8 weeks")

    ## Get previous summary
    try:
        previous_summaries = []
        for i in range(2, 6):
            summary = paper_db.get_weekly_content(
                prev_mondays[-i], content_type="content"
            )
            if summary:
                previous_summaries.append(summary)
        
        previous_themes = "\n".join([s.split("\n")[0] for s in previous_summaries])
        logger.info("Retrieved previous 3 weeks' summaries")
    except:
        previous_summary = paper_db.get_weekly_summary_old(prev_mondays[-2])
        if previous_summary is None:
            previous_themes = "N/A"
            logger.warning("Could not find previous week's summary")
        else:
            previous_themes = previous_summary.split("\n##")[2]
            logger.info("Retrieved previous week's summary from old format")

    ## Format content
    date_end = date_st + pd.Timedelta(days=6)
    date_st_long = date_st.strftime("%B %d, %Y")
    date_end_long = date_end.strftime("%B %d, %Y")
    weekly_content_md = f"# Weekly Review ({date_st_long} to {date_end_long})\n\n"

    ## Add table of weekly paper counts.
    logger.info("Generating weekly content markdown")
    weekly_content_md += f"## Weekly Publication Trends\n"
    weekly_content_md += "| Week | Total Papers |\n"
    weekly_content_md += "| --- | --- |\n"
    for tmp_date_str, count in weekly_counts.items():
        date = pd.to_datetime(tmp_date_str)
        date_st_long = date.strftime("%B %d, %Y")
        date_end = date + pd.Timedelta(days=7)
        date_end_long = date_end.strftime("%B %d, %Y")
        if tmp_date_str == date_str:
            weekly_content_md += (
                f"| **{date_st_long} to {date_end_long}** | **{count}** |\n\n\n"
            )
        else:
            weekly_content_md += f"| {date_st_long} to {date_end_long} | {count} |\n"

    weekly_content_md += f"## Papers Published This Week\n"

    html_markdowns = []
    for idx, row in weekly_content_df.iterrows():
        paper_markdown = pu.format_paper_summary(row)
        weekly_content_md += paper_markdown
        if "http" in paper_markdown:
            html_markdowns.append(paper_markdown)

    weekly_content_md += f"## Previous Week's Submissions for New Developments and Themes\nBelow is the introduction you published last week. Play close attention to the themes mentioned so you don't repeat them.\n"
    # weekly_content_md += f"```{previous_themes}```"

    ## Generate summary.
    logger.info("Generating weekly report")
    weekly_summary_obj = vs.generate_weekly_report(
        weekly_content_md, model="claude-3-7-sonnet-20250219"
    )
    weekly_summary_obj = (
        weekly_summary_obj.replace("<new_developments_findings>", "")
        .replace("</new_developments_findings>", "")
        .strip()
    )

    logger.info("Generating weekly highlight")
    weekly_highlight = vs.generate_weekly_highlight(
        weekly_content_md, model="claude-3-7-sonnet-20250219"
    )

    ## Format content.
    logger.info("Formatting and preparing data for storage")
    date = pd.to_datetime(date_str)
    tstp_now = pd.Timestamp.now()

    # weekly_topics_df = pd.DataFrame(
    #     [
    #         {"topic": topic, "arxiv_codes": db.list_to_pg_array(arxiv_codes)}
    #         for topic, arxiv_codes in weekly_summary_obj.themes_mapping.items()
    #     ]
    # )
    # weekly_topics_df["date"] = date
    # weekly_topics_df["tstp"] = tstp_now

    # weekly_content_df = pd.DataFrame.from_dict(
    #     weekly_summary_obj.dict(), orient="index"
    # ).T
    weekly_content_df = pd.DataFrame(
        {
            "content": [weekly_summary_obj],
            "highlight": [weekly_highlight],
            "scratchpad_papers": [None],
            "scratchpad_themes": [None],
            "date": [date],
            "tstp": [tstp_now],
        }
    )

    ## Store.
    logger.info("Uploading weekly content to database")
    db_utils.upload_dataframe(weekly_content_df, "weekly_content")
    # db.upload_df_to_db(weekly_topics_df, "weekly_topics", pu.db_params)
    logger.info(f"Successfully completed weekly review for {date_str}")


if __name__ == "__main__":
    ## Use provided date or default to 2024-10-07
    date_str = sys.argv[1] if len(sys.argv) == 2 else "2025-02-12"
    main(date_str)
