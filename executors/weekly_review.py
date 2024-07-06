import sys, os
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
import time
import re

load_dotenv()

from langchain_community.callbacks import get_openai_callback

sys.path.append(os.environ.get("PROJECT_PATH"))
os.chdir(os.environ.get("PROJECT_PATH"))

import utils.paper_utils as pu
import utils.vector_store as vs
import utils.db as db

summaries_path = os.path.join(os.environ.get("PROJECT_PATH"), "data", "summaries")
meta_path = os.path.join(os.environ.get("PROJECT_PATH"), "data", "arxiv_meta")
review_path = os.path.join(os.environ.get("PROJECT_PATH"), "data", "weekly_reviews")


def main(date_str: str):
    """Generate a weekly review of highlights and takeaways from papers."""
    ## Check if we have the summary.
    vs.validate_openai_env()
    if db.check_weekly_summary_exists(date_str):
        return

    ## Get data to generate summary.
    date_st = pd.to_datetime(date_str)
    weekly_content_df = db.get_weekly_summary_inputs(date_str)

    ## Get weekly total counts for the last 4 weeks.
    prev_mondays = pd.date_range(
        date_st - pd.Timedelta(days=7 * 4), date_st, freq="W-MON"
    )
    prev_mondays = [date.strftime("%Y-%m-%d") for date in prev_mondays]
    weekly_counts = {
        date_str: len(db.get_weekly_summary_inputs(date_str))
        for date_str in prev_mondays
    }

    previous_summary = db.get_weekly_summary(prev_mondays[-2])
    previous_themes = previous_summary.split("\n##")[2]

    date_end = date_st + pd.Timedelta(days=6)
    date_st_long = date_st.strftime("%B %d, %Y")
    date_end_long = date_end.strftime("%B %d, %Y")
    weekly_content_md = f"# Weekly Review ({date_st_long} to {date_end_long})\n\n"

    ## Add table of weekly paper counts.
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

    # weekly_content_md += f"*Total papers published this week: {len(weekly_content_df)}*\n"
    weekly_content_md += f"## Papers Published This Week\n\n"

    for idx, row in weekly_content_df.iterrows():
        paper_markdown = pu.format_paper_summary(row)
        weekly_content_md += paper_markdown
        if idx >= 50:
            weekly_content_md += f"\n*...and {len(weekly_content_df) - idx} more.*"
            break

    ## Add previous "New Developments and Findings" section.
    weekly_content_md += f"\n\n## Last Week's Submissions for New Developments and Themes\n"
    weekly_content_md += previous_themes

    with get_openai_callback() as cb:
        ## Generate summary.
        weekly_summary_obj = vs.generate_weekly_report(
            weekly_content_md
        )
        tstp_now = pd.Timestamp.now()
        date = pd.to_datetime(date_str)
        weekly_markdown = vs.ps.generate_weekly_review_markdown(weekly_summary_obj, date)
        weekly_summary_df = pd.DataFrame(
            {"date": [date], "tstp": [tstp_now], "review": [weekly_markdown], "review_json": [weekly_summary_obj.json()]}
        )
        db.upload_df_to_db(weekly_summary_df, "weekly_reviews", pu.db_params)


if __name__ == "__main__":
    ## Read dates from arguments.
    start_dt = sys.argv[1]
    end_dt = sys.argv[2]

    date_range = pd.date_range(start_dt, end_dt, freq="W-MON")
    date_range = [date.strftime("%Y-%m-%d") for date in date_range]
    for date_str in tqdm(date_range):
        main(date_str)
        time.sleep(5)

