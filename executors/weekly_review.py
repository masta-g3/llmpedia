import sys, os
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
import datetime
import time
import re

load_dotenv()

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

    ## Get weekly total counts for the last 8 weeks.
    prev_mondays = pd.date_range(
        date_st - pd.Timedelta(days=7 * 8), date_st, freq="W-MON"
    )
    prev_mondays = [date.strftime("%Y-%m-%d") for date in prev_mondays]
    weekly_counts = {
        date_str: len(db.get_weekly_summary_inputs(date_str))
        for date_str in prev_mondays
    }

    ## ToDo: Remove this block.
    ## -------------------------
    try:
        previous_summary = db.get_weekly_content(prev_mondays[-2], content_type="content")
        previous_themes = previous_summary.split("\n")[0]
    except:
        previous_summary = db.get_weekly_summary_old(prev_mondays[-2])
        if previous_summary is None:
            previous_themes = "N/A"
        else:
            previous_themes = previous_summary.split("\n##")[2]

    ## -------------------------
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
    weekly_content_md += f"## Papers Published This Week\n"

    html_markdowns = []
    for idx, row in weekly_content_df.iterrows():
        paper_markdown = pu.format_paper_summary(row)
        weekly_content_md += paper_markdown
        if "http" in paper_markdown:
            html_markdowns.append(paper_markdown)

    weekly_content_md += f"## Last Week's Submissions for New Developments and Themes\nBelow is the introduction you published last week.\n"
    weekly_content_md += f"```{previous_themes}```"

    ## Generate summary.
    weekly_summary_obj = vs.generate_weekly_report(weekly_content_md, model="gpt-4o")
    weekly_highlight = vs.generate_weekly_highlight(weekly_content_md, model="gpt-4o")

    ## Format content.
    date = pd.to_datetime(date_str)
    tstp_now = pd.Timestamp.now()

    weekly_topics_df = pd.DataFrame(
        [
            {"topic": topic, "arxiv_codes": db.list_to_pg_array(arxiv_codes)}
            for topic, arxiv_codes in weekly_summary_obj.themes_mapping.items()
        ]
    )
    weekly_topics_df["date"] = date
    weekly_topics_df["tstp"] = tstp_now

    weekly_content_df = pd.DataFrame.from_dict(
        weekly_summary_obj.dict(), orient="index"
    ).T
    weekly_content_df["highlight"] = weekly_highlight
    weekly_content_df.drop(columns=["themes_mapping"], inplace=True)
    weekly_content_df.rename(
        columns={"new_developments_findings": "content"}, inplace=True
    )
    weekly_content_df["date"] = date
    weekly_content_df["tstp"] = tstp_now

    ## Store.
    db.upload_df_to_db(weekly_content_df, "weekly_content", pu.db_params)
    db.upload_df_to_db(weekly_topics_df, "weekly_topics", pu.db_params)


if __name__ == "__main__":
    ## Read dates from arguments.
    start_dt = sys.argv[1]
    end_dt = sys.argv[2]

    date_range = pd.date_range(start_dt, end_dt, freq="W-MON")
    date_range = [date.strftime("%Y-%m-%d") for date in date_range]
    for date_str in tqdm(date_range):
        main(date_str)
        time.sleep(5)
