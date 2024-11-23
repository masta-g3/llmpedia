import sys, os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from tqdm import tqdm

PROJECT_PATH = os.environ.get("PROJECT_PATH")
PROJECT_PATH = os.getenv('PROJECT_PATH', '/app')
sys.path.append(PROJECT_PATH)

os.chdir(PROJECT_PATH)

import utils.paper_utils as pu
import utils.db as db
import utils.prompts as p
from utils.instruct import run_instructor_query

def main():
    arxiv_codes = pu.get_local_arxiv_codes()
    done_codes = db.get_arxiv_id_list(db.db_params, "arxiv_dashboards")
    arxiv_codes = list(set(arxiv_codes) - set(done_codes))
    arxiv_codes = sorted(arxiv_codes)[::-1][:20]

    for arxiv_code in arxiv_codes:
        title = db.get_arxiv_title_dict()[arxiv_code]
        content = db.get_extended_notes(arxiv_code, expected_tokens=3000)
        res_str = run_instructor_query(
            p.DATA_CARD_SYSTEM_PROMPT,
            p.PDATA_CARD_USER_PROMPT.format(title=title, content=content),
            llm_model="claude-3-5-sonnet-20241022",
            temperature=0.5,
            process_id="data_card"
        )
        summary = res_str.split("<summary>")[1].split("</summary>")[0].strip()
        script = res_str.split("<script>")[1].split("</script>")[0].strip()
        scratchpad = ""
        db.save_arxiv_dashboard_script(arxiv_code, summary, scratchpad, script)

    print("Done!")


if __name__ == "__main__":
    main()
