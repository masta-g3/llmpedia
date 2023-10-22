import sys, os
from dotenv import load_dotenv
load_dotenv()

sys.path.append(os.environ.get("PROJECT_PATH"))
os.chdir(os.environ.get("PROJECT_PATH"))

import re, json
from tqdm import tqdm
import tiktoken
import copy

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import get_openai_callback
from langchain.chains.openai_functions import (
    create_structured_output_chain,
)
import utils.paper_utils as pu
from utils.prompts import summarizer_system_prompt, PaperReview

## LLM model.
llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=0.4)
llm_aux = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.)
token_encoder = tiktoken.encoding_for_model("gpt-3.5-turbo")


def main():
    ## Initialize LLM.
    parsed_list = []

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", summarizer_system_prompt),
            (
                "human",
                "Tip: Make sure to provide your response in the correct format. Do not forget to include the 'applied_example' under 'takeaways'!",
            ),
        ]
    )
    parser = pu.CustomFixParser(pydantic_schema=PaperReview)
    chain = create_structured_output_chain(PaperReview, llm, prompt,
                                           output_parser=parser,
                                           verbose=False)

    ## Get paper list.
    gist_id = "1dd189493c1890df6e04aaea6d049643"
    gist_filename = "llm_queue.txt"
    paper_list = pu.fetch_queue_gist(gist_id, gist_filename)
    paper_list_iter = paper_list[:]

    ## Compare similarity with existing papers.
    existing_papers = pu.get_arxiv_title_dict(pu.db_params)
    existing_paper_names = list(existing_papers.values())
    existing_paper_ids = list(existing_papers.keys())

    ## Iterate.
    gist_url = None
    with get_openai_callback() as cb:
        for paper_name in tqdm(paper_list_iter):
            existing = pu.check_if_exists(paper_name, existing_paper_names, existing_paper_ids)
            if existing:
                print(f"\nSkipping '{paper_name}' as it is already in the database.")
                ## Update gist.
                parsed_list.append(paper_name)
                paper_list = list(set(paper_list) - set(parsed_list))
                gist_url = pu.update_gist(
                    os.environ["GITHUB_TOKEN"],
                    gist_id,
                    gist_filename,
                    "Updated LLM queue.",
                    "\n".join(paper_list),
                )
                continue

            new_doc = pu.search_arxiv_doc(paper_name)
            if new_doc is None:
                print(f"\nCould not find '{paper_name}' in Arxiv. Skipping...")
                continue

            new_meta = new_doc.metadata
            new_content = pu.preprocess_arxiv_doc(new_doc, token_encoder)
            prev_summary = new_meta["Summary"].replace("\n", " ")
            arxiv_code = new_meta["entry_id"].split("/")[-1]
            arxiv_code = re.sub(r"v\d+$", "", arxiv_code)

            ## Check if we have a summary locally.
            local_paper_codes = os.path.join(
                os.environ.get("PROJECT_PATH"), "data", "summaries"
            )
            local_paper_codes = [
                f.split(".json")[0] for f in os.listdir(local_paper_codes)
            ]
            if arxiv_code in local_paper_codes:
                print(f"\nFound '{paper_name}' locally. Skipping...")
                ## Update gist.
                parsed_list.append(paper_name)
                paper_list = list(set(paper_list) - set(parsed_list))
                gist_url = pu.update_gist(
                    os.environ["GITHUB_TOKEN"],
                    gist_id,
                    gist_filename,
                    "Updated LLM queue.",
                    "\n".join(paper_list),
                )
                continue

            ## Try to run LLM process up to 3 times.
            success = False
            for i in range(3):
                try:
                    content = {"content": new_content}#, "prev_summary": prev_summary}
                    summary = chain.run(content)
                    success = True
                    break
                except Exception as e:
                    print(f"\nFailed to run LLM for '{paper_name}'. Attempt {i+1}/3.")
                    print(e)
                    continue
            if not success:
                print(f"Failed to run LLM for '{paper_name}'. Skipping...")
                continue

            ## Extract and combine results.
            parsed_summary = summary.json()
            parsed_summary = json.loads(parsed_summary)
            keep_meta_keys = ["Published", "Title", "Authors", "Summary"]
            keep_meta = {k: v for k, v in new_meta.items() if k in keep_meta_keys}
            result_dict = {**keep_meta, **parsed_summary}
            result_dict["Summary"] = prev_summary

            ## Store.
            pu.store_local(result_dict, arxiv_code, "summaries")
            print(f"\nSummary for '{paper_name}' stored locally.")

            ## Update gist.
            parsed_list.append(paper_name)
            paper_list = list(set(paper_list) - set(parsed_list))
            gist_url = pu.update_gist(
                os.environ["GITHUB_TOKEN"],
                gist_id,
                gist_filename,
                "Updated LLM queue.",
                "\n".join(paper_list),
            )

    if gist_url:
        print(f"Done! Updated queue gist URL: {gist_url}")
    print(cb)


if __name__ == "__main__":
    main()
