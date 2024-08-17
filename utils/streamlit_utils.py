import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
from typing import Dict, Tuple
import time

import utils.app_utils as au
import utils.data_cards as dc
import utils.db as db


def create_sidebar(full_papers_df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    ## Filter sidebar.
    st.sidebar.markdown("# 📁 Filters")
    ## Filter by year or select all of them.
    year_cols = st.sidebar.columns((1, 2))
    _ = year_cols[0].markdown("#### Year")
    all_years = year_cols[0].checkbox("All ", value=st.session_state.all_years)
    st.session_state.all_years = all_years

    _ = year_cols[1].markdown("####")
    year = year_cols[1].slider(
        "Year",
        min_value=2016,
        max_value=2024,
        value=2024,
        step=1,
        label_visibility="collapsed",
        disabled=st.session_state.all_years,
    )

    search_term = st.sidebar.text_input("Search", "")
    search_opt_cols = st.sidebar.columns((1, 1))
    title_only = search_opt_cols[0].checkbox("`Title Only`", value=True)
    code_only = search_opt_cols[1].checkbox("`Arxiv Code`", value=False)
    categories = st.sidebar.multiselect(
        "Categories",
        list(full_papers_df["category"].unique()),
    )
    topics = st.sidebar.multiselect(
        "Topic Group",
        list(full_papers_df["topic"].unique()),
    )

    min_citations = st.sidebar.select_slider(
        "Min Citations",
        options=[0, 1, 5, 10, 100],
        value=0,
    )

    ## Sort by.
    sort_by = st.sidebar.selectbox(
        "Sort By",
        ["Published Date", "Last Updated", "Citations", "Random"],
    )

    ## Year filter.
    if not st.session_state.all_years:
        papers_df = full_papers_df[full_papers_df["published"].dt.year == int(year)]
    else:
        papers_df = full_papers_df.copy()

    ## Search terms.
    if len(search_term) > 0 and title_only:
        search_term = search_term.lower()
        papers_df = papers_df[papers_df["title"].str.lower().str.contains(search_term)]
    elif len(search_term) > 0 and code_only:
        search_term = search_term.lower()
        papers_df = papers_df[
            papers_df["arxiv_code"].str.lower().str.contains(search_term)
        ]
        st.session_state.arxiv_code = search_term
    elif len(search_term) > 0:
        search_term = search_term.lower()
        papers_df = papers_df[
            papers_df["title"].str.lower().str.contains(search_term)
            | papers_df["arxiv_code"].str.lower().str.contains(search_term)
            | papers_df["authors"].str.lower().str.contains(search_term)
            | papers_df["summary"].str.lower().str.contains(search_term)
            | papers_df["contribution_title"].str.lower().str.contains(search_term)
            | papers_df["contribution_content"].str.lower().str.contains(search_term)
            | papers_df["takeaway_title"].str.lower().str.contains(search_term)
            | papers_df["takeaway_content"].str.lower().str.contains(search_term)
        ]

    ## Categories.
    if len(categories) > 0:
        papers_df = papers_df[papers_df["category"].isin(categories)]

    # Cluster.
    if len(topics) > 0:
        papers_df = papers_df[papers_df["topic"].isin(topics)]

    ## Citations.
    papers_df = papers_df[papers_df["citation_count"] >= min_citations]

    ## Order.
    if sort_by == "Last Updated":
        papers_df = papers_df.sort_values("updated", ascending=False)
    elif sort_by == "Published Date":
        papers_df = papers_df.sort_values("published", ascending=False)
    elif sort_by == "Citations":
        papers_df = papers_df.sort_values("citation_count", ascending=False)
    elif sort_by == "Random":
        papers_df = papers_df.sample(frac=1)

    return papers_df, year


def create_paper_card(paper: Dict, mode="closed", name=""):
    """Creates card UI for paper details."""
    img_cols = st.columns((1, 3))
    expanded = False
    if mode == "open":
        expanded = True
    paper_code = paper["arxiv_code"]
    try:
        img_cols[0].image(
            f"https://llmpedia.s3.amazonaws.com/{paper_code}.png", use_column_width=True
        )
    except:
        pass

    paper_title = paper["title"]
    paper_url = paper["url"]
    img_cols[1].markdown(
        f'<h2><a href="{paper_url}" style="color: #FF4B4B;">{paper_title}</a></h2>',
        unsafe_allow_html=True,
    )

    pub_date = pd.to_datetime(paper["published"]).strftime("%B %d, %Y")
    upd_date = pd.to_datetime(paper["updated"]).strftime("%B %d, %Y")
    tweet_insight = paper["tweet_insight"]
    if not pd.isna(tweet_insight):
        tweet_insight = tweet_insight.split("):")[1].strip()
        img_cols[1].markdown(f"🐦 *{tweet_insight}*")
    img_cols[1].markdown(f"#### Published: {pub_date}")
    if pub_date != upd_date:
        img_cols[1].caption(f"Last Updated: {upd_date}")
    img_cols[1].markdown(f"*{paper['authors']}*")
    influential_citations = int(paper["influential_citation_count"])
    postpend = ""
    if influential_citations > 0:
        postpend = f" ({influential_citations} influential)"
    img_cols[1].markdown(f"`{int(paper['citation_count'])} citations {postpend}`")
    arxiv_comment = paper["arxiv_comment"]
    if arxiv_comment:
        img_cols[1].caption(f"*{arxiv_comment}*")

    report_log_space = img_cols[1].empty()
    action_btn_cols = img_cols[1].columns((1, 1, 1))

    report_btn = action_btn_cols[0].popover("🚨 Report")
    if report_btn.checkbox("Report bad image", key=f"report_v1_{paper_code}_{name}"):
        db.report_issue(paper_code, "bad_image")
        report_log_space.success("Reported bad image. Thanks!")
        time.sleep(3)
        report_log_space.empty()
    if report_btn.checkbox("Report bad summary", key=f"report_v2_{paper_code}_{name}"):
        db.report_issue(paper_code, "bad_summary")
        report_log_space.success("Reported bad summary. Thanks!")
        time.sleep(3)
        report_log_space.empty()
    if report_btn.checkbox(
        "Report non-LLM paper", key=f"report_v3_{paper_code}_{name}"
    ):
        db.report_issue(paper_code, "non_llm")
        report_log_space.success("Reported non-LLM paper. Thanks!")
        time.sleep(3)
        report_log_space.empty()
    if report_btn.checkbox(
        "Report bad data card", key=f"report_v4_{paper_code}_{name}"
    ):
        db.report_issue(paper_code, "bad_datacard")
        report_log_space.success("Reported bad data-card. Thanks!")
        time.sleep(3)
        report_log_space.empty()

    datacard_btn = action_btn_cols[1].button(
        "🃏 Data Card", key=f"dashboard_{paper_code}", type="primary"
    )
    if datacard_btn:
        with st.spinner("*Loading data card...*"):
            html_card = dc.generate_data_card_html(paper_code)
            if html_card:

                @st.experimental_dialog(paper_title, width="large")
                def render():
                    components.html(html_card, height=700, scrolling=True)

                render()
            else:
                error_container = st.empty()
                error_container.warning("Data card not available yet. Check back soon!")
                time.sleep(2)
                error_container.empty()

    with st.expander(f"💭 Abstract (arXiv:{paper_code})", expanded=False):
        st.markdown(paper["summary"])

    with st.expander(f"🗒 **Notes**", expanded=True):
        level_select = st.selectbox(
            "Detail",
            [
                "🔖 Most Interesting Findings",
                "📝 High-Level Overview",
                "🔎 Detailed Research Notes",
            ],
            label_visibility="collapsed",
            index=1,
            key=f"level_select_{paper_code}{name}",
        )

        summary = (
            paper["recursive_summary"]
            if not pd.isna(paper["recursive_summary"])
            else paper["contribution_content"]
        )
        markdown_summary = paper["markdown_notes"]
        bullet_summary = (
            paper["bullet_list_summary"]
            if not pd.isna(paper["bullet_list_summary"])
            else "Not available yet, check back soon!"
        )

        if level_select == "🔖 Most Interesting Findings":
            st.markdown(bullet_summary)
        elif level_select == "📝 High-Level Overview":
            st.markdown(summary)
        elif level_select == "🔎 Detailed Research Notes":
            if not pd.isna(markdown_summary):
                markdown_summary = markdown_summary.replace("#", "###")
                st.markdown(markdown_summary)
            else:
                st.markdown("Currently unavailable. Check again soon!")

    with st.expander("👮 **Interrogate** (Chat)", expanded=expanded):
        paper_question = st.text_area(
            "Ask GPT Maestro about this paper.",
            height=100,
            key=f"chat_{paper_code}{name}",
        )
        if st.button("Send", key=f"send_{paper_code}{name}"):
            response = au.interrogate_paper(paper_question, paper_code)
            db.log_qna_db(f"[{paper_code}] ::: {paper_question}", response)
            st.write(response)

    # if not pd.isna(paper["repo_url"]):
    #     with st.expander("🔗 **Repositories & Libraries**", expanded=False):
    #         repo_link = paper["repo_url"]
    #         repo_title = paper["repo_title"]
    #         repo_description = paper["repo_description"]
    #
    #         st.markdown(f"**{repo_title}**: {repo_link}")
    #         st.markdown(repo_description)

    # with st.expander("🌟 **GPT Assessments**", expanded=False):
    #     assessment_cols = st.columns((1, 3, 1, 3, 1, 3))
    #     assessment_cols[0].metric("Novelty", f"{paper['novelty_score']}/3", "🚀")
    #     assessment_cols[1].caption(f"{paper['novelty_analysis']}")
    #     assessment_cols[2].metric(
    #         "Technical Depth", f"{paper['technical_score']}/3", "🔧"
    #     )
    #     assessment_cols[3].caption(f"{paper['technical_analysis']}")
    #     assessment_cols[4].metric("Readability", f"{paper['enjoyable_score']}/3", "📚")
    #     assessment_cols[5].caption(f"{paper['enjoyable_analysis']}")

    paper_repos = st.session_state["repos"]
    paper_repos = paper_repos.loc[paper_code]

    if len(paper_repos) > 0:
        with st.expander("🔗 **Repos & Other Resources**", expanded=False):
            for i, row in paper_repos.iterrows():
                repo_link = row["repo_url"]
                repo_title = row["repo_title"]
                repo_description = row["repo_description"]
                st.markdown(f"**{repo_title}**: {repo_link}")
                st.markdown(repo_description)

    with st.expander(
        f"✏️ **Takeaways & Applications**:  {paper['takeaway_title']}", expanded=False
    ):
        st.markdown(f"{paper['takeaway_example']}")

    with st.expander(f"📚 **Similar Papers**", expanded=False):
        papers_df = st.session_state["papers"]
        if paper_code in papers_df.index:
            similar_codes = papers_df.loc[paper_code]["similar_docs"]
            if any(pd.isna(similar_codes)):
                st.write("Not available yet. Check back soon!")
            else:
                similar_codes = [d for d in similar_codes if d in papers_df.index]
                if len(similar_codes) > 5:
                    similar_codes = np.random.choice(similar_codes, 5, replace=False)
                similar_df = papers_df.loc[similar_codes]
                generate_grid_gallery(similar_df, extra_key="_sim", n_cols=5)
    st.markdown("---")


def generate_grid_gallery(df, n_cols=5, extra_key=""):
    """Create streamlit grid gallery of paper cards with thumbnail."""
    n_rows = int(np.ceil(len(df) / n_cols))
    for i in range(n_rows):
        cols = st.columns(n_cols)
        for j in range(n_cols):
            if i * n_cols + j < len(df):
                with cols[j]:
                    try:
                        # st.image(f"imgs/{df.iloc[i*n_cols+j]['arxiv_code']}.png")
                        st.image(
                            f"https://llmpedia.s3.amazonaws.com/{df.iloc[i*n_cols+j]['arxiv_code']}.png"
                        )
                    except:
                        pass
                    paper_url = df.iloc[i * n_cols + j]["url"]
                    paper_title = df.iloc[i * n_cols + j]["title"].replace("\n", "")
                    star_count = (
                        df.iloc[i * n_cols + j]["influential_citation_count"] > 0
                    )
                    publish_date = pd.to_datetime(
                        df.iloc[i * n_cols + j]["published"]
                    ).strftime("%b %d, %Y")
                    star = ""
                    if star_count:
                        star = "⭐️"

                    centered_code = f"""
                    <div class="centered">
                        <code>{star} {publish_date}</code>
                    </div>
                    """
                    st.markdown(centered_code, unsafe_allow_html=True)

                    paper_code = df.iloc[i * n_cols + j]["arxiv_code"]
                    focus_btn = st.button(
                        "Focus",
                        key=f"focus_{paper_code}{extra_key}",
                        use_container_width=True,
                    )
                    if focus_btn:
                        st.session_state.arxiv_code = paper_code
                        click_tab(2)

                    st.markdown(
                        f'<p style="text-align: center"><strong><a href="{paper_url}" style="color: #FF4B4B;">{paper_title}</a></strong></p>',
                        unsafe_allow_html=True,
                    )

                    last_updated = pd.to_datetime(
                        df.iloc[i * n_cols + j]["published"]
                    ).strftime("%b %d, %Y")
                    authors_str = df.iloc[i * n_cols + j]["authors"]
                    authors_str = (
                        authors_str[:30] + "..."
                        if len(authors_str) > 30
                        else authors_str
                    )
                    # st.markdown(authors_str)


def create_pagination(items, items_per_page, label="summaries"):
    num_items = len(items)
    num_pages = num_items // items_per_page
    if num_items % items_per_page != 0:
        num_pages += 1

    st.session_state["num_pages"] = num_pages

    st.markdown(f"**{num_items} items found.**")
    st.markdown(f"**Pg. {st.session_state.page_number + 1} of {num_pages}**")
    prev_button, mid, next_button = st.columns((1, 10, 1))
    prev_clicked = prev_button.button("Prev", key=f"prev_{label}")
    next_clicked = next_button.button("Next", key=f"next_{label}")

    if prev_clicked and "page_number" in st.session_state:
        st.session_state.page_number = max(0, st.session_state.page_number - 1)
    if next_clicked and "page_number" in st.session_state:
        st.session_state.page_number = min(
            num_pages - 1, st.session_state.page_number + 1
        )

    start_index = st.session_state.page_number * items_per_page
    end_index = min(start_index + items_per_page, num_items)

    return items[start_index:end_index]


def create_bottom_navigation(label: str):
    num_pages = st.session_state["num_pages"]
    st.write(f"**Pg. {st.session_state.page_number + 1} of {num_pages}**")
    prev_button_btm, _, next_button_btm = st.columns((1, 10, 1))
    prev_clicked_btm = prev_button_btm.button("Prev", key=f"prev_{label}_btm")
    next_clicked_btm = next_button_btm.button("Next", key=f"next_{label}_btm")
    if prev_clicked_btm and "page_number" in st.session_state:
        st.session_state.page_number = max(0, st.session_state.page_number - 1)
        st.rerun()
    if next_clicked_btm and "page_number" in st.session_state:
        st.session_state.page_number = min(
            num_pages - 1, st.session_state.page_number + 1
        )
        st.rerun()


def click_tab(tab_num):
    js = f"""
    <script>
        var tabs = window.parent.document.querySelectorAll("[id^='tabs-bui'][id$='-tab-{tab_num}']");
        if (tabs.length > 0) {{
            tabs[0].click();
        }}
    </script>
    """
    st.components.v1.html(js)
