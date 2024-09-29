import streamlit as st
from datetime import timedelta

from streamlit_plotly_events import plotly_events
from typing import List, Tuple
import pandas as pd

import utils.streamlit_utils as su
import utils.app_utils as au
import utils.plots as pt
import utils.db as db


## Page config.
st.set_page_config(
    layout="wide",
    page_title="📚 LLMpedia",
    page_icon="📚",
    initial_sidebar_state="expanded",
)

# Initialization of state variables
if "papers" not in st.session_state:
    st.session_state.papers = None

if "page_number" not in st.session_state:
    st.session_state.page_number = 0

if "num_pages" not in st.session_state:
    st.session_state.num_pages = 0

if "arxiv_code" not in st.session_state:
    st.session_state.arxiv_code = ""

if "openai_api_key" not in st.session_state:
    st.session_state.openai_api_key = ""

if "all_years" not in st.session_state:
    st.session_state.all_years = False


collection_map = {
    "GTE-Large": "arxiv_vectors",
    "🆕 Cohere V3": "arxiv_vectors_cv3",
}

st.markdown(
    """
    <style>
        @import 'https://fonts.googleapis.com/css2?family=Orbitron&display=swap';
        .pixel-font {
            font-family: 'Orbitron', sans-serif;
            font-size: 32px;
            margin-bottom: 1rem;
        }
    </style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
    <style>
        .centered {
            display: flex;
            justify-content: center;
            margin-bottom: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def combine_input_data():
    arxiv_df = db.load_arxiv()
    summaries_df = db.load_summaries()
    topics_df = db.load_topics()
    citations_df = db.load_citations()
    recursive_summaries_df = db.load_recursive_summaries()
    bullet_list_df = db.load_bullet_list_summaries()
    markdown_summaries = db.load_summary_markdown()
    tweets = db.load_tweet_insights()
    similar_docs_df = db.load_similar_documents()

    papers_df = summaries_df.join(arxiv_df, how="left")
    papers_df = papers_df.join(topics_df, how="left")
    papers_df = papers_df.join(citations_df, how="left")
    papers_df = papers_df.join(recursive_summaries_df, how="left")
    papers_df = papers_df.join(bullet_list_df, how="left")
    papers_df = papers_df.join(markdown_summaries, how="left")
    papers_df = papers_df.join(tweets, how="left")
    papers_df = papers_df.join(similar_docs_df, how="left")

    papers_df["arxiv_code"] = papers_df.index
    papers_df["url"] = papers_df["arxiv_code"].map(
        lambda l: f"https://arxiv.org/abs/{l}"
    )
    papers_df.sort_values("published", ascending=False, inplace=True)
    return papers_df


@st.cache_data(ttl=timedelta(hours=6))
def load_data():
    """Load data from compiled dataframe."""
    result_df = combine_input_data()

    ## Remapping with emotion.
    classification_map = {
        "TRAINING": "🏋️‍ TRAINING",
        "FINE-TUNING": "🔧 FINE-TUNING",
        "ARCHITECTURES": "⚗️MODELS",
        "BEHAVIOR": "🧠 BEHAVIOR",
        "PROMPTING": "✍️ PROMPTING",
        "USE CASES": "💰 USE CASES",
        "OTHER": "🤷 OTHER",
    }
    ## Round published and updated columns.
    result_df["updated"] = pd.to_datetime(result_df["updated"]).dt.date
    result_df["published"] = pd.to_datetime(result_df["published"].dt.date)
    result_df["category"] = result_df["category"].apply(
        lambda x: classification_map.get(x, "🤷 OTHER")
    )
    result_df[["citation_count", "influential_citation_count"]] = result_df[
        ["citation_count", "influential_citation_count"]
    ].fillna(0)

    return result_df


@st.cache_data
def load_repositories(year: int, filter_by_year=True):
    repos_df = db.load_repositories()
    topics_df = db.load_topics()
    meta_df = db.load_arxiv()
    topics_df.drop(columns=["dim1", "dim2"], inplace=True)
    repos_df = repos_df.join(topics_df, how="left")
    repos_df = repos_df.join(meta_df[["published"]], how="left")
    repos_df["domain"] = repos_df["repo_url"].apply(
        lambda x: x.split("/")[2].split(".")[-2]
    )
    if filter_by_year:
        repos_df = repos_df[repos_df["published"].dt.year == year]
    return repos_df


@st.cache_data
def get_weekly_summary(date_str: str):
    au.get_weekly_summary(date_str)


@st.cache_data(ttl=timedelta(hours=6))
def initialize_weekly_summary(date_report: str):
    if (
        "weekly_summary" not in st.session_state
        or st.session_state["weekly_summary_date"] != date_report
    ):
        weekly_content, weekly_highlight, weekly_repos = au.get_weekly_summary(
            date_report
        )
        st.session_state["weekly_summary"] = (
            weekly_content,
            weekly_highlight,
            weekly_repos,
        )
        st.session_state["weekly_summary_date"] = date_report
    else:
        weekly_content, weekly_highlight, weekly_repos = st.session_state[
            "weekly_summary"
        ]
    return weekly_content, weekly_highlight, weekly_repos


@st.cache_data
def get_max_report_date():
    max_date = db.get_max_table_date(
        db.db_params,
        "weekly_content",
    )
    if max_date.weekday() != 6:
        max_date = max_date + pd.Timedelta(days=6 - max_date.weekday())
    return max_date


@st.cache_data
def generate_calendar_df(df: pd.DataFrame):
    """Daily counts of papers."""
    published_df = df.groupby("published").count()["title"]
    published_df = published_df.reindex(
        pd.date_range(
            start=published_df.index.min(), end=published_df.index.max(), freq="D"
        )
    ).fillna(0)
    published_df = published_df.reset_index()
    published_df.columns = ["published", "Count"]
    return published_df


@st.cache_data
def get_similar_docs(
    arxiv_code: str, df: pd.DataFrame, n: int = 5
) -> Tuple[List[str], List[str], List[str]]:
    return au.get_similar_docs(arxiv_code, df, n)


def main():
    ## URL info extraction.
    url_query = st.query_params
    if "arxiv_code" in url_query and len(st.session_state.arxiv_code) == 0:
        db.log_visit(url_query["arxiv_code"])
        paper_code = url_query["arxiv_code"]
        st.session_state.arxiv_code = paper_code
        su.click_tab(2)

    st.markdown(
        """<div class="pixel-font">LLMpedia</div>
    """,
        unsafe_allow_html=True,
    )
    st.markdown("##### The Illustrated Large Language Model Encyclopedia")
    ## Humorous and poetic introduction.
    st.markdown(
        "LLMpedia is a curated collection of key papers on Large Language Models, selected and analyzed by GPT Maestro. "
        " With pixel art and structured summaries, the encyclopedia is designed to guide you through the extensive research on LLMs. "
        "If you have any questions or want to do LLM research, go to the *Chat* section and talk to the Maestro. "
        "And don't forget to follow us [@GPTMaestro](https://twitter.com/GPTMaestro) for the latest updates and paper reviews.\n\n"
        "*Buona lettura!*"
    )

    ## Main content.
    full_papers_df = load_data()
    papers_df, year = su.create_sidebar(full_papers_df)

    filter_by_year = not st.session_state.all_years
    repositories_df = load_repositories(year, filter_by_year=filter_by_year)

    st.session_state["papers"] = full_papers_df
    st.session_state["repos"] = repositories_df

    if len(papers_df) == 0:
        st.error("No papers found.")
        return

    published_df = generate_calendar_df(papers_df)
    if not st.session_state.all_years:
        heatmap_data = au.prepare_calendar_data(published_df, year)
        release_calendar, padded_date = pt.plot_activity_map(heatmap_data)
        st.markdown(f"### 📅 {year} Release Calendar")
        calendar_select = plotly_events(release_calendar, override_height=220)

        # Published date.
        if len(calendar_select) > 0:
            ## Select from padded dates.
            x_coord = 6 - calendar_select[0]["pointNumber"][0]
            y_coord = calendar_select[0]["pointNumber"][1]
            publish_date = pd.to_datetime(
                padded_date.loc[x_coord, y_coord] + f" {year}"
            )

            if len(papers_df[papers_df["published"] == publish_date]) > 0:
                papers_df = papers_df[papers_df["published"] == publish_date]
                ## Add option to clear filter on sidebar.
                if st.sidebar.button(
                    f"📅 **Publish Date Filter:** {publish_date.strftime('%B %d, %Y')}",
                    help="Double click on calendar chart to remove date filter.",
                ):
                    pass

    st.sidebar.markdown(
        """
    <style>
        .reportview-container .main footer {visibility: hidden;}
        .footer {
            position: fixed;
            bottom: 0;
            width: 0%;
            text-align: center;
            color: #888;
            font-size: 0.75rem;
        }
        .footer a {
            color: inherit;
            text-decoration: none;
        }
    </style>
    <div class="footer">
        <a href="https://github.com/masta-g3/llmpedia/blob/main/VERSIONS.md" target="_blank">v1.3.2</a>
    </div>

    <div style="font-size: 0.7em; font-style: italic; text-align: center; position: relative; top: 20px; left: 0; right: 0; color: #888;">
        🖤 Thanks to Anthropic for supporting this project.
    </div>
    """,
        unsafe_allow_html=True,
    )

    if len(papers_df) == 0:
        st.markdown("No papers found.")
        return

    ## Content tabs.
    content_tabs = st.tabs(
        [
            "🧮 Release Feed",
            "🗺️ Statistics",
            "🔍 Paper Details",
            "🤖 Chat Assistant",
            "⚙️  Links & Repositories",
            "🗞 Weekly Report",
        ]
    )

    with content_tabs[0]:
        ## Gried view.
        if "page_number" not in st.session_state:
            st.session_state.page_number = 0

        papers_df_subset = su.create_pagination(
            papers_df, items_per_page=25, label="grid"
        )
        su.generate_grid_gallery(papers_df_subset)
        su.create_bottom_navigation(label="grid")

    with content_tabs[1]:
        total_papers = len(papers_df)
        st.markdown(f"### 📈 Total Publication Counts: {total_papers}")
        plot_type = st.radio(
            label="Plot Type",
            options=["Daily", "Cumulative"],
            index=1,
            label_visibility="collapsed",
            horizontal=True,
        )
        cumulative = plot_type == "Cumulative"
        ts_plot = pt.plot_publication_counts(papers_df, cumulative=cumulative)
        st.plotly_chart(ts_plot, use_container_width=True)

        ## Cluster map.
        st.markdown(f"### {year} Topic Model Map")
        cluster_map = pt.plot_cluster_map(papers_df)
        st.plotly_chart(cluster_map, use_container_width=True)

    with content_tabs[2]:
        ## Focus on a paper.
        arxiv_code = st.text_input("arXiv Code", st.session_state.arxiv_code)
        st.session_state.arxiv_code = arxiv_code
        if len(arxiv_code) > 0:
            if arxiv_code in full_papers_df.index:
                paper = full_papers_df.loc[arxiv_code].to_dict()
                su.create_paper_card(paper, mode="open", name="_focus")
            else:
                st.error("Paper not found.")

    with content_tabs[3]:
        st.markdown("##### 🤖 Chat with the GPT maestro.")
        user_question = st.text_area(
            label="Ask any question about LLMs or the arxiv papers.", value=""
        )
        chat_btn_disabled = len(user_question) == 0
        chat_cols = st.columns((1, 2, 1))
        chat_btn = chat_cols[0].button("Send", disabled=chat_btn_disabled)
        # response_length = chat_cols[2].select_slider(
        #     "Response Length",
        #     options=["Short Answer", "Normal"],
        #     value="Short Answer",
        #     label_visibility="collapsed",
        # )
        response_length = "Short Answer"
        if chat_btn:
            if user_question != "":
                with st.spinner(
                    "Consulting the GPT maestro, this might take a minute..."
                ):
                    response, referenced_codes, relevant_codes = au.query_llmpedia_new(
                        user_question=user_question,
                        response_length=response_length,
                        query_llm_model="gpt-4o",
                        rerank_llm_model="gpt-4o-mini",
                        response_llm_model="gpt-4o",
                    )
                    db.log_qna_db(user_question, response)
                    st.divider()
                    st.markdown(response)
                    if len(referenced_codes) > 0:
                        st.divider()
                        st.markdown(
                            "<h4>Referenced Papers:</h4>", unsafe_allow_html=True
                        )
                        reference_df = st.session_state["papers"].loc[referenced_codes]
                        su.generate_grid_gallery(
                            reference_df, n_cols=5, extra_key="_chat"
                        )
                    if len(relevant_codes) > 0:
                        st.divider()
                        st.markdown(
                            "<h4>Other Relevant Papers:</h4>", unsafe_allow_html=True
                        )
                        relevant_df = st.session_state["papers"].loc[relevant_codes]
                        su.generate_grid_gallery(
                            relevant_df, n_cols=5, extra_key="_chat"
                        )

    with content_tabs[4]:
        ## Repositories.
        repos_df = st.session_state["repos"]
        repos_search_cols = st.columns((1, 1, 1))
        topic_filter = repos_search_cols[0].multiselect(
            "Filter by Topic",
            options=repos_df["topic"].unique().tolist(),
            default=[],
        )

        domain_filter = repos_search_cols[1].multiselect(
            "Filter by Domain",
            options=repos_df["domain"].value_counts().index.tolist(),
            default=[],
        )

        search_term = repos_search_cols[2].text_input("Search by Title", value="")

        def filter_repos(
            df: pd.DataFrame,
            search_term: str,
            topic_filter: List[str],
            domain_filter: List[str],
        ):
            if len(search_term) > 0:
                df = df[df["title"].str.contains(search_term, case=False)]
            if len(topic_filter) > 0:
                df = df[df["topic"].isin(topic_filter)]
            if len(domain_filter) > 0:
                df = df[df["domain"].isin(domain_filter)]
            return df

        filtered_repos = filter_repos(
            repos_df, search_term, topic_filter, domain_filter
        )
        repo_count = len(filtered_repos)
        repos_title = f"### 📦 Total resources found: {repo_count}"
        st.markdown(repos_title)
        st.data_editor(
            filtered_repos.drop(columns=["published"]).sort_index(ascending=False),
            column_config={
                "topic": st.column_config.ListColumn(
                    "Topic",
                    width="medium",
                ),
                "domain": st.column_config.ListColumn(
                    "Domain",
                    width="medium",
                ),
                "repo_url": st.column_config.LinkColumn(
                    "Repository URL",
                ),
                "repo_title": st.column_config.TextColumn(
                    "Repository Title",
                    width="medium",
                ),
                "repo_description": st.column_config.TextColumn(
                    "Repository Description",
                ),
            },
            disabled=True,
        )

        plot_by = st.selectbox(
            "Plot total resources by",
            options=["topic", "domain", "published"],
            index=0,
        )

        if len(filtered_repos) > 0:
            plot_repos = pt.plot_repos_by_feature(filtered_repos, plot_by)
            st.plotly_chart(plot_repos, use_container_width=True)

    with content_tabs[5]:
        weekly_plot_container = st.empty()

        report_top_cols = st.columns((5, 2))
        with report_top_cols[0]:
            st.markdown("# 📰 LLM Weekly Review")
        with report_top_cols[1]:
            ## ToDo: Make dynamic?
            if year == 2024:
                max_date = get_max_report_date()
            else:
                max_date = pd.to_datetime(f"{year}-12-31").date()
            week_select = st.date_input(
                "Select Week",
                value=pd.to_datetime(max_date),
                min_value=pd.to_datetime(f"{year}-01-01"),
                max_value=pd.to_datetime(max_date),
            )
            ## Convert selection to previous monday.
            date_report = week_select - pd.Timedelta(days=week_select.weekday())

            ## Plot.
            weekly_ts_plot = pt.plot_weekly_activity_ts(published_df, date_report)
            weekly_plot_container.plotly_chart(weekly_ts_plot, use_container_width=True)

        if year < 2023:
            st.warning("Weekly reports are available from 2023 onwards.")
        else:
            weekly_content, weekly_highlight, weekly_repos = initialize_weekly_summary(
                date_report
            )

            weekly_report = (
                f"##### ({date_report.strftime('%B %d, %Y')} to "
                f"{(date_report + pd.Timedelta(days=6)).strftime('%B %d, %Y')})\n\n"
                f"## 🔬 New Developments & Findings\n\n{weekly_content}\n\n"
                f"## 🌟 Highlight of the Week\n\n"
            )

            st.write(weekly_report)
            report_highlights_cols = st.columns((1, 2.5))
            highlight_img = au.get_img_link_for_blob(weekly_highlight)
            report_highlights_cols[0].image(highlight_img, use_column_width=True)
            report_highlights_cols[1].markdown(weekly_highlight)
            st.markdown(weekly_repos)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        db.log_error_db(e)
        st.error("Something went wrong. Please refresh the app and try again, we will look into it.")
