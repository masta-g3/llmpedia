import streamlit as st
from datetime import timedelta
from pathlib import Path

from typing import List, Tuple, Dict
import pandas as pd

import utils.streamlit_utils as su
import utils.app_utils as au
import utils.plots as pt
import utils.db.db_utils as db_utils
import utils.db.db as db
import utils.db.logging_db as logging_db
import utils.styling as styling
import time
import streamlit.components.v1 as components

if hasattr(st, "_is_running_with_streamlit"):
    import streamlit.watcher.path_watcher

    streamlit.watcher.path_watcher.IGNORE_MODULES.add("torch")

## Page config.
st.set_page_config(
    layout="wide",
    page_title="LLMpedia - The Illustrated Large Language Model Encyclopedia",
    page_icon="🤖",
    initial_sidebar_state="collapsed",
    menu_items={
        'About': "LLMpedia - The Illustrated Large Language Model Encyclopedia"
    }
)

# Apply styling
styling.apply_arxiv_theme()
styling.apply_custom_fonts()
styling.apply_centered_style()

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

# Chat-related state variables
if "chat_response" not in st.session_state:
    st.session_state.chat_response = None

if "referenced_codes" not in st.session_state:
    st.session_state.referenced_codes = []

if "relevant_codes" not in st.session_state:
    st.session_state.relevant_codes = []

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
    arxiv_df = db.load_arxiv(drop_tstp=False)
    summaries_df = db.load_summaries()
    topics_df = db.load_topics()
    citations_df = db.load_citations()
    recursive_summaries_df = db.load_recursive_summaries()
    bullet_list_df = db.load_bullet_list_summaries()
    markdown_summaries = db.load_summary_markdown()
    tweets = db.load_tweet_insights()
    similar_docs_df = db.load_similar_documents()
    punchlines_df = db.load_punchlines()

    papers_df = summaries_df.join(arxiv_df, how="left")
    papers_df = papers_df.join(topics_df, how="left")
    papers_df = papers_df.join(citations_df, how="left")
    papers_df = papers_df.join(recursive_summaries_df, how="left")
    papers_df = papers_df.join(bullet_list_df, how="left")
    papers_df = papers_df.join(markdown_summaries, how="left")
    papers_df = papers_df.join(tweets, how="left")
    papers_df = papers_df.join(similar_docs_df, how="left")
    papers_df = papers_df.join(punchlines_df, how="left")

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
    repos_df["repo_url"].fillna("", inplace=True)
    repos_df["domain"] = repos_df["repo_url"].apply(
        lambda x: x.split("/")[2].split(".")[-2] if len(x.split("/")) > 2 else ""
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


@st.cache_data(ttl=timedelta(hours=6))
def get_random_interesting_facts(n=10, recency_days=7) -> List[Dict]:
    """Get random interesting facts from the database with caching."""
    return db.get_random_interesting_facts(n=n, recency_days=recency_days)


@st.cache_data
def get_max_report_date():
    max_date = db_utils.get_max_table_date("weekly_content")
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
    st.markdown(
        """<div class="pixel-font">LLMpedia</div>
    """,
        unsafe_allow_html=True,
    )
    st.markdown(
        "##### The Illustrated Large Language Model Encyclopedia",
        help="Welcome to LLMpedia, your curated guide to Large Language Model research, brought to you by GPT Maestro. "
        "Our pixel art illustrations and structured summaries make complex research accessible. "
        "Have questions or interested in LLM research? Chat with the Maestro or follow us [@GPTMaestro](https://twitter.com/GPTMaestro) for the latest updates.\n\n"
        "*Buona lettura!*",
    )
    ## Main content.
    full_papers_df = load_data()
    papers_df, year = su.create_sidebar(full_papers_df)

    filter_by_year = not st.session_state.all_years
    repositories_df = load_repositories(year, filter_by_year=filter_by_year)

    st.session_state["papers"] = full_papers_df
    st.session_state["repos"] = repositories_df

    if len(papers_df) == 0:
        st.error("No papers found. Try a different year.")
        return

    published_df = generate_calendar_df(papers_df)
    if not st.session_state.all_years:
        heatmap_data = au.prepare_calendar_data(published_df, year)
        release_calendar, padded_date = pt.plot_activity_map(heatmap_data)
        st.markdown(f"### 📅 {year} Release Calendar")

        # Use native Streamlit plotly_chart with on_select
        calendar_event = st.plotly_chart(
            release_calendar,
            key="calendar_heatmap",
            on_select="rerun",
            use_container_width=True,
            height=220,
        )

        if calendar_event.selection and calendar_event.selection.get("points"):
            # Get the first selected point's data
            point = calendar_event.selection["points"][0]
            coords_text = point.get("text")

            if coords_text:
                y_coord, x_coord = map(int, coords_text.split(","))
                publish_date = pd.to_datetime(
                    padded_date.loc[y_coord, x_coord] + f" {year}"
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
        <a href="https://github.com/masta-g3/llmpedia/blob/main/VERSIONS.md" target="_blank">v1.6.0</a>
    </div>

    <div style="font-size: 0.7em; font-style: italic; text-align: center; position: relative; top: 20px; left: 0; right: 0; color: #888;">
        🖤 Thanks to Anthropic for supporting this project.
    </div>
    """,
        unsafe_allow_html=True,
    )

    if len(papers_df) == 0:
        st.markdown("No papers found. Try changing some filters.")
        return

    ## Content tabs.
    content_tabs = st.tabs(
        [
            "📰 News",
            "🧮 Release Feed",
            "🗺️ Statistics & Topics",
            "🔍 Paper Details",
            "🤖 Online Research",
            "⚙️  Links & Repositories",
            "🗞 Weekly Report",
        ]
    )

    with content_tabs[0]:
        # Calculate recent papers (1 day and 7 days)
        today = pd.Timestamp.now()
        yesterday = today - pd.Timedelta(days=1)
        last_week = today.date() - pd.Timedelta(days=7)

        # Filter dataframes for recent papers - convert datetime64[ns] to date for comparison
        papers_1d = papers_df[papers_df["tstp"] >= yesterday]
        papers_7d = papers_df[papers_df["published"].dt.date >= last_week]

        # Display all metrics in a single row of 4 columns
        metric_cols = st.columns(4)
        with metric_cols[0]:
            st.metric(
                label="Total Papers in Archive",
                value=f"{len(full_papers_df):,d}",
            )
        with metric_cols[1]:
            if not st.session_state.all_years:
                value = f"{len(papers_df[papers_df['published'].dt.year == year]):,d}"
                st.metric(
                    label=f"Published in {year}",
                    value=value,
                )
        with metric_cols[2]:
            st.metric(label="Last 7 days", value=len(papers_7d))
        with metric_cols[3]:
            st.metric(label="Added in last 24 hours", value=len(papers_1d))

        # Display interesting facts section
        with st.expander("**💡 Recent Findings**", expanded=True):
            # Use the cached function directly
            facts = get_random_interesting_facts(n=6, recency_days=7)
            su.display_interesting_facts(facts, n_cols=2, papers_df=full_papers_df)

        # Create a 4-panel layout (2 rows, 2 columns)
        st.write("### Recent Activity (~7 days)")
        row1_cols = st.columns([3, 2])
        row2_cols = st.columns([4, 2])

        # Panel 1.1: Trending Topics (Left)
        with row1_cols[0]:
            ## Get trending topics from the most recent papers.
            if len(papers_7d) > 0:
                # Use utility function to extract trending topics
                trending_terms = au.get_trending_topics_from_papers(
                    papers_df=papers_7d, time_window_days=7, n=15
                )
                if trending_terms:
                    trend_chart = pt.plot_trending_words(trending_terms)
                    st.plotly_chart(trend_chart, use_container_width=True)
                else:
                    st.info("Not enough data to identify trending topics.")

        # Panel 1.2: Topic Distribution (Right)
        with row1_cols[1]:
            topic_chart = pt.plot_top_topics(papers_7d, n=5)
            st.plotly_chart(topic_chart, use_container_width=True)

        # Panel 2.1: Top Cited Papers (Left)
        with row2_cols[0]:
            citation_window = 90
            citation_top_n = 5
            st.markdown(f"### Top Cited Papers (Last {citation_window} days)")
            top_papers = au.get_top_cited_papers(
                papers_df, n=citation_top_n, time_window_days=citation_window
            )
            su.generate_mini_paper_table(top_papers, n=citation_top_n, extra_key="_dashboard")

        # Panel 2.2: Featured Paper (Right)
        with row2_cols[1]:
            arxiv_code = au.get_latest_weekly_highlight()
            highlight_paper = (
                papers_df[papers_df["arxiv_code"] == arxiv_code].iloc[0].to_dict()
            )
            su.create_featured_paper_card(highlight_paper)

    with content_tabs[1]:
        ## Grid view or Table view
        if "page_number" not in st.session_state:
            st.session_state.page_number = 0

        # Add view selector with radio buttons
        if "view_mode" not in st.session_state:
            st.session_state.view_mode = "Grid View"

        view_mode = st.radio(
            "Display Mode",
            options=["Grid View", "Table View"],
            horizontal=True,
            key="view_selector",
        )

        # Get paginated data
        papers_df_subset = su.create_pagination(
            papers_df, items_per_page=25, label="grid", year=year
        )

        # Display content based on selected view
        if view_mode == "Grid View":
            su.generate_grid_gallery(papers_df_subset)
        else:
            su.generate_paper_table(papers_df_subset)

        su.create_bottom_navigation(label="grid")

    with content_tabs[2]:
        total_papers = len(papers_df)
        if not st.session_state.all_years:
            st.markdown(f"### 📈 {year} Publication Counts: {total_papers}")
        else:
            st.markdown(f"### 📈 Total Publication Counts: {total_papers}")
        plot_type = st.radio(
            label="Plot Type",
            options=["Daily", "Cumulative"],
            index=0,
            label_visibility="collapsed",
            horizontal=True,
        )
        cumulative = plot_type == "Cumulative"
        ts_plot = pt.plot_publication_counts(papers_df, cumulative=cumulative)
        st.plotly_chart(ts_plot, use_container_width=True)

        ## Cluster map.
        if not st.session_state.all_years:
            st.markdown(f"### Topic Model Map ({year})")
        else:
            st.markdown(f"### Topic Model Map")

        cluster_map = pt.plot_cluster_map(papers_df)

        # Use native Streamlit plotly_chart with on_select
        cluster_event = st.plotly_chart(
            cluster_map,
            key="cluster_map",
            on_select="rerun",
            use_container_width=True,
            height=800,
        )

        # Handle cluster map selection
        if cluster_event.selection and cluster_event.selection.get("points"):
            # Only proceed if exactly one point is selected
            if len(cluster_event.selection["points"]) == 1:
                point = cluster_event.selection["points"][0]
                custom_data = point.get("customdata", [])

                # Custom data is an array with [title, arxiv_code]
                if len(custom_data) > 1:
                    arxiv_code = custom_data[1]  # The second element is the arxiv_code
                    if arxiv_code:
                        # Redirect to the paper details
                        st.query_params["arxiv_code"] = arxiv_code
                        st.session_state.arxiv_code = arxiv_code
                        su.click_tab(3)

    ## URL info extraction.
    url_query = st.query_params
    if "arxiv_code" in url_query and len(st.session_state.arxiv_code) == 0:
        paper_code = url_query["arxiv_code"]
        logging_db.log_visit(paper_code)
        st.session_state.arxiv_code = paper_code
        su.click_tab(3)

    with content_tabs[3]:
        ## Focus on a paper.
        if len(st.session_state.arxiv_code) == 0:
            st.markdown("<div style='font-size: 0.9em; opacity: 0.8; margin-bottom: 1.5em;'>💡 <em>Search a paper by its arXiv code, or use the sidebar to search and filter papers by title, author, or other attributes.</em></div>", unsafe_allow_html=True)
        arxiv_code = st.text_input("arXiv Code", st.session_state.arxiv_code)
        st.session_state.arxiv_code = arxiv_code
        if len(arxiv_code) > 0:
            if arxiv_code in full_papers_df.index:
                paper = full_papers_df.loc[arxiv_code].to_dict()
                su.create_paper_card(paper, mode="open", name="_focus")
            else:
                st.error("Paper not found.")

    with content_tabs[4]:
        st.markdown("##### 🤖 Chat with the GPT maestro.")
        user_question = st.text_area(
            label="Ask any question about LLMs or the arxiv papers.", value=""
        )
        chat_btn_disabled = len(user_question) == 0

        ## Advanced response settings in an expander
        with st.expander("⚙️ Response Settings", expanded=False):
            settings_cols = st.columns(2)

            with settings_cols[0]:
                response_length = st.select_slider(
                    "Response Length (words)",
                    options=[250, 500, 1000, 3000],
                    value=500,
                    format_func=lambda x: f"{x} words",
                )
            with settings_cols[1]:
                max_sources = st.select_slider(
                    "Maximum Sources",
                    options=[1, 3, 5, 7, 10, 15, 20, 25, 30],
                    value=10,
                )

            custom_instructions = st.text_area(
                "Custom Instructions (Optional)",
                placeholder="Add any specific instructions for how you would like the response to be structured or formatted...",
                help="Provide custom style guidelines, instructions on what to focus on, etc.",
            )

            show_only_sources = st.checkbox(
                "Show me only the sources",
                help="Skip generating a response and just show the most relevant papers for this query.",
            )

        chat_cols = st.columns((1, 1, 1))
        chat_btn = chat_cols[0].button("Send", disabled=chat_btn_disabled)

        # Show clear button only when we have a response
        if st.session_state.chat_response:
            if chat_cols[1].button("Clear", type="secondary"):
                st.session_state.chat_response = None
                st.session_state.referenced_codes = []
                st.session_state.relevant_codes = []
                st.rerun()

        if chat_btn:
            if user_question != "":
                progress_placeholder = st.empty()

                def update_progress(message: str):
                    with progress_placeholder:
                        st.info(message)

                response, referenced_codes, relevant_codes = au.query_llmpedia_new(
                    user_question=user_question,
                    response_length=response_length,
                    query_llm_model="claude-3-5-sonnet-20241022",
                    rerank_llm_model="gemini/gemini-2.0-flash",
                    response_llm_model="claude-3-5-sonnet-20241022",
                    max_sources=max_sources,
                    debug=True,
                    progress_callback=update_progress,
                    custom_instructions=(
                        custom_instructions if custom_instructions.strip() else None
                    ),
                    show_only_sources=show_only_sources,
                )
                progress_placeholder.empty()

                # Store results in session state
                st.session_state.chat_response = response
                st.session_state.referenced_codes = referenced_codes
                st.session_state.relevant_codes = relevant_codes

                logging_db.log_qna_db(user_question, response)

        # Display results if they exist in session state
        if st.session_state.chat_response:
            st.divider()
            st.markdown(st.session_state.chat_response)

            if len(st.session_state.referenced_codes) > 0:
                st.divider()

                # View selector for paper display format
                display_format = st.radio(
                    "Display Format",
                    options=["Grid View", "Citation List"],
                    horizontal=True,
                    label_visibility="collapsed",
                    key="papers_display_format",
                )

                st.markdown("<h4>Referenced Papers:</h4>", unsafe_allow_html=True)
                reference_df = st.session_state["papers"].loc[
                    st.session_state.referenced_codes
                ]
                if display_format == "Grid View":
                    su.generate_grid_gallery(reference_df, n_cols=5, extra_key="_chat")
                else:
                    su.generate_citations_list(reference_df)

                if len(st.session_state.relevant_codes) > 0:
                    st.divider()
                    st.markdown(
                        "<h4>Other Relevant Papers:</h4>", unsafe_allow_html=True
                    )
                    relevant_df = st.session_state["papers"].loc[
                        st.session_state.relevant_codes
                    ]
                    if display_format == "Grid View":
                        su.generate_grid_gallery(
                            relevant_df, n_cols=5, extra_key="_chat"
                        )
                    else:
                        su.generate_citations_list(relevant_df)

    with content_tabs[5]:
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
            df.dropna(subset=["repo_url"], inplace=True)
            df = df[df["repo_url"].map(lambda x: len(x) > 0)]
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

    with content_tabs[6]:

        report_top_cols = st.columns((5, 2))
        with report_top_cols[0]:
            st.markdown("# 📰 LLM Weekly Review")

        with report_top_cols[1]:
            ## ToDo: Make dynamic?
            if year == 2025:
                max_date = max(
                    get_max_report_date(), pd.to_datetime(f"{year}-01-01").date()
                )
            else:
                max_date = get_max_report_date()
            week_select = st.date_input(
                "Select Week",
                value=pd.to_datetime(max_date),
                min_value=pd.to_datetime(f"{year}-01-01"),
                max_value=pd.to_datetime(max_date),
            )
            ## Convert selection to previous monday.
            date_report = week_select - pd.Timedelta(days=week_select.weekday())

        if year < 2023:
            st.error("Weekly reports are available from 2023 onwards.")
            return

        else:
            weekly_content, weekly_highlight, weekly_repos = initialize_weekly_summary(
                date_report
            )
            if weekly_content is None:
                st.error("No weekly report found for this week.")
                return

            else:
                weekly_report = (
                    f"## 🔬 New Developments & Findings\n\n{weekly_content}\n\n"
                    f"## 🌟 Highlight of the Week\n\n"
                )

            ## Plot.
            st.write(
                f"##### ({date_report.strftime('%B %d, %Y')} to "
                f"{(date_report + pd.Timedelta(days=6)).strftime('%B %d, %Y')})\n\n"
            )
            weekly_ts_plot = pt.plot_weekly_activity_ts(published_df, date_report)
            st.plotly_chart(weekly_ts_plot, use_container_width=True)

            ## Report.
            st.write(weekly_report)
            report_highlights_cols = st.columns((1, 2.5))
            highlight_img = au.get_img_link_for_blob(weekly_highlight)
            report_highlights_cols[0].image(highlight_img, use_container_width=True)
            report_highlights_cols[1].markdown(weekly_highlight)
            st.markdown(weekly_repos)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging_db.log_error_db(e)
        st.error("Something went wrong. Please refresh the app and try again, we will look into it.")