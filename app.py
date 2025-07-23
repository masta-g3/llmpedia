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
        "About": "LLMpedia - The Illustrated Large Language Model Encyclopedia"
    },
)

# Apply all styling through single master function
styling.apply_complete_app_styles()

# Initialization of state variables
if "papers" not in st.session_state:
    st.session_state.papers = None

if "page_number" not in st.session_state:
    st.session_state.page_number = 0

if "num_pages" not in st.session_state:
    st.session_state.num_pages = 0

if "arxiv_code" not in st.session_state:
    st.session_state.arxiv_code = ""

if "all_years" not in st.session_state:
    st.session_state.all_years = False

if "facts_refresh_trigger" not in st.session_state:
    st.session_state.facts_refresh_trigger = 0

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


@st.cache_data(ttl=timedelta(hours=3))
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


@st.cache_data(ttl=timedelta(hours=1))
def get_random_interesting_facts(n=10, recency_days=7, _trigger: int = 0) -> List[Dict]:
    """Get random interesting facts from the database with caching."""
    return db.get_random_interesting_facts(n=n, recency_days=recency_days)


@st.cache_data(ttl=timedelta(hours=6))
def get_featured_paper(papers_df: pd.DataFrame) -> Dict:
    """Get featured paper with caching."""
    arxiv_code = au.get_latest_weekly_highlight()
    return papers_df[papers_df["arxiv_code"] == arxiv_code].iloc[0].to_dict()


@st.cache_data(ttl=timedelta(minutes=15))
def get_active_users_count() -> int:
    """Get active users count with caching."""
    return logging_db.get_active_users_last_24h()


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


@st.cache_data
def get_cached_top_cited_papers_app(
    papers_df_fragment: pd.DataFrame, n: int, time_window_days: int
) -> pd.DataFrame:
    """Cached wrapper in app.py to get top cited papers."""
    return au.get_top_cited_papers(
        papers_df_fragment, n=n, time_window_days=time_window_days
    )


@st.cache_data(ttl=timedelta(minutes=30))
def get_cached_raw_trending_data_app(
    n_fetch: int, time_window_days_db: int
) -> pd.DataFrame:
    """Cached wrapper in app.py to fetch raw trending paper data from DB with TTL."""
    return db.get_trending_papers(n=n_fetch, time_window_days=time_window_days_db)


def get_processed_trending_papers(
    papers_df_fragment: pd.DataFrame,
    raw_trending_data: pd.DataFrame,
    top_n_display: int,
) -> pd.DataFrame:
    """
    Cached wrapper in app.py to process raw trending data with current paper selection.
    Relies on a new utility function in app_utils.py (au.process_trending_data).
    """
    if papers_df_fragment.empty or raw_trending_data.empty:
        return pd.DataFrame()
    # This will call a new function in app_utils we'll define next.
    return au.process_trending_data(
        papers_df_fragment, raw_trending_data, top_n_display
    )


@st.fragment
def chat_fragment():
    """Handles the entire chat interface logic within a Streamlit fragment."""
    # Get initial query value and render header
    user_question = su.get_initial_query_value()
    su.render_research_header()
    
    # User input
    user_question = st.text_area(
        label="Ask any question about LLMs or the arxiv papers.",
        value=user_question,
        key="chat_user_question_area",
    )
    chat_btn_disabled = len(user_question) == 0

    # Render settings panel and get configuration
    settings = su.render_research_settings_panel()
    
    status_placeholder = st.empty()
    
    # Action buttons
    chat_cols = st.columns((1, 1, 1))
    chat_btn = chat_cols[0].button("Send", disabled=chat_btn_disabled)

    # Show clear button only when we have a response
    if st.session_state.chat_response:
        if chat_cols[1].button("Clear", type="secondary"):
            st.session_state.chat_response = None
            st.session_state.referenced_codes = []
            st.session_state.relevant_codes = []
            st.rerun(scope="fragment")

    # Execute research when Send is clicked
    if chat_btn and user_question:
        with status_placeholder.container():
            with st.status("Processing your query...", expanded=True) as status:
                
                # Initialize progress tracking with functional approach
                progress_state = {
                    "current_phase": "Initializing",
                    "phase_details": "",
                    "agents_total": 0,
                    "agents_completed": 0,
                    "current_agent": 0,
                    "activity_log": [],
                    "insights_found": 0,
                    "papers_found": 0,
                }
                
                def update_progress(message: str):
                    print(message)  # Keep console logging
                    
                    # Parse message and update state
                    updates = su.parse_research_progress_message(message)
                    progress_state.update(updates)
                    
                    # Add to activity log (keep last 5 entries)
                    progress_state["activity_log"].append(message)
                    if len(progress_state["activity_log"]) > 5:
                        progress_state["activity_log"] = progress_state["activity_log"][-5:]
                    
                    # Render updated progress
                    su.render_research_progress(status, progress_state)

                try:
                    response_title, response, referenced_codes, relevant_codes = (
                        au.query_llmpedia_new(
                            user_question=user_question,
                            response_length=settings["response_length"],
                            llm_model="openai/gpt-4.1-nano",
                            max_sources=settings["max_sources"],
                            max_agents=settings["max_agents"],
                            debug=True,
                            progress_callback=update_progress,
                            show_only_sources=settings["show_only_sources"],
                        )
                    )
                    status.update(
                        label="Processing complete!",
                        state="complete",
                        expanded=False,
                    )
                    time.sleep(1)
                    status_placeholder.empty()

                except Exception as e:
                    status.update(
                        label="🤖 Error detected. System malfunction. Returning to nebular state.",
                        state="error",
                        expanded=True,
                    )
                    import traceback
                    print("Error details:")
                    print(e)
                    print(traceback.format_exc())
                    logging_db.log_error_db(e)
                    st.error("Sorry, an error occurred while processing your request.")

        # Store results in session state only if successful
        if "response" in locals():
            st.session_state.chat_response = response
            st.session_state.chat_response_title = response_title
            st.session_state.referenced_codes = referenced_codes
            st.session_state.relevant_codes = relevant_codes
            logging_db.log_qna_db(user_question, response)
            st.rerun(scope="fragment")

    # Display results if they exist in session state
    if st.session_state.chat_response:
        su.display_research_results(
            title=st.session_state.chat_response_title,
            response=st.session_state.chat_response,
            referenced_codes=st.session_state.referenced_codes,
            relevant_codes=st.session_state.relevant_codes,
            papers_df=st.session_state["papers"]
        )


@st.fragment
def display_top_cited_trending_panel(papers_df_fragment: pd.DataFrame):
    """Displays the Top Cited / Trending Papers panel with toggle and caching."""
    citation_window = 90
    trending_window = 7  # For fetching raw data from DB (e.g., last 7 days of tweets)
    top_n = 5  # Number of papers to display

    current_actual_toggle_state = st.session_state.get("toggle_trending_papers", True)


    # Create enhanced header
    if current_actual_toggle_state:
        header_html = f"""
        <div class="trending-panel-header">
            <div class="trending-panel-title">
                📈 Trending on X.com
            </div>
            <div class="trending-panel-subtitle">
                Most liked papers in the last {trending_window} days
            </div>
        </div>
        """
    else:
        header_html = f"""
        <div class="trending-panel-header">
            <div class="trending-panel-title">
                🏆 Top Cited Papers
            </div>
            <div class="trending-panel-subtitle">
                Most cited papers in the last {citation_window} days
            </div>
        </div>
        """
    
    st.markdown(header_html, unsafe_allow_html=True)

    # Toggle with improved styling
    toggle_label = (
        "🏆 Switch to Citations"
        if current_actual_toggle_state
        else "📈 Switch to Trending"
    )
    st.toggle(toggle_label, value=current_actual_toggle_state, key="toggle_trending_papers")

    if current_actual_toggle_state:  # Show trending table
        raw_trending_df = get_cached_raw_trending_data_app(
            n_fetch=top_n + 10,  # Fetch a bit more for robust joining
            time_window_days_db=trending_window,
        )
        trending_papers = get_processed_trending_papers(
            papers_df_fragment, raw_trending_df, top_n_display=top_n
        )

        if not trending_papers.empty:
            su.generate_mini_paper_table(
                trending_papers,
                n=top_n,
                extra_key="_dashboard_trending",
                metric_name="Likes",
                metric_col="like_count",
                show_tweets_toggle=True,
            )
        else:
            st.info("No trending data found or papers not in current view.")
    else:  # Show top cited table
        # Call the app.py cached wrapper for top cited papers
        top_papers = get_cached_top_cited_papers_app(
            papers_df_fragment, n=top_n, time_window_days=citation_window
        )

        if not top_papers.empty:
            su.generate_mini_paper_table(top_papers, n=top_n, extra_key="_dashboard")
        else:
            st.info("No top cited papers found for the current selection.")


def main():
    st.markdown(
        """<div class="pixel-font" style="margin-bottom: -0.5em;">LLMpedia</div>
    """,
        unsafe_allow_html=True,
    )
    st.markdown(
        "##### The Illustrated Large Language Model Encyclopedia"
        # help="Welcome to LLMpedia, your curated guide to Large Language Model research, brought to you by GPT Maestro. "
        # "Our pixel art illustrations and structured summaries make complex research accessible. "
        # "Have questions or interested in LLM research? Chat with the Maestro or follow us [@GPTMaestro](https://twitter.com/GPTMaestro) for the latest updates.\n\n"
        # "*Buona lettura!*",
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
        
        # Section header with consistent styling
        header_html = f"""
        <div class="trending-panel-header">
            <div class="trending-panel-title">
                📅 {year} Release Calendar
            </div>
            <div class="trending-panel-subtitle">
                Click on any day to filter papers by publication date
            </div>
        </div>
        """
        st.markdown(header_html, unsafe_allow_html=True)

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
    <div class="llmp-sidebar-footer">
        <a href="https://github.com/masta-g3/llmpedia/blob/main/VERSIONS.md" target="_blank">v1.6.0</a>
    </div>

    <div class="llmp-acknowledgment">
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
            "🏠 Main",
            "🧮 Release Feed",
            "🗺️ Statistics & Topics",
            "🔍 Paper Details",
            "🔬 Deep Research",
            "⚙️ Links & Repositories",
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

        # Display all metrics in a single row of 5 columns
        metric_cols = st.columns(5)  # Adjusted for 5 metrics
        with metric_cols[0]:
            st.metric(
                label="🗄️ Total Papers in Archive",
                value=f"{len(full_papers_df):,d}",
            )
        with metric_cols[1]:
            if not st.session_state.all_years:
                value = f"{len(papers_df[papers_df['published'].dt.year == year]):,d}"
                st.metric(
                    label=f"🔮 Published in {year}",
                    value=value,
                )
        with metric_cols[2]:
            st.metric(label="📅 Last 7 days", value=len(papers_7d))
        with metric_cols[3]:
            st.metric(label="⏰ Added in last 24 hours", value=len(papers_1d))

        with metric_cols[4]:  # New metric for Active Users
            active_users_count = get_active_users_count()
            st.metric(label="👥 Active Users", value=f"{active_users_count:,d}")

        st.divider()
        # Deep Research promotion section - moved to top for prominence
        header_html = """
        <div class="trending-panel-header">
            <div class="trending-panel-title">
                🔬 Ask the GPT Maestro
            </div>
            <div class="trending-panel-subtitle">
                AI-powered multi-agent research with cited sources
            </div>
        </div>
        """
        st.markdown(header_html, unsafe_allow_html=True)
        
        st.markdown(
            "Got specific questions about LLMs, arXiv papers, or need to find relevant research? "
            "Our AI-powered Deep Research tool uses specialized agents to find answers, synthesize content, and discover related work with cited sources. "
            "Ask the GPT maestro!"
        )
        shared_query_news_tab = st.text_input(
            "Ask your question here first:",
            key="news_tab_shared_query_input",
            placeholder="E.g., Why do LLMs sometimes exhibit ADHD like symptoms?",
        )
        if st.button(
            "Explore Deep Research", key="explore_deep_research_news_promo"
        ):
            query_to_pass = st.session_state.get("news_tab_shared_query_input", "")
            if query_to_pass:  # Only pass if there's actual text
                st.session_state.query_to_pass_to_chat = query_to_pass
            su.click_tab(4)  # Navigate to the Deep Research tab

        st.divider()
        row2_cols = st.columns([4, 0.1, 2])

        # Panel 2.1: Top Cited / Trending Papers (Left)
        with row2_cols[0]:
            # Make sure papers_df is available here
            # If papers_df can be None or empty, handle appropriately before calling
            if papers_df is not None and not papers_df.empty:
                display_top_cited_trending_panel(papers_df)
            else:
                st.info("Paper data is not available for this panel.")

        # Panel 2.2: X.com/Reddit & Featured Paper (Right)
        with row2_cols[2]:
            @st.fragment
            def discussions_toggle_panel():
                """Displays discussions toggle panel similar to trending papers toggle."""
                current_discussions_toggle = st.session_state.get("toggle_discussions_platform", True)
                
                # Create enhanced header
                if current_discussions_toggle:
                    header_html = """
                    <div class="trending-panel-header">
                        <div class="trending-panel-title">
                            🐦 Latest LLM Discussions on X
                        </div>
                        <div class="trending-panel-subtitle">
                            Timestamped summaries updated every ~24 hours
                        </div>
                    </div>
                    """
                else:
                    header_html = """
                    <div class="trending-panel-header">
                        <div class="trending-panel-title">
                            🦙 Latest LLM Discussions on Reddit
                        </div>
                        <div class="trending-panel-subtitle">
                            Cross-subreddit summaries updated every ~24 hours
                        </div>
                    </div>
                    """
                
                st.markdown(header_html, unsafe_allow_html=True)

                # Toggle with improved styling
                toggle_label = (
                    "🦙 Switch to Reddit"
                    if current_discussions_toggle
                    else "🐦 Switch to X.com"
                )
                st.toggle(toggle_label, value=current_discussions_toggle, key="toggle_discussions_platform")

                if current_discussions_toggle:  # Show X.com discussions
                    tweet_summaries_df = db.read_last_n_tweet_analyses(n=8)
                    tweet_summaries_df = tweet_summaries_df[tweet_summaries_df["tstp"] >= "2025-05-14"]
                    if tweet_summaries_df is not None and not tweet_summaries_df.empty:
                        su.display_tweet_summaries(tweet_summaries_df, max_entries=8)
                    else:
                        st.info("No recent X.com discussions found.")
                else:  # Show Reddit discussions
                    reddit_summaries_df = db.read_last_n_reddit_analyses(n=8)
                    if reddit_summaries_df is not None and not reddit_summaries_df.empty:
                        su.display_reddit_summaries(reddit_summaries_df, max_entries=8)
                    else:
                        st.info("No recent Reddit discussions found.")

            discussions_toggle_panel()
            st.divider()

            highlight_paper = get_featured_paper(papers_df)
            su.create_featured_paper_card(highlight_paper)

        st.divider()
        row3_cols = st.columns([1, 1])

        # Column 1: Random Interesting Fact
        with row3_cols[0]:

            @st.fragment
            def interesting_fact_display():
                # Section header with consistent styling
                header_html = """
                <div class="trending-panel-header">
                    <div class="trending-panel-title">
                        💡 Interesting Fact
                    </div>
                    <div class="trending-panel-subtitle">
                        Random discovery from recent research
                    </div>
                </div>
                """
                st.markdown(header_html, unsafe_allow_html=True)
                
                # Retrieve a single random fact (cached) and display it
                fact_list = get_random_interesting_facts(
                    n=1,
                    recency_days=30,  # Consider if this window is too narrow or too wide
                    _trigger=st.session_state.facts_refresh_trigger,
                )
                # full_papers_df is defined in the main() scope and should be accessible
                # If not, it might need to be passed or accessed via st.session_state.papers
                su.display_interesting_facts(
                    fact_list, n_cols=1, papers_df=full_papers_df
                )
                # Refresh button to get a new fact
                if st.button("🔄 New Fact", key="refresh_fact_single_fragment"):
                    st.session_state.facts_refresh_trigger += 1
                    st.rerun()  # This will rerun only this fragment

            interesting_fact_display()

        # Column 2: Feature Poll
        with row3_cols[1]:

            @st.fragment
            def feature_poll_fragment():
                # Section header with consistent styling
                header_html = """
                <div class="trending-panel-header">
                    <div class="trending-panel-title">
                        🗳️ Feature Poll
                    </div>
                    <div class="trending-panel-subtitle">
                        Help us shape LLMpedia's future
                    </div>
                </div>
                """
                st.markdown(header_html, unsafe_allow_html=True)

                has_voted = st.session_state.get("user_has_voted_poll", False)

                if has_voted:
                    st.info("You have already voted in this session. Thank you for your feedback!")
                else:
                    pass
                default_options = [
                    "**Aggregate Model Scores**: Get a summary of how different models perform on tests test results presented in papers.",
                    "**Improved Deep Research**: Support for long running (30 min+) agentic research.",
                    "**Concept Glossary**: A searchable glossary of key concepts used in papers.",
                    "Other (specify)",
                ]
                selected_option = st.radio(
                    "*Most desired upcoming feature:*",
                    options=default_options,
                    key="feature_poll_option",
                    disabled=has_voted,
                )

                # If user chooses Other, allow a custom input
                custom_feature = ""
                if selected_option == "Other (specify)" and not has_voted: # Added 'and not has_voted'
                    custom_feature = st.text_input(
                        "Your feature suggestion",
                        key="custom_feature_suggestion",
                        disabled=has_voted, # Added disabled state
                    )

                if st.button("Vote", key="vote_feature_poll_button", disabled=has_voted):
                    feature_name_to_log = ""
                    is_custom = False

                    if selected_option == "Other (specify)":
                        if custom_feature.strip():
                            feature_name_to_log = custom_feature.strip()
                            is_custom = True
                        else:
                            st.warning("Please provide a feature name before voting.")
                    else:
                        feature_name_to_log = selected_option
                        is_custom = False

                    if feature_name_to_log:
                        try:
                            # Placeholder for session_id; implement actual session tracking if needed
                            current_session_id = st.session_state.get(
                                "user_session_id", None
                            )

                            logging_db.log_feature_poll_vote(
                                feature_name=feature_name_to_log,
                                is_custom_suggestion=is_custom,
                                session_id=current_session_id,
                            )
                            st.session_state.user_has_voted_poll = True  # Set flag after successful vote
                            st.success("Thanks for voting!")
                            # Rerun only this fragment
                            st.rerun()
                        except Exception as e:
                            st.error("Sorry, there was an issue recording your vote.")
                            # Log the error to your error_logs table
                            logging_db.log_error_db(
                                f"Feature poll vote logging error: {e}"
                            )
                    elif (
                        selected_option == "Other (specify)"
                        and not custom_feature.strip()
                    ):
                        # This case is already handled by the warning above, but kept for clarity
                        pass  # Warning already shown

            feature_poll_fragment()


    with content_tabs[1]:
        ## Grid view or Table view
        if "page_number" not in st.session_state:
            st.session_state.page_number = 0

        # Add view selector with radio buttons
        if "view_mode" not in st.session_state:
            st.session_state.view_mode = "Grid View"

        view_mode = st.radio(
            "Display Mode",
            options=["Artwork Grid", "First Page Grid", "Table View"],
            horizontal=True,
            key="view_selector",
            index=0,
        )

        # Get paginated data
        papers_df_subset = su.create_pagination(
            papers_df, items_per_page=25, label="grid", year=year
        )

        # Display content based on selected view
        if view_mode == "Artwork Grid":
            su.generate_grid_gallery(papers_df_subset, image_type="artwork")
        elif view_mode == "First Page Grid":
            su.generate_grid_gallery(papers_df_subset, image_type="first_page")
        elif view_mode == "Table View":
            su.generate_paper_table(papers_df_subset)

        # Bottom navigation
        su.create_bottom_navigation("grid")

    with content_tabs[2]:
        total_papers = len(papers_df)
        
        # Publication Counts Section with consistent header styling
        if not st.session_state.all_years:
            publication_header_html = f"""
            <div class="trending-panel-header">
                <div class="trending-panel-title">
                    📈 {year} Publication Trends
                </div>
                <div class="trending-panel-subtitle">
                    {total_papers:,} papers published • Interactive visualizations with filtering options
                </div>
            </div>
            """
        else:
            publication_header_html = f"""
            <div class="trending-panel-header">
                <div class="trending-panel-title">
                    📈 Publication Trends Overview
                </div>
                <div class="trending-panel-subtitle">
                    {total_papers:,} papers total • Comprehensive analysis across all years
                </div>
            </div>
            """
        
        st.markdown(publication_header_html, unsafe_allow_html=True)
        
        # Default values for initial plot generation
        plot_view = st.session_state.get("stats_plot_view", "Total Volume")
        plot_type = st.session_state.get("stats_plot_type", "Daily")
        cumulative = plot_type == "Cumulative"
        
        ## Generate appropriate plot based on selections
        if plot_view == "Total Volume":
            ts_plot = pt.plot_publication_counts(papers_df, cumulative=cumulative)
        else:
            ts_plot = pt.plot_publication_counts_by_topics(papers_df, cumulative=cumulative, top_n=10)
        
        # Reduce top margin for the publication counts chart
        ts_plot.update_layout(margin=dict(t=0, b=0.5))
        st.plotly_chart(ts_plot, use_container_width=True)
        
        ## Controls for plotting
        plot_controls_cols = st.columns([0.5, 1, 1, 0.5])
        
        with plot_controls_cols[1]:
            plot_view = st.radio(
                label="View Mode",
                options=["Total Volume", "By Topics"],
                index=0 if plot_view == "Total Volume" else 1,
                horizontal=True,
                help="Choose between total publication volume or breakdown by research topics",
                key="stats_plot_view"
            )
        
        with plot_controls_cols[2]:
            plot_type = st.radio(
                label="Chart Type",
                options=["Daily", "Cumulative"],
                index=0 if plot_type == "Daily" else 1,
                horizontal=True,
                help="Daily shows publications per day, Cumulative shows running totals",
                key="stats_plot_type"
            )

        st.divider()

        # Topic Model Map Section with consistent header styling
        if not st.session_state.all_years:
            topic_header_html = f"""
            <div class="trending-panel-header">
                <div class="trending-panel-title">
                    🗺️ {year} Research Topic Map
                </div>
                <div class="trending-panel-subtitle">
                    Interactive clustering visualization • Click any point to explore paper details
                </div>
            </div>
            """
        else:
            topic_header_html = f"""
            <div class="trending-panel-header">
                <div class="trending-panel-title">
                    🗺️ Research Topic Landscape
                </div>
                <div class="trending-panel-subtitle">
                    Complete topic model visualization • Click any point to explore paper details
                </div>
            </div>
            """
        
        st.markdown(topic_header_html, unsafe_allow_html=True)

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
                        # st.query_params["arxiv_code"] = arxiv_code
                        st.session_state.arxiv_code = arxiv_code
                        su.click_tab(3)

    with content_tabs[3]:
        ## Focus on a paper.
        if len(st.session_state.arxiv_code) == 0:
            st.markdown(
                "<div style='font-size: 0.9em; opacity: 0.8; margin-bottom: 1.5em;'>💡 <em>Search a paper by its arXiv code, or use the sidebar to search and filter papers by title, author, or other attributes.</em></div>",
                unsafe_allow_html=True,
            )

        # Text input remains outside the fragment to update session state
        search_cols = st.columns((7, 1))
        st.session_state.details_canvas = st.container()
        with search_cols[0]:
            arxiv_code_input = st.text_input("arXiv Code", "")
        with search_cols[1]:
            st.write("  ")
            st.write("  ")
            if st.button("Search"):
                st.session_state.arxiv_code = arxiv_code_input
                su.click_tab(3)

    with content_tabs[4]:
        chat_fragment()

    with content_tabs[5]:
        # Links & Repositories header with consistent styling
        repos_header_html = """
        <div class="trending-panel-header">
            <div class="trending-panel-title">
                ⚙️ Links & Repositories
            </div>
            <div class="trending-panel-subtitle">
                Discover code repositories and resources related to research papers
            </div>
        </div>
        """
        st.markdown(repos_header_html, unsafe_allow_html=True)
        
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
        
        # Results summary with enhanced styling
        if repo_count > 0:
            results_html = f"""
            <div style="margin: 1rem 0; padding: 0.75rem; background: linear-gradient(135deg, var(--background-color, #ffffff) 0%, var(--secondary-background-color, #f8f9fa) 100%); border: 1px solid rgba(179, 27, 27, 0.08); border-radius: var(--radius-base); font-size: var(--font-size-sm);">
                <strong>📊 {repo_count:,} resources found</strong> • Use filters above to refine results
            </div>
            """
        else:
            results_html = """
            <div style="margin: 1rem 0; padding: 0.75rem; background: rgba(255, 193, 7, 0.1); border: 1px solid rgba(255, 193, 7, 0.3); border-radius: var(--radius-base); font-size: var(--font-size-sm);">
                <strong>⚠️ No resources found</strong> • Try adjusting your filters
            </div>
            """
        st.markdown(results_html, unsafe_allow_html=True)
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
        # Weekly Report header with consistent styling
        weekly_header_html = """
        <div class="trending-panel-header">
            <div class="trending-panel-title">
                🗞️ Weekly Research Report
            </div>
            <div class="trending-panel-subtitle">
                Curated weekly summaries of key developments in LLM research
            </div>
        </div>
        """
        st.markdown(weekly_header_html, unsafe_allow_html=True)
        
        report_top_cols = st.columns((5, 2))
        with report_top_cols[0]:
            pass  # Header is now handled above

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

    ## URL info extraction (moved to end after all components are initialized).
    su.parse_query_params()


if __name__ == "__main__":
    # try:
        main()
    # except Exception as e:
    #     logging_db.log_error_db(e)
    #     st.error(
    #         "Something went wrong. Please refresh the app and try again, we will look into it."
    #     )
