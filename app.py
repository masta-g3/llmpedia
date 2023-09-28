import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from streamlit_plotly_events import plotly_events

from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import psycopg2
import json
import re, os

import plotly.io as pio

pio.templates.default = "plotly"

db_params = {**st.secrets["postgres"]}

## Page config.
st.set_page_config(
    layout="wide",
    page_title="📚 LLMpedia",
    page_icon="📚",
    initial_sidebar_state="expanded",
)

if "papers" not in st.session_state:
    st.session_state.papers = None

if "page_number" not in st.session_state:
    st.session_state.page_number = 0

if "num_pages" not in st.session_state:
    st.session_state.num_pages = 0

if "arxiv_code" not in st.session_state:
    st.session_state.arxiv_code = ""

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


def load_arxiv():
    query = "SELECT * FROM arxiv_details;"
    conn = psycopg2.connect(**db_params)
    arxiv_df = pd.read_sql(query, conn)
    arxiv_df.set_index("arxiv_code", inplace=True)
    conn.close()
    return arxiv_df


def load_reviews():
    query = "SELECT * FROM summaries;"
    conn = psycopg2.connect(**db_params)
    summaries_df = pd.read_sql(query, conn)
    summaries_df.set_index("arxiv_code", inplace=True)
    conn.close()
    return summaries_df


def load_topics():
    query = "SELECT * FROM topics;"
    conn = psycopg2.connect(**db_params)
    topics_df = pd.read_sql(query, conn)
    topics_df.set_index("arxiv_code", inplace=True)
    conn.close()
    return topics_df


def load_citations():
    query = "SELECT * FROM semantic_details;"
    conn = psycopg2.connect(**db_params)
    citations_df = pd.read_sql(query, conn)
    citations_df.set_index("arxiv_code", inplace=True)
    citations_df.drop(columns=["paper_id"], inplace=True)
    conn.close()
    return citations_df


def combine_input_data():
    arxiv_df = load_arxiv()
    reviews_df = load_reviews()
    topics_df = load_topics()
    citations_df = load_citations()
    papers_df = reviews_df.join(arxiv_df, how="left")
    papers_df = papers_df.join(topics_df, how="left")
    papers_df = papers_df.join(citations_df, how="left")
    papers_df["arxiv_code"] = papers_df.index
    papers_df["url"] = papers_df["arxiv_code"].map(
        lambda l: f"https://arxiv.org/abs/{l}"
    )
    papers_df.sort_values("published", ascending=False, inplace=True)
    return papers_df


def prepare_calendar_data(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Prepares data for the creation of a calendar heatmap."""
    df["published"] = pd.to_datetime(df["published"])
    df_year = df[df["published"].dt.year == year].copy()
    ## publishes dates with zero 'Counts' with full year dates.
    df_year = (
        df_year.set_index("published")
        .reindex(pd.date_range(start=f"{year}-01-01", end=f"{year}-12-31", freq="D"))
        .fillna(0)
        .reset_index()
    )
    df_year.columns = ["published", "Count"]
    df_year["week"] = df_year["published"].dt.isocalendar().week - 1
    df_year["weekday"] = df_year["published"].dt.weekday
    return df_year


def plot_publication_counts(df: pd.DataFrame, cumulative=False) -> go.Figure:
    """Plot line chart of total number of papers updated per day."""
    df["published"] = pd.to_datetime(df["published"])
    df["published"] = df["published"].dt.date
    df = df.groupby("published")["title"].nunique().reset_index()
    df.columns = ["published", "Count"]
    df["published"] = pd.to_datetime(df["published"])
    df.sort_values("published", inplace=True)
    df["Cumulative Count"] = df["Count"].cumsum()
    if cumulative:
        fig = px.area(
            df,
            x="published",
            y="Cumulative Count",
            title=None,
        )
    else:
        fig = px.bar(
            df,
            x="published",
            y="Count",
        )
    return fig


def plot_activity_map(df_year: pd.DataFrame) -> (go.Figure, pd.DataFrame):
    """Creates a calendar heatmap plot along with corresponding map of dates in a DF."""
    colors = ["#003366", "#005599", "#0077CC", "#3399FF", "#66B2FF", "#99CCFF"]
    colors = ["#994400", "#CC6600", "#FF8833", "#FFAA66", "#FFCC99"]

    week_max_dates = (
        df_year.groupby(df_year["published"].dt.isocalendar().week)["published"]
        .max()
        .dt.strftime("%b %d")
        .tolist()
    )

    padded_count = df_year.pivot_table(
        index="weekday", columns="week", values="Count", aggfunc="sum"
    ).fillna(0)
    padded_date = df_year.pivot_table(
        index="weekday", columns="week", values="published", aggfunc="last"
    ).fillna(pd.NaT)
    padded_date = padded_date.applymap(
        lambda x: x.strftime("%b %d") if pd.notna(x) else ""
    )
    padded_count = padded_count.iloc[::-1]
    padded_date = padded_date.iloc[::-1]

    fig = go.Figure(
        data=go.Heatmap(
            z=padded_count.values,  # df_year["Count"].values.reshape(54, 7).T,
            x=padded_date.iloc[0].values,  # df_year["weekday"].values.reshape(54, 7).T,
            y=["Sun", "Sat", "Fri", "Thu", "Wed", "Tue", "Mon"],
            hoverongaps=False,
            hovertext=padded_date.values,  # df_year["hovertext"].values.reshape(53, 7).T,
            colorscale=colors,
            showscale=False,
        )
    )
    fig.update_layout(
        height=210,
        margin=dict(t=0, b=0, l=0, r=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(tickfont=dict(color="grey"), showgrid=False, zeroline=False)
    fig.update_yaxes(tickfont=dict(color="grey"), showgrid=False, zeroline=False)
    padded_date = padded_date.iloc[::-1]

    return fig, padded_date


def plot_cluster_map(df: pd.DataFrame) -> go.Figure:
    """Creates a scatter plot of the UMAP embeddings of the papers."""
    fig = px.scatter(
        df,
        x="dim1",
        y="dim2",
        color="topic",
        hover_name="title",
    )
    fig.update_layout(
        legend=dict(
            title=None,
            font=dict(size=14),
        ),
        margin=dict(t=0, b=0, l=0, r=0),
    )
    fig.update_xaxes(title_text=None)
    fig.update_yaxes(title_text=None)
    fig.update_traces(marker=dict(line=dict(width=1, color="DarkSlateGrey"), size=8))
    return fig


@st.cache_data
def load_data():
    """Load data from compiled dataframe."""
    result_df = combine_input_data()
    # result_df = result_df[result_df["published"] >= "2021-01-01"]

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
    result_df["category"] = result_df["category"].apply(lambda x: classification_map[x])

    return result_df


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


def get_similar_titles(
    title: str, df: pd.DataFrame, n: int = 5
) -> Tuple[List[str], str]:
    """Returns titles of papers from the same cluster, along with cluster name"""
    title = title.lower()
    if title in df["title"].str.lower().values:
        cluster = df[df["title"].str.lower() == title]["topic"].values[0]
        size = df[df["topic"] == cluster].shape[0]
        similar_titles = (
            df[df["topic"] == cluster]["title"].sample(min(n, size)).tolist()
        )
        similar_titles = [t for t in similar_titles if t.lower() != title]
        return similar_titles, cluster
    else:
        return [], ""


def create_paper_card(paper: Dict, mode="preview"):
    """Creates card UI for paper details."""
    img_cols = st.columns((1, 3))
    expanded = False
    if mode == "open":
        expanded = True
    paper_code = paper["arxiv_code"]
    try:
        img_cols[0].image(f"imgs/{paper_code}.png", use_column_width=True)
    except:
        pass

    paper_title = paper["title"]
    similar_titles, cluster_name = get_similar_titles(
        paper_title, st.session_state["papers"], n=5
    )

    paper_url = paper["url"]
    img_cols[1].markdown(
        f'<h2><a href="{paper_url}" style="color: #FF4B4B;">{paper_title}</a></h2>',
        unsafe_allow_html=True,
    )

    pub_date = pd.to_datetime(paper["published"]).strftime("%B %d, %Y")
    upd_date = pd.to_datetime(paper["updated"]).strftime("%B %d, %Y")
    img_cols[1].markdown(f"#### Published: {pub_date}")
    if pub_date != upd_date:
        img_cols[1].caption(f"Last Updated: {upd_date}")
    img_cols[1].markdown(f"*{paper['authors']}*")
    img_cols[1].markdown(f"`{int(paper['citation_count'])} citations`")

    with st.expander(f"💭 Abstract (arXiv:{paper_code})", expanded=expanded):
        st.markdown(paper["summary"])

    with st.expander(
        f"➕ **Contributions** - {paper['contribution_title']}", expanded=True
    ):
        st.markdown(f"{paper['contribution_content']}")

    with st.expander(
        f"✏️ **Takeaways** - {paper['takeaway_title']}", expanded=expanded
    ):
        st.markdown(f"{paper['takeaway_content']}")
        st.markdown(f"{paper['takeaway_example']}")

    with st.expander("🥉 **GPT Assessments**", expanded=expanded):
        ## GPT Cluster category.
        st.markdown(f"**GPT Cluster Group**: {paper['topic']}")

        novelty_cols = st.columns((1, 10))
        novelty_cols[0].metric("Novelty", f"{paper['novelty_score']}/3", "🚀")
        novelty_cols[1].markdown(f"{paper['novelty_analysis']}")

        tech_cols = st.columns((1, 10))
        tech_cols[0].metric("Technical Depth", f"{paper['technical_score']}/3", "🔧")
        tech_cols[1].markdown(f"{paper['technical_analysis']}")

        enjoy_cols = st.columns((1, 10))
        enjoy_cols[0].metric("Readability", f"{paper['enjoyable_score']}/3", "📚")
        enjoy_cols[1].markdown(f"{paper['enjoyable_analysis']}")

    with st.expander(f"📚 **Similar Papers** (Topic: {cluster_name})"):
        for title in similar_titles:
            st.markdown(f"* {title}")

    st.markdown("---")


def generate_grid_gallery(df, n_cols=5):
    """Create streamlit grid gallery of paper cards with thumbnail."""
    n_rows = int(np.ceil(len(df) / n_cols))
    for i in range(n_rows):
        cols = st.columns(n_cols)
        for j in range(n_cols):
            if i * n_cols + j < len(df):
                with cols[j]:
                    try:
                        st.image(f"imgs/{df.iloc[i*n_cols+j]['arxiv_code']}.png")
                    except:
                        pass
                    paper_code = df.iloc[i * n_cols + j]["arxiv_code"]
                    focus_btn = st.button(
                        "Focus", key=f"focus_{paper_code}", use_container_width=True
                    )
                    if focus_btn:
                        st.session_state.arxiv_code = paper_code
                        click_tab(3)
                    paper_url = df.iloc[i * n_cols + j]["url"]
                    paper_title = df.iloc[i * n_cols + j]["title"].replace("\n", "")
                    star_count = df.iloc[i * n_cols + j]["influential_citation_count"] > 0
                    publish_date = pd.to_datetime(
                        df.iloc[i * n_cols + j]["published"]
                    ).strftime("%B %d, %Y")
                    star = ""
                    if star_count:
                        star = "⭐️"
                    st.code(f"{star} {publish_date}", language="html")
                    st.markdown(
                        f'<h6 style="text-align: center"><a href="{paper_url}" style="color: #FF4B4B;">{paper_title}</a></h6>',
                        unsafe_allow_html=True,
                    )
                    last_updated = pd.to_datetime(
                        df.iloc[i * n_cols + j]["published"]
                    ).strftime("%B %d, %Y")
                    # st.markdown(f"{last_updated}")
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


def create_bottom_navigation(label):
    num_pages = st.session_state["num_pages"]
    st.write(f"**Pg. {st.session_state.page_number + 1} of {num_pages}**")
    prev_button_btm, _, next_button_btm = st.columns((1, 10, 1))
    prev_clicked_btm = prev_button_btm.button("Prev", key=f"prev_{label}_btm")
    next_clicked_btm = next_button_btm.button("Next", key=f"next_{label}_btm")
    if prev_clicked_btm and "page_number" in st.session_state:
        st.session_state.page_number = max(0, st.session_state.page_number - 1)
        st.experimental_rerun()
    if next_clicked_btm and "page_number" in st.session_state:
        st.session_state.page_number = min(
            num_pages - 1, st.session_state.page_number + 1
        )
        st.experimental_rerun()


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


def main():
    ## URL info extraction.
    url_query = st.experimental_get_query_params()
    if "arxiv_code" in url_query:
        arxiv_code = url_query["arxiv_code"][0]
        st.session_state.arxiv_code = arxiv_code

    st.markdown(
        """<div class="pixel-font">LLMpedia</div>
    """,
        unsafe_allow_html=True,
    )
    st.markdown(
        "##### A collection of research papers on Language Models curated by the GPT maestro itself."
    )
    ## Humorous and poetic introduction.
    st.markdown(
        "Every week dozens of papers are published on Language Models. It is impossible to keep up with the latest research. "
        "That's why we created LLMpedia, a collection of papers on Language Models curated by the GPT maestro itself.\n\n"
        "Each week GPT will sweep through the latest LLM related papers and select the most interesting ones. "
        "The maestro will then summarize the papers and provide its own analysis, including a novelty, technical depth and readability score. "
        "We hope you enjoy this collection and find it useful.\n\n"
        "*Buona lettura!*"
    )

    ## Main content.
    papers_df = load_data()
    st.session_state["papers"] = papers_df

    ## Filter sidebar.
    st.sidebar.markdown("# 📁 Filters")
    year = st.sidebar.slider(
        "Year",
        min_value=2021,
        max_value=2023,
        value=2023,
        step=1,
    )
    search_term = st.sidebar.text_input("Search", "")
    search_opt_cols = st.sidebar.columns((1, 1))
    title_only = search_opt_cols[0].checkbox("`Title Only`", value=False)
    code_only = search_opt_cols[1].checkbox("`Arxiv Code`", value=False)
    categories = st.sidebar.multiselect(
        "Categories",
        list(papers_df["category"].unique()),
    )
    topics = st.sidebar.multiselect(
        "Topic Group",
        list(papers_df["topic"].unique()),
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
    papers_df = papers_df[papers_df["published"].dt.year == year]

    ## Search terms.
    if len(search_term) > 0 and title_only:
        search_term = search_term.lower()
        papers_df = papers_df[papers_df["title"].str.lower().str.contains(search_term)]
    elif len(search_term) > 0 and code_only:
        search_term = search_term.lower()
        papers_df = papers_df[
            papers_df["arxiv_code"].str.lower().str.contains(search_term)
        ]
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

    if len(papers_df) == 0:
        st.error("No papers found.")
        return

    ## Calendar selector.
    published_df = generate_calendar_df(papers_df)
    heatmap_data = prepare_calendar_data(published_df, year)
    release_calendar, padded_date = plot_activity_map(heatmap_data)
    st.markdown(f"### 📅 {year} Release Calendar")
    calendar_select = plotly_events(release_calendar, override_height=220)

    ## Published date.
    if len(calendar_select) > 0:
        ## Select from padded dates.
        x_coord = 6 - calendar_select[0]["pointNumber"][0]
        y_coord = calendar_select[0]["pointNumber"][1]
        publish_date = pd.to_datetime(padded_date.loc[x_coord, y_coord] + f" {year}")

        if len(papers_df[papers_df["published"] == publish_date]) > 0:
            papers_df = papers_df[papers_df["published"] == publish_date]
            ## Add option to clear filter on sidebar.
            if st.sidebar.button(
                f"📅 **Publish Date Filter:** {publish_date.strftime('%B %d, %Y')}"
            ):
                st.experimental_rerun()

    if len(papers_df) == 0:
        st.markdown("No papers found.")
        return

    papers = papers_df.to_dict("records")

    ## Content tabs.
    content_tabs = st.tabs(["Grid View", "Feed View", "Over View", "Focus View"])

    with content_tabs[0]:
        if "page_number" not in st.session_state:
            st.session_state.page_number = 0

        papers_df_subset = create_pagination(papers_df, items_per_page=25, label="grid")
        generate_grid_gallery(papers_df_subset)
        create_bottom_navigation(label="grid")

    with content_tabs[1]:
        if "page_number" not in st.session_state:
            st.session_state.page_number = 0

        papers_subset = create_pagination(papers, items_per_page=7, label="summaries")
        st.markdown(f"**{len(papers)} papers found.**")
        for paper in papers_subset:
            create_paper_card(paper)
        create_bottom_navigation(label="summaries")

    with content_tabs[2]:
        ## Publication counts.
        total_papers = len(papers_df)
        st.markdown(f"### 📈 Publication Counts (Total Tracked: {total_papers})")
        plot_type = st.radio(
            label="Plot Type",
            options=["Daily", "Cumulative"],
            index=1,
            label_visibility="collapsed",
            horizontal=True,
        )
        cumulative = plot_type == "Cumulative"
        ts_plot = plot_publication_counts(papers_df, cumulative=cumulative)
        st.plotly_chart(ts_plot, use_container_width=True)

        ## Cluster map.
        st.markdown("### Topic Model Map")
        cluster_map = plot_cluster_map(papers_df)
        st.plotly_chart(cluster_map, use_container_width=True)

    with content_tabs[3]:
        ## Focus on a paper.
        arxiv_code = st.text_input("arXiv Code", st.session_state.arxiv_code)
        st.session_state.arxiv_code = arxiv_code
        if len(arxiv_code) > 0:
            if arxiv_code in papers_df.index:
                paper = papers_df.loc[arxiv_code].to_dict()
                create_paper_card(paper, mode="open")
            else:
                st.error("Paper not found.")

    ## URL tab selection.
    # if "tab_num" in url_query:
    #     index_tab = int(url_query["tab_num"][0])
    #     js = f"""
    #     <script>
    #         var tabs = window.parent.document.querySelectorAll("[id^='tabs-bui'][id$='-tab-{index_tab}']");
    #         if (tabs.length > 0) {{
    #             tabs[0].click();
    #         }}
    #     </script>
    #     """
    #     st.components.v1.html(js)


if __name__ == "__main__":
    main()
