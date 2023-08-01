import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from streamlit_plotly_events import plotly_events

from typing import Dict
import pandas as pd
import numpy as np
import json

import plotly.io as pio

pio.templates.default = "plotly"

## Page config.
st.set_page_config(
    layout="wide",
    page_title="📚 LLMpedia",
    page_icon="📚",
    initial_sidebar_state="expanded",
)

if 'page_number' not in st.session_state:
    st.session_state.page_number = 0

def combine_input_data():
    with open("arxiv_code_map.json", "r") as f:
        arxiv_code_map = json.load(f)
    arxiv_df = pd.read_pickle("data/arxiv.pkl")
    reviews_df = pd.read_pickle("data/reviews.pkl")
    topics_df = pd.read_pickle("data/topics.pkl")
    papers_df = pd.concat([arxiv_df, reviews_df, topics_df], axis=1)
    papers_df.sort_values("Updated", ascending=False, inplace=True)
    return papers_df


def prepare_calendar_data(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Prepares data for the creation of a calendar heatmap."""
    df["Published"] = pd.to_datetime(df["Published"])
    df_year = df[df["Published"].dt.year == year].copy()
    df_year["week"] = df_year["Published"].dt.isocalendar().week
    df_year["weekday"] = df_year["Published"].dt.weekday

    all_dates = pd.DataFrame(
        [(week, weekday) for week in range(1, 54) for weekday in range(7)],
        columns=["week", "weekday"],
    )
    heatmap_data = (
        df_year.groupby(["week", "weekday"])
        .agg({"Count": "sum", "Published": "first"})
        .reset_index()
    )
    heatmap_data = pd.merge(all_dates, heatmap_data, how="left", on=["week", "weekday"])
    heatmap_data["Count"] = heatmap_data["Count"].fillna(0)
    heatmap_data["Published"] = heatmap_data["Published"].fillna(pd.NaT)

    return heatmap_data


def plot_activity_map(df_year: pd.DataFrame) -> go.Figure:
    """Creates a calendar heatmap plot."""
    df_year["hovertext"] = np.where(
        df_year["Published"].isna(), "", df_year["Published"].dt.strftime("%b %d, %Y")
    )

    fig = go.Figure(
        data=go.Heatmap(
            z=df_year["Count"].values.reshape(53, 7).T,
            x=["W" + str(i) for i in range(1, 54)],
            y=["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"],
            hoverongaps=False,
            hovertext=df_year["hovertext"].values.reshape(53, 7).T,
            colorscale="amp",
            showscale=False,
        )
    )
    fig.update_layout(
        height=200,
        margin=dict(t=0, b=0, l=0, r=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def plot_cluster_map(df: pd.DataFrame) -> go.Figure:
    """ Creates a scatter plot of the UMAP embeddings of the papers."""
    fig = px.scatter(
        df,
        x="dim1",
        y="dim2",
        color="topic",
        hover_name="Title",
    )
    fig.update_layout(
        autosize=False,
        width=1200,
        height=500,
        font=dict(
            size=16,
        ),
        legend=dict(
            title=None,
            font=dict(
                size=14,
            ),
        ),
        margin=dict(t=0, b=0, l=0, r=0),
    )
    fig.update_xaxes(title_text="UMAP Dim 1")
    fig.update_yaxes(title_text="UMAP Dim 2")
    fig.update_traces(marker=dict(line=dict(width=1, color="DarkSlateGrey"), size=10))
    return fig


@st.cache_data
def load_data():
    """Load data from compiled dataframe."""
    result_df = combine_input_data()
    result_df = result_df[result_df["Published"] > "2021-01-01"]

    ## Remapping with emotion.
    classification_map = {
        "TRAINING": "🏋️‍ TRAINING",
        "FINE-TUNING": "🔧 FINE-TUNING",
        "ARCHITECTURES": "🏗️ ARCHS",
        "BEHAVIOR": "🔮 BEHAVIOR",
        "PROMPTING": "📣 PROMPTING",
        "USE CASES": "💰 USE CASES",
        "OTHER": "🤷 OTHER",
    }
    ## Round published and updated columns.
    result_df["Updated"] = pd.to_datetime(result_df["Updated"]).dt.date
    result_df["Published"] = pd.to_datetime(result_df["Published"]).dt.date
    result_df["Published"] = result_df["Updated"]
    result_df["category"] = result_df["category"].apply(lambda x: classification_map[x])
    return result_df


@st.cache_data
def generate_calendar_df(df: pd.DataFrame):
    """Daily counts of papers."""
    published_df = df.groupby("Published").count()["Title"]
    published_df = published_df.reindex(
        pd.date_range(
            start=published_df.index.min(), end=published_df.index.max(), freq="D"
        )
    ).fillna(0)
    published_df = published_df.reset_index()
    published_df.columns = ["Published", "Count"]
    return published_df


def create_paper_card(paper: Dict):
    """Creates card UI for paper details."""
    title_cols = st.columns((10, 1))
    paper_title = paper['Title'].replace("\n","")
    paper_url = paper['URL'].replace("http","https")
    title_cols[0].markdown(f"## 📄 [{paper_title}]({paper_url})")
    title_cols[1].markdown(f"###### {paper['category']}")

    date = pd.to_datetime(paper["Published"]).strftime("%B %d, %Y")
    st.markdown(f"#### {date}")
    authors_str = ", ".join(paper["Authors"])
    st.markdown(f"*{authors_str}*")

    with st.expander("💭 Summary"):
        st.markdown(paper["Summary"])

    with st.expander(f"➕ **Contributions** - {paper['main_contribution']['headline']}",
                     expanded=True):
        st.markdown(f"{paper['main_contribution']['description']}")

    with st.expander(f"✏️ **Takeaways** - {paper['takeaways']['headline']}"):
        st.markdown(f"{paper['takeaways']['description']}")
        st.markdown(f"{paper['takeaways']['example']}")

    with st.expander("🥉 **GPT Assessments**"):
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

    st.markdown("---")


def main():
    st.title("📚 LLMpedia")
    st.markdown(
        "##### A collection of research papers on Language Models curated by the GPT maestro itself."
    )
    ## Humorous and poetic introduction.
    st.markdown(
        "Every week hundreds of papers are published on Language Models. It is impossible to keep up with the latest research. "
        "That's why we created LLMpedia, a collection of papers on Language Models curated by the GPT maestro itself.\n\n"
        "Each week GPT-4 will sweep through the latest LLM related papers and select the most interesting ones. "
        "The maestro will then summarize the papers and provide his own analysis, including a novelty, technical depth and readability score. "
        "We hope you enjoy this collection and find it useful. Leave a comment on the sidebar if you have any feedback or suggestions.\n\n"
        "*Bonne lecture!*"
    )

    ## Main content.
    papers_df = load_data()

    ## Filter sidebar.
    st.sidebar.markdown("# 📁 Filters")
    search_term = st.sidebar.text_input("Search Term", "")
    categories = st.sidebar.multiselect(
        "Categories",
        list(papers_df["category"].unique()),
    )
    cluster = st.sidebar.multiselect(
        "Cluster Group",
        list(papers_df["topic"].unique()),
    )

    ## Calendar selector.
    published_df = generate_calendar_df(papers_df)
    heatmap_data = prepare_calendar_data(published_df, 2023)

    release_calendar = plot_activity_map(heatmap_data)
    st.markdown("### 📅 2023 Release Calendar")
    calendar_select = plotly_events(release_calendar, override_height=200)

    ## Cluster map
    with st.expander("📊 Cluster Map"):
        cluster_map = plot_cluster_map(papers_df)
        st.plotly_chart(cluster_map, use_container_width=True)

    st.markdown("---")

    # papers = papers_df.to_dict("records")

    ## Search terms.
    if len(search_term) > 0:
        papers_df = papers_df[
            papers_df["Title"].str.lower().str.contains(search_term)
            | papers_df["Summary"].str.lower().str.contains(search_term)
            | papers_df["main_contribution"].map(
                lambda l: search_term.lower() in l["description"].lower()
            )
            | papers_df["takeaways"].map(
                lambda l: search_term.lower() in l["description"].lower()
            )
        ]

    ## Categories.
    if len(categories) > 0:
        papers_df = papers_df[papers_df["category"].isin(categories)]

    ## Cluster.
    if len(cluster) > 0:
        papers_df = papers_df[papers_df["topic"].isin(cluster)]

    ## Published date.
    if len(calendar_select) > 0:
        week_num = calendar_select[0]["x"]
        weekday = calendar_select[0]["y"]
        week_num = int(week_num[1:])
        weekday = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"].index(weekday)

        publish_date = heatmap_data[
            (heatmap_data["week"] == week_num) & (heatmap_data["weekday"] == weekday)
        ]["Published"].values[0]
        publish_date = pd.to_datetime(publish_date)

        if len(papers_df[papers_df["Published"] == publish_date]) > 0:
            papers_df = papers_df[papers_df["Published"] == publish_date]
            ## Add option to clear filter on sidebar.
            if st.sidebar.button(
                f"📅 **Publish Date Filter:** {publish_date.strftime('%B %d, %Y')}"
            ):
                st.experimental_rerun()

    if len(papers_df) == 0:
        st.markdown("No papers found.")
        return

    papers = papers_df.to_dict("records")

    items_per_page = 5
    num_pages = len(papers) // items_per_page
    if len(papers) % items_per_page:
        num_pages += 1

    prev_button, _, next_button = st.columns((1, 10, 1))
    if prev_button.button("Prev"):
        st.session_state.page_number = max(0, st.session_state.page_number - 1)
    if next_button.button("Next"):
        st.session_state.page_number = min(num_pages - 1, st.session_state.page_number + 1)

    # Display the page number
    st.markdown(f"**Pg. {st.session_state.page_number + 1} of {num_pages}**")

    # Get the indices of the items for the current page
    start_index = st.session_state.page_number * items_per_page
    end_index = start_index + items_per_page

    # Display items for the current page
    for paper in papers[start_index:end_index]:
        create_paper_card(paper)


if __name__ == "__main__":
    main()
