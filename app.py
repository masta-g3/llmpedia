from plotly_calplot import calplot
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from typing import Union, Dict, Any, List
import pandas as pd
import json
import os

## Page config.
st.set_page_config(
    layout="wide",
    page_title="📚 LLMPedia",
    page_icon="📚",
    initial_sidebar_state="expanded",
)

def plot_activity_map(df: pd.DataFrame, year: Union[int, str]) -> go.Figure:
    """ Function to generate a Github-style activity heatmap for a given year. """
    df['Published'] = pd.to_datetime(df['Published'])
    df.set_index('Published', inplace=True)
    df_year = df[df.index.year == int(year)].copy()
    first_day_of_year = pd.Timestamp(year=int(year), month=1, day=1)
    df_year['week'] = ((df_year.index - first_day_of_year) / pd.Timedelta(days=7)).astype(int)
    df_year['weekday'] = df_year.index.weekday
    heatmap_data = pd.DataFrame(0, index=pd.MultiIndex.from_product([range(53), range(7)], names=['week', 'weekday']), columns=['Count'])
    for index, row in df_year.iterrows():
        heatmap_data.loc[(row['week'], row['weekday']), 'Count'] = row['Count']
    fig = go.Figure(data=go.Heatmap(z=heatmap_data['Count'].values.reshape(53, 7).T, x=["W" + str(i) for i in range(1, 54)], y=["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"],  showscale=False))
    fig.update_layout(title=f'Activity Map for {year}', height=350)
    return fig

@st.cache_data
def load_data():
    ## ToDo: Replace with DF from DB.
    fnames = os.listdir("summaries")
    result_dict = {}
    for fname in fnames:
        with open(f"summaries/{fname}", "r") as f:
            result_dict[fname] = json.load(f)

    result_df = pd.DataFrame(result_dict).T
    result_df["Published"] = pd.to_datetime(result_df["Published"])
    result_df = result_df[result_df["Published"] > "2021-01-01"]
    result_df = result_df.sort_values("Published", ascending=False)

    ## Remapping with emotion.
    classification_map = {
        "TRAINING": "🏋️‍ TRAINING",
        "FINE-TUNING": "🔧 FINE-TUNING",
        "ARCHITECTURES": "🏗️ ARCHITECTURE",
        "BEHAVIOR": "🔮 BEHAVIOR",
        "PROMPTING": "📣 PROMPTING",
        "USE CASES": "💰 USE CASES",
        "OTHER": "🤷 OTHER",
    }
    result_df["category"] = result_df["category"].apply(lambda x: classification_map[x])
    return result_df

@st.cache_data
def generate_calendar_df(df: pd.DataFrame):
    """ Daily counts of papers."""
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
    """ Creates card UI for paper details. """
    title_cols = st.columns((10, 1))
    title_cols[0].markdown(f"## {paper['Title']}")
    title_cols[1].markdown(f"{paper['category']}")

    date = pd.to_datetime(paper["Published"]).strftime("%B %d, %Y")
    st.markdown(f"#### {date}")
    st.markdown(f"*{paper['Authors']}*")
    with st.expander("Summary"):
        st.markdown(paper["Summary"])
    with st.expander("Main Contribution"):
        st.markdown(f"{paper['main_contribution']}")
    with st.expander("Takeaways"):
        st.markdown(f"{paper['takeaways']}")
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
    papers_df = load_data()
    published_df = generate_calendar_df(papers_df)
    papers = papers_df.to_dict("records")

    ## Search sidebar.
    search_term = st.sidebar.text_input("Search", "")

    ## Category multi-select, for display show only 2 categories max.
    categories = st.sidebar.multiselect(
        "Categories", list(papers_df["category"].unique()),
    )

    ## Calendar selector.
    release_calendar = plot_activity_map(published_df, 2023)
    st.plotly_chart(release_calendar, use_container_width=True)


    for i, paper in enumerate(papers[:10]):
        create_paper_card(paper)


if __name__ == "__main__":
    main()