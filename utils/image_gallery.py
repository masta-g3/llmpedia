import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import time
import json
import os

st.set_page_config(
    layout="wide",
    page_title="LLM Card Selector",
    page_icon="🌆",
    initial_sidebar_state="expanded",
)

versions = ["v1", "v2", "v3", "v4", "v5", "v6"]


def load_data():
    with open("../arxiv_code_map.json", "r") as f:
        arxiv_map = json.load(f)

    img_scores = pd.read_pickle("../data/img_scores.pkl")
    df = pd.DataFrame.from_dict(arxiv_map, orient="index", columns=["title"])
    df = df.merge(img_scores, how="left", left_index=True, right_index=True)
    ## Find version with higest score and assign it as preferred.
    df["preferred_version"] = df[versions].idxmax(axis=1, skipna=True)
    df.index.name = "arxiv_code"
    df.index = df.index.map(str)
    df.reset_index(inplace=True)
    return df


def ensure_column_exists(df):
    if "preferred_version" not in df.columns:
        df["preferred_version"] = [None] * len(df)
    return df


def generate_heatmap_data(df):
    heatmap_data = pd.DataFrame(0, index=df.index, columns=versions)

    for idx, row in df.iterrows():
        if pd.notna(row["preferred_version"]):
            heatmap_data.at[idx, row["preferred_version"]] = 1
    return heatmap_data


if st.session_state.get("df", None) is None:
    df = load_data()
    st.session_state["df"] = df


def main():
    items_per_page = 20
    page_cols = st.columns(3)
    page = page_cols[1].number_input(
        "Choose a page",
        min_value=1,
        max_value=(len(st.session_state["df"]) // items_per_page) + 1,
        value=1,
        step=1,
    )

    heatmap_data = generate_heatmap_data(st.session_state["df"])
    fig = go.Figure(
        data=go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns,
            y=heatmap_data.index,
            showscale=False,
            colorscale=[[0, "#282434"], [1, "#ff7f0e"]],
        )
    )
    fig.update_layout(
        height=500,
        margin=dict(l=0, r=20, t=0, b=0),
    )
    st.sidebar.plotly_chart(fig, use_container_width=True)

    start_idx = (page - 1) * items_per_page
    end_idx = start_idx + items_per_page
    sub_df = st.session_state["df"].iloc[start_idx:end_idx]

    for index, row in sub_df.iterrows():
        item_name = row["arxiv_code"]
        title = row["title"]
        preferred = row["preferred_version"]
        pref_score = row[preferred] if pd.notna(preferred) else None

        st.write(f"### Item: {item_name} - {title}")
        # st.write(f"### AXV-{item_name}")

        selected_version = st.selectbox(
            label=f"Select version for {item_name}",
            label_visibility="collapsed",
            options=[None] + versions,
            index=0 if pd.isna(preferred) else 1+versions.index(preferred),
        )

        st.session_state["df"].at[index, "preferred_version"] = selected_version

        image_cols = st.columns(6)
        for idx, ver in enumerate(["v1", "v2", "v3", "v4", "v5", "v6"]):
            img_path = f"llm_cards_{ver}/{item_name}.png"
            if os.path.exists(img_path):
                image_cols[idx].image(
                    img_path, caption=f"{ver}", use_column_width=True
                )
                selected = selected_version == ver
                if not selected:
                    image_cols[idx].caption(f"Score: {row[ver]:.2f}")
                else:
                    image_cols[idx].caption(f"**Score: {row[ver]:.2f} (Preferred)**")



    if st.sidebar.button("Save Preferences"):
        st.session_state["df"].to_pickle("content.pkl")
        st.sidebar.success("Saved preferences to content.pkl!")
        time.sleep(1)
        st.experimental_rerun()

    if st.sidebar.button("Load Preferences"):
        st.session_state["df"] = pd.read_pickle("../data/content.pkl")
        st.sidebar.success("Loaded preferences from content.pkl!")
        time.sleep(1)
        st.experimental_rerun()

if __name__ == "__main__":
    main()
