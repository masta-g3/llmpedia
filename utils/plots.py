import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import pandas as pd
import numpy as np
import colorcet as cc
import datetime
from typing import Tuple

pio.templates.default = "plotly"


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
            color_discrete_sequence=["#b31b1b"],
        )
    else:
        fig = px.bar(
            df,
            x="published",
            y="Count",
            color_discrete_sequence=["#b31b1b"],
        )
    fig.update_xaxes(title=None, tickfont=dict(size=17))
    fig.update_yaxes(titlefont=dict(size=18), tickfont=dict(size=17))

    return fig


def plot_activity_map(df_year: pd.DataFrame) -> Tuple[go.Figure, pd.DataFrame]:
    """Creates a calendar heatmap plot along with corresponding map of dates in a DF."""
    colors = ["#b31b1b", "#c93232", "#e04848", "#e87070", "#f09898", "#f8c0c0"]

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
            z=padded_count.values,
            x=padded_date.iloc[0].values,
            y=["Sun", "Sat", "Fri", "Thu", "Wed", "Tue", "Mon"],
            hoverongaps=False,
            hovertext=padded_date.values,
            hovertemplate="%{hovertext}<extra>Count: %{z}</extra>",
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


def plot_weekly_activity_ts(
    df: pd.DataFrame, date_report: datetime.date = None
) -> go.Figure:
    """Calculate weekly activity and plot a time series."""
    df = df.copy()
    df["published"] = pd.to_datetime(df["published"])
    year_range = df["published"].dt.year.unique()
    date_format = "%b %d, %y" if len(year_range) > 1 else "%b %d"
    df = df.sort_values("published")
    df["week_start"] = df["published"].dt.to_period("W").apply(lambda r: r.start_time)
    df = df.groupby(["week_start"])["Count"].sum().reset_index()
    df["publish_str"] = df["week_start"].dt.strftime(date_format)

    highlight_date_str = date_report.strftime(date_format)

    fig = px.area(
        df,
        x="publish_str",
        y="Count",
        title=None,
        labels={"title": "Papers Published"},
        height=250,
        color_discrete_sequence=["#b31b1b"],
    )
    fig.update_xaxes(title=None, tickfont=dict(size=17))
    fig.update_yaxes(titlefont=dict(size=18), tickfont=dict(size=17))
    fig.add_vline(x=highlight_date_str, line_width=2, line_dash="dash", line_color="#c93232")
    fig.update_layout(
        margin=dict(t=0, b=0, l=0, r=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis_title="# Published",
    )

    bar_height = df[df["publish_str"] == highlight_date_str]["Count"]
    if len(bar_height) > 0:
        bar_height = bar_height.values[0]
    else:
        bar_height = 0
    fig.add_trace(
        go.Scatter(
            x=[highlight_date_str],
            y=[bar_height],
            mode="markers",
            showlegend=False,
            marker=dict(size=20, color="#b31b1b"),
        )
    )
    return fig


def plot_cluster_map(df: pd.DataFrame) -> go.Figure:
    """Creates a scatter plot of the UMAP embeddings of the papers."""
    # Calculate marker size based on number of points
    n_points = len(df)
    marker_size = min(20, max(4, int(400 / n_points)))  # Size between 4 and 20, inverse to number of points
    
    # Create base contour plot
    fig = go.Figure()
    
    ## Add density contours.
    fig.add_trace(go.Histogram2dContour(
        x=df["dim1"],
        y=df["dim2"],
        colorscale="Blues",
        showscale=False,
        ncontours=20,
        contours=dict(coloring="fill"),
        opacity=0.2,
    ))
    
    ## Add scatter points on top.
    for topic in df["topic"].unique():
        mask = df["topic"] == topic
        fig.add_trace(go.Scatter(
            x=df[mask]["dim1"],
            y=df[mask]["dim2"],
            mode="markers",
            name=topic,
            marker=dict(
                size=marker_size,
                line=dict(width=0.5, color="Black"),
            ),
            hovertemplate="<b>%{customdata}</b><extra></extra>",
            customdata=df[mask]["title"],
        ))
    
    fig.update_layout(
        legend=dict(
            title=None,
            font=dict(size=14),
        ),
        margin=dict(t=0, b=0, l=0, r=0),
        xaxis=dict(showgrid=True, gridwidth=0.1),
        yaxis=dict(showgrid=True, gridwidth=0.1)
    )
    fig.update_xaxes(title_text=None)
    fig.update_yaxes(title_text=None)
    return fig


def plot_repos_by_feature(
    df: pd.DataFrame, plot_by: str, max_chars: int = 30
) -> go.Figure:
    """Plot bar chart of repositories by a feature."""
    count_df = df.groupby(plot_by).count()[["repo_title"]].reset_index()

    if plot_by != "published":
        count_df[plot_by] = np.where(
            count_df["repo_title"] < 10, "Other", count_df[plot_by]
        )
        count_df = count_df.sort_values("repo_title", ascending=False)
        count_df["topic_label"] = count_df[plot_by].apply(
            lambda x: (x[:max_chars] + "...") if len(x) > max_chars else x
        )
    else:
        count_df[plot_by] = pd.to_datetime(count_df[plot_by])
        count_df[plot_by] = (
            count_df[plot_by]
            .dt.tz_localize(None)
            .dt.to_period("W")
            .apply(lambda r: r.start_time)
        )
        count_df = count_df.groupby(plot_by).sum().reset_index()
        count_df = count_df.sort_values(plot_by, ascending=True)
        count_df["topic_label"] = count_df[plot_by].dt.strftime("%b %d")

    fig = px.bar(count_df, x=plot_by, y="repo_title", title=None, hover_data=[plot_by])
    fig.update_xaxes(title=None, tickfont=dict(size=13), tickangle=75)
    fig.update_yaxes(titlefont=dict(size=14), title="# Resources")
    fig.update_traces(marker_color="#b31b1b", marker_line_color="#c93232")
    return fig
