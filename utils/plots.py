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
    df.columns = ["published", "# Papers Published"]
    df["published"] = pd.to_datetime(df["published"])
    df.sort_values("published", inplace=True)
    df["# Papers Published (Cumulative)"] = df["# Papers Published"].cumsum()
    if cumulative:
        fig = px.area(
            df,
            x="published",
            y="# Papers Published (Cumulative)",
            title=None,
            color_discrete_sequence=["#b31b1b"],
        )
    else:
        fig = px.bar(
            df,
            x="published",
            y="# Papers Published",
            color_discrete_sequence=["#b31b1b"],
        )
    fig.update_xaxes(title=None, tickfont=dict(size=17))
    fig.update_yaxes(title_font=dict(size=18), tickfont=dict(size=17))

    return fig


def plot_activity_map(df_year: pd.DataFrame) -> Tuple[go.Figure, pd.DataFrame]:
    """Creates a calendar heatmap plot using scatter markers instead of a heatmap."""
    colors = ["#f8c0c0", "#f09898", "#e87070", "#e04848", "#c93232", "#b31b1b"]
    
    # Create a colorscale function to map values to colors
    max_count = df_year["Count"].max() if not df_year.empty else 1
    min_count = df_year[df_year["Count"] > 0]["Count"].min() if not df_year.empty else 0
    
    def get_color(count):
        if count == 0:
            return "rgba(240, 240, 240, 0.5)"  # Light gray for zero values
        
        log_count = np.log1p(count - min_count + 1)
        log_max = np.log1p(max_count - min_count + 1)
        
        normalized = (log_count / log_max) ** 2
        color_idx = int(normalized * (len(colors) - 1))
        return colors[color_idx]
    
    week_max_dates = (
        df_year.groupby(df_year["published"].dt.isocalendar().week)["published"]
        .max()
        .dt.strftime("%b %d")
        .tolist()
    )

    # Create pivot tables for counts and dates
    padded_count = df_year.pivot_table(
        index="weekday", columns="week", values="Count", aggfunc="sum"
    ).fillna(0)
    padded_date = df_year.pivot_table(
        index="weekday", columns="week", values="published", aggfunc="last"
    ).fillna(pd.NaT)
    
    # Convert dates to string format
    padded_date = padded_date.applymap(
        lambda x: x.strftime("%b %d") if pd.notna(x) else ""
    )
    
    # Days of the week in display order (top to bottom)
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    
    # Create figure
    fig = go.Figure()
    
    # Add scatter markers to simulate heatmap
    for y_idx, y_val in enumerate(days):
        for x_idx, x_val in enumerate(padded_date.iloc[0].values):
            if x_val:  # Only add points for non-empty dates
                count = int(padded_count.values[y_idx, x_idx])
                date_str = padded_date.values[y_idx, x_idx]
                
                # Store coordinates for selection handling
                coords = f"{y_idx},{x_idx}"
                
                fig.add_trace(
                    go.Scatter(
                        x=[x_val],
                        y=[y_val],
                        mode="markers",
                        marker=dict(
                            size=22,
                            color=get_color(count),
                            symbol="square",
                            line=dict(width=1, color="white"),
                        ),
                        name="",
                        showlegend=False,
                        hovertemplate=f"{date_str}<br>Count: {count}<extra></extra>",
                        text=[coords],
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
        # line_shape="hv"
    )
    fig.update_xaxes(title=None, tickfont=dict(size=17))
    fig.update_yaxes(title_font=dict(size=18), tickfont=dict(size=17))
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
    marker_size = min(20, max(6, int(400 / n_points)))  # Size between 4 and 20, inverse to number of points
    
    # Create base contour plot
    fig = go.Figure()
    
    ## Add density contours - lines only, no fill
    fig.add_trace(go.Histogram2dContour(
        x=df["dim1"],
        y=df["dim2"],
        colorscale=[[0, "rgba(200,200,255,0.3)"], [1, "rgba(100,100,255,0.3)"]],
        showscale=False,
        ncontours=15,
        contours=dict(
            coloring="lines",
            showlabels=False,
            start=0,
            end=1,
            size=0.1,
        ),
        line=dict(width=1),
        opacity=0.4,
    ))
    
    ## Add scatter points on top.
    for topic in df["topic"].unique():
        mask = df["topic"] == topic
        topic_df = df[mask]
        
        # Prepare customdata including title, arxiv_code, published date, topic, and punchline
        customdata = []
        for _, row in topic_df.iterrows():
            custom_item = [
                row["title"],
                row.get("arxiv_code", ""),
                row.get("published", "").strftime("%b %d, %Y") if pd.notna(row.get("published", "")) else "",
                row.get("topic", ""),
                row.get("punchline", "")[:150] + "..." if len(str(row.get("punchline", ""))) > 150 else row.get("punchline", "")
            ]
            customdata.append(custom_item)
        
        fig.add_trace(go.Scatter(
            x=topic_df["dim1"],
            y=topic_df["dim2"],
            mode="markers",
            name=topic,
            marker=dict(
                size=marker_size,
                line=dict(width=0.5, color="Black"),
            ),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br><br>" +
                "<b>Topic:</b> %{customdata[3]}<br>" +
                "<b>Published:</b> %{customdata[2]}<br>" +
                "<b>Summary:</b> %{customdata[4]}<extra></extra>"
            ),
            customdata=customdata,
        ))
    
    fig.update_layout(
        legend=dict(
            title=None,
            font=dict(size=14),
        ),
        margin=dict(t=0, b=0, l=0, r=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=True, gridwidth=0.1, gridcolor="rgba(128,128,128,0.1)"),
        yaxis=dict(showgrid=True, gridwidth=0.1, gridcolor="rgba(128,128,128,0.1)")
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
    fig.update_yaxes(title_font=dict(size=14), title="# Resources")
    fig.update_traces(marker_color="#b31b1b", marker_line_color="#c93232")
    return fig


def plot_category_distribution(categories: pd.Series) -> go.Figure:
    """Creates a horizontal bar chart of paper categories."""
    # Sort by count
    categories = categories.sort_values(ascending=True)
    
    # Create horizontal bar chart
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=categories.index,
            x=categories.values,
            orientation='h',
            marker_color='#b31b1b',
            hovertemplate='%{y}: %{x} papers<extra></extra>'
        )
    )
    
    fig.update_layout(
        height=300,
        margin=dict(t=0, b=0, l=0, r=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Number of Papers",
        yaxis=dict(
            title=None,
            tickfont=dict(size=14)
        )
    )
    
    return fig


def plot_trending_words(trending_words: list) -> go.Figure:
    """Creates a horizontal bar chart of trending words/phrases in paper titles.
    
    The input is expected to be a list of tuples (word/phrase, score) where 
    score can be either a count or a TF-IDF score.
    """
    words = [word for word, score in trending_words]
    scores = [score for word, score in trending_words]
    
    # Capitalize each term for better readability
    formatted_words = [' '.join(word.split()).title() for word in words]
    
    # Create a gradient color scale based on scores
    colors = []
    max_score = max(scores)
    for score in scores:
        # Normalize score between 0.5 and 1.0 for color intensity
        intensity = 0.5 + 0.5 * (score / max_score)
        # Create a shade of red with varying intensity
        colors.append(f'rgba(179, 27, 27, {intensity})')
    
    # Create hover text based on the score type
    if any(isinstance(score, float) and score < 1.0 for _, score in trending_words):
        # This is likely TF-IDF data
        hover_template = '%{y}: TF-IDF Score: %{x:.3f}<extra></extra>'
    else:
        # This is likely count data
        hover_template = '%{y}: %{x} occurrences<extra></extra>'
    
    # Create horizontal bar chart
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=formatted_words,
            x=scores,
            orientation='h',
            marker_color=colors,
            hovertemplate=hover_template
        )
    )
    
    # Determine an appropriate title based on the data
    title = "Trending Phrases (TF-IDF)" if len(words[0].split()) > 1 else "Trending Words"
    
    fig.update_layout(
        height=350,
        margin=dict(t=30, b=0, l=0, r=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Relevance Score",
        yaxis=dict(
            title=None,
            autorange="reversed",  # Show highest score at the top
            tickfont=dict(size=14)
        ),
        title=dict(
            text=title,
            font=dict(size=14),
            x=0.5,
            xanchor='center'
        )
    )
    
    return fig


def plot_top_topics(df: pd.DataFrame, n: int = 5) -> go.Figure:
    """ Creates a pie chart of the top n topics in the dataset. """
    
    topic_counts = df["topic"].value_counts()
    
    if len(topic_counts) > n:
        top_topics = topic_counts.iloc[:n]
        other_count = topic_counts.iloc[n:].sum()
        
        plot_data = pd.Series({**top_topics.to_dict(), "Other": other_count})
    else:
        plot_data = topic_counts
    
    ## Truncate long topic names for display.
    max_label_length = 40
    display_labels = [label[:max_label_length] + "..." if len(label) > max_label_length else label 
                     for label in plot_data.index]
    
    colors = ["#b31b1b", "#c93232", "#e04848", "#e87070", "#f09898", "#f8c0c0"]
    
    fig = go.Figure(data=[go.Pie(
        labels=display_labels,
        values=plot_data.values,
        hole=0.4,  # Create a donut chart
        marker_colors=colors[:len(plot_data)],
        textinfo='label+percent',
        textposition='outside',
        textfont=dict(size=12),
        insidetextorientation='radial',
        pull=[0.05 if i == 0 else 0 for i in range(len(plot_data))],  # Pull out the largest segment slightly
        hoverinfo='label+value',
        hovertemplate='%{label}: %{value} papers<extra></extra>'
    )])
    
    fig.update_layout(
        height=350,
        margin=dict(t=55, b=0, l=0, r=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        title=dict(
            text="Popular Topics",
            font=dict(size=14),
            x=0.5,
            xanchor='center',
            y=0.95
        )
    )
    
    return fig
