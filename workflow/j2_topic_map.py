import sys, os
from dotenv import load_dotenv

load_dotenv()

PROJECT_PATH = os.getenv("PROJECT_PATH", "/app")
sys.path.append(PROJECT_PATH)

os.chdir(PROJECT_PATH)

import warnings
warnings.filterwarnings('ignore', category=Warning)

import numpy as np
import pandas as pd
import datamapplot
import seaborn as sns
from matplotlib.colors import rgb2hex
from pathlib import Path

import utils.db as db
from utils.logging_utils import setup_logger

# Set up logging
logger = setup_logger(__name__, "j2_topic_map.log")


def create_topic_map(topics_df: pd.DataFrame, citations_df: pd.DataFrame, arxiv_df: pd.DataFrame) -> datamapplot.interactive_rendering.InteractiveFigure:
    """Create interactive topic map visualization."""
    logger.info("Creating topic map visualization...")
    
    # Prepare data
    arxiv_ids = topics_df.index.tolist()
    topics_df["title"] = arxiv_df.loc[arxiv_ids, "title"]
    embeddings = topics_df[["dim1", "dim2"]].to_numpy()
    labels = topics_df["topic"].tolist()
    titles = topics_df["title"].tolist()
    citation_values = np.array([citations_df.get("citation_count", {}).get(idx, 0) for idx in arxiv_ids])
    marker_sizes = 4 + np.log1p(citation_values) * 1.2
    
    # Get publication dates and citations
    publication_dates = pd.to_datetime(arxiv_df.loc[arxiv_ids, "published"]).values
    citation_values = np.array([citations_df.get("citation_count", {}).get(idx, 0) for idx in arxiv_ids])
    marker_sizes = 4 + np.log1p(citation_values) * 1.2
    
    date_range = (publication_dates.min(), publication_dates.max())
    logger.info(f"Publication date range: {date_range[0]} to {date_range[1]}")
    logger.info(f"Number of papers: {len(publication_dates)}")
    
    # Create color mapping for topics
    unique_labels = sorted(list(set(labels)))
    colors = sns.color_palette("husl", len(unique_labels))
    color_map = {label: rgb2hex(color) for label, color in zip(unique_labels, colors)}
    marker_colors = np.array([color_map[label] for label in labels])

    # Create hover template
    hover_template = """
    <div style="max-width: 500px; font-family: system-ui, -apple-system, sans-serif;">
        <div style="font-size: 14px; font-weight: bold; padding: 4px; color: #2a2a2a;">
            {hover_text}
            <a href="http://llmpedia.streamlit.app/?arxiv_code={arxiv_id}" target="_blank" style="text-decoration: none; margin-left: 6px; color: #666;">🔗</a>
        </div>
        <div style="display: flex; gap: 8px; margin-top: 4px;">
            <div style="background-color: {color}; color: white; border-radius: 4px; padding: 4px 8px; font-size: 12px;">{topic}</div>
            <div style="background-color: #f0f0f0; color: #666; border-radius: 4px; padding: 4px 8px; font-size: 12px;">{citation_count} citations</div>
        </div>
    </div>
    """
    
    logger.info("Configuring interactive plot...")
    # Create visualization
    plot = datamapplot.create_interactive_plot(
        embeddings,
        labels,
        marker_color_array=marker_colors,
        marker_size_array=marker_sizes,
        width=1200,
        height=800,
        darkmode=False,
        enable_search=True,
        noise_color="#aaaaaa22",
        color_label_text=False,
        inline_data=True,
        extra_point_data=pd.DataFrame({
            "hover_text": titles,
            "topic": labels,
            "color": marker_colors,
            "citation_count": citation_values,
            "arxiv_id": arxiv_ids
        }),
        hover_text_html_template=hover_template,
        title="LLM Research Landscape",
        sub_title="A data map of LLM research topics based on ArXiv papers.",
        point_radius_max_pixels=24,
        text_outline_width=4,
        text_min_pixel_size=16,
        text_max_pixel_size=48,
        min_fontsize=16,
        max_fontsize=32,
        font_family="system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
        cluster_boundary_polygons=True,
        color_cluster_boundaries=False,
        initial_zoom_fraction=0.87,
        on_click="window.open(`http://llmpedia.streamlit.app/?arxiv_code={arxiv_id}`)",
        histogram_data=publication_dates,
        histogram_group_datetime_by="quarter",
        histogram_range=(pd.Timestamp("2020-01-01"), pd.Timestamp("2024-12-31"))
    )
    
    return plot


def main():
    """Main function to generate and save the topic map visualization."""
    try:
        logger.info("Starting topic map generation process")
        
        # Load data
        logger.info("Loading required data from database")
        topics_df = db.load_topics()
        citations_df = db.load_citations()
        arxiv_df = db.load_arxiv()
        
        # Create visualization
        plot = create_topic_map(topics_df, citations_df, arxiv_df)
        
        # Save plot
        output_path = Path(PROJECT_PATH) / "artifacts" / "arxiv_cluster_map.html"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving visualization to {output_path}")
        plot.save(str(output_path))
        
        logger.info("Topic map generation completed successfully")
        
    except Exception as e:
        logger.error(f"Error in topic map generation: {str(e)}")
        raise
    

if __name__ == "__main__":
    main() 