# LLMpedia Project Structure

## Main Files

- `app.py`: The main Streamlit application file that contains the UI and application logic
- `deep_research.py`: Multi-agent deep research system with comprehensive workflow logging
- `README.md`: Project documentation and setup instructions
- `requirements.txt`: Python dependencies for the project
- `Procfile`: Configuration for deployment on platforms like Heroku (primarily relevant if deploying outside Streamlit Cloud)
- `.env.template` & `.env`: Environment variable configuration for local development
- `STRUCTURE.md`: This file, describing the project layout
- `PLANNING.md`: Temporary file for planning complex tasks
- `CHANGES.md`: Log of deviations from initial implementation plans

## Directories

- `utils/`: Utility functions and modules for the application
  - `plots.py`: Visualization functions for creating charts and plots (e.g., publication trends, topic maps)
  - `streamlit_utils.py`: Utility functions specific to Streamlit UI components (e.g., custom pagination, styling helpers)
  - `app_utils.py`: General utility functions for the application logic (e.g., data formatting, helper calculations)
  - `styling.py`: CSS and styling functions, including custom CSS injection
  - `db/`: Database-related utilities
    - `db_utils.py`: Lower-level database connection and query execution helpers (consider merging or refactoring if overlap with `db.py`)
    - `db.py`: Consolidated database access functions abstracting queries for papers, embeddings, tweets, repositories, etc. This is the primary interface for fetching data for the UI.
    - `logging_db.py`: Functions specifically for logging user interactions or application events to the database
  - `instruct.py`: Functions for interacting with external LLM APIs (OpenAI, Anthropic, etc.) used in the chat feature
  - `pydantic_objects.py`: Pydantic models used for data validation, especially for data fetched from the database or external APIs.
  - `prompts.py`: Predefined prompt templates used for generating queries or instructions for the LLMs in `instruct.py`.
- `notebooks/`: Jupyter notebooks used for experimentation, data analysis, or one-off tasks during development. Not part of the core application runtime.
- `components/`: Custom Streamlit components
- `deployment/`: Configuration files and guides for deployment
  - `database/`: Database migration scripts and schema updates
    - `add_workflow_logging.sql`: Migration script to add workflow logging columns to qna_logs table
    - `rename_workflow_columns.sql`: Migration script to rename columns to match their actual usage
  - `digitalocean/`: Files specific to DigitalOcean deployment
    - `nginx_streamlit.conf`: Nginx configuration template
    - `streamlit_app.service`: Systemd service file template
    - `DEPLOY_DIGITALOCEAN.md`: Step-by-step deployment guide for DigitalOcean

## Design Philosophy & Styling

### Visual Identity & Aesthetics

LLMpedia follows a **sophisticated academic minimalism** that combines scholarly authority with digital precision. The design system is comprehensively implemented in `utils/styling.py` and follows these core principles:

#### **arXiv-Inspired Color Palette**
- **Primary Brand Color**: arXiv red (`#b31b1b`) with subtle variations (`#c93232` light, `#8f1414` dark)
- **Purpose**: Maintains visual connection to the academic arXiv ecosystem while establishing brand identity
- **Application**: Used for interactive elements, accents, gradients, and call-to-action components

#### **Zen-Like Minimalism**
- **Clean Typography**: System fonts with Orbitron display font for headers, creating subtle pixel art influence
- **Generous Whitespace**: Strategic use of spacing scales (`--space-xs` to `--space-2xl`) for visual breathing room
- **Subtle Interactions**: Gentle hover effects, smooth transitions, and elevated shadows that enhance without overwhelming
- **Card-Based Layout**: Clean, floating card components with soft shadows and gradient backgrounds

#### **Sophisticated Academic Typography**
- **Libertinus Serif Font**: Scholarly, authoritative typeface for headers that conveys academic credibility
- **Geometric Elements**: Clean borders, consistent border radius scale, and structured layouts
- **Digital Precision**: Systematic design tokens and precise spacing that reflect modern web standards

#### **Adaptive Design System**
- **CSS Custom Properties**: Centralized design tokens for consistency across components
- **Dark/Light Mode Support**: Automatic theme adaptation respecting user preferences
- **Responsive Design**: Mobile-first approach with fluid layouts and adaptive components
- **Component Library**: Reusable patterns for cards, buttons, badges, and interactive elements

### Technical Implementation

The styling system (`utils/styling.py`) provides:

- **`apply_complete_app_styles()`**: Master function that applies the entire design system
- **Design Tokens**: CSS custom properties defining colors, typography, spacing, and effects
- **Component Patterns**: Reusable styles for cards, buttons, tables, and specialized components
- **Theme Adaptations**: Automatic dark mode support with enhanced contrast and accessibility
- **Animation System**: Consistent transitions and micro-interactions that feel responsive yet subtle

### Style Guidelines

1. **Minimalist First**: Every element serves a purpose; decorative elements are meaningful and subtle
2. **Academic Elegance**: Professional appearance suitable for research content with modern web sensibilities  
3. **Consistent Interactions**: Predictable hover states, transitions, and feedback across all components
4. **Accessibility**: High contrast ratios, readable typography, and keyboard navigation support
5. **Performance**: Lightweight CSS with efficient selectors and minimal reflows

The design creates an environment that feels both **academically credible** and **digitally precise**, allowing users to focus on research content while enjoying a polished, systematic interface that reflects the rigor and clarity of academic scholarship.

## Key Features

### Main Application (app.py)

The application is organized into tabs:

1. **📰 News**: Landing page showing recent paper statistics, trending topics, and platform discussions
   - Recent papers (1-day and 7-day statistics)
   - Top cited papers vs trending papers toggle panel
   - **X.com/Reddit discussions toggle**: Switch between X.com and Reddit LLM discussion summaries
   - Featured paper (weekly highlight)
   - Interesting facts and feature poll
   - Quick navigation to Online Research

2. **🧮 Release Feed**: Grid or table view of papers with pagination
   - Supports both grid and table views
   - Paginated navigation

3. **🗺️ Statistics & Topics**: Interactive visualizations of publication trends and research topics
   - **Publication Trends**: Enhanced charts with dual-mode visualization (Total Volume/By Topics, Daily/Cumulative)
   - **Research Topic Map**: Interactive UMAP-based clustering visualization for exploring paper relationships
   - Consistent header styling and improved UX with centered controls and better visual hierarchy

4. **🔍 Paper Details**: Detailed view of a specific paper
   - Search by arXiv code
   - Detailed paper information and summaries

5. **🤖 Online Research**: AI chat interface for querying paper data
   - Natural language questions about LLM research
   - Response settings for customization
   - Source citations and references

6. **⚙️ Links & Repositories**: Related repositories and resources
   - Filterable list of repositories
   - Visualization of repositories by category

7. **🗞 Weekly Report**: Weekly summaries of LLM research
   - Activity visualization for the selected week
   - Summaries of key developments
   - Highlight of the week

### Visualization (utils/plots.py)

- `plot_publication_counts()`: Line/bar chart of total papers published over time
- `plot_publication_counts_by_topics()`: Enhanced chart showing publication counts grouped by research topics with top 10 filtering
- `plot_activity_map()`: Calendar heatmap of publication activity
- `plot_weekly_activity_ts()`: Time series of weekly publication activity
- `plot_cluster_map()`: Scatter plot of paper topics using UMAP embeddings
- `plot_repos_by_feature()`: Bar chart of repositories by feature
- `plot_category_distribution()`: Horizontal bar chart of paper categories
- `plot_trending_words()`: Horizontal bar chart of trending words in paper titles
- `plot_top_topics()`: Pie chart of the most popular research topics

## Data Flow Overview

The core data (research papers, metadata) originates from arXiv and is processed by a separate system detailed in the [llmpedia_workflows repository](https://github.com/masta-g3/llmpedia_workflows). This workflow handles:
- Fetching new papers.
- Generating AI-derived content (summaries, artwork, insights).
- Performing topic modeling (e.g., using UMAP) and generating embeddings.
- Storing the raw and processed data into a PostgreSQL database.

This LLMpedia application primarily reads from that pre-populated database to display information through the Streamlit interface.

## Database Structure (Conceptual)

While the exact schema resides in the database managed by `llmpedia_workflows`, the application primarily interacts with tables representing:
- **Papers**: Core metadata (title, authors, abstract, arXiv ID, publication date), derived content (summaries, keywords), and links to generated assets (artwork).
- **Embeddings**: Vector representations of papers used for topic modeling and similarity searches.
- **Topics**: Clusters or categories derived from paper embeddings.
- **Repositories**: Information about related code repositories.
- **Tweets**: Relevant tweets associated with papers or topics (if applicable).
- **Logs**: Records of user activity or application events

The `utils/db/db.py` module provides functions to query these conceptual entities.