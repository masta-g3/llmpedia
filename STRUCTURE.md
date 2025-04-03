# LLMpedia Project Structure

## Main Files

- `app.py`: The main Streamlit application file that contains the UI and application logic
- `README.md`: Project documentation and setup instructions
- `requirements.txt`: Python dependencies for the project
- `Procfile`: Configuration for deployment on platforms like Heroku

## Directories

- `utils/`: Utility functions and modules for the application
  - `plots.py`: Visualization functions for creating charts and plots
  - `streamlit_utils.py`: Utility functions for Streamlit UI components
  - `app_utils.py`: General utility functions for the application
  - `styling.py`: CSS and styling functions
  - `db/`: Database-related utilities
    - `db_utils.py`: General database utilities
    - `db.py`: Consolidated database functions for papers, embeddings, and tweets
    - `logging_db.py`: Functions for logging user activity
  - `instruct.py`: LLM instruction/query functions
  - `pydantic_objects.py`: Pydantic models for data validation
  - `prompts.py`: Prompt templates for LLM interactions
- `notebooks/`: Jupyter notebooks for analysis and development
- `components/`: Custom Streamlit components
- `.streamlit/`: Streamlit configuration
- `.vscode/`: VS Code configuration

## Key Features

### Main Application (app.py)

The application is organized into tabs:

1. **📰 Main Page**: Landing page showing recent paper statistics, trending topics, and a featured paper
   - Recent papers (1-day and 7-day statistics)
   - Category distribution visualization
   - Trending topics based on paper titles
   - Featured paper (most cited or recent)
   - Quick navigation buttons

2. **🧮 Release Feed**: Grid or table view of papers with pagination
   - Supports both grid and table views
   - Paginated navigation

3. **🗺️ Statistics & Topics**: Visualizations of publication trends and topics
   - Publication count charts (daily and cumulative)
   - Topic model map for exploring paper clusters

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

- `plot_publication_counts()`: Line/bar chart of papers published
- `plot_activity_map()`: Calendar heatmap of publication activity
- `plot_weekly_activity_ts()`: Time series of weekly publication activity
- `plot_cluster_map()`: Scatter plot of paper topics using UMAP embeddings
- `plot_repos_by_feature()`: Bar chart of repositories by feature
- `plot_category_distribution()`: Horizontal bar chart of paper categories
- `plot_trending_words()`: Horizontal bar chart of trending words in paper titles 