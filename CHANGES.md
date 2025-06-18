# Changes Log

## Enhanced Time Series Visualization (2024-12-XX)

Enhanced the Statistics & Topics tab with improved publication time series visualization:

### Key Improvements
- Added dual-mode view selector: "Total Volume" vs "By Topics" for publication charts
- Implemented topic-grouped time series showing the top 10 most popular research topics
- Maintained existing daily/cumulative toggle functionality for both view modes
- Added intelligent topic grouping with "Others" category for less common topics
- Enhanced visual design with consistent color palette and improved legend styling

### Technical Changes
- Added `plot_publication_counts_by_topics()` function in `utils/plots.py` for topic-based visualization
- Enhanced UI controls with clear help text and improved user guidance
- Implemented smart color assignment with theme-consistent palette and distinct "Others" coloring
- Added proper topic ordering (by popularity) and legend styling for better readability
- Maintained backward compatibility with existing `plot_publication_counts()` function
- Added category ordering to ensure consistent topic display across charts

### User Experience Improvements
- Clean, modern UI controls with helpful tooltips
- Seamless switching between total volume and topic breakdown views
- Visual consistency with existing app theme and color scheme
- Responsive legend placement and styling for optimal chart readability

## Code Refactoring (2024-06-10)

Refactored the trending topics analysis code for better maintainability:

### Key Improvements
- Extracted text analysis code from app.py into reusable utility functions in app_utils.py
- Created a new TEXT ANALYSIS section in app_utils.py with proper documentation
- Generalized the functions to be reusable across the application
- Improved error handling and input validation
- Enhanced documentation with proper docstrings

### Technical Changes
- Added `get_domain_stopwords()` to centralize stopword management
- Added `preprocess_text()` with comprehensive text normalization
- Added `extract_trending_topics()` for general-purpose topic extraction
- Added `get_trending_topics_from_papers()` for convenient paper analysis
- Simplified the main app.py code by calling the new utility functions
- Added proper error handling throughout the text analysis pipeline

## Trending Topics Improvements (2024-06-10)

Enhanced the trending topics analysis on the Main Page:

### Key Improvements
- Replaced simple word counting with TF-IDF vectorization for more meaningful topic extraction
- Added support for bi-grams and tri-grams to capture multi-word phrases
- Implemented proper text preprocessing with NLTK stopwords
- Added lemmatization to normalize word forms and improve topic coherence
- Enhanced hyphenated word handling for better phrase extraction
- Added domain-specific stopword filtering for LLM research, including common multi-word phrases
- Enhanced visualization with better formatting and hover information
- Fixed datatype comparison issues in date filtering

### Technical Changes
- Integrated scikit-learn's TfidfVectorizer for advanced text analysis
- Added NLTK's WordNetLemmatizer for basic lemmatization
- Optimized TF-IDF parameters (sublinear term frequency, smoothed IDF, etc.) for better results
- Enhanced token pattern to filter out single-character words
- Improved handling of hyphenated terms in preprocessing
- Updated `plot_trending_words()` function to handle TF-IDF scores
- Added NLTK and scikit-learn to requirements.txt
- Implemented basic text preprocessing with appropriate cleaning
- Enhanced visualization with dynamic hover templates and formatting

## Main Page Addition (2024-06-10)

Added a new "Main Page" tab to serve as the landing page for the LLMpedia app:

### Key Features
- Added a new "📰 Main Page" tab as the first tab in the application
- Added recent paper statistics (1-day and 7-day counts)
- Added category distribution visualization for recent papers
- Added trending topics analysis based on words in recent paper titles
- Added a featured paper section that displays the most cited or recent paper
- Added quick navigation buttons to other tabs
- Added activity statistics (total papers and yearly counts)

### Technical Changes
- Added new plotting functions in `utils/plots.py`:
  - `plot_category_distribution()`: Horizontal bar chart of paper categories
  - `plot_trending_words()`: Horizontal bar chart of trending words in paper titles
- Updated tab indices in `app.py` to accommodate the new tab
- Created documentation files:
  - `STRUCTURE.md`: Documentation of the project structure
  - `CHANGES.md`: This changelog file

### Future Improvements
- Add more advanced trending topic analysis using NLP techniques
- Add personalized recommendations based on user activity
- Add more interactive elements to the Main Page
- Integrate with external APIs for additional research metrics

## Dashboard Weekly Highlight Integration (2024-06-14)

Enhanced the dashboard with dynamic weekly highlights:

- Added utility function `get_latest_weekly_highlight()` in `app_utils.py` to retrieve the most recent weekly highlight from the database
- Modified `create_featured_paper_card()` in `streamlit_utils.py` to automatically use the weekly highlight in the featured content section
- The weekly highlight now includes:
  - The highlight image from arXiv
  - The highlight title extracted from the content
  - A "Read More" button that navigates to the paper details
  - Fallback mechanisms if no highlight is available
- Improved user experience by showing the date of the weekly highlight
- Added better title extraction using regular expressions

These changes ensure the dashboard always shows fresh, curated content and leverages existing weekly highlights to provide value to users immediately upon application load.

## Dashboard Interesting Facts Integration (2024-06-15)

Enhanced the News tab with interesting research facts:

- Added `get_random_interesting_facts()` function to `utils/db/db_utils.py` to retrieve random facts from the database with a bias toward recent ones
- Added `display_interesting_facts()` function to `streamlit_utils.py` to present facts in a clean, consistent style
- Implemented facts display in the News tab as an expandable section before the Recent Activity section
- Each fact includes:
  - The interesting research finding 
  - A link to the source paper
  - Topic badge matching the paper's topic category
  - Attractive styling with subtle borders and proper spacing
- Facts are cached for 6 hours to minimize database queries
- Added weighting to favor more recent facts while still including diversity
- Maintained consistent code organization by placing database queries in the appropriate db module
- Efficiently reused existing data by leveraging the papers DataFrame already in memory

These changes enhance the educational value of the dashboard by highlighting interesting research findings for users to discover at a glance. 