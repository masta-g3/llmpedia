# Repository Structure

```
llmpedia/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Core production dependencies
├── requirements_dev.txt      # Development dependencies
├── Dockerfile               # Container configuration
├── docker-compose.yml       # Container orchestration
├── workflow.sh              # Main workflow execution script
├── tweet_collector.sh       # Tweet collection script
├── update_and_restart.sh    # Deployment update script
├── daily_update.sh          # Daily update automation script
├── weekly_review.sh         # Weekly review automation script
│
├── config/                  # Configuration files
│   └── tweet_types.yaml        # Tweet generation configuration
│
├── workflow/                # Core processing pipeline modules
│   ├── a0_scrape_lists.py      # Twitter list scraping
│   ├── a1_scrape_tweets.py     # Tweet scraping
│   ├── b0_download_paper.py    # Paper download logic
│   ├── b1_download_paper_marker.py  # Paper download tracking
│   ├── c0_fetch_meta.py        # Metadata fetching
│   ├── d0_summarize.py         # Paper summarization
│   ├── e0_narrate.py           # Narrative generation
│   ├── e1_narrate_bullet.py    # Bullet point narratives
│   ├── e2_narrate_punchline.py # Punchline generation
│   ├── f0_review.py            # Review generation
│   ├── g0_create_thumbnail.py  # Thumbnail creation
│   ├── h0_citations.py         # Citation processing
│   ├── i0_generate_embeddings.py  # Embedding generation
│   ├── i1_topic_model.py       # Topic modeling
│   ├── i2_similar_docs.py      # Document similarity
│   ├── i3_topic_map.py         # Topic mapping
│   ├── j0_doc_chunker.py       # Document chunking
│   ├── k0_rag_embedder.py      # RAG embedding
│   ├── l0_abstract_embedder.py # Abstract embedding
│   ├── m0_page_extractor.py    # Page extraction
│   ├── n0_repo_extractor.py    # Repository extraction
│   ├── t0_analyze_tweets.py    # Tweet analysis
│   ├── z0_daily_update.py      # Daily update workflow
│   └── z2_generate_tweet.py    # Configuration-driven tweet generation
│
├── utils/                   # Utility modules and helpers
│   ├── app_utils.py           # Application utilities
│   ├── db/                    # Database operations modules
│   │   ├── __init__.py          # Package initialization
│   │   ├── db.py                # Re-exports for backward compatibility
│   │   ├── db_utils.py          # Core database utilities
│   │   ├── paper_db.py          # Paper-related operations
│   │   ├── tweet_db.py          # Tweet-related operations
│   │   ├── embedding_db.py      # Embedding-related operations
│   │   └── logging_db.py        # Logging-related operations
│   ├── prompts.py             # LLM prompt templates
│   ├── vector_store.py        # Vector storage operations
│   ├── paper_utils.py         # Paper processing utilities
│   ├── tweet.py               # Tweet processing utilities
│   ├── streamlit_utils.py     # Streamlit UI utilities
│   ├── pydantic_objects.py    # Data models
│   ├── plots.py               # Visualization utilities
│   ├── styling.py             # UI styling
│   ├── logging_utils.py       # Logging configuration
│   ├── notifications.py       # Notification system
│   └── models.py              # ML model utilities
│
├── executors/               # Task execution modules
├── notebooks/               # Jupyter notebooks for analysis
├── sql/                     # SQL scripts and schemas
├── data/                    # Data storage
├── artifacts/              # Generated artifacts
├── logs/                   # Application logs
│
├── .streamlit/             # Streamlit configuration
├── .env                    # Environment variables
├── .gitignore             # Git ignore rules
├── README.md              # Project documentation
├── VERSIONS.md            # Version history
└── planning.md            # Development planning
```

## Directory Overview

### Core Components
- `app.py`: Main Streamlit application serving the web interface
- `workflow/`: Contains the sequential processing pipeline modules (a0 → z2)
- `utils/`: Shared utility functions and helper modules
- `executors/`: Task execution and scheduling modules

### Data and Storage
- `data/`: Raw and processed data storage
- `artifacts/`: Generated outputs and artifacts
- `logs/`: Application and process logs
- `sql/`: Database schemas and SQL scripts

### Configuration
- `.env`: Environment variables and configuration
- `.streamlit/`: Streamlit-specific configuration
- `requirements.txt`: Production dependencies
- `requirements_dev.txt`: Development dependencies

### Documentation
- `README.md`: Project overview and setup instructions
- `VERSIONS.md`: Version history and changelog
- `planning.md`: Development planning and tracking
- `notebooks/`: Analysis and development notebooks

### Deployment
- `Dockerfile`: Container image definition
- `docker-compose.yml`: Container orchestration
- `update_and_restart.sh`: Deployment update script
- `workflow.sh`: Main workflow execution script
- `tweet_collector.sh`: Tweet collection automation
- `daily_update.sh`: Shell script that runs daily updates at 7 PM PST/PDT, with progress tracking and logging functionality
- `weekly_review.sh`: Shell script that runs weekly reviews every Monday at 2:00 PM PST/PDT, using the previous Monday's date as input

## Utils Directory (`utils/`)

### Database Operations (`utils/db/`)
- `db.py`: Re-exports database functions from specialized modules for backward compatibility
- `db_utils.py`: Core database utilities and helper functions
  - Connection management
  - Query execution
  - Common SQL operations
- `paper_db.py`: Paper-related database operations
  - Loading paper details
  - Managing summaries and topics
  - Handling citations and repositories
- `tweet_db.py`: Tweet-related database operations
  - Storing and reading tweets
  - Managing tweet analyses and replies
- `embedding_db.py`: Embedding-related database operations
  - Storing and loading embeddings
  - Managing embedding dimensions
- `logging_db.py`: Logging-related database operations
  - Token usage tracking
  - Error logging
  - Q&A and visit logging
  - Workflow execution logging

### Vector Store Operations (`utils/vector_store.py`)
- `vector_store.py`: Vector storage operations
  - Connection management
  - Embedding operations
  - LLM query functions
  - Tweet generation and reply functions:
    - `select_tweet_reply`: Selects a tweet to reply to and determines the appropriate response type
    - `write_tweet_reply_academic`: Generates a technical response based on academic papers
    - `write_tweet_reply_funny`: Generates a humorous, light-hearted response
    - `write_tweet_reply_commonsense`: Generates a response based on common-sense insights

### Paper Processing (`utils/paper_utils.py`)
- `paper_utils.py`: Paper processing utilities
