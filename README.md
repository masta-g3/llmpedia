# LLMpedia App

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://llmpedia.streamlit.app)

LLMpedia is a Streamlit application that provides a user-friendly interface to explore and interact with Large Language Model research papers.

## Features

- Browse and search LLM research papers
- View paper summaries, illustrations, and key insights
- Explore topics through interactive visualizations
- Chat with an AI assistant about LLM research
- Access weekly research reviews

## Setup

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Copy `.env.template` to `.env` and configure your environment variables
4. Run the app: `streamlit run app.py`

## Database Configuration

This app requires access to a PostgreSQL database containing LLMpedia data. Configure the database connection in the `.env` file.

## Environment Variables

The following environment variables are required to run the app:

```
# Database Configuration
DB_NAME
DB_HOST
DB_PORT
DB_USER
DB_PASSWORD

# LLM API Keys (for chat functionality)
OPENAI_API_KEY
ANTHROPIC_API_KEY
COHERE_API_KEY
HUGGINGFACE_API_KEY
GROQ_API_KEY

# AWS Configuration (for S3 access)
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
AWS_REGION
```

## Development

See the `notebooks/` directory for analysis and development notebooks related to app visualizations.