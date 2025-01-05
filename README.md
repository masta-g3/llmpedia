[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://llmpedia.streamlit.app)

# LLMpedia
A Streamlit app for keeping up with LLM-related research. By maintaining an automated workflow and leveraging multiple language models, LLMpedia enables continuous discovery, analysis, and summarization of new publications.

Up-to-date coverage: https://gist.github.com/masta-g3/8f7227397b1053b42e727bbd6abf1d2e

## Overview
LLMpedia tracks newly uploaded or recently revised papers relevant to Large Language Models (LLMs). It automatically:
- Retrieves paper titles from a provided gist.
- Fetches meta-data and content using the arxiv API or other data sources.
- Summarizes, reviews, and highlights key points using a variety of LLM pipelines.
- Uncovers topics using topic modeling techniques (e.g., BERTopic).
- Generates thumbnails or visual content for each paper.
- Produces weekly research reviews and social media content.
- Tracks citations and community engagement.

The end result is a cohesive research dashboard that is easily accessible via Streamlit.

## Features
Below are some of the key features of LLMpedia. This is not an exhaustive list, as the project includes many more capabilities and is continuously evolving.

### Multi-Modal Analysis
- **Multiple LLM Providers**: Leverages various LLM endpoints (OpenAI, Anthropic, Groq, etc.) for diverse analysis perspectives
- **Quality Verification**: Automated checks to verify LLM-relevance and content quality
- **Visual Content**: Generates paper thumbnails using RetrodiffusionAI's pixel art API

### Comprehensive Summaries
- **Multiple Formats**: Generates narrative summaries, bullet points, and single-sentence punchlines
- **Weekly Reviews**: Produces comprehensive weekly research reviews with trending topics
- **Citation Tracking**: Monitors paper impact and citation networks

### Community Integration
- **Social Media**: Automated generation and sharing of paper insights on social platforms
- **Community Engagement**: Tracks and analyzes community discussions around papers
- **Repository Collection**: Aggregates and organizes related code repositories and resources

## How it Works
1. Title Gathering  
   - New paper titles are stored or updated in a gist (refer to https://gist.github.com/masta-g3/1dd189493c1890df6e04aaea6d049643).

2. Automated Pipeline (workflow.sh)  
   - The shell script serves as the orchestration layer.  
   - It runs multiple steps in sequence (a0_scrape_lists → a1_scrape_tweets → b0_download_paper → … → z1_generate_tweet).  
   - Each step is logged in a timestamped log file, allowing easy monitoring and debugging.  
   - After completing a cycle, the script sleeps for a random duration before starting again, thereby ensuring continuous updates without overloading services.

3. Data Storage  
   - A database (PostgreSQL by default) keeps track of all metadata, paper content, summaries, and more.  
   - Each step updates or retrieves data from the database.

4. Summaries and Reviews from LLMs  
   - The scripts in vector_store.py and other supporting modules handle text splitting, summarization, bullet-point extraction, and more.  
   - They use different LLM endpoints to generate concise overviews, convert them to bullet-point lists, or produce copywritten narratives.  
   - Additional prompts check if papers are genuinely about LLMs or relevant subfields.

5. Topic Modeling  
   - BERTopic (or other topic modeling libraries) clusters papers to identify key research areas and trends. Results are displayed in the Streamlit app.

6. Streaming the Results  
   - The final results (summaries, topics, and metadata) are published automatically to the Streamlit app, providing a near real-time overview of new LLM research.

## Workflow
Everything is automated and stored in a DB.  
1. New paper title gets added to https://gist.github.com/masta-g3/1dd189493c1890df6e04aaea6d049643.  
2. Paper meta-data and content are fetched via the `arxiv` library.  
3. LLM runs read and summarization processes over paper content, generating a template of output review.  
4. BERTopic model is run over the full paper set to generate topic groups and labels.  
5. Paper thumbnail is generated using RetrodiffusionAI's pixel art API.  
6. Streamlit app is updated and deployed with new content.

## Dev Dependencies
- All dependencies under `dev_requirements.txt`.  
- AWS S3 storage credentials.

## Environment Variables
The following environment variables are required to run the app (API keys + DB config):
```
# LLM API Keys
OPENAI_API_KEY
COHERE_API_KEY
TOGETHER_API_KEY
ANTHROPIC_API_KEY
HUGGINGFACE_API_KEY
GROQ_API_KEY

# Image Generation
RD_API_KEY

# Database Configuration
DB_NAME
DB_HOST
DB_PORT
DB_USER
DB_PASS

# External Services
SEMANTIC_SCHOLAR_API_KEY
GITHUB_TOKEN

# Optional: Twitter Integration
TWITTER_EMAIL
TWITTER_PASSWORD
TWITTER_PHONE

# Optional: Email Notifications
SENDER_EMAIL
SENDER_EMAIL_PASSWORD
RECEIVER_EMAIL

# Optional: Project Configuration
PROJECT_PATH
```

A populated database is also required to run the app; instructions for setting it up coming soon.