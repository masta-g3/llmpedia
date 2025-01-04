[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://llmpedia.streamlit.app)

# LLMpedia
A streamlit app for keeping up with LLM related research.

Up-to-date coverage: https://gist.github.com/masta-g3/8f7227397b1053b42e727bbd6abf1d2e

## Workflow
Everything is automated and stored in a DB. 
1. New paper title gets added to https://gist.github.com/masta-g3/1dd189493c1890df6e04aaea6d049643.
2. Paper meta-data and content are fetched via the `arxiv` library.
3. LLM runs read and summarization process over paper content, generating template of output review.
4. BERTopic model is run over full paper set to generate topic groups and labels.
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