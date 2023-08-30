[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://llmpedia.streamlit.app)

# LLMpedia
A streamlit app for keeping up with LLM related research.

## Workflow
1. New paper title gets added to https://gist.github.com/masta-g3/8f7227397b1053b42e727bbd6abf1d2e.
2. Paper meta-data and content is fetched via the `arxiv` library and langchain's `ArxivLoader`.
3. GPT-4 runs reading and summarization process over paper content, generating template of output review.
4. BERTopic model is run over full paper set to generate topic groups and labels.
5. SDXL pipeline runs to generate paper thumbnail.
6. Streamlit app is updated and deployed with new content.

## Dev Dependencies
- ComfyUI (https://github.com/comfyanonymous/ComfyUI)
- And all under `dev_requirements.txt`.