#!/bin/bash
echo ">> [0] Scraping paper list..."
python workflow/a0_scrape_lists.py
echo ">> [1] Downloading papers.."
python workflow/b0_download_paper.py
echo ">> [2] Fetching meta..."
python workflow/c0_fetch_meta.py
echo ">> [3] Summarizing..."
python workflow/d0_summarize.py
echo ">> [4] Narrating..."
python workflow/e0_narrate.py
echo ">> [5] Reviewing..."
python workflow/f0_review.py
echo ">> [6] Creating thumbnails..."
python workflow/g0_create_thumbnail.py
echo ">> [7] Citations..."
python workflow/h0_citations.py
echo ">> [8] Running topic model..."
python workflow/i0_topic_model.py
echo ">> [9] Chunking documents..."
python workflow/j0_doc_chunker.py
echo ">> [10] Creating embeddings..."
python workflow/k0_rag_embedder.py
echo ">> [11] Updating gist..."
python workflow/l0_update_gist.py

echo "Done! Please enjoy the rest of your day and spread love around your neighbourhood."