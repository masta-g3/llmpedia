#!/bin/bash

set -e  # Exit immediately if any command fails
set -a
source /app/.env
set +a

PROJECT_PATH=${PROJECT_PATH:-/app}
LOG_FILE="${PROJECT_PATH}/workflow.log"

echo "Workflow started at $(date)" | tee -a "$LOG_FILE"

function run_step() {
    local step_name="$1"
    local script="$2"
    echo ">> [$step_name]" | tee -a "$LOG_FILE"
    python "${PROJECT_PATH}/${script}" 2>&1 | tee -a "$LOG_FILE"
}

run_step "0: Web Scraper" "workflow/a0_scrape_lists.py"
run_step "1 Tweet Scraper" "workflow/a1_scrape_tweets.py"
run_step "1: Document Fetcher" "workflow/b0_download_paper.py"
run_step "2: Meta-Data Collect" "workflow/c0_fetch_meta.py"
run_step "3: Summarizer" "workflow/d0_summarize.py"
run_step "4: Narrator" "workflow/e0_narrate.py"
run_step "4.1: Bullet List" "workflow/e1_narrate_bullet.py"
# run_step "4.2: Data Card" "workflow/e2_data_card.py" # BY DEMAND
run_step "5: Reviewer" "workflow/f0_review.py"
# run_step "6: Visual Artist" "workflow/g0_create_thumbnail.py" # HIGH MEMORY
run_step "7: Scholar" "workflow/h0_citations.py"
# run_step "8: Topic Model" "workflow/i0_topic_model.py" # HIGH MEMORY
run_step "8.1: Similar Documents" "workflow/i1_similar_docs.py"
# run_step "9: Document Chunker" "workflow/j0_doc_chunker.py" # DEPRECATED
# run_step "10: Document Embedder" "workflow/k0_rag_embedder.py" # DEPRECATED
run_step "11: Abstract Embedder" "workflow/l0_abstract_embedder.py"
run_step "12: Page Extractor" "workflow/m0_page_extractor.py"
run_step "13:  Repo Extractor" "workflow/n0_repo_extractor.py"
run_step "14: GIST Updater" "workflow/z0_update_gist.py"
#run_step "15: Generate tweet" "workflow/z1_generate_tweet.py" # REVIEW

echo "Done! Please enjoy the rest of your day and spread love around your neighbourhood."
