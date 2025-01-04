#!/bin/bash

set -e  # Exit immediately if any command fails
set -a
source .venv/bin/activate
set +a

PROJECT_PATH=${PROJECT_PATH:-.}

while true; do
    # Create timestamped log file
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    LOG_FILE="${PROJECT_PATH}/logs/workflow_${TIMESTAMP}.log"
    
    echo "Workflow started at $(date)" | tee -a "$LOG_FILE" 2>/dev/null || true

    function run_step() {
        local step_name="$1"
        local script="$2"
        echo ">> [$step_name] Started at $(date)" >> "$LOG_FILE"
        python "${PROJECT_PATH}/${script}" 2>&1 | tee -a "$LOG_FILE"
        echo ">> [$step_name] Completed at $(date)" >> "$LOG_FILE"
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
    run_step "6: Visual Artist" "workflow/g0_create_thumbnail.py"
    run_step "7: Scholar" "workflow/h0_citations.py"
    run_step "8: Topic Model" "workflow/i0_topic_model.py"
    run_step "8.1: Similar Documents" "workflow/i1_similar_docs.py"
    # run_step "9: Document Chunker" "workflow/j0_doc_chunker.py" # DEPRECATED
    # run_step "10: Document Embedder" "workflow/k0_rag_embedder.py" # DEPRECATED
    run_step "11: Abstract Embedder" "workflow/l0_abstract_embedder.py"
    run_step "12: Page Extractor" "workflow/m0_page_extractor.py"
    run_step "13:  Repo Extractor" "workflow/n0_repo_extractor.py"
    run_step "14: GIST Updater" "workflow/z0_update_gist.py"
    run_step "15: Generate tweet" "workflow/z1_generate_tweet.py"

    echo "Cycle completed at $(date)" | tee -a "$LOG_FILE"
    echo "Starting next cycle..."

    sleep_minutes=$(( (RANDOM % 81) + 120 ))
    total_seconds=$((sleep_minutes * 60))
    echo "Sleeping for ${sleep_minutes} minutes..." | tee -a "$LOG_FILE"
    
    for ((i=0; i<=$total_seconds; i++)); do
        # Calculate percentages and counts
        pct=$((i * 100 / total_seconds))
        filled=$((pct / 2))
        unfilled=$((50 - filled))
        
        # Create the progress bar
        printf "\r["
        printf "%${filled}s" '' | tr ' ' '#'
        printf "%${unfilled}s" '' | tr ' ' '-'
        printf "] %d%% (%dm %ds/%dm)" $pct $((i / 60)) $((i % 60)) $sleep_minutes
        
        sleep 1
    done
    printf "\n"
    
    echo "Waking up after ${sleep_minutes} minute sleep..." | tee -a "$LOG_FILE"
done