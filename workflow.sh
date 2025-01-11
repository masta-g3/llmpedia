#!/bin/bash

set -e  # Exit immediately if any command fails
set -a
source .venv/bin/activate
set +a

PROJECT_PATH=${PROJECT_PATH:-.}

function show_progress() {
    local elapsed=$1
    local total=$2
    local pct=$((elapsed * 100 / total))
    local filled=$((pct / 2))
    local unfilled=$((50 - filled))
    
    printf "\r["
    printf "%${filled}s" '' | tr ' ' '#'
    printf "%${unfilled}s" '' | tr ' ' '-'
    printf "] %d%% (%dm %ds/%dm)" $pct $((elapsed / 60)) $((elapsed % 60)) $((total / 60))
}

function sleep_with_progress() {
    local minutes=$1
    local total_seconds=$((minutes * 60))
    
    echo "Sleeping for ${minutes} minutes..." | tee -a "$LOG_FILE"
    
    for ((i=0; i<=$total_seconds; i++)); do
        show_progress $i $total_seconds
        sleep 1
    done
    printf "\n"
    
    echo "Waking up after ${minutes} minute sleep..." | tee -a "$LOG_FILE"
}

function run_step() {
    local step_name="$1"
    local script="$2"
    local temp_error_file="/tmp/workflow_error_$TIMESTAMP.txt"
    
    echo ">> [$step_name] Started at $(date)" | tee -a "$LOG_FILE"
    
    ## Run the Python script and capture output.
    python "${PROJECT_PATH}/${script}" 2>&1 | tee -a "$LOG_FILE" "$temp_error_file"
    local exit_status=${PIPESTATUS[0]}
    
    if [ $exit_status -ne 0 ]; then
        ## If the script failed, log the error.
        local error_msg=$(cat "$temp_error_file")
        python -c "from utils.db import log_workflow_run; log_workflow_run('$step_name', '$script', 'error', '''$error_msg''')"
        rm -f "$temp_error_file"
        echo ">> [$step_name] Failed at $(date)" | tee -a "$LOG_FILE"
        return 1
    fi
    
    ## Log successful run.
    python -c "from utils.db import log_workflow_run; log_workflow_run('$step_name', '$script', 'success')"
    rm -f "$temp_error_file"
    echo ">> [$step_name] Completed at $(date)" | tee -a "$LOG_FILE"
    return 0
}

while true; do
    ## Create timestamped log file.
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    LOG_FILE="${PROJECT_PATH}/logs/workflow_${TIMESTAMP}.log"
    
    echo "Workflow started at $(date)" | tee -a "$LOG_FILE" 2>/dev/null || true

    run_step "0: Web Scraper" "workflow/a0_scrape_lists.py"
    run_step "1 Tweet Scraper" "workflow/a1_scrape_tweets.py"
    run_step "1: Document Fetcher" "workflow/b0_download_paper.py"
    run_step "2: Marker Fetcher" "workflow/b1_download_paper_marker.py"
    run_step "2: Meta-Data Collect" "workflow/c0_fetch_meta.py"
    run_step "3: Summarizer" "workflow/d0_summarize.py"
    run_step "4: Narrator" "workflow/e0_narrate.py"
    run_step "4.1: Bullet List" "workflow/e1_narrate_bullet.py"
    run_step "4.2: Punchline" "workflow/e2_narrate_punchline.py"
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

    sleep_minutes=$(( (RANDOM % 151) + 150 ))
    sleep_with_progress $sleep_minutes
done