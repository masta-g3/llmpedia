#!/bin/bash

set -e  # Exit immediately if any command fails.
set -a
source .venv/bin/activate
set +a

PROJECT_PATH=${PROJECT_PATH:-.}

while true; do
    ## Create timestamped log file.
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    LOG_FILE="${PROJECT_PATH}/logs/tweet_collector_${TIMESTAMP}.log"
    
    echo "Tweet collection started at $(date)" | tee -a "$LOG_FILE" 2>/dev/null || true
    START_TIME=$(date +"%Y-%m-%d %H:%M:%S")

    ## Random sleep between 2.5 and 3 hours to avoid detection.
    sleep_minutes=$(( (RANDOM % 31) + 150 ))
    total_seconds=$((sleep_minutes * 60))
    echo "Will sleep for ${sleep_minutes} minutes after collection..." | tee -a "$LOG_FILE"
    
    ## Run the tweet collector.
    python "executors/collect_tweets.py" 2>&1 | tee -a "$LOG_FILE"
    
    ## Run tweet analysis.
    echo "Running tweet analysis..." | tee -a "$LOG_FILE"
    python "executors/t0_analyze_tweets.py" --start-time "$START_TIME" 2>&1 | tee -a "$LOG_FILE"
    
    echo "Tweet collection completed at $(date)" | tee -a "$LOG_FILE"
    echo "Sleeping for ${sleep_minutes} minutes..." | tee -a "$LOG_FILE"
    
    ## Progress bar during sleep.
    for ((i=0; i<=$total_seconds; i++)); do
        ## Calculate percentages and counts.
        pct=$((i * 100 / total_seconds))
        filled=$((pct / 2))
        unfilled=$((50 - filled))
        
        ## Create the progress bar.
        printf "\r["
        printf "%${filled}s" '' | tr ' ' '#'
        printf "%${unfilled}s" '' | tr ' ' '-'
        printf "] %d%% (%dm %ds/%dm)" $pct $((i / 60)) $((i % 60)) $sleep_minutes
        
        sleep 1
    done
    printf "\n"
    
    echo "Waking up after ${sleep_minutes} minute sleep..." | tee -a "$LOG_FILE"
done 