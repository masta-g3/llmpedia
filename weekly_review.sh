#!/bin/bash

set -e  ## Exit immediately if any command fails
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

function sleep_until_monday() {
    ## Get current time in seconds since epoch.
    current_time=$(date +%s)
    
    ## Get current day of week (0=Sunday, 1=Monday, ..., 6=Saturday)
    current_day=$(date +%u)
    
    ## Calculate target time (2:00 PM PST) for next Monday.
    if [ $current_day -eq 1 ]; then
        ## If today is Monday and it's before 2:00 PM, set target to today at 2:00 PM
        target_time=$(TZ=PST8PDT date -v14H -v00M -v00S +%s)
        if [ $current_time -gt $target_time ]; then
            ## If it's past 2:00 PM, set target to next Monday
            target_time=$(TZ=PST8PDT date -v+7d -v14H -v00M -v00S +%s)
        fi
    else
        ## Calculate days until next Monday
        days_until_monday=$((8 - current_day))
        if [ $days_until_monday -eq 7 ]; then
            days_until_monday=0
        fi
        
        ## Set target to next Monday at 2:00 PM
        target_time=$(TZ=PST8PDT date -v+${days_until_monday}d -v14H -v00M -v00S +%s)
    fi
    
    ## Calculate seconds until target.
    seconds_to_wait=$((target_time - current_time))
    
    echo "Waiting until next Monday at 2:00 PM PST..." | tee -a "$LOG_FILE"
    echo "Current time: $(date)" | tee -a "$LOG_FILE"
    echo "Target time: $(date -r $target_time)" | tee -a "$LOG_FILE"
    
    ## Show progress bar while waiting.
    for ((i=0; i<=$seconds_to_wait; i++)); do
        show_progress $i $seconds_to_wait
        sleep 1
    done
    printf "\n"
}

function get_previous_monday_date() {
    ## Get date for Monday of the previous week (7 days ago)
    TZ=PST8PDT date -v-7d -v-mon +%Y-%m-%d
}

function run_weekly_review() {
    local temp_error_file="/tmp/weekly_review_error_$TIMESTAMP.txt"
    local prev_monday=$(get_previous_monday_date)
    
    echo ">> [Weekly Review] Started at $(date)" | tee -a "$LOG_FILE"
    echo ">> [Weekly Review] Using date: $prev_monday" | tee -a "$LOG_FILE"
    
    ## Run the Python script and capture output.
    python "${PROJECT_PATH}/executors/b1_weekly_review.py" "$prev_monday" 2>&1 | tee -a "$LOG_FILE" "$temp_error_file"
    local exit_status=${PIPESTATUS[0]}
    
    if [ $exit_status -ne 0 ]; then
        ## If the script failed, log the error.
        local error_msg=$(cat "$temp_error_file")
        python -c "from utils.db.logging_db import log_workflow_run; log_workflow_run('Weekly Review', 'executors/b1_weekly_review.py', 'error', '''$error_msg''')"
        rm -f "$temp_error_file"
        echo ">> [Weekly Review] Failed at $(date)" | tee -a "$LOG_FILE"
        return 1
    fi
    
    ## Log successful run.
    python -c "from utils.db.logging_db import log_workflow_run; log_workflow_run('Weekly Review', 'executors/b1_weekly_review.py', 'success')"
    rm -f "$temp_error_file"
    echo ">> [Weekly Review] Completed at $(date)" | tee -a "$LOG_FILE"
    return 0
}

while true; do
    ## Create timestamped log file.
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    LOG_FILE="${PROJECT_PATH}/logs/weekly_review_${TIMESTAMP}.log"
    
    echo "Weekly Review process started at $(date)" | tee -a "$LOG_FILE"
    
    ## Wait until next Monday at 2:00 PM PST/PDT.
    sleep_until_monday
    
    ## Run the weekly review.
    run_weekly_review
    
    echo "Weekly Review cycle completed at $(date)" | tee -a "$LOG_FILE"
    echo "Waiting for next cycle..."
done 