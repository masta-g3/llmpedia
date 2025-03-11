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

function sleep_until_target() {
    ## Get current time in seconds since epoch.
    current_time=$(date +%s)
    
    ## Calculate target time (18:30 PST) for today.
    target_time=$(TZ=PST8PDT date -v18H -v30M -v00S +%s)
    
    ## If current time is past target time, set target to tomorrow.
    if [ $current_time -gt $target_time ]; then
        target_time=$(TZ=PST8PDT date -v+1d -v18H -v30M -v00S +%s)
    fi
    
    ## Calculate seconds until target.
    seconds_to_wait=$((target_time - current_time))
    
    echo "Waiting until 6:30 PM PST..." | tee -a "$LOG_FILE"
    echo "Current time: $(date)" | tee -a "$LOG_FILE"
    echo "Target time: $(date -r $target_time)" | tee -a "$LOG_FILE"
    
    ## Show progress bar while waiting.
    for ((i=0; i<=$seconds_to_wait; i++)); do
        show_progress $i $seconds_to_wait
        sleep 1
    done
    printf "\n"
}

function run_daily_update() {
    local temp_error_file="/tmp/daily_update_error_$TIMESTAMP.txt"
    
    echo ">> [Daily Update] Started at $(date)" | tee -a "$LOG_FILE"
    
    ## Run the Python script and capture output.
    python "${PROJECT_PATH}/executors/a1_daily_update.py" 2>&1 | tee -a "$LOG_FILE" "$temp_error_file"
    local exit_status=${PIPESTATUS[0]}
    
    if [ $exit_status -eq 2 ]; then
        ## If there are too few papers, log it as info and continue.
        echo ">> [Daily Update] Skipped - Too few papers (less than 4) in the last 24 hours" | tee -a "$LOG_FILE"
        python -c "from utils.db.logging_db import log_workflow_run; log_workflow_run('Daily Update', 'executors/a1_daily_update.py', 'skipped', 'Too few papers')"
        rm -f "$temp_error_file"
        return 0
    elif [ $exit_status -ne 0 ]; then
        ## If the script failed, log the error.
        local error_msg=$(cat "$temp_error_file")
        python -c "from utils.db.logging_db import log_workflow_run; log_workflow_run('Daily Update', 'executors/a1_daily_update.py', 'error', '''$error_msg''')"
        rm -f "$temp_error_file"
        echo ">> [Daily Update] Failed at $(date)" | tee -a "$LOG_FILE"
        return 1
    fi
    
    ## Log successful run.
    python -c "from utils.db.logging_db import log_workflow_run; log_workflow_run('Daily Update', 'executors/a1_daily_update.py', 'success')"
    rm -f "$temp_error_file"
    echo ">> [Daily Update] Completed at $(date)" | tee -a "$LOG_FILE"
    return 0
}

while true; do
    ## Create timestamped log file.
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    LOG_FILE="${PROJECT_PATH}/logs/daily_update_${TIMESTAMP}.log"
    
    echo "Daily Update process started at $(date)" | tee -a "$LOG_FILE"
    
    ## Wait until 7 PM PST/PDT.
    sleep_until_target
    
    ## Run the daily update.
    run_daily_update
    
    echo "Daily Update cycle completed at $(date)" | tee -a "$LOG_FILE"
    echo "Waiting for next cycle..."
done 