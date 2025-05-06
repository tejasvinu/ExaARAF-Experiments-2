#!/bin/bash
# System resource monitoring script for MCTS experiments

# Get the job ID from Slurm or use PID if not in Slurm
if [ -n "${SLURM_JOB_ID}" ]; then
    MONITOR_ID="${SLURM_JOB_ID}"
    METRIC_FILE="system_metrics.job${SLURM_JOB_ID}.log"
else
    MONITOR_ID="$$"
    METRIC_FILE="system_metrics.pid$$.log"
fi

# Create header for metrics file
echo "timestamp,cpu_pct,mem_rss_kb,mem_vms_kb,io_read_kb,io_write_kb,cpu_user_time,cpu_system_time" > "$METRIC_FILE"

# Function to get process stats
get_process_stats() {
    local pids=("$@")
    local stats=""
    
    # Sum resources across all relevant processes
    local total_cpu=0
    local total_rss=0
    local total_vms=0
    local total_read=0
    local total_write=0
    local total_utime=0
    local total_stime=0
    
    for pid in "${pids[@]}"; do
        if [ -d "/proc/$pid" ]; then
            # Get CPU usage
            local cpu_stat=$(top -b -n 1 -p $pid | grep $pid)
            local cpu_pct=$(echo "$cpu_stat" | awk '{print $9}')
            
            # Get memory usage
            local mem_stat=$(cat /proc/$pid/status | grep -E 'VmRSS|VmSize')
            local rss=$(echo "$mem_stat" | grep 'VmRSS' | awk '{print $2}')
            local vms=$(echo "$mem_stat" | grep 'VmSize' | awk '{print $2}')
            
            # Get I/O stats
            local io_stat=$(cat /proc/$pid/io 2>/dev/null)
            local read_bytes=$(echo "$io_stat" | grep 'read_bytes' | awk '{print $2}')
            local write_bytes=$(echo "$io_stat" | grep 'write_bytes' | awk '{print $2}')
            
            # Get CPU time
            local cpu_time=$(cat /proc/$pid/stat | awk '{print $14, $15}')
            local utime=$(echo "$cpu_time" | awk '{print $1}')
            local stime=$(echo "$cpu_time" | awk '{print $2}')
            
            # Add to totals
            total_cpu=$(echo "$total_cpu + $cpu_pct" | bc)
            total_rss=$(echo "$total_rss + $rss" | bc)
            total_vms=$(echo "$total_vms + $vms" | bc)
            total_read=$(echo "$total_read + $read_bytes" | bc)
            total_write=$(echo "$total_write + $write_bytes" | bc)
            total_utime=$(echo "$total_utime + $utime" | bc)
            total_stime=$(echo "$total_stime + $stime" | bc)
        fi
    done
    
    # Return comma-separated stats
    total_read_kb=$(echo "$total_read / 1024" | bc)
    total_write_kb=$(echo "$total_write / 1024" | bc)
    echo "$total_cpu,$total_rss,$total_vms,$total_read_kb,$total_write_kb,$total_utime,$total_stime"
}

# Monitor interval in seconds
INTERVAL=5

# If we're in a Slurm job, get all PIDs associated with it
if [ -n "${SLURM_JOB_ID}" ]; then
    echo "Monitoring all processes in Slurm job ${SLURM_JOB_ID}"
    
    while true; do
        # Get all PIDs in this Slurm job
        pids=($(ps -u $USER -o pid= --sort=-pid | xargs -I{} sh -c "grep -l ${SLURM_JOB_ID} /proc/{}/environ 2>/dev/null | grep -o '[0-9]*'"))
        
        if [ ${#pids[@]} -eq 0 ]; then
            echo "No processes found for job ${SLURM_JOB_ID}, sleeping..."
            sleep $INTERVAL
            continue
        fi
        
        # Get timestamp
        timestamp=$(date +%s)
        
        # Get process stats
        stats=$(get_process_stats "${pids[@]}")
        
        # Write to log
        echo "$timestamp,$stats" >> "$METRIC_FILE"
        
        sleep $INTERVAL
    done
else
    # Non-Slurm mode: monitor a specific process and its children
    target_pid=$1
    if [ -z "$target_pid" ]; then
        echo "Error: When not in Slurm, you must provide a PID to monitor"
        exit 1
    fi
    
    echo "Monitoring process $target_pid and its children"
    
    while kill -0 $target_pid 2>/dev/null; do
        # Get all child PIDs
        pids=($(pstree -p $target_pid | grep -o '([0-9]\+)' | grep -o '[0-9]\+'))
        pids+=($target_pid)
        
        # Get timestamp
        timestamp=$(date +%s)
        
        # Get process stats
        stats=$(get_process_stats "${pids[@]}")
        
        # Write to log
        echo "$timestamp,$stats" >> "$METRIC_FILE"
        
        sleep $INTERVAL
    done
fi

echo "Monitoring finished"
