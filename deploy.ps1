#!/bin/bash

# ==============================================================================
# V12.0 AUTOMATED DEPLOYMENT LIFECYCLE (COMMANDER EDITION)
# TARGET: Azure VM (Ubuntu)
# SOURCE: Local Windows PC (via Git Bash/WSL)
# FEATURES: Retro UI, Live Dashboard, SSH Tunneling, Auto-Retrieval
# ==============================================================================

# --- CONFIGURATION ---
VM_IP="20.186.178.188"
VM_USER="jake240501"
SSH_KEY="./IRER-V11-LAUNCH-R_ID2.txt"
REMOTE_DIR="~/v11_hpc_suite"
LOCAL_SAVE_DIR="./run_data_$(date +%Y%m%d_%H%M%S)"
RUNTIME_SECONDS=36000 # 10 Hours

# --- HELPER 1: RETRO SPINNER ---
show_spinner() {
    local pid=$!
    local delay=0.1
    local spinstr='|/-\'
    local msg="$1"
    
    # ASCII Banner on first run
    if [ -z "$BANNER_SHOWN" ]; then
        clear
        echo -e "\033[36m"
        echo "    ___   _____ ______ ______"
        echo "   /   | / ___//_  __// ____/"
        echo "  / /| | \__ \  / /  / __/   "
        echo " / ___ |___/ / / /  / /___   "
        echo "/_/  |_/____/ /_/  /_____/   "
        echo "      V12.0  H P C  C O R E  "
        echo -e "\033[0m"
        export BANNER_SHOWN=1
    fi

    echo -ne "$msg... "
    
    # Spin until the task (passed as function) finishes or for a set time
    # Here we just simulate a spin for cosmetic effect if no PID provided
    for i in {1..20}; do
        local temp=${spinstr#?}
        printf " [%c]  " "$spinstr"
        local spinstr=$temp${spinstr%"$temp"}
        sleep $delay
        printf "\b\b\b\b\b\b"
    done
    printf "    \b\b\b\b"
    echo -e "\033[32m[OK]\033[0m"
}

# --- HELPER 2: LIVE DASHBOARD ---
draw_dashboard() {
    local time_left="$1"
    local gen="$2"
    local sse="$3"
    local stab="$4"
    local status="$5"

    clear
    echo -e "\033[36m"
    echo "========================================================"
    echo "   IRER V12.0  |  MISSION CONTROL  |  ROBUST MODE"
    echo "========================================================"
    echo "   STATUS:      $status"
    echo "   TIME LEFT:   $time_left"
    echo "--------------------------------------------------------"
    echo "   GENERATION:  $gen"
    echo "   LAST SSE:    $sse"
    echo "   STABILITY:   $stab"
    echo "========================================================"
    echo "   [ ACTION ]   Keep window open to maintain Tunnel."
    echo "   [ UI ]       http://localhost:8080"
    echo "========================================================"
    echo -e "\033[0m"
}

# --- HELPER 3: ROBUST JSON PARSER ---
# We use Python for parsing to avoid dependency on 'jq'
get_remote_value() {
    local json="$1"
    local key="$2"
    echo "$json" | grep -o "\"$key\": [^,}]*" | awk -F': ' '{print $2}' | tr -d '"'
}

# --- [PHASE 1] PRE-FLIGHT CHECKS ---
if [ ! -f "$SSH_KEY" ]; then
    echo "❌ ERROR: SSH Key not found at $SSH_KEY"
    exit 1
fi
chmod 400 "$SSH_KEY" 2>/dev/null

show_spinner "Connecting to Azure VM ($VM_IP)"

# --- [PHASE 2] UPLOADING SUITE ---
ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no "$VM_USER@$VM_IP" "mkdir -p $REMOTE_DIR/templates" > /dev/null 2>&1
show_spinner "Initializing Remote Structure"

scp -i "$SSH_KEY" -q app.py settings.py core_engine.py worker_sncgl_sdg.py \
    validation_pipeline.py solver_sdg.py aste_hunter.py requirements.txt \
    "$VM_USER@$VM_IP:$REMOTE_DIR/"
scp -i "$SSH_KEY" -q templates/index.html "$VM_USER@$VM_IP:$REMOTE_DIR/templates/"
show_spinner "Payload Synchronization Complete"

# --- [PHASE 3] REMOTE LAUNCH ---
REMOTE_SCRIPT="
    set -e
    mkdir -p $REMOTE_DIR; cd $REMOTE_DIR
    export DEBIAN_FRONTEND=noninteractive
    
    if ! command -v pip3 &> /dev/null; then sudo apt-get update -qq; sudo apt-get install -y python3-pip -qq; fi
    pip3 install -r requirements.txt > /dev/null 2>&1
    mkdir -p input_configs simulation_data provenance_reports logs
    
    pkill -f app.py || true
    nohup python3 app.py > app.log 2>&1 &
"

ssh -i "$SSH_KEY" "$VM_USER@$VM_IP" "$REMOTE_SCRIPT" > /dev/null 2>&1
show_spinner "Remote Kernels Ignited"

# --- [PHASE 4] TUNNEL & DASHBOARD LOOP ---
show_spinner "Establishing Secure Tunnel (8080)"
ssh -i "$SSH_KEY" -N -L 8080:localhost:8080 "$VM_USER@$VM_IP" &
TUNNEL_PID=$!

START_TIME=$(date +%s)
END_TIME=$((START_TIME + RUNTIME_SECONDS))

while [ $(date +%s) -lt $END_TIME ]; do
    CURRENT_TIME=$(date +%s)
    REMAINING=$((END_TIME - CURRENT_TIME))
    
    # Calculate formatted time
    DAYS=$((REMAINING / 86400))
    HOURS=$(( (REMAINING % 86400) / 3600 ))
    MINS=$(( (REMAINING % 3600) / 60 ))
    SECS=$((REMAINING % 60))
    TIME_STR="${DAYS}d ${HOURS}h ${MINS}m ${SECS}s"

    # Fetch Status JSON
    JSON_RAW=$(ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no "$VM_USER@$VM_IP" "cat $REMOTE_DIR/status.json 2>/dev/null")
    
    if [ -z "$JSON_RAW" ]; then
        GEN="?"
        SSE="?"
        STAB="?"
        STAT="Connecting..."
    else
        # Parse without jq
        GEN=$(get_remote_value "$JSON_RAW" "current_gen")
        SSE=$(get_remote_value "$JSON_RAW" "last_sse")
        STAB=$(get_remote_value "$JSON_RAW" "last_h_norm")
        STAT=$(get_remote_value "$JSON_RAW" "hunt_status")
    fi

    draw_dashboard "$TIME_STR" "$GEN" "$SSE" "$STAB" "$STAT"

    # Check Tunnel
    if ! kill -0 $TUNNEL_PID 2>/dev/null; then
        ssh -i "$SSH_KEY" -N -L 8080:localhost:8080 "$VM_USER@$VM_IP" &
        TUNNEL_PID=$!
    fi

    sleep 5
done

# --- [PHASE 5] SHUTDOWN & RETRIEVAL ---
echo -e "\n\033[33mMission Ended. Retrieving Data...\033[0m"
ssh -i "$SSH_KEY" "$VM_USER@$VM_IP" "pkill -f app.py"

mkdir -p "$LOCAL_SAVE_DIR"
scp -i "$SSH_KEY" -r "$VM_USER@$VM_IP:$REMOTE_DIR/simulation_data" "$LOCAL_SAVE_DIR/"
scp -i "$SSH_KEY" -r "$VM_USER@$VM_IP:$REMOTE_DIR/provenance_reports" "$LOCAL_SAVE_DIR/"
scp -i "$SSH_KEY" "$VM_USER@$VM_IP:$REMOTE_DIR/simulation_ledger.csv" "$LOCAL_SAVE_DIR/"

kill $TUNNEL_PID
echo -e "\033[32mDone. Data in $LOCAL_SAVE_DIR\033[0m"