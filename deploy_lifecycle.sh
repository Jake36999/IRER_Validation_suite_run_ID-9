#!/bin/bash

# ==============================================================================
# V11.0 AUTOMATED DEPLOYMENT LIFECYCLE
# TARGET: Azure VM (Ubuntu)
# SOURCE: Local Windows PC (via Git Bash/WSL)
# ==============================================================================

# --- CONFIGURATION ---
# 1. YOUR AZURE VM IP (You must paste this!)
VM_IP="REPLACE_WITH_YOUR_VM_IP" 

# 2. SSH KEY PATH (Converted for Git Bash/WSL compatibility)
# We convert "C:\Users..." to "/c/Users..." for the shell environment
SSH_KEY="/c/Users/jakem/Downloads/IRER-V11-LAUNCH-R_ID1.pem"

# 3. VM USERNAME (Default for Azure Ubuntu images)
VM_USER="azureuser"

# 4. PATHS
REMOTE_DIR="~/v11_hpc_suite"
# This saves data to a "run_data" folder inside your current project folder
LOCAL_DATA_SAVE_PATH="./run_data_$(date +%Y%m%d_%H%M%S)"

# 5. DURATION (10 Hours - buffer = 35800 seconds)
RUNTIME_SECONDS=35800 

# --- [PHASE 1] PRE-FLIGHT CHECKS ---
echo "--- [PHASE 1] CHECKING CREDENTIALS ---"

if [ "$VM_IP" == "REPLACE_WITH_YOUR_VM_IP" ]; then
    echo "❌ ERROR: You forgot to put your Azure VM IP in the script!"
    echo "Please open deploy_lifecycle.sh and edit line 11."
    exit 1
fi

if [ ! -f "$SSH_KEY" ]; then
    echo "❌ ERROR: SSH Key not found at $SSH_KEY"
    echo "Please ensure the .pem file exists and the path is correct."
    exit 1
fi

# Fix key permissions (crucial for ssh on some systems)
chmod 400 "$SSH_KEY" 2>/dev/null

echo "Testing connection to $VM_IP..."
ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no "$VM_USER@$VM_IP" "echo '✅ Connection Successful'"
if [ $? -ne 0 ]; then
    echo "❌ Could not connect to VM. Please check your IP address and Network Security Group (NSG) rules in Azure."
    exit 1
fi

# --- [PHASE 2] UPLOADING SUITE ---
echo "--- [PHASE 2] UPLOADING V11.0 SUITE ---"
# Create remote structure
ssh -i "$SSH_KEY" "$VM_USER@$VM_IP" "mkdir -p $REMOTE_DIR/templates"

# Upload Python Core
echo "Uploading core files..."
scp -i "$SSH_KEY" app.py settings.py core_engine.py worker_sncgl_sdg.py \
    validation_pipeline.py solver_sdg.py aste_hunter.py requirements.txt \
    "$VM_USER@$VM_IP:$REMOTE_DIR/"

# Upload Templates
echo "Uploading UI templates..."
scp -i "$SSH_KEY" templates/index.html "$VM_USER@$VM_IP:$REMOTE_DIR/templates/"

# --- [PHASE 3] REMOTE SETUP & LAUNCH ---
echo "--- [PHASE 3] REMOTE INSTALL & LAUNCH ---"
ssh -i "$SSH_KEY" "$VM_USER@$VM_IP" << EOF
    cd $REMOTE_DIR
    
    # 1. System Updates (Silent)
    echo "Updating system packages..."
    sudo apt-get update -qq > /dev/null
    sudo apt-get install -y python3-pip -qq > /dev/null

    # 2. Python Dependencies
    echo "Installing Python requirements..."
    pip3 install -r requirements.txt > /dev/null 2>&1

    # 3. Create Data Contract Directories
    mkdir -p input_configs simulation_data provenance_reports logs

    # 4. Launch Control Hub (Background Mode)
    echo "Launching V11.0 Control Hub..."
    # Kill any old instance first
    pkill -f app.py || true
    # Start new instance
    nohup python3 app.py > app.log 2>&1 &
    sleep 3 # Give it a moment to start
EOF

# --- [PHASE 4] TUNNELING UI ---
echo "--- [PHASE 4] ESTABLISHING SECURE TUNNEL ---"
echo "Mapping Remote:8080 -> Local:8080"
# This creates the bridge so you can see the site on your PC
ssh -i "$SSH_KEY" -N -L 8080:localhost:8080 "$VM_USER@$VM_IP" &
TUNNEL_PID=$!

echo "========================================================"
echo "🚀 SYSTEM LIVE!"
echo "--------------------------------------------------------"
echo "1. Open your browser: http://localhost:8080"
echo "2. Click 'Start New Hunt'"
echo "--------------------------------------------------------"
echo "⏳ Running for 10 hours. DO NOT CLOSE THIS WINDOW."
echo "   (Data will auto-download when finished)"
echo "========================================================"

# --- WAIT LOOP ---
sleep $RUNTIME_SECONDS

# --- [PHASE 5] SHUTDOWN & DATA RETRIEVAL ---
echo "--- [PHASE 5] TIMEOUT REACHED - RETRIEVING DATA ---"

# 1. Stop the Remote Process
echo "Stopping remote simulation..."
ssh -i "$SSH_KEY" "$VM_USER@$VM_IP" "pkill -f app.py"

# 2. Download Data
echo "Downloading artifacts to $LOCAL_DATA_SAVE_PATH..."
mkdir -p "$LOCAL_DATA_SAVE_PATH"

echo "Downloading large datasets..."
scp -i "$SSH_KEY" -r "$VM_USER@$VM_IP:$REMOTE_DIR/simulation_data" "$LOCAL_DATA_SAVE_PATH/"
scp -i "$SSH_KEY" -r "$VM_USER@$VM_IP:$REMOTE_DIR/provenance_reports" "$LOCAL_DATA_SAVE_PATH/"
scp -i "$SSH_KEY" "$VM_USER@$VM_IP:$REMOTE_DIR/simulation_ledger.csv" "$LOCAL_DATA_SAVE_PATH/"

# 3. Kill Tunnel
kill $TUNNEL_PID

echo "✅ RUN COMPLETE. Data saved securely."
echo "⚠️ ACTION REQUIRED: Go to Azure Portal and STOP the VM to save credits!"