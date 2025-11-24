<#
.SYNOPSIS
    V11.0 Deployment Automator (Unified Commander Edition)
.DESCRIPTION
    Combines Retro UI styling with Real-Time Telemetry fetching.
    HARDENING: Implemented SSH retry loop for dashboard stability.
#>


# --- CONFIGURATION ---
$VM_IP = "20.186.178.188"         # The new East US 2 IP
$VM_USER = "jake240501"           # The new lowercase username
# Absolute Path to your new key file (copied from your second path)
$KEY_FILE = "C:\Users\jakem\OneDrive\Documents\IRER_SUITE_V11_assembly\IRER_VALIDATION_SUITE_V11\Run_ID=6\IRER_v11_suite_RUN_ID-6\draft_8\IRER-V11-LAUNCH-R_ID2.txt"
$REMOTE_DIR = "/home/$VM_USER/v11_hpc_suite"
$LOCAL_SAVE_DIR = ".\run_data_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
$DURATION_SECONDS = 36000 # 10 Hours

$REMOTE_DIR = "/home/$VM_USER/v11_hpc_suite"
$LOCAL_SAVE_DIR = ".\run_data_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
$DURATION_SECONDS = 36000 # 10 Hours

# --- HELPER 1: RETRO SPINNER (SETUP PHASE) ---
$script:BannerShown = $false
function Show-Spinner {
    param([string]$Message, [int]$Cycles = 8)
    
    if (-not $script:BannerShown) {
$asteBanner = @"
    ___   _____ ______ ______
   /   | / ___//_  __// ____/
  / /| | \__ \  / /  / __/   
 / ___ |___/ / / /  / /___   
/_/  |_/____/ /_/  /_____/   
      V11.0  H P C  C O R E
"@
        Clear-Host
        Write-Host $asteBanner -ForegroundColor Cyan
        Write-Host ""
        $script:BannerShown = $true
    }

    Write-Host -NoNewline ("{0,-50}" -f $Message)
    $frames = @("|", "/", "-", "\")
    for ($i = 0; $i -lt $Cycles; $i++) {
        foreach ($frame in $frames) {
            Write-Host -NoNewline -ForegroundColor Yellow ("`r{0,-50} [{1}]" -f $Message, $frame)
            Start-Sleep -Milliseconds 80
        }
    }
    Write-Host -ForegroundColor Green ("`r{0,-50} [OK]" -f $Message)
}

# --- HELPER 2: LIVE DASHBOARD (RUNTIME PHASE) ---
function Draw-Dashboard {
    param($TimeStr, $Gen, $SSE, $Stab, $Status)
    $dash = @"
========================================================
   IRER V11.0  |  MISSION CONTROL  |  ROBUST MODE
========================================================
   STATUS:      $Status
   TIME LEFT:   $TimeStr
--------------------------------------------------------
   GENERATION:  $Gen
   LAST SSE:    $SSE
   STABILITY:   $Stab
========================================================
   [ ACTION ]   Keep window open to maintain Tunnel.
   [ UI ]       http://localhost:8081
========================================================
"@
    Clear-Host
    Write-Host $dash -ForegroundColor Cyan
}

# --- HELPER 3: ROBUST JSON RETRIEVAL (HARDENING) ---
function Get-RemoteJson {
    param([string]$KeyFile, [string]$User, [string]$IP, [string]$RemotePath, [int]$Retries=3, [int]$Delay=1)
    
    for ($i=0; $i -lt $Retries; $i++) {
        try {
            # Use strict ASCII command to prevent encoding issues with JSON
            $jsonRaw = ssh -i $KeyFile -o StrictHostKeyChecking=no "$User@${IP}" "cat $RemotePath" 2>$null
            
            # Check if JSON is non-empty and valid before returning
            if ($jsonRaw -and $jsonRaw.Trim().Length -gt 0 -and $jsonRaw -notlike "*cat:*") {
                return $jsonRaw | ConvertFrom-Json
            }
        } catch {
            Start-Sleep -Seconds $Delay
        }
    }
    return $null
}

# --- PHASE 1: PRE-FLIGHT ---
if (-not (Test-Path $KEY_FILE)) { Write-Error "Key missing!"; exit }
Show-Spinner "Connecting to Azure VM ($VM_IP)..."
ssh -i $KEY_FILE -o StrictHostKeyChecking=no "$VM_USER@${VM_IP}" "echo ''" 2>$null

# --- PHASE 2: UPLOAD ---
Show-Spinner "Initializing Remote Directory Structure..." 2
ssh -i $KEY_FILE "$VM_USER@${VM_IP}" "mkdir -p $REMOTE_DIR/templates"

$Files = @("app.py", "settings.py", "core_engine.py", "worker_sncgl_sdg.py", "validation_pipeline.py", "solver_sdg.py", "aste_hunter.py", "requirements.txt", "templates\index.html")
foreach ($f in $Files) {
    if ($f -eq "templates\index.html") { scp -q -i $KEY_FILE $f "$VM_USER@${VM_IP}:$REMOTE_DIR/templates/" }
    else { scp -q -i $KEY_FILE $f "$VM_USER@${VM_IP}:$REMOTE_DIR/" }
}
Show-Spinner "Payload Synchronization Complete" 2

# --- PHASE 3: REMOTE LAUNCH ---
$RemoteScript = @"
    set -e
    mkdir -p $REMOTE_DIR; cd $REMOTE_DIR
    export DEBIAN_FRONTEND=noninteractive
    
    # 1. KERNEL TUNING
    if ! grep -q "net.ipv4.ip_forward=1" /etc/sysctl.conf; then
        echo "net.ipv4.ip_forward=1" | sudo tee -a /etc/sysctl.conf > /dev/null; sudo sysctl -p
    fi
    
    # 2. DEPENDENCIES
    if ! command -v pip3 &> /dev/null; then sudo apt-get update -qq; sudo apt-get install -y python3-pip python3-venv -qq; fi
    
    # 3. ENVIRONMENT SETUP
    if [ ! -d "venv" ]; then python3 -m venv venv; fi
    source venv/bin/activate
    pip install -r requirements.txt > /dev/null 2>&1
    
    mkdir -p input_configs simulation_data provenance_reports logs
    
    # 4. SYSTEMD SERVICE
    sudo tee /etc/systemd/system/irer_hpc.service > /dev/null <<EOL
[Unit]
Description=IRER V11.0 HPC Core
After=network.target
[Service]
User=$VM_USER
WorkingDirectory=$REMOTE_DIR
ExecStart=$REMOTE_DIR/venv/bin/python3 $REMOTE_DIR/app.py
Restart=always
RestartSec=5
StandardOutput=append:$REMOTE_DIR/app.log
StandardError=append:$REMOTE_DIR/app.log
[Install]
WantedBy=multi-user.target
EOL

    # 5. LAUNCH
    sudo systemctl daemon-reload
    sudo systemctl enable irer_hpc.service
    sudo systemctl restart irer_hpc.service
"@ -replace "`r`n", "`n"

Show-Spinner "Configuring Systemd & Tuning Kernel..." 15
ssh -i $KEY_FILE "$VM_USER@${VM_IP}" $RemoteScript | Out-Null

# --- PHASE 4: TUNNEL & DASHBOARD ---
Show-Spinner "Establishing Secure Tunnel (8081)..." 5
# Using Port 8081 locally to map to 8080 remotely (avoids local conflicts)
$TunnelJob = Start-Job -ScriptBlock { param($k, $u, $ip) ssh -i $k -o StrictHostKeyChecking=no -N -L 8081:localhost:8080 "$u@$ip" } -ArgumentList $KEY_FILE, $VM_USER, $VM_IP
Start-Sleep 5

$startTime = Get-Date
$endTime = $startTime.AddSeconds($DURATION_SECONDS)

while ((Get-Date) -lt $endTime) {
    $remaining = $endTime - (Get-Date)
    $timeStr = "{0:dd}d {0:hh}h {0:mm}m {0:ss}s" -f $remaining
    
    # HARDENED: Use retry helper to fetch JSON status
    $statusObj = Get-RemoteJson $KEY_FILE $VM_USER $VM_IP "$REMOTE_DIR/status.json"
    
    if (-not $statusObj) {
        $gen = "?"; $sse = "?"; $stab = "?"; $stat = "Connecting..."
    } else {
        $gen = $statusObj.current_gen;
        $sse = $statusObj.last_sse;
        $stab = $statusObj.last_h_norm;
        $stat = $statusObj.hunt_status;
    }

    Draw-Dashboard $timeStr $gen $sse $stab $stat

    if ($TunnelJob.State -ne 'Running') {
        # Auto-heal tunnel if it drops
        Remove-Job $TunnelJob -Force
        $TunnelJob = Start-Job -ScriptBlock { param($k, $u, $ip) ssh -i $k -o StrictHostKeyChecking=no -N -L 8081:localhost:8080 "$u@$ip" } -ArgumentList $KEY_FILE, $VM_USER, $VM_IP
    }
    Start-Sleep 5
}

# --- PHASE 5: RETRIEVAL ---
Write-Host "`nMission Ended. Retrieving Data..." -ForegroundColor Yellow
ssh -i $KEY_FILE "$VM_USER@${VM_IP}" "sudo systemctl stop irer_hpc.service" 2>$null

New-Item -ItemType Directory -Force -Path $LOCAL_SAVE_DIR | Out-Null

scp -i $KEY_FILE -r "$VM_USER@${VM_IP}:$REMOTE_DIR/simulation_data" "$LOCAL_SAVE_DIR"
scp -i $KEY_FILE -r "$VM_USER@${VM_IP}:$REMOTE_DIR/provenance_reports" "$LOCAL_SAVE_DIR"
scp -i $KEY_FILE "$VM_USER@${VM_IP}:$REMOTE_DIR/simulation_ledger.csv" "$LOCAL_SAVE_DIR"

Stop-Job $TunnelJob; Remove-Job $TunnelJob
Write-Host "Done. Data in $LOCAL_SAVE_DIR" -ForegroundColor Green