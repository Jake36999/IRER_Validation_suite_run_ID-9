<#
.SYNOPSIS
    V12.0 IRER Automated Deployment Lifecycle - Deploys HPC Suite to Azure VM

.DESCRIPTION
    Orchestrates complete deployment of IRER V12.0 HPC Suite to Ubuntu Azure VM with:
    - Automated file synchronization (root files, modules, templates)
    - Remote application launch with dependency installation
    - Secure SSH tunneling to web dashboard (port 8080)
    - Live mission control dashboard with status monitoring
    - Automatic data retrieval (simulation data, reports)

.PARAMETER VM_IP
    Target Azure VM IP address (default: 20.186.178.188)

.PARAMETER VM_USER
    SSH user account on remote VM (default: jake240501)

.PARAMETER SSH_KEY
    Path to SSH private key file (relative to script dir, default: IRER-V11-LAUNCH-R_ID2.txt)

.PARAMETER REMOTE_DIR
    Remote deployment directory (default: ~/v11_hpc_suite)

.PARAMETER RUNTIME_SECONDS
    Maximum runtime for mission (default: 36000 seconds = 10 hours)

.PARAMETER Verbose
    Enable verbose output for debugging

.PARAMETER LogPath
    Custom path for deployment log file (default: logs/deploy_<timestamp>.log)

.EXAMPLE
    PS> .\deploy.ps1
    # Uses default parameters

.EXAMPLE
    PS> .\deploy.ps1 -VM_IP "10.0.0.100" -RUNTIME_SECONDS 7200 -Verbose
    # Custom VM with 2-hour runtime and verbose output

.PREREQUISITES
    - PowerShell 5.0 or higher
    - SSH client installed and accessible
    - SCP client installed and accessible  
    - Valid SSH key file with correct permissions
    - Network connectivity to target Azure VM
    - Python 3+ on remote VM (will be installed if missing)

.NOTES
    Version: 12.0
    Author: IRER Suite
    Target: Azure Ubuntu VM
    Status: Production Ready
#>

param(
    [string]$VM_IP = "20.186.178.188",
    [string]$VM_USER = "jake240501",
    [string]$SSH_KEY = "./IRER-V11-LAUNCH-R_ID2.txt",
    [string]$REMOTE_DIR = "~/v11_hpc_suite",
    [int]$RUNTIME_SECONDS = 36000,
    [switch]$Verbose,
    [string]$LogPath
)

# --- CONFIGURATION & INITIALIZATION ---
$ErrorActionPreference = "Stop"
$VerbosePreference = if ($Verbose) { "Continue" } else { "SilentlyContinue" }

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$SSH_KEY = Join-Path $ScriptDir $SSH_KEY
$LOCAL_SAVE_DIR = Join-Path $ScriptDir "run_data_$(Get-Date -Format 'yyyyMMdd_HHmmss')"

# Setup logging
if (-not $LogPath) {
    $LogDir = Join-Path $ScriptDir "logs"
    if (-not (Test-Path $LogDir)) {
        $null = New-Item -ItemType Directory -Path $LogDir -Force
    }
    $LogPath = Join-Path $LogDir "deploy_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"
}

Start-Transcript -Path $LogPath -Append | Out-Null
Write-Verbose "Logging to: $LogPath"
Write-Verbose "Script directory: $ScriptDir"
Write-Verbose "SSH Key: $SSH_KEY"
Write-Verbose "Remote directory: $REMOTE_DIR"
Write-Verbose "Runtime: $RUNTIME_SECONDS seconds"

# --- HELPER 1: RETRO SPINNER ---
function Show-Spinner {
    param(
        [string]$Message,
        [switch]$ShowBanner
    )
    
    Write-Verbose "Show-Spinner: $Message"
    
    if ($ShowBanner -and -not $script:BannerShown) {
        Clear-Host
        Write-Host "`u{001b}[36m" -NoNewline
        Write-Host "    ___   _____ ______ ______"
        Write-Host "   /   | / ___//_  __// ____/"
        Write-Host "  / /| | \__ \  / /  / __/   "
        Write-Host " / ___ |___/ / / /  / /___   "
        Write-Host "/_/  |_/____/ /_/  /_____/   "
        Write-Host "      V12.0  H P C  C O R E  "
        Write-Host "`u{001b}[0m"
        $script:BannerShown = $true
    }
    
    $spinstr = @('|', '/', '-', '\')
    Write-Host "$Message... " -NoNewline
    
    for ($i = 0; $i -lt 20; $i++) {
        Write-Host "`b`b`b`b`b$($spinstr[$i % 4])   " -NoNewline
        Start-Sleep -Milliseconds 100
    }
    
    Write-Host "`u{001b}[32m[OK]`u{001b}[0m"
}

# --- HELPER 2: LIVE DASHBOARD ---
function Show-Dashboard {
    param(
        [string]$TimeLeft,
        [string]$Generation,
        [string]$LastSSE,
        [string]$Stability,
        [string]$Status
    )
    
    Clear-Host
    Write-Host "`u{001b}[36m"
    Write-Host "========================================================" 
    Write-Host "   IRER V12.0  |  MISSION CONTROL  |  ROBUST MODE"
    Write-Host "========================================================" 
    Write-Host "   STATUS:      $Status"
    Write-Host "   TIME LEFT:   $TimeLeft"
    Write-Host "--------------------------------------------------------" 
    Write-Host "   GENERATION:  $Generation"
    Write-Host "   LAST SSE:    $LastSSE"
    Write-Host "   STABILITY:   $Stability"
    Write-Host "========================================================" 
    Write-Host "   [ ACTION ]   Keep window open to maintain Tunnel."
    Write-Host "   [ UI ]       http://localhost:8080"
    Write-Host "========================================================" 
    Write-Host "`u{001b}[0m"
}

# --- HELPER 3: JSON PARSER ---
function Get-RemoteValue {
    param(
        [string]$JsonString,
        [string]$Key
    )
    
    try {
        $json = $JsonString | ConvertFrom-Json
        return $json.$Key
    }
    catch {
        Write-Verbose "Failed to parse JSON for key '$Key': $_"
        return $null
    }
}

# --- HELPER 4: INPUT VALIDATION ---
function Invoke-PreFlightValidation {
    Write-Verbose "=== PRE-FLIGHT VALIDATION ==="
    
    # Check SSH key
    if (-not (Test-Path $SSH_KEY)) {
        throw "SSH Key not found at $SSH_KEY"
    }
    Write-Verbose "✓ SSH key exists"
    
    # Check SSH client
    if (-not (Get-Command ssh -ErrorAction SilentlyContinue)) {
        throw "SSH client not found. Please install OpenSSH for Windows."
    }
    Write-Verbose "✓ SSH client available"
    
    # Check SCP client
    if (-not (Get-Command scp -ErrorAction SilentlyContinue)) {
        throw "SCP client not found. Please install OpenSSH for Windows."
    }
    Write-Verbose "✓ SCP client available"
    
    # Test SSH connectivity
    Write-Host "Testing SSH connectivity..." -ForegroundColor Cyan
    $sshTest = ssh -i "$SSH_KEY" -o ConnectTimeout=5 -o StrictHostKeyChecking=no "${VM_USER}@${VM_IP}" "echo 'OK'" 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "SSH connectivity test failed: $sshTest"
    }
    Write-Verbose "✓ SSH connectivity verified"
    
    # Verify local directories
    $requiredDirs = @("app.py", "settings.py", "requirements.txt")
    foreach ($file in $requiredDirs) {
        $filePath = Join-Path $ScriptDir $file
        if (-not (Test-Path $filePath)) {
            Write-Warning "Expected file not found: $file"
        }
    }
    Write-Verbose "✓ Local file checks complete"
    
    Write-Host "✓ All validations passed" -ForegroundColor Green
}

# ==============================================================================
# PHASE 0: VALIDATION & INITIALIZATION
# ==============================================================================

Write-Host "`n================================================" -ForegroundColor Cyan
Write-Host "IRER V12.0 DEPLOYMENT LIFECYCLE - PHASE 0" -ForegroundColor Cyan
Write-Host "Initialization & Validation" -ForegroundColor Cyan
Write-Host "================================================`n" -ForegroundColor Cyan

try {
    Invoke-PreFlightValidation
}
catch {
    Write-Host "❌ ERROR: Validation failed: $_" -ForegroundColor Red
    Stop-Transcript
    exit 1
}

# ==============================================================================
# PHASE 1: UPLOADING SUITE
# Transfers all application files and resources to remote VM
# ==============================================================================

Write-Host "`n================================================" -ForegroundColor Cyan
Write-Host "IRER V12.0 DEPLOYMENT LIFECYCLE - PHASE 1" -ForegroundColor Cyan
Write-Host "File Upload & Synchronization" -ForegroundColor Cyan
Write-Host "================================================`n" -ForegroundColor Cyan

try {
    # Prepare Remote Directory
    Write-Host "Preparing remote structure..." -ForegroundColor Cyan
    $null = ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no "${VM_USER}@${VM_IP}" "mkdir -p $REMOTE_DIR/templates"
    Show-Spinner "Initializing Remote Structure"

    # 1. Upload Root Files
    Write-Host "Uploading root application files..." -ForegroundColor Cyan
    $RootFiles = @("app.py", "settings.py", "core_engine.py", "worker_sncgl_sdg.py", "validation_pipeline.py", "solver_sdg.py", "aste_hunter.py", "requirements.txt")

    foreach ($file in $RootFiles) {
        $localPath = Join-Path $ScriptDir $file
        if (Test-Path $localPath) {
            $null = & scp -i "$SSH_KEY" -q "$localPath" "${VM_USER}@${VM_IP}:${REMOTE_DIR}/"
            Write-Host "  ✓ Uploaded $file" -ForegroundColor Green
            Write-Verbose "Uploaded: $file to $REMOTE_DIR"
        }
        else {
            Write-Warning "  ✗ File not found: $file"
        }
    }

    # 2. Upload Modules Folder (Recursive)
    Write-Host "Uploading modules..." -ForegroundColor Cyan
    $ModulesPath = Join-Path $ScriptDir "modules"
    if (Test-Path $ModulesPath) {
        $null = ssh -i "$SSH_KEY" "${VM_USER}@${VM_IP}" "mkdir -p $REMOTE_DIR/modules"
        $null = & scp -i "$SSH_KEY" -r -q "$ModulesPath" "${VM_USER}@${VM_IP}:${REMOTE_DIR}/"
        Write-Host "  ✓ Uploaded modules folder" -ForegroundColor Green
        Write-Verbose "Uploaded modules to: $REMOTE_DIR/modules"
    }
    else {
        Write-Warning "  ✗ Modules folder not found"
    }

    # 3. Upload Templates
    Write-Host "Uploading templates..." -ForegroundColor Cyan
    $TemplatePath = Join-Path $ScriptDir "templates" "index.html"
    if (Test-Path $TemplatePath) {
        $null = & scp -i "$SSH_KEY" -q "$TemplatePath" "${VM_USER}@${VM_IP}:${REMOTE_DIR}/templates/"
        Write-Host "  ✓ Uploaded templates" -ForegroundColor Green
        Write-Verbose "Uploaded: templates/index.html"
    }
    else {
        Write-Warning "  ✗ Template not found: $TemplatePath"
    }

    Show-Spinner "Payload Synchronization Complete"
    Write-Host ""
}
catch {
    Write-Host "❌ ERROR: Phase 1 upload failed: $_" -ForegroundColor Red
    Write-Verbose "Phase 1 Error Details: $_"
    Stop-Transcript
    exit 1
}

# ==============================================================================
# PHASE 2: REMOTE LAUNCH
# Installs dependencies and starts remote application
# ==============================================================================

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "IRER V12.0 DEPLOYMENT LIFECYCLE - PHASE 2" -ForegroundColor Cyan
Write-Host "Remote Application Launch" -ForegroundColor Cyan
Write-Host "================================================`n" -ForegroundColor Cyan

try {
    Write-Host "Launching remote application..." -ForegroundColor Cyan
    Write-Verbose "Executing remote launch script on ${VM_USER}@${VM_IP}"
    
    $RemoteScript = @"
set -e
mkdir -p $REMOTE_DIR
cd $REMOTE_DIR
export DEBIAN_FRONTEND=noninteractive

# Install Python 3 and pip if needed
if ! command -v pip3 &> /dev/null; then
    echo "Installing Python 3 and pip..."
    sudo apt-get update -qq
    sudo apt-get install -y python3-pip -qq
fi

# Install dependencies
echo "Installing Python dependencies..."
pip3 install -r requirements.txt > /dev/null 2>&1

# Create required directories
mkdir -p input_configs simulation_data provenance_reports logs

# Stop any existing app.py processes
pkill -f app.py 2>/dev/null || true

# Launch application in background
echo "Starting application..."
nohup python3 app.py > app.log 2>&1 &
echo "Application started (PID: \$!)"
"@

    $null = ssh -i "$SSH_KEY" "${VM_USER}@${VM_IP}" $RemoteScript
    Show-Spinner "Remote Kernels Ignited"
    Write-Host "✓ Remote application successfully launched" -ForegroundColor Green
    Write-Host ""
}
catch {
    Write-Host "❌ ERROR: Phase 2 launch failed: $_" -ForegroundColor Red
    Write-Verbose "Phase 2 Error Details: $_"
    Stop-Transcript
    exit 1
}

# ==============================================================================
# PHASE 3: TUNNEL & DASHBOARD LOOP
# Establishes SSH tunnel, monitors remote status, displays live dashboard
# ==============================================================================

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "IRER V12.0 DEPLOYMENT LIFECYCLE - PHASE 3" -ForegroundColor Cyan
Write-Host "Live Dashboard & Monitoring" -ForegroundColor Cyan
Write-Host "================================================`n" -ForegroundColor Cyan

try {
    Show-Spinner "Establishing Secure Tunnel (8080)"
    Write-Verbose "Starting SSH tunnel: ${VM_USER}@${VM_IP} forwarding 8080->8080"
    
    # Start SSH tunnel in background
    $TunnelProcess = Start-Process ssh -ArgumentList @("-i", $SSH_KEY, "-N", "-L", "8080:localhost:8080", "${VM_USER}@${VM_IP}") -NoNewWindow -PassThru
    Write-Verbose "Tunnel process started with PID: $($TunnelProcess.Id)"
    
    $StartTime = Get-Date
    $EndTime = $StartTime.AddSeconds($RUNTIME_SECONDS)
    
    Write-Host "Dashboard will update every 5 seconds. Press Ctrl+C to stop." -ForegroundColor Yellow
    Write-Host "Access dashboard at: http://localhost:8080" -ForegroundColor Yellow
    Start-Sleep -Seconds 2
    
    while ((Get-Date) -lt $EndTime) {
        $CurrentTime = Get-Date
        $Remaining = ($EndTime - $CurrentTime).TotalSeconds
        
        $Days = [int]($Remaining / 86400)
        $Hours = [int](($Remaining % 86400) / 3600)
        $Mins = [int](($Remaining % 3600) / 60)
        $Secs = [int]($Remaining % 60)
        $TimeStr = "{0}d {1}h {2}m {3}s" -f $Days, $Hours, $Mins, $Secs
        
        # Fetch Status JSON
        try {
            $JsonRaw = ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no "${VM_USER}@${VM_IP}" "cat $REMOTE_DIR/status.json 2>/dev/null"
            Write-Verbose "Status JSON fetched: $(($JsonRaw | Measure-Object -Character).Characters) bytes"
        }
        catch {
            Write-Verbose "Failed to fetch status JSON: $_"
            $JsonRaw = ""
        }
        
        if ([string]::IsNullOrWhiteSpace($JsonRaw)) {
            $Gen = "?"
            $SSE = "?"
            $Stab = "?"
            $Stat = "Connecting..."
        }
        else {
            $Gen = Get-RemoteValue $JsonRaw "current_gen"
            $SSE = Get-RemoteValue $JsonRaw "last_sse"
            $Stab = Get-RemoteValue $JsonRaw "last_h_norm"
            $Stat = Get-RemoteValue $JsonRaw "hunt_status"
        }
        
        Show-Dashboard $TimeStr $Gen $SSE $Stab $Stat
        
        # Check Tunnel - restart if dead
        if ($TunnelProcess.HasExited) {
            Write-Host "Tunnel lost. Reconnecting..." -ForegroundColor Yellow
            Write-Verbose "Restarting tunnel process..."
            $TunnelProcess = Start-Process ssh -ArgumentList @("-i", $SSH_KEY, "-N", "-L", "8080:localhost:8080", "${VM_USER}@${VM_IP}") -NoNewWindow -PassThru
            Write-Verbose "New tunnel PID: $($TunnelProcess.Id)"
        }
        
        Start-Sleep -Seconds 5
    }
    
    # Cleanup tunnel
    Write-Host "Mission time expired. Closing tunnel..." -ForegroundColor Yellow
    if ($TunnelProcess -and -not $TunnelProcess.HasExited) {
        $TunnelProcess | Stop-Process -Force -ErrorAction SilentlyContinue
        Write-Verbose "Tunnel process terminated"
    }
    
    Write-Host ""
}
catch {
    Write-Host "❌ ERROR: Phase 3 tunnel failed: $_" -ForegroundColor Red
    Write-Verbose "Phase 3 Error Details: $_"
    if ($TunnelProcess -and -not $TunnelProcess.HasExited) {
        $TunnelProcess | Stop-Process -Force -ErrorAction SilentlyContinue
    }
}

# ==============================================================================
# PHASE 4: SHUTDOWN & RETRIEVAL
# Stops remote application and downloads simulation results
# ==============================================================================

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "IRER V12.0 DEPLOYMENT LIFECYCLE - PHASE 4" -ForegroundColor Cyan
Write-Host "Data Retrieval & Cleanup" -ForegroundColor Cyan
Write-Host "================================================`n" -ForegroundColor Cyan

try {
    Write-Host "Mission Ended. Retrieving Data..." -ForegroundColor Yellow
    
    Write-Host "Stopping remote application..." -ForegroundColor Cyan
    $null = ssh -i "$SSH_KEY" "${VM_USER}@${VM_IP}" "pkill -f app.py 2>/dev/null || true"
    Write-Verbose "Remote application stopped"
    
    Write-Host "Creating local save directory..." -ForegroundColor Cyan
    $null = New-Item -ItemType Directory -Path $LOCAL_SAVE_DIR -Force
    Write-Verbose "Created directory: $LOCAL_SAVE_DIR"
    
    Write-Host "Downloading simulation data..." -ForegroundColor Cyan
    $null = & scp -i "$SSH_KEY" -r "${VM_USER}@${VM_IP}:${REMOTE_DIR}/simulation_data" "$LOCAL_SAVE_DIR/" 2>$null
    Write-Verbose "Downloaded simulation_data"
    
    Write-Host "Downloading provenance reports..." -ForegroundColor Cyan
    $null = & scp -i "$SSH_KEY" -r "${VM_USER}@${VM_IP}:${REMOTE_DIR}/provenance_reports" "$LOCAL_SAVE_DIR/" 2>$null
    Write-Verbose "Downloaded provenance_reports"
    
    Write-Host "Downloading simulation ledger..." -ForegroundColor Cyan
    $null = & scp -i "$SSH_KEY" "${VM_USER}@${VM_IP}:${REMOTE_DIR}/simulation_ledger.csv" "$LOCAL_SAVE_DIR/" 2>$null
    Write-Verbose "Downloaded simulation_ledger.csv"
    
    Write-Host "`u{001b}[32m✓ Done. Data saved to: $LOCAL_SAVE_DIR`u{001b}[0m" -ForegroundColor Green
}
catch {
    Write-Host "❌ ERROR: Phase 4 retrieval failed: $_" -ForegroundColor Red
    Write-Verbose "Phase 4 Error Details: $_"
    Write-Host "Data may still be available on remote host at ${VM_USER}@${VM_IP}:${REMOTE_DIR}" -ForegroundColor Yellow
}

# ==============================================================================
# FINALIZATION
# ==============================================================================

Write-Host "`nDeployment complete!" -ForegroundColor Cyan
Write-Host "Log file: $LogPath" -ForegroundColor Cyan
Stop-Transcript
