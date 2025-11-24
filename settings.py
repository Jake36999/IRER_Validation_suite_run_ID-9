"""
IRER V11.5 — Unified Configuration Authority
--------------------------------------------
This file extends the V11.0 GOLD MASTER settings.py with:

• FMIA module defaults
• BSSN / SDG geometry module defaults
• Closed-loop emergent gravity driver parameters
• HDF5 I/O contract for standalone + integrated runs
• Real-SSE spectral validation defaults
• Standard run directory resolver

This version is compatible with:
    fmia_dynamics_solver.py
    fmia_rhs.py
    bssn_solver.py
    bssn_source_hook.py
    emergent_gravity_core.py
    geometry_metric.py
    spectral_analysis.py
    spectral_validation.py
    certification_runner.py
"""
import os
import math
from pathlib import Path

# -------------------------------
# BASE DIRECTORY
# -------------------------------
BASE_DIR = Path(os.getcwd())

# -------------------------------
# DIRECTORY STRUCTURE (Unified I/O Contract)
# -------------------------------
PROVENANCE_DIR = BASE_DIR / "provenance_reports"
DATA_DIR = BASE_DIR / "simulation_data"
CONFIG_DIR = BASE_DIR / "input_configs"
LOG_DIR = BASE_DIR / "logs"
LEDGER_FILE = BASE_DIR / "simulation_ledger.csv"
STATUS_FILE = BASE_DIR / "status.json"

for d in (PROVENANCE_DIR, DATA_DIR, CONFIG_DIR, LOG_DIR):
    d.mkdir(exist_ok=True)

# -------------------------------
# SCRIPT POINTERS
# -------------------------------
WORKER_SCRIPT = "worker_sncgl_sdg.py"
VALIDATOR_SCRIPT = "validation_pipeline.py"

# -------------------------------
# EVOLUTIONARY SEARCH DEFAULTS
# -------------------------------
DEFAULT_NUM_GENERATIONS = 10
DEFAULT_POPULATION_SIZE = 10
DEFAULT_GRID_SIZE = 64
DEFAULT_T_STEPS = 200
DEFAULT_DT = 0.01
LAMBDA_FALSIFIABILITY = 0.1
MUTATION_RATE = 0.3
MUTATION_STRENGTH = 0.05

# -------------------------------
# FMIA DYNAMICS: DEFAULT PARAMETERS
# -------------------------------
FMIA_DEFAULTS = {
    "dx": 0.1,           # Spatial step (used by fmia_rhs)
    "epsilon": 0.5,      # Linear growth
    "D": 1.0,            # Diffusion coefficient
    "lam": 1.0,          # Cubic saturation (rho^3)
    "eta": 0.02,         # Damping
    "metric_enabled": False,
}

# -------------------------------
# GEOMETRY (SDG / CONFORMAL METRIC)
# -------------------------------
GEOMETRY_DEFAULTS = {
    "rho_vac": 1.0,
    "alpha": 1.0,  # Conformal exponent
}

# -------------------------------
# BSSN / GR-LIKE SUBSYSTEM DEFAULTS
# -------------------------------
BSSN_DEFAULTS = {
    "dr": 0.1,
    "kappa_g": 0.1,
}

# -------------------------------
# CLOSED-LOOP SIMULATION DEFAULTS
# -------------------------------
CLOSED_LOOP_DEFAULTS = {
    "T_TOTAL": 5.0,
    "DT": 0.01,
    "SAVE_EVERY": 5,
}

# -------------------------------
# HDF5 SCHEMA CONTRACT
# -------------------------------
HDF5_SLOTS = {
    "fmia_only": {
        "rho": "rho",
        "time": "time",
    },
    "closed_loop": {
        "rho": "rho",
        "time": "time",
        "metric": "g_munu",
        "bssn": "bssn_state",
    }
}

# -------------------------------
# REAL-SSE VALIDATION Defaults
# -------------------------------
VALIDATION_DEFAULTS = {
    "max_prime": 31,
    "lnp_tolerance": 0.1,
}

# -------------------------------
# SENTINEL CODES (FAIL-OPEN)
# -------------------------------
SENTINEL_SUCCESS = 0.0
SENTINEL_SCIENTIFIC_FAILURE = 999.0
SENTINEL_GEOMETRIC_SINGULARITY = 1002.0
SENTINEL_TIMEOUT_COMPUTATIONAL = 1004.0
SENTINEL_ARTIFACT_MISSING = 998.0

# -------------------------------
# Log-Prime Targets (legacy compatibility)
# -------------------------------
LOG_PRIME_TARGETS = [
    math.log(2), math.log(3), math.log(5), math.log(7),
    math.log(11), math.log(13), math.log(17), math.log(19), math.log(23)
]

# -------------------------------
# API KEYS (Frontend Status Layer)
# -------------------------------
API_KEY_HUNT_STATUS = "hunt_status"
API_KEY_LAST_EVENT = "last_event"
API_KEY_LAST_SSE = "last_sse"
API_KEY_LAST_STABILITY = "last_h_norm"
API_KEY_FINAL_RESULT = "final_result"

# -------------------------------
# LEGACY BRIDGE (Critical for V11.0 Consumers)
# -------------------------------
# These aliases ensure `app.py` and `core_engine.py` do not break
# when referencing V11.0 specific constants.
SENTINEL_TIMEOUT = SENTINEL_TIMEOUT_COMPUTATIONAL
METRIC_BLOCK_SPECTRAL = "spectral_fidelity"

# -------------------------------
# HELPER: Resolve per-run output directory
# -------------------------------
def resolve_run_dir(run_id: str) -> Path:
    run_dir = DATA_DIR / f"run_{run_id}"
    run_dir.mkdir(exist_ok=True)
    return run_dir

# -------------------------------
# HELPER: Path builder for standard HDF5 output
# -------------------------------
def hdf5_output_path(run_id: str) -> Path:
    return resolve_run_dir(run_id) / "simulation_results.hdf5"
# --- API KEYS (FRONTEND INTERFACE) ---
API_KEY_HUNT_STATUS = "hunt_status"
API_KEY_LAST_EVENT = "last_event"
API_KEY_LAST_SSE = "last_sse"
API_KEY_LAST_STABILITY = "last_h_norm"
API_KEY_FINAL_RESULT = "final_result"

# --- V12 NETWORK BRIDGE CONFIGURATION ---
# Configure this for your Headless PC / Azure VM
V12_DCO_CONFIG = {
    "USE_REMOTE": False,  # Set to True to enable the Network Bridge
    "HOST": "20.186.178.188", # Your Azure VM IP
    "USER": "jake240501",     # Your VM Username
    "KEY_PATH": "IRER-V11-LAUNCH-R_ID2.txt", # Path to your Private Key
    "REMOTE_DIR": "/home/jake240501/v11_hpc_suite" # Directory on the VM
}

# --- V11.0 END OF FILE ---
# INTEGRITY CHECK: SHA1_PLACEHOLDER
# SENTINEL REFERENCE: 999.0, 1002.0
# -------------------------------
# END FILE
# -------------------------------