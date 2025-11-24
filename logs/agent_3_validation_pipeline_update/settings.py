"""
MODULE: settings.py
CLASSIFICATION: V11.0 Central Configuration File
GOAL: Acts as the single source of truth for all configuration parameters.
CONTRACT ID: IO-CFG-V11
HASHING MANDATE: Variant A (Deterministic SHA1)
STATUS: GOLD MASTER (Merged Governance + Science)
"""
import os
import math
from pathlib import Path

# Adjust BASE_DIR if running in a nested structure
BASE_DIR = Path(os.getcwd())

# --- DIRECTORY CONFIGURATION (Pathlib - Modern) ---
PROVENANCE_DIR = BASE_DIR / "provenance_reports"
DATA_DIR = BASE_DIR / "simulation_data"
CONFIG_DIR = BASE_DIR / "input_configs"
LOG_DIR = BASE_DIR / "logs"
LEDGER_FILE = BASE_DIR / "simulation_ledger.csv"
STATUS_FILE = BASE_DIR / "status.json"

# --- SCRIPT POINTERS ---
WORKER_SCRIPT = "worker_sncgl_sdg.py"
VALIDATOR_SCRIPT = "validation_pipeline.py"

# --- EVOLUTIONARY ALGORITHM DEFAULTS ---
DEFAULT_NUM_GENERATIONS = 10
DEFAULT_POPULATION_SIZE = 10
DEFAULT_GRID_SIZE = 64
DEFAULT_T_STEPS = 200
DEFAULT_DT = 0.01
LAMBDA_FALSIFIABILITY = 0.1
MUTATION_RATE = 0.3
MUTATION_STRENGTH = 0.05

# --- RESOURCE LIMITS ---
JOB_TIMEOUT_SECONDS = 600
VALIDATOR_TIMEOUT_SECONDS = 300

# --- DATA CONTRACT KEYS (BACKEND) ---
HASH_KEY = "job_uuid"  
SSE_METRIC_KEY = "log_prime_sse"
STABILITY_METRIC_KEY = "sdg_h_norm_l2"
METRIC_BLOCK_SPECTRAL = "spectral_fidelity"

# --- SENTINEL CODES (FAIL-OPEN PROTOCOL) ---
# "The System must distinguish between a crash and a singularity."
SENTINEL_SUCCESS = 0.0
SENTINEL_SCIENTIFIC_FAILURE = 999.0  # Physics Divergence / NaN / Flatline
SENTINEL_GEOMETRIC_SINGULARITY = 1002.0 # H-Norm Violation
SENTINEL_TIMEOUT_COMPUTATIONAL = 1004.0 # Resource exhaustion
SENTINEL_ARTIFACT_MISSING = 998.0 # System/IO Error

# --- SCIENTIFIC CONSTANTS (V9 TRANSPLANT) ---
# The Theoretical Targets for the Log-Prime Attractor
LOG_PRIME_TARGETS = [
    math.log(2),  # ~0.693
    math.log(3),  # ~1.098
    math.log(5),  # ~1.609
    math.log(7),  # ~1.945
    math.log(11), # ~2.397
    math.log(13), # ~2.564
    math.log(17), # ~2.833
    math.log(19), # ~2.944
    math.log(23)  # ~3.135
]

# --- API KEYS (FRONTEND INTERFACE) ---
API_KEY_HUNT_STATUS = "hunt_status"
API_KEY_LAST_EVENT = "last_event"
API_KEY_LAST_SSE = "last_sse"
API_KEY_LAST_STABILITY = "last_h_norm"
API_KEY_FINAL_RESULT = "final_result"

# --- V11.0 END OF FILE ---
# INTEGRITY CHECK: SHA1_PLACEHOLDER
# SENTINEL REFERENCE: 999.0, 1002.0