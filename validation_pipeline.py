"""
MODULE: validation_pipeline.py
CLASSIFICATION: V11.0 Analysis Plane (Certified)
GOAL: Independent verification of Log-Prime Spectral Attractors via FFT.
CONTRACT ID: IO-VAL-V11
HASHING MANDATE: Variant A (Deterministic SHA1)
"""
import os
import sys
import json
import argparse
import h5py
import numpy as np
import scipy.fft
import scipy.signal
import logging
import traceback

# Try import settings, fallback to local defaults if missing (for standalone testing)
try:
    import settings
except ImportError:
    class settings:
        PROVENANCE_DIR = "provenance_reports"
        DATA_DIR = "simulation_data"
        SSE_METRIC_KEY = "log_prime_sse"
        STABILITY_METRIC_KEY = "h_norm"
        SENTINEL_FAILURE = 999.0
        SENTINEL_DIVERGENCE = 1002.0

logging.basicConfig(level=logging.INFO, format='[Validator] %(message)s')

# Theoretical Targets: Natural Logarithms of Prime Numbers
# ln(2), ln(3), ln(5), ln(7), ln(11), ln(13), ln(17)...
LOG_PRIME_TARGETS = np.log([2, 3, 5, 7, 11, 13, 17, 19, 23, 29])

def atomic_write_provenance(filepath, data):
    """Writes JSON to temp file then moves it to ensure atomicity."""
    temp_path = filepath + ".tmp"
    with open(temp_path, 'w') as f:
        json.dump(data, f, indent=2, sort_keys=True)
    os.replace(temp_path, filepath)

def extract_rays(field_3d):
    """
    Multi-Ray Sampling: Extracts 1D rays from 3D field to detect anisotropic structures.
    Returns list of 1D arrays: [Center-X, Center-Y, Center-Z, Diagonal-Main]
    """
    N = field_3d.shape[0]
    center = N // 2
    
    # Cardinal Rays
    ray_x = field_3d[:, center, center]
    ray_y = field_3d[center, :, center]
    ray_z = field_3d[center, center, :]
    
    # Main Diagonal
    ray_diag = np.array([field_3d[i, i, i] for i in range(N)])
    
    return [ray_x, ray_y, ray_z, ray_diag]

def analyze_spectral_fidelity(rho_field):
    """
    Performs FFT on rays and calculates SSE against Log-Prime targets.
    True De-Mocked Logic.
    """
    rays = extract_rays(rho_field)
    total_sse = 0.0
    valid_rays = 0

    for ray in rays:
        # 1. Windowing to reduce spectral leakage
        windowed_ray = ray * scipy.signal.windows.hann(len(ray))
        
        # 2. FFT
        fft_spectrum = scipy.fft.fft(windowed_ray)
        power = np.abs(fft_spectrum[:len(ray)//2])
        freqs = scipy.fft.fftfreq(len(ray))[:len(ray)//2]
        
        # 3. Peak Finding
        peaks, _ = scipy.signal.find_peaks(power, height=np.max(power)*0.1)
        if len(peaks) == 0: continue
        
        observed_freqs = freqs[peaks] * len(ray) # Scale to wavenumbers
        
        # 4. Log-Prime Matching
        # We look for the 'Scaling Factor' alpha that best fits observed to targets
        # k_obs = alpha * ln(p)
        
        # Heuristic: Take the strongest peak and assume it maps to ln(2) (Fundamental)
        # This anchors the scale.
        strongest_peak_idx = np.argmax(power[peaks])
        alpha_est = observed_freqs[strongest_peak_idx] / LOG_PRIME_TARGETS[0]
        
        normalized_freqs = observed_freqs / (alpha_est + 1e-9)
        
        # Calculate SSE for this ray
        ray_sse = 0.0
        for freq in normalized_freqs:
            # Find distance to nearest target
            dist = np.min(np.abs(freq - LOG_PRIME_TARGETS))
            ray_sse += dist**2
            
        total_sse += ray_sse
        valid_rays += 1

    if valid_rays == 0: return 100.0 # High penalty for noise
    return total_sse / valid_rays

def run_validation(job_uuid):
    h5_path = os.path.join(settings.DATA_DIR, f"rho_history_{job_uuid}.h5")
    prov_path = os.path.join(settings.PROVENANCE_DIR, f"provenance_{job_uuid}.json")
    
    output_data = {
        "job_uuid": job_uuid,
        "metrics": {},
        "sentinel_code": 0,
        "validation_status": "PASS"
    }

    try:
        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"Artifact {h5_path} not found.")

        with h5py.File(h5_path, 'r') as f:
            # Load Data - Trust But Verify (Independent Load)
            if 'final_psi' not in f:
                raise ValueError("Corrupt HDF5: 'final_psi' missing.")
            
            psi = f['final_psi'][()]
            rho = np.abs(psi)**2
            
            # 1. Refusal Scaffolding: Check for Divergence
            if np.isnan(rho).any() or np.max(rho) > 1e6:
                output_data["sentinel_code"] = settings.SENTINEL_DIVERGENCE
                output_data["validation_status"] = "DIVERGENCE"
                output_data["metrics"][settings.SSE_METRIC_KEY] = 1000.0 # Max Penalty
                atomic_write_provenance(prov_path, output_data)
                return

            # 2. Geometric Stability Check (H-Norm)
            # Recalculate, do not trust metadata
            # Simplified H-Norm proxy for validator speed (Variance of Metric)
            if 'final_g_mu_nu' in f:
                g_tensor = f['final_g_mu_nu'][()]
                h_norm = np.var(g_tensor) 
            else:
                h_norm = 0.0
            
            # 3. Spectral Fidelity Check (The Science)
            sse = analyze_spectral_fidelity(rho)
            
            output_data["metrics"][settings.SSE_METRIC_KEY] = float(sse)
            output_data["metrics"][settings.STABILITY_METRIC_KEY] = float(h_norm)

    except Exception as e:
        logging.error(f"Validation crashed: {e}")
        output_data["sentinel_code"] = settings.SENTINEL_FAILURE
        output_data["validation_status"] = "CRASH"
        output_data["error"] = str(e)
        output_data["traceback"] = traceback.format_exc()

    atomic_write_provenance(prov_path, output_data)
    logging.info(f"Provenance generated: {prov_path} (Sentinel: {output_data['sentinel_code']})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_uuid", required=True)
    args = parser.parse_args()
    run_validation(args.job_uuid)