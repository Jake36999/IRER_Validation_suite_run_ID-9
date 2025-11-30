"""
MODULE: tda_taxonomy_validator.py
CLASSIFICATION: Structural Validation Module (ASTE / IRER)
GOAL: Implements the "Quantule Taxonomy" by applying Topological
      Data Analysis (persistent homology via ripser) to a single
      simulation run's collapse events.
CONTRACT ID: IO-VAL-TDA-V11
"""

from __future__ import annotations

import os
import sys
import argparse
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd

# --- Import Shared Settings (BASE_DIR, DATA_DIR, PROVENANCE_DIR) ---
try:
    import settings
except ImportError:
    print(
        "FATAL: 'settings.py' not found. "
        "Please ensure it exists and defines DATA_DIR / PROVENANCE_DIR.",
        file=sys.stderr,
    )
    sys.exit(1)

# --- Handle Specialized TDA Dependencies ---
TDA_LIBS_AVAILABLE = False
try:
    from ripser import ripser
    import matplotlib.pyplot as plt
    from persim import plot_diagrams

    TDA_LIBS_AVAILABLE = True
except ImportError:
    print("=" * 60, file=sys.stderr)
    print("WARNING: TDA libraries 'ripser', 'persim', 'matplotlib' not found.", file=sys.stderr)
    print("TDA Module is BLOCKED. Please install dependencies:", file=sys.stderr)
    print("  pip install ripser persim matplotlib pandas", file=sys.stderr)
    print("=" * 60, file=sys.stderr)


# ---------------------------------------------------------------------------
# 1. Data Loading
# ---------------------------------------------------------------------------

def load_collapse_data(filepath: str) -> Optional[np.ndarray]:
    """
    Load (x, y, z) coordinates from a <hash>_quantule_events.csv file.

    Returns
    -------
    point_cloud : np.ndarray shape (N, 3) or None on failure
    """
    print(f"[TDA] Loading collapse data from: {filepath}...")
    if not os.path.exists(filepath):
        print(f"ERROR: File not found: {filepath}", file=sys.stderr)
        return None

    try:
        df = pd.read_csv(filepath)

        if not all(col in df.columns for col in ("x", "y", "z")):
            print("ERROR: CSV must contain 'x', 'y', and 'z' columns.", file=sys.stderr)
            return None

        point_cloud = df[["x", "y", "z"]].values
        if point_cloud.shape[0] == 0:
            print("ERROR: CSV contains no data points.", file=sys.stderr)
            return None

        print(f"[TDA] Loaded {len(point_cloud)} collapse events.")
        return point_cloud

    except Exception as e:
        print(f"ERROR: Could not load data. {e}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# 2. Persistent Homology Core
# ---------------------------------------------------------------------------

def compute_persistence(data: np.ndarray, max_dim: int = 2) -> List[np.ndarray]:
    """
    Compute persistent homology of the point cloud with ripser.

    Parameters
    ----------
    data : np.ndarray
        Point cloud, shape (N, D).
    max_dim : int
        Maximum homology dimension (e.g. 2 => H0, H1, H2).

    Returns
    -------
    dgms : list of np.ndarray
        dgms[d] is the diagram for H_d with columns [birth, death].
    """
    if not TDA_LIBS_AVAILABLE:
        raise RuntimeError("TDA libraries not available; cannot compute persistence.")

    print(f"[TDA] Computing persistent homology (max_dim={max_dim})...")
    result = ripser(data, maxdim=max_dim)
    dgms = result["dgms"]
    print("[TDA] Persistent homology computation complete.")
    return dgms


# ---------------------------------------------------------------------------
# 3. Taxonomy Analysis
# ---------------------------------------------------------------------------

def analyze_taxonomy(
    dgms: List[np.ndarray],
    persistence_threshold: float = 0.5,
) -> str:
    """
    Turn persistence diagrams into a human-readable "Quantule Taxonomy".

    Parameters
    ----------
    dgms : list of np.ndarray
        ripser diagrams, dgms[d] is H_d.
    persistence_threshold : float
        Minimum (death - birth) to consider a feature "persistent".

    Returns
    -------
    report : str
        Multi-line report summarizing H0/H1/H2 persistent feature counts.
    """
    if not dgms:
        return "Taxonomy: FAILED (No diagrams computed)."

    def count_persistent_features(diagram: np.ndarray, dim: int) -> int:
        if diagram.size == 0:
            return 0

        persistence = diagram[:, 1] - diagram[:, 0]

        # For H0, ignore infinite bars (the one connected component that never dies)
        if dim == 0:
            persistent_features = persistence[
                (persistence > persistence_threshold) & (persistence != np.inf)
            ]
        else:
            persistent_features = persistence[persistence > persistence_threshold]

        return int(len(persistent_features))

    h0_count = count_persistent_features(dgms[0], 0)
    h1_count = count_persistent_features(dgms[1], 1) if len(dgms) > 1 else 0
    h2_count = count_persistent_features(dgms[2], 2) if len(dgms) > 2 else 0

    taxonomy_str = (
        "--- Quantule Taxonomy Report ---\n"
        f"  - H0 (Components / Spots): {h0_count} persistent features\n"
        f"  - H1 (Loops / Tunnels):    {h1_count} persistent features\n"
        f"  - H2 (Cavities / Voids):   {h2_count} persistent features"
    )
    return taxonomy_str


# ---------------------------------------------------------------------------
# 4. Plotting (Persistence Diagram)
# ---------------------------------------------------------------------------

def plot_taxonomy(
    dgms: List[np.ndarray],
    run_id: str,
    output_dir: str,
) -> str:
    """
    Generate and save a persistence-diagram-based taxonomy plot.

    Parameters
    ----------
    dgms : list of diagrams from ripser
    run_id : str
        Hash / UUID of the run.
    output_dir : str
        Directory for the PNG output.

    Returns
    -------
    filepath : str
        Path to saved PNG file.
    """
    if not TDA_LIBS_AVAILABLE:
        raise RuntimeError("TDA libraries not available; cannot plot taxonomy.")

    os.makedirs(output_dir, exist_ok=True)

    print(f"[TDA] Generating persistence diagram plot for run {run_id}...")
    plt.figure(figsize=(15, 5))

    # H0
    plt.subplot(1, 3, 1)
    plot_diagrams(dgms[0], show=False, labels=["H0 (Components)"])
    plt.title("H0 Features (Components)")

    # H1
    plt.subplot(1, 3, 2)
    if len(dgms) > 1 and dgms[1].size > 0:
        plot_diagrams(dgms[1], show=False, labels=["H1 (Loops)"])
    plt.title("H1 Features (Loops/Tunnels)")

    # H2
    plt.subplot(1, 3, 3)
    if len(dgms) > 2 and dgms[2].size > 0:
        plot_diagrams(dgms[2], show=False, labels=["H2 (Cavities)"])
    plt.title("H2 Features (Cavities/Voids)")

    plt.suptitle(f"Quantule Taxonomy (Persistence Diagram) for Run-ID: {run_id}")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    filename = os.path.join(output_dir, f"tda_taxonomy_{run_id}.png")
    plt.savefig(filename)
    print(f"[TDA] Taxonomy plot saved to: {filename}")
    plt.close()

    return filename


# ---------------------------------------------------------------------------
# 5. Programmatic Entry Point (non-CLI)
# ---------------------------------------------------------------------------

def run_tda_taxonomy_for_hash(config_hash: str) -> dict:
    """
    Run the full TDA taxonomy pipeline for a given config hash.

    Returns
    -------
    result : dict
        {
            "status": "ok" | "error",
            "hash": config_hash,
            "taxonomy": <str> | None,
            "plot_path": <str> | None,
            "error": <str> | None
        }
    """
    if not TDA_LIBS_AVAILABLE:
        return {
            "status": "error",
            "hash": config_hash,
            "taxonomy": None,
            "plot_path": None,
            "error": "TDA libraries not available (ripser / persim / matplotlib).",
        }

    # Resolve paths using settings
    data_filepath = os.path.join(settings.DATA_DIR, f"{config_hash}_quantule_events.csv")
    
    # Fallback output dir if PROVENANCE_DIR missing
    output_dir = getattr(settings, "PROVENANCE_DIR", settings.DATA_DIR)

    point_cloud = load_collapse_data(data_filepath)
    if point_cloud is None:
        return {
            "status": "error",
            "hash": config_hash,
            "taxonomy": None,
            "plot_path": None,
            "error": f"No valid point cloud for hash {config_hash}.",
        }

    max_dim = 2 if point_cloud.shape[1] == 3 else 1
    dgms = compute_persistence(point_cloud, max_dim=max_dim)
    plot_path = plot_taxonomy(dgms, config_hash, str(output_dir))
    taxonomy_str = analyze_taxonomy(dgms)

    return {
        "status": "ok",
        "hash": config_hash,
        "taxonomy": taxonomy_str,
        "plot_path": plot_path,
        "error": None,
    }


# ---------------------------------------------------------------------------
# 6. CLI Entrypoint
# ---------------------------------------------------------------------------

def main():
    """
    CLI entrypoint for TDA Taxonomy Validator.

    Usage:
        python tda_taxonomy_validator.py --hash <CONFIG_HASH>
    """
    print("--- TDA Structural Validation Module ---")

    if not TDA_LIBS_AVAILABLE:
        print("FATAL: TDA Module is BLOCKED. Please install dependencies.", file=sys.stderr)
        sys.exit(1)

    parser = argparse.ArgumentParser(description="ASTE / IRER TDA Taxonomy Validator")
    parser.add_argument(
        "--hash",
        type=str,
        required=True,
        help="Config hash of the run to analyze (prefix of quantule_events CSV).",
    )
    args = parser.parse_args()

    result = run_tda_taxonomy_for_hash(args.hash)

    if result["status"] != "ok":
        print(f"FATAL: {result['error']}", file=sys.stderr)
        sys.exit(1)

    print("\n--- Validation Result ---")
    print(f"Analysis for: {args.hash}")
    print(result["taxonomy"])
    print(f"Plot saved to: {result['plot_path']}")
    print("-------------------------")


if __name__ == "__main__":
    main()