"""
MODULE: tda_analyzer.py
CLASSIFICATION: V11.0 Analysis Tool
GOAL: Spectral / TDA-adjacent tools for radial FFT and peak finding.
      Extracted from IRER dev runs for V11.5+ compatibility.
CONTRACT ID: IO-TDA-V11
"""
from __future__ import annotations
from typing import Tuple, Optional, List

# ---------------------------------------------------------------------------
# Lazy imports (as used in the notebook patches)
# ---------------------------------------------------------------------------

try:
    _lazy_imports  # type: ignore[name-defined]
except NameError:
    def _lazy_imports():
        """
        Late-import numpy, h5py, and scipy.signal (if available).

        Returns
        -------
        np : module
        h5py : module
        sp_signal : module | None
        """
        import importlib
        np = importlib.import_module("numpy")
        h5py = importlib.import_module("h5py")
        try:
            sp_signal = importlib.import_module("scipy.signal")
        except Exception:
            sp_signal = None
        return np, h5py, sp_signal


# ---------------------------------------------------------------------------
# Core spectral helper: multi-ray radial FFT
# ---------------------------------------------------------------------------

def _center_rays_indices(shape: Tuple[int, int], n_rays: int = 96) -> List[List[Tuple[int, int]]]:
    """
    Compute integer pixel coordinates for n_rays radial rays
    from the centre of a 2D field.

    Parameters
    ----------
    shape : (H, W)
        Height and width of the field.
    n_rays : int
        Number of rays to sample.

    Returns
    -------
    rays : list[list[tuple[int,int]]]
        Each element is a list of (iy, ix) indices along a ray.
    """
    np, _, _ = _lazy_imports()
    H, W = shape
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    R_max = int(np.hypot(cy, cx))  # max radius to stay in bounds

    thetas = np.linspace(0.0, 2.0 * np.pi, n_rays, endpoint=False)
    rays = []

    for theta in thetas:
        coords = []
        for r in range(R_max):
            iy = int(round(cy + r * np.sin(theta)))
            ix = int(round(cx + r * np.cos(theta)))
            if 0 <= iy < H and 0 <= ix < W:
                coords.append((iy, ix))
            else:
                break
        if coords:
            rays.append(coords)

    return rays


def _multi_ray_fft_1d(
    field2d,
    n_rays: int = 96,
    detrend: bool = True,
    window: bool = True,
) -> Tuple["np.ndarray", "np.ndarray"]:
    """
    Radial multi-ray FFT over a 2D field.

    Contract: returns (k, mean_power) with k.shape == mean_power.shape.

    Parameters
    ----------
    field2d : array_like, shape (H, W)
        2D scalar field (e.g. final rho slice).
    n_rays : int
        Number of radial rays from centre used for averaging.
    detrend : bool
        If True and scipy.signal is available, remove linear trend per ray.
    window : bool
        If True, apply a Hann window before FFT.

    Returns
    -------
    k : np.ndarray
        1D wavenumber vector (rFFT frequencies).
    mean_power : np.ndarray
        Mean power spectrum over all valid rays.
    """
    np, _, sp_signal = _lazy_imports()
    field2d = np.asarray(field2d, dtype=float)
    H, W = field2d.shape

    rays = _center_rays_indices((H, W), n_rays=n_rays)
    spectra = []

    for coords in rays:
        sig = np.array([field2d[iy, ix] for (iy, ix) in coords], dtype=float)
        if sig.size < 4:
            continue

        if detrend and sp_signal is not None:
            sig = sp_signal.detrend(sig, type="linear")

        if window:
            n = len(sig)
            if n > 1:
                w = 0.5 * (1 - np.cos(2 * np.pi * np.arange(n) / (n - 1)))
            else:
                w = 1.0
            sig = sig * w

        fft = np.fft.rfft(sig)
        power = (fft.conj() * fft).real
        spectra.append(power)

    if not spectra:
        raise ValueError("No valid rays for FFT (field too small).")

    # Pad/align all rays to common length and only then compute k
    maxL = max(map(len, spectra))
    P = np.zeros((len(spectra), maxL))
    for i, p in enumerate(spectra):
        P[i, :len(p)] = p

    mean_power = P.mean(axis=0)
    k = np.fft.rfftfreq(maxL, d=1.0)

    # CONTRACT: k.shape == power.shape
    assert k.shape == mean_power.shape, (
        f"Internal contract violated: k{ k.shape } vs P{ mean_power.shape }"
    )

    return k, mean_power


# ---------------------------------------------------------------------------
# Peak finder
# ---------------------------------------------------------------------------

def _find_peaks(
    k,
    power,
    max_peaks: int = 12,
    prominence: float = 0.02,
    strict: bool = True,
):
    """
    Select dominant peaks in a 1D power spectrum.

    If strict=True, enforce k.shape == power.shape and raise a descriptive
    ValueError on mismatch (this is the guard against upstream bugs in
    _multi_ray_fft_1d etc).

    Parameters
    ----------
    k : array_like
        Wavenumber axis (must match power.shape if strict=True).
    power : array_like
        Power spectrum values.
    max_peaks : int
        Maximum number of peaks to return.
    prominence : float
        Prominence threshold passed to scipy.signal.find_peaks if available.
    strict : bool
        If True, enforce shape equality; if False, truncate to min length.

    Returns
    -------
    peak_k : np.ndarray
    peak_power : np.ndarray
    """
    np, _, sp_signal = _lazy_imports()
    k = np.asarray(k)
    power = np.asarray(power)

    if strict:
        if k.shape != power.shape:
            raise ValueError(
                f"_find_peaks input contract violated: "
                f"k.shape {k.shape} != power.shape {power.shape} "
                f"(this indicates an upstream bug; investigate _multi_ray_fft_1d)."
            )
    else:
        n = min(k.size, power.size)
        k, power = k[:n], power[:n]

    # Drop DC
    mask = k > 0
    k_filtered, power_filtered = k[mask], power[mask]

    if k_filtered.size == 0:
        return np.array([]), np.array([])

    if sp_signal is None:
        # naive: pick top-N excluding k=0
        order = np.argsort(power_filtered)[::-1][:max_peaks]
        order = order[(order >= 0) & (order < k_filtered.size)]
        sel = np.array(sorted(order, key=lambda i: k_filtered[i]))
        return k_filtered[sel], power_filtered[sel]

    idx, _ = sp_signal.find_peaks(power_filtered, prominence=prominence)
    if idx.size == 0:
        idx = np.argsort(power_filtered)[::-1][:max_peaks]

    idx = idx[(idx >= 0) & (idx < k_filtered.size)]
    idx = idx[np.argsort(power_filtered[idx])[::-1]][:max_peaks]
    idx = np.array(sorted(idx, key=lambda i: k_filtered[i]))

    return k_filtered[idx], power_filtered[idx]


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------

def analyze_field_spectrum(
    field2d,
    n_rays: int = 96,
    max_peaks: int = 12,
    prominence: float = 0.02,
    strict: bool = True,
):
    """
    Convenience wrapper: take a 2D field, compute radial spectrum,
    and extract dominant peaks.

    Returns
    -------
    result : dict
        {
            "k": k,
            "power": power,
            "peak_k": peak_k,
            "peak_power": peak_power,
        }
    """
    k, power = _multi_ray_fft_1d(field2d, n_rays=n_rays)
    peak_k, peak_power = _find_peaks(
        k, power,
        max_peaks=max_peaks,
        prominence=prominence,
        strict=strict,
    )
    return {
        "k": k,
        "power": power,
        "peak_k": peak_k,
        "peak_power": peak_power,
    }


def plot_spectrum_with_peaks(
    k,
    power,
    peak_k: Optional["np.ndarray"] = None,
    peak_power: Optional["np.ndarray"] = None,
    title: str = "Radial Power Spectrum",
):
    """
    Simple Matplotlib plot helper.

    Parameters
    ----------
    k : array_like
    power : array_like
    peak_k : array_like, optional
    peak_power : array_like, optional
    title : str
    """
    np, _, _ = _lazy_imports()
    import matplotlib.pyplot as plt

    k = np.asarray(k)
    power = np.asarray(power)

    plt.figure()
    plt.plot(k, power, label="mean power")
    if peak_k is not None and peak_power is not None and len(peak_k) > 0:
        plt.scatter(peak_k, peak_power, marker="x", label="peaks")
    plt.xlabel("k")
    plt.ylabel("Power")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()