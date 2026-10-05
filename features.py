"""Shared signal features and project-relative data paths."""

import os
from pathlib import Path

import numpy as np
import pywt
from scipy.stats import kurtosis, skew

SAMPLING_RATE = 836
PROJECT_DIR = Path(__file__).resolve().parent


def resolve_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    root = Path(os.getenv("DRIVESENSE_DATA_DIR", str(PROJECT_DIR)))
    resolved = root / path
    if not resolved.is_file():
        raise FileNotFoundError(
            f"Missing dataset: {resolved}. See DATA_FORMAT.md for required CSVs."
        )
    return resolved


def signal_features(signal, *, extended=True):
    """Return 17 features, or 15 without zero crossing rate and entropy.

    Invalid samples are rejected rather than deleted, preserving segment timing.
    Empty signals return a correctly sized zero vector for API compatibility.
    """
    data = np.asarray(signal, dtype=float)
    if data.ndim != 1:
        raise ValueError("Expected a one-dimensional signal.")
    count = 17 if extended else 15
    if data.size == 0:
        return [0.0] * count
    if not np.isfinite(data).all():
        raise ValueError("Segment contains missing or nonfinite samples.")
    constant = np.ptp(data) == 0
    values = [
        np.mean(data), np.std(data),
        0.0 if constant else skew(data),
        0.0 if constant else kurtosis(data),
        np.ptp(data), np.percentile(data, 95),
        np.std(data) / (np.mean(np.abs(data)) + 1e-10),
    ]
    power = np.abs(np.fft.rfft(data)) ** 2 / len(data)
    freqs = np.fft.rfftfreq(len(data), d=1 / SAMPLING_RATE)
    total = power.sum() + 1e-10
    for low, high in [(0, 5), (5, 20), (20, 50), (50, 100)]:
        values.append(power[(freqs >= low) & (freqs < high)].sum() / total)
    scales = pywt.central_frequency("mexh") * SAMPLING_RATE / np.arange(1, 128)
    coefficients, _ = pywt.cwt(data, scales, "mexh")
    magnitude = np.abs(coefficients)
    values.extend([
        np.mean(magnitude), np.std(magnitude),
        np.percentile(magnitude, 95), np.sum(magnitude ** 2),
    ])
    if extended:
        values.append(np.count_nonzero(np.diff(np.signbit(data))) / len(data))
        counts, _ = np.histogram(data, bins=20)
        probabilities = counts[counts > 0] / counts.sum()
        values.append(-np.sum(probabilities * np.log2(probabilities)))
    result = np.asarray(values, dtype=float)
    if not np.isfinite(result).all():
        raise ValueError("Feature extraction produced nonfinite values.")
    return result.tolist()
