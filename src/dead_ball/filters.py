"""Signal processing filters for tracking and kinematic data.

Provides a fast, pure NumPy implementation of the Savitzky-Golay filter
to smooth noisy player speeds and trajectories without introducing external
heavy dependencies like scipy.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt


def _calc_savgol_coeffs(window_length: int, polyorder: int, deriv: int = 0) -> np.ndarray:
    """Computes Savitzky-Golay filter coefficients using polynomial least squares."""
    if window_length % 2 == 0:
        raise ValueError("window_length must be an odd positive integer")
    if window_length <= polyorder:
        raise ValueError("window_length must be greater than polyorder")

    half_window = (window_length - 1) // 2
    # Coordinate system around 0: [-half_window, ..., 0, ..., half_window]
    k = np.arange(-half_window, half_window + 1, dtype=np.float64)

    # Vandermonde matrix: V[i, j] = k[i]^j
    V = k[:, np.newaxis] ** np.arange(polyorder + 1, dtype=np.float64)

    # Solve least squares for unit vector selecting derivative
    # (V^T * V)^(-1) * V^T
    pinv = np.linalg.pinv(V)

    # Select row corresponding to requested derivative order
    coeffs = pinv[deriv] * float(math.factorial(deriv))
    return coeffs


def savitzky_golay(
    y: npt.NDArray[np.floating],
    window_length: int = 25,
    polyorder: int = 2,
    mode: str = "reflect",
) -> np.ndarray:
    """Smooths a 1D signal using a Savitzky-Golay polynomial filter.

    Unlike a simple moving average, Savitzky-Golay preserves steep transitions
    and impulse peaks (e.g. sudden accelerations at dead ball restarts)
    while removing high-frequency tracking jitter.

    Args:
        y: 1D array of floats (e.g. player speeds or ball speed).
        window_length: Length of the filter window (must be odd).
        polyorder: Order of the polynomial used to fit the samples.
        mode: Boundary padding mode ('reflect', 'nearest', 'constant').

    Returns:
        Smoothed 1D numpy array of the same shape as y.
    """
    y_arr = np.asarray(y, dtype=np.float64)
    if y_arr.ndim != 1:
        raise ValueError("Input signal y must be 1-dimensional")

    n = len(y_arr)
    if n < window_length:
        return y_arr.copy()

    coeffs = _calc_savgol_coeffs(window_length, polyorder, deriv=0)
    half_window = (window_length - 1) // 2

    # Pad boundaries to keep output array identical in length
    # Linear continuation (odd reflection) preserves trends across boundaries
    if mode == "interp" or mode == "reflect":
        pad_left = 2.0 * y_arr[0] - y_arr[1 : half_window + 1][::-1]
        pad_right = 2.0 * y_arr[-1] - y_arr[-half_window - 1 : -1][::-1]
    elif mode == "nearest":
        pad_left = np.full(half_window, y_arr[0])
        pad_right = np.full(half_window, y_arr[-1])
    else:  # constant 0
        pad_left = np.zeros(half_window, dtype=np.float64)
        pad_right = np.zeros(half_window, dtype=np.float64)

    padded = np.concatenate([pad_left, y_arr, pad_right])

    # Convolve with reversed coefficients
    smoothed = np.convolve(padded, coeffs[::-1], mode="valid")
    return smoothed.astype(y.dtype)
