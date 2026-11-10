"""Two-sided CUSUM changepoint detector (pure NumPy, deterministic)."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def cusum_changepoints(
    x: npt.NDArray[np.floating],
    threshold: float = 5.0,
    drift: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    """Detect mean downshifts and upshifts.

    Returns:
        (downshift_indices, upshift_indices) as int32 arrays.
        Downshifts (speed collapse) are candidate dead-ball starts.
        Upshifts (restart impulse) are candidate dead-ball ends.
    """
    y = np.asarray(x, dtype=np.float64)
    n = len(y)
    empty = np.array([], dtype=np.int32)
    if n == 0:
        return empty, empty

    med = np.nanmedian(y)
    if np.isnan(med):
        med = 0.0
    y = np.where(np.isnan(y), med, y)
    mu = float(np.mean(y))

    gp = 0.0
    gn = 0.0
    down: list[int] = []
    up: list[int] = []
    for i in range(n):
        gp = max(0.0, gp + (y[i] - mu - drift))
        gn = max(0.0, gn + (mu - drift - y[i]))
        if gp > threshold:
            up.append(i)
            gp = 0.0
        if gn > threshold:
            down.append(i)
            gn = 0.0

    return np.array(down, dtype=np.int32), np.array(up, dtype=np.int32)
