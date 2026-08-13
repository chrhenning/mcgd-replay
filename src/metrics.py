"""Diagnostics shared across experiments.

Correlated streams are compared on a per-sample axis, so how much independent
information a run of ``n`` chain steps actually carries is the quantity everything
downstream is normalised by.
"""

import numpy as np

__all__ = [
    "autocorrelation",
    "integrated_autocorrelation_time",
    "effective_sample_size",
]


def autocorrelation(x):
    """Autocorrelations of a scalar series at lags ``0..n-1``, normalised to 1 at lag 0.

    Via FFT: on the ~1e5-step series this runs on, the direct O(n^2) sum is the
    bottleneck.
    """
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n = len(x)

    # Zero-pad to at least 2n so the circular FFT correlation matches the linear one.
    nfft = 1 << (2 * n - 1).bit_length()
    f = np.fft.rfft(x, nfft)
    acf = np.fft.irfft(f * np.conjugate(f), nfft)[:n]
    return acf / acf[0]


def integrated_autocorrelation_time(x, c=5.0, tol=50.0):
    """Integrated autocorrelation time via Sokal's automatic windowing.

    ``tau(M) = 1 + 2 * sum_{k<=M} acf(k)`` is unbiased but its variance grows with
    ``M``; Sokal truncates at the first ``M >= c * tau(M)``, the standard compromise.

    Such an ``M`` always exists, but not always for the right reason: autocovariances of
    a mean-centred series sum to ``c_0 / 2``, so ``tau(n - 1)`` is identically zero and
    the rule always triggers, if need be down in the noise floor of the tail. Hence
    ``tol``: a truncated estimate is otherwise indistinguishable from a converged one.

    Returns the estimate in chain steps; roughly 1 for i.i.d. samples.

    Raises:
        ValueError: If the series is shorter than ``tol`` correlation times.
    """
    acf = autocorrelation(x)
    tau = 2.0 * np.cumsum(acf) - 1.0

    windows = np.arange(len(tau))
    m = int(np.argmax(windows >= c * tau))
    tau_int = float(tau[m])

    if len(x) < tol * tau_int:
        needed = int(np.ceil(tol * tau_int))
        raise ValueError(
            f"series of {len(x)} steps is too short to resolve tau_int, estimated at "
            f"{tau_int:.0f} steps; needs at least tol * tau_int = {needed} steps"
        )
    return tau_int


def effective_sample_size(x, c=5.0, tol=50.0):
    """Number of independent samples the series is worth, ``n / tau_int``."""
    return len(x) / integrated_autocorrelation_time(x, c=c, tol=tol)
