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

    Via FFT because the sweep needs this on series of ~1e5 steps, where the direct
    O(n^2) sum is the bottleneck.
    """
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n = len(x)

    # Zero-pad to at least 2n so the circular FFT correlation matches the linear one.
    nfft = 1 << (2 * n - 1).bit_length()
    f = np.fft.rfft(x, nfft)
    acf = np.fft.irfft(f * np.conjugate(f), nfft)[:n]
    return acf / acf[0]


def integrated_autocorrelation_time(x, c=5.0):
    """Integrated autocorrelation time via Sokal's automatic windowing.

    The estimator ``tau(M) = 1 + 2 * sum_{k<=M} acf(k)`` is unbiased but its variance
    grows with ``M``, since high lags contribute noise rather than signal. Sokal's rule
    truncates at the first ``M >= c * tau(M)`` — the smallest window still several
    correlation times long — which is the standard bias/variance compromise.

    Returns the estimate in chain steps; roughly 1 for i.i.d. samples.
    """
    acf = autocorrelation(x)
    tau = 2.0 * np.cumsum(acf) - 1.0

    windows = np.arange(len(tau))
    converged = windows >= c * tau
    # Falling back to the longest window means the series is too short to resolve tau;
    # the estimate is then a lower bound rather than an error.
    m = int(np.argmax(converged)) if converged.any() else len(tau) - 1
    return float(tau[m])


def effective_sample_size(x, c=5.0):
    """Number of independent samples the series is worth, ``n / tau_int``."""
    return len(x) / integrated_autocorrelation_time(x, c=c)
