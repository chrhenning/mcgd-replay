import numpy as np
import pytest

from src.metrics import integrated_autocorrelation_time


def ar1(rho, n, seed=0):
    """Stationary AR(1) series — the one process whose tau_int is known exactly."""
    rng = np.random.default_rng(seed)
    x = np.empty(n)
    x[0] = rng.standard_normal()
    innovation = np.sqrt(1.0 - rho**2) * rng.standard_normal(n)
    for k in range(1, n):
        x[k] = rho * x[k - 1] + innovation[k]
    return x


@pytest.mark.parametrize("rho", [0.0, 0.9])
def test_tau_int_matches_ar1_closed_form(rho):
    """tau_int of an AR(1) process is (1+rho)/(1-rho)."""
    expected = (1.0 + rho) / (1.0 - rho)
    x = ar1(rho, 200_000)
    assert integrated_autocorrelation_time(x) == pytest.approx(expected, rel=0.15)


def test_tau_int_rejects_a_series_too_short_to_resolve_it():
    """Sokal's rule always finds a window, so under-resolution needs its own check."""
    x = ar1(0.99, 2_000)  # tau_int = 199, so tol=50 wants 10,000 steps
    with pytest.raises(ValueError, match="too short to resolve"):
        integrated_autocorrelation_time(x)
