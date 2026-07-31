import numpy as np
import pytest

from src.metrics import integrated_autocorrelation_time


@pytest.mark.parametrize("rho", [0.0, 0.9])
def test_tau_int_matches_ar1_closed_form(rho):
    """AR(1) is the one process whose tau_int is known exactly: (1+rho)/(1-rho)."""
    rng = np.random.default_rng(0)
    n = 200_000

    x = np.empty(n)
    x[0] = rng.standard_normal()
    innovation = np.sqrt(1.0 - rho**2) * rng.standard_normal(n)
    for k in range(1, n):
        x[k] = rho * x[k - 1] + innovation[k]

    expected = (1.0 + rho) / (1.0 - rho)
    assert integrated_autocorrelation_time(x) == pytest.approx(expected, rel=0.15)
