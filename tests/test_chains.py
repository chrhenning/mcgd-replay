import numpy as np
import pytest
from scipy.stats import ks_2samp
from sklearn.datasets import make_moons

from src.chains import MoonChain, fold

# The correlation grid the Phase-1 sweep runs on. Stationarity has to hold at every
# setting, since the whole design rests on correlation being the only difference
# between arms.
DELTAS = [0.02, 0.05, 0.12, 0.3, 0.8]
RHOS = [0.0, 0.5, 0.9, 0.99]

N_CHAINS = 10_000
N_STEPS = 8
ALPHA = 1e-3


def assert_stationary(moon, delta, rho):
    """Assert the chain's law after ``N_STEPS`` is still the moon's own law.

    Many short chains rather than one long run, because the two answer different
    questions. A long run tests *mixing*, which we deliberately vary — at ``delta=0.02``
    the angle needs ~(pi/delta)^2 steps just to cross its interval, so any feasible run
    has an effective sample size near 1 and KS rejects however exact the kernel is.
    Stepping independent chains from a stationary start tests ``pi P^k = pi`` directly,
    on genuinely independent samples, which is what KS needs to be valid.
    """
    rng = np.random.default_rng(0)
    chains = [MoonChain(moon, delta, rho, rng=rng) for _ in range(N_CHAINS)]
    sample = chains[0].decode([chain.run(N_STEPS)[-1] for chain in chains])

    x, label = make_moons(n_samples=8 * N_CHAINS, noise=0.08, random_state=1)
    reference = x[label == moon]
    for coord in (0, 1):
        assert ks_2samp(sample[:, coord], reference[:, coord]).pvalue > ALPHA


def test_fold_reflects():
    assert fold(np.pi + 0.3) == pytest.approx(np.pi - 0.3)
    assert fold(-0.3) == pytest.approx(0.3)

    inside = np.linspace(0.0, np.pi, 17)
    assert fold(inside) == pytest.approx(inside)

    far = fold(np.linspace(-50.0, 50.0, 1001))
    assert np.all((far >= 0.0) & (far <= np.pi))


@pytest.mark.parametrize("rho", RHOS)
@pytest.mark.parametrize("delta", DELTAS)
def test_chain_is_stationary_at_the_moon_law(delta, rho):
    """Milestone M0's acceptance criterion.

    Nothing downstream is trustworthy unless the day stream targets exactly the day's
    distribution.
    """
    assert_stationary(moon=0, delta=delta, rho=rho)


def test_second_moon_decodes_to_its_own_half():
    """The kernel is moon-independent, so moon 1 only needs its decoder checked."""
    assert_stationary(moon=1, delta=0.05, rho=0.9)
