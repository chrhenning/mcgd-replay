"""Per-day sampling chains — the correlated stream a day-model trains on.

The chains here stand in for a robot's sensor stream: consecutive samples are near
neighbours in data space, so the day explores its distribution by walking through it
rather than by drawing from it. Each chain is restricted to a single mode (one "day")
and is exactly stationary at that mode's conditional law, so correlation is the only
thing that separates it from i.i.d. sampling.
"""

import numpy as np

__all__ = ["fold", "MoonChain"]


def fold(x, hi=np.pi):
    """Reflect the real line onto ``[0, hi]`` as a triangle wave.

    Reflection rather than clipping or wrapping, because it is what keeps the angle
    kernel symmetric: ``Unif[0, hi]`` is then preserved exactly and no probability mass
    piles up at the endpoints.
    """
    return hi - np.abs(np.mod(x, 2.0 * hi) - hi)


class MoonChain:
    """Autocorrelated stream for one moon of the two-moons distribution.

    The state is the **latent** ``xi = (t, eps) in [0, pi] x R^2``, never the decoded
    point ``x``. This matters: the decoded sequence alone is a hidden Markov process,
    not a Markov chain, so treating ``R^2`` as the sample space would violate the MCGD
    assumptions outright. Callers keep the chain in latent space and treat ``x`` as an
    observation, defining the per-sample loss as ``F(theta; xi) := loss(theta,
    decode(xi))``.

    Both components are reversible w.r.t. their stationary laws — a symmetric kernel
    against a uniform target, an OU/AR(1) kernel against a Gaussian target — so the
    product chain is reversible, which the nonconvex continuous-state MCGD result
    requires. The chain is started *in* stationarity, so there is no burn-in and every
    sample is marginally exact.

    ``delta`` (angle step size) and ``rho`` (noise correlation) are the two correlation
    knobs, and are ordinary attributes that may be reassigned between steps: every
    kernel in this family has the same stationary law for any value of them, so a
    time-inhomogeneous schedule leaves the marginal exactly ``Pi`` at every step.
    """

    def __init__(self, moon=0, delta=0.05, rho=0.9, sigma=0.08, rng=None):
        """Initialise the chain in its stationary distribution.

        ``sigma`` is the marginal noise scale, matching ``make_moons(noise=sigma)``;
        ``rng`` is anything ``np.random.default_rng`` accepts.

        Raises:
            ValueError: If ``moon`` is not 0 or 1, or ``rho`` is outside ``[0, 1)``.
        """
        if moon not in (0, 1):
            raise ValueError(f"moon must be 0 or 1, got {moon}")
        if not 0.0 <= rho < 1.0:
            raise ValueError(f"rho must lie in [0, 1), got {rho}")

        self.moon = moon
        self.delta = delta
        self.rho = rho
        self.sigma = sigma
        self.rng = np.random.default_rng(rng)

        self.t = self.rng.uniform(0.0, np.pi)
        self.eps = self.rng.normal(0.0, sigma, size=2)

    @property
    def state(self):
        """Current latent state as a ``(3,)`` array ``[t, eps_x, eps_y]``."""
        return np.array([self.t, self.eps[0], self.eps[1]])

    def step(self):
        """Advance the chain one step and return the new latent state."""
        self.t = fold(self.t + self.delta * self.rng.standard_normal())
        # Scaling the innovation by sqrt(1 - rho^2) is what holds the marginal variance
        # at sigma^2 for every rho, so rho moves correlation without moving the target.
        self.eps = self.rho * self.eps + self.sigma * np.sqrt(
            1.0 - self.rho**2
        ) * self.rng.standard_normal(2)
        return self.state

    def run(self, n):
        """Advance the chain ``n`` steps, returning the ``(n, 3)`` latent trajectory."""
        return np.array([self.step() for _ in range(n)])

    def decode(self, latent):
        """Map a ``(..., 3)`` latent state or trajectory to observations on this moon."""
        latent = np.asarray(latent, dtype=float)
        t, eps = latent[..., 0], latent[..., 1:]
        if self.moon == 0:
            mu = np.stack([np.cos(t), np.sin(t)], axis=-1)
        else:
            mu = np.stack([1.0 - np.cos(t), 0.5 - np.sin(t)], axis=-1)
        return mu + eps
