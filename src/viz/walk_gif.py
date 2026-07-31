"""Animation of a day's stream walking through its own distribution.

The point the figure has to make is that the day never *draws* from its
distribution, it *traverses* it: the walk is local in data space, and the uniform
marginal over the angle only appears once enough of the day has elapsed. Hence the
two panels — where the stream is now, and what it has covered so far.

Rendering takes an already-simulated trajectory, so simulation and drawing stay
separable and later figures can reuse the same arrays.
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba
from sklearn.datasets import make_moons

__all__ = ["render_walk_gif"]

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
WALKER = "#2a78d6"
CLOUD = "#c9c8c3"

N_BINS = 30


def _style(ax):
    """Keep the frame recessive so the marks carry the figure."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(CLOUD)
    ax.tick_params(colors=INK_MUTED, labelsize=8, length=3)


def render_walk_gif(
    chain, latents, out_path, stride=4, fps=25, trail=150, tau_int=None
):
    """Render the walk and its running angle histogram to a GIF.

    ``chain`` is the :class:`~src.chains.MoonChain` that produced the ``(n, 3)``
    ``latents``, and supplies the decoder and the parameters in the readout.
    ``stride`` is chain steps per frame and ``trail`` the fading trail length, both in
    chain steps; ``tau_int`` is shown in the readout if already computed.
    """
    latents = np.asarray(latents, dtype=float)
    angles, points = latents[:, 0], chain.decode(latents)

    # Both moons, so the half the day never visits is visibly unvisited.
    cloud, _ = make_moons(n_samples=2000, noise=chain.sigma, random_state=0)

    fig, (ax_walk, ax_hist) = plt.subplots(
        1, 2, figsize=(9.0, 4.0), gridspec_kw={"width_ratios": [1.25, 1]}
    )
    fig.patch.set_facecolor(SURFACE)
    for ax in (ax_walk, ax_hist):
        _style(ax)

    ax_walk.scatter(*cloud.T, s=5, c=CLOUD, linewidths=0, zorder=1)
    ax_walk.set_aspect("equal")
    ax_walk.set_xlabel("$x_1$", color=INK_MUTED, fontsize=9)
    ax_walk.set_ylabel("$x_2$", color=INK_MUTED, fontsize=9)
    ax_walk.set_title(
        f"day stream, moon {chain.moon}", color=INK, fontsize=10, loc="left"
    )

    path = LineCollection([], linewidths=1.6, zorder=2)
    ax_walk.add_collection(path)
    (head,) = ax_walk.plot(
        [], [], "o", ms=8, mfc=WALKER, mec=SURFACE, mew=1.5, zorder=3
    )
    counter = ax_walk.text(
        0.02, 0.04, "", transform=ax_walk.transAxes, color=INK_MUTED, fontsize=8
    )

    uniform = 1.0 / np.pi
    edges = np.linspace(0.0, np.pi, N_BINS + 1)
    bars = ax_hist.bar(
        edges[:-1],
        np.zeros(N_BINS),
        width=np.diff(edges),
        align="edge",
        color=WALKER,
        linewidth=0,
        clip_on=True,
    )
    ax_hist.axhline(uniform, color=INK_MUTED, lw=1.2, ls=(0, (4, 3)), zorder=3)
    ax_hist.text(
        np.pi,
        uniform * 1.06,
        r"$\mathrm{Unif}[0,\pi]$",
        color=INK_MUTED,
        fontsize=8,
        ha="right",
        zorder=4,
        # The bars grow into this corner, so the label needs its own backing.
        bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1.5},
    )
    ax_hist.set_xlim(0.0, np.pi)
    ax_hist.set_ylim(0.0, 2.4 * uniform)
    ax_hist.set_xticks([0.0, np.pi / 2, np.pi], ["0", r"$\pi/2$", r"$\pi$"])
    ax_hist.set_xlabel("latent angle $t$", color=INK_MUTED, fontsize=9)
    ax_hist.set_ylabel("density", color=INK_MUTED, fontsize=9)
    ax_hist.set_title("coverage so far", color=INK, fontsize=10, loc="left")

    readout = rf"$\delta={chain.delta:g}$   $\rho={chain.rho:g}$"
    if tau_int is not None:
        readout += rf"   $\tau_{{\mathrm{{int}}}}={tau_int:.0f}$"
    fig.suptitle(readout, color=INK, fontsize=10, x=0.99, ha="right")

    trail_rgba = np.tile(to_rgba(WALKER), (trail, 1))

    def draw(k):
        recent = points[max(0, k - trail) : k]
        segments = np.stack([recent[:-1], recent[1:]], axis=1)
        path.set_segments(segments)
        # Fade the tail out so the eye reads direction, not just occupancy.
        colors = trail_rgba[: len(segments)].copy()
        colors[:, 3] = np.linspace(0.05, 0.85, len(segments))
        path.set_color(colors)

        head.set_data(points[k - 1 : k, 0], points[k - 1 : k, 1])
        counter.set_text(f"step {k:,} / {len(points):,}")

        density, _ = np.histogram(angles[:k], bins=edges, density=True)
        for bar, height in zip(bars, density, strict=True):
            bar.set_height(height)

        return (path, head, counter, *bars)

    fig.tight_layout()
    anim = FuncAnimation(
        fig, draw, frames=range(stride, len(points) + 1, stride), blit=False
    )
    # dpi is set here rather than on the figure: GIF size scales with the pixel count,
    # and a fast chain redraws most of the frame every step, so it compresses poorly.
    anim.save(
        out_path,
        writer=PillowWriter(fps=fps),
        dpi=100,
        savefig_kwargs={"facecolor": SURFACE},
    )
    plt.close(fig)
