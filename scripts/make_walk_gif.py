"""Render the two-moons day-stream GIF.

Usage:
    python -m scripts.make_walk_gif
    python -m scripts.make_walk_gif --delta 0.8 --rho 0.0
"""

import argparse
from pathlib import Path

from src.chains import MoonChain
from src.metrics import integrated_autocorrelation_time
from src.viz.walk_gif import render_walk_gif

OUT_DIR = Path("results/figures")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--moon", type=int, default=0, choices=(0, 1))
    p.add_argument("--delta", type=float, default=0.05, help="angle step size")
    p.add_argument("--rho", type=float, default=0.9, help="AR(1) noise coefficient")
    p.add_argument("--sigma", type=float, default=0.08, help="marginal noise scale")
    p.add_argument("--steps", type=int, default=6000, help="chain steps to simulate")
    p.add_argument("--stride", type=int, default=20, help="chain steps per frame")
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--trail", type=int, default=150, help="trail length, in steps")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def main():
    args = parse_args()

    chain = MoonChain(args.moon, args.delta, args.rho, args.sigma, rng=args.seed)
    latents = chain.run(args.steps)

    # The angle is the scalar summary the correlation sweep reports tau_int on, so the
    # figure and the sweep quote the same number.
    tau_int = integrated_autocorrelation_time(latents[:, 0])

    out = args.out or OUT_DIR / (
        f"walk_moon{args.moon}_d{args.delta:g}_r{args.rho:g}.gif"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    render_walk_gif(
        chain,
        latents,
        out,
        stride=args.stride,
        fps=args.fps,
        trail=args.trail,
        tau_int=tau_int,
    )
    print(f"{out}  (tau_int = {tau_int:.1f} steps)")


if __name__ == "__main__":
    main()
