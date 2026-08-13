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
    p.add_argument(
        "--tau-steps",
        type=int,
        default=200_000,
        help="steps of the pilot run tau_int is measured on",
    )
    p.add_argument("--stride", type=int, default=20, help="chain steps per frame")
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--trail", type=int, default=150, help="trail length, in steps")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def main():
    args = parse_args()

    def new_chain(seed):
        return MoonChain(args.moon, args.delta, args.rho, args.sigma, rng=seed)

    chain = new_chain(args.seed)
    latents = chain.run(args.steps)

    # tau_int belongs to the kernel, not to the stretch being animated — a walk short
    # enough to watch cannot resolve it — so measure it on a longer pilot run. The
    # slowest latent coordinate is not always the angle: only the noise carries rho.
    pilot = new_chain(args.seed + 1).run(args.tau_steps)
    tau_int = max(
        integrated_autocorrelation_time(pilot[:, i]) for i in range(pilot.shape[1])
    )

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
