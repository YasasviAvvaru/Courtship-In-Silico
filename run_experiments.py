"""Run silencing experiments and produce figures + a demo video.

    python run_experiments.py            # all conditions, 6 seeds each
    python run_experiments.py --quick    # 2 seeds, shorter trials
"""
import argparse
import pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sim import Courtship, courtship_index

OUT = Path(__file__).parent / "results"
CONDITIONS = {
    "intact": (),
    "pC1 silenced": ("pC1",),
    "pheromone ORNs silenced": ("pher_ORN",),
    "contact neurons silenced": ("contact_asc",),
    "LC10a silenced (blind)": ("LC10a",),
}


def one(args):
    name, seed, dur = args
    tr = Courtship(silence=CONDITIONS[name], seed=seed, duration_s=dur).run()
    ci, dist = courtship_index(tr)
    return name, seed, ci, dist, tr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    seeds, dur = (2, 15) if a.quick else (6, 30)
    OUT.mkdir(exist_ok=True)

    jobs = [(n, s, dur) for n in CONDITIONS for s in range(seeds)]
    with ProcessPoolExecutor() as ex:
        results = list(ex.map(one, jobs))

    df = pd.DataFrame([r[:4] for r in results], columns=["condition", "seed", "courtship_index", "mean_distance_mm"])
    df.to_csv(OUT / "summary.csv", index=False)
    print(df.groupby("condition", sort=False)[["courtship_index", "mean_distance_mm"]].agg(["mean", "sem"]).round(3))
    trials = {(r[0], r[1]): r[4] for r in results}
    with open(OUT / "trials.pkl", "wb") as f:
        pickle.dump(trials, f)

    plot_summary(df)
    plot_trajectories(trials)


def plot_summary(df):
    g = df.groupby("condition", sort=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    colors = ["#2a6f97", "#c44536", "#e09f3e", "#9e2a2b", "#6c757d"]
    for ax, col, label in zip(axes, ["courtship_index", "mean_distance_mm"],
                              ["Courtship index (following, facing)", "Mean distance to female (mm)"]):
        m, s = g[col].mean(), g[col].sem()
        ax.bar(range(len(m)), m, yerr=s, color=colors, capsize=4)
        for i, (_, grp) in enumerate(g):
            ax.scatter(np.full(len(grp), i) + np.random.uniform(-.15, .15, len(grp)), grp[col], s=12, c="k", zorder=3)
        ax.set_xticks(range(len(m)), [c.replace(" (", "\n(").replace(" silenced", "\nsilenced") for c in m.index], fontsize=8)
        ax.set_ylabel(label)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Silencing connectome-defined neurons in a virtual courting fly")
    fig.tight_layout()
    fig.savefig(OUT / "silencing_summary.png", dpi=160)


def plot_trajectories(trials, seed=0):
    names = list(CONDITIONS)
    fig, axes = plt.subplots(1, len(names), figsize=(4 * len(names), 4.2))
    for ax, n in zip(axes, names):
        tr = trials[(n, seed)]
        ax.add_patch(plt.Circle((0, 0), 20, fill=False, color="0.7"))
        ax.plot(tr.female[:, 0], tr.female[:, 1], color="#d6336c", lw=1, label="female")
        ax.plot(tr.male[:, 0], tr.male[:, 1], color="#1c7ed6", lw=1, label="male")
        ax.plot(*tr.male[0, :2], "o", color="#1c7ed6")
        ax.plot(*tr.female[0, :2], "o", color="#d6336c")
        ax.set_title(n, fontsize=10)
        ax.set_aspect("equal"); ax.set_xlim(-21, 21); ax.set_ylim(-21, 21); ax.axis("off")
    axes[0].legend(loc="lower left", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "trajectories.png", dpi=160)


if __name__ == "__main__":
    main()
