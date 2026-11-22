"""Render the demo video: intact vs pC1-silenced male, with live neural activity.

    python make_video.py            # needs results/trials.pkl from run_experiments.py
"""
import pickle
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import imageio_ffmpeg

from sim import READOUT

matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
OUT = Path(__file__).parent / "results"
PANELS = [("intact", "Intact connectome"), ("pC1 silenced", "pC1 (arousal hub) silenced")]
LABELS = ["pheromone\nORNs", "pheromone\nPNs", "contact", "pC1", "LC10a L", "LC10a R", "DNa02 L", "DNa02 R"]
MALE, FEMALE = "#1c7ed6", "#d6336c"


def fly_marker(ax, color):
    body, = ax.plot([], [], "o", ms=9, color=color, zorder=4)
    head, = ax.plot([], [], "-", lw=3, color=color, zorder=4)
    trail, = ax.plot([], [], "-", lw=1, color=color, alpha=.45)
    return body, head, trail


def main(seed=0, fps=25, trail_s=3.0):
    trials = pickle.load(open(OUT / "trials.pkl", "rb"))
    trs = [trials[(k, seed)] for k, _ in PANELS]
    T = min(len(t.t) for t in trs)
    stride = max(1, int(round(1000 / 20 / fps)))   # body steps are 20 ms
    frames = range(0, T, stride)
    tl = int(trail_s * 50)

    fig = plt.figure(figsize=(12, 7.2), facecolor="white")
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1.35], hspace=.25, wspace=.12)
    artists = []
    for col, (tr, (_, title)) in enumerate(zip(trs, PANELS)):
        ax = fig.add_subplot(gs[0, col])
        ax.add_patch(plt.Circle((0, 0), 20, fill=False, color="0.75", lw=1.5))
        ax.set_xlim(-21, 21); ax.set_ylim(-21, 21); ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(title, fontsize=13, weight="bold")
        m = fly_marker(ax, MALE); f = fly_marker(ax, FEMALE)
        plume = ax.add_patch(plt.Circle((0, 0), 12, color=FEMALE, alpha=.07, lw=0))
        txt = ax.text(-20, -21.5, "", fontsize=10, family="monospace")
        bx = fig.add_subplot(gs[1, col])
        bars = bx.bar(range(len(LABELS)), np.zeros(len(LABELS)),
                      color=["#e09f3e"] * 3 + ["#c44536"] + ["#2a9d8f"] * 2 + ["#264653"] * 2)
        bx.set_xticks(range(len(LABELS)), LABELS, fontsize=7.5)
        bx.set_ylim(0, 400); bx.set_ylabel("firing rate (Hz)", fontsize=8)
        bx.spines[["top", "right"]].set_visible(False)
        artists.append((tr, m, f, plume, txt, bars))
    fig.text(.5, .015, "FlyWire v783 connectome · 1,827 neurons · 88.5k synaptic connections · leaky integrate-and-fire",
             ha="center", fontsize=9, color="0.4")
    if True:
        ax0 = fig.axes[0]
        ax0.plot([], [], "o", color=MALE, label="male (connectome brain)")
        ax0.plot([], [], "o", color=FEMALE, label="female")
        ax0.legend(loc="upper left", fontsize=8, frameon=False)

    def smooth(r, i, w=10):
        return r[max(0, i - w):i + 1].mean(0)

    def draw(i):
        out = []
        for tr, m, f, plume, txt, bars in artists:
            for (body, head, trail), xyz in ((m, tr.male), (f, tr.female)):
                x, y, h = xyz[i]
                body.set_data([x], [y])
                head.set_data([x, x + 1.4 * np.cos(h)], [y, y + 1.4 * np.sin(h)])
                trail.set_data(xyz[max(0, i - tl):i + 1, 0], xyz[max(0, i - tl):i + 1, 1])
                out += [body, head, trail]
            plume.center = tr.female[i, :2]
            d = np.hypot(*(tr.female[i, :2] - tr.male[i, :2]))
            txt.set_text(f"t={tr.t[i]:5.1f}s  dist={d:5.1f}mm  arousal={tr.arousal[i]:.2f}")
            for b, v in zip(bars, smooth(tr.rates, i)):
                b.set_height(v)
            out += [plume, txt, *bars]
        return out

    anim = FuncAnimation(fig, draw, frames=frames, blit=False)
    path = OUT / "courtship_demo.mp4"
    anim.save(path, writer=FFMpegWriter(fps=fps, bitrate=3000))
    print("saved", path)


if __name__ == "__main__":
    main()
