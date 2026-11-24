"""재평가 결과 그림 생성. 입력: reeval.py가 만든 *_videos.csv

figures/summary.png          규칙별 재현율·오탐률 (반복 10회 분포)
figures/score_separation.png 영상 점수 분포로 본 개선 이유
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8f8e88", "#e7e6e1"
BASE, BLUE, ORANGE = "#a3a29b", "#2a78d6", "#eb6834"
plt.rcParams.update({
    "font.family": ["Segoe UI", "Malgun Gothic", "DejaVu Sans"],
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": GRID, "axes.unicode_minus": False,
})

src = sys.argv[1] if len(sys.argv) > 1 else "reeval_summary_videos.csv"
out = sys.argv[2] if len(sys.argv) > 2 else "figures"
os.makedirs(out, exist_ok=True)
v = pd.read_csv(src)
n_rep = v.rep.nunique()
pos, neg = v[v.y == 1], v[v.y == 0]
n_pos, n_neg = len(pos) // n_rep, len(neg) // n_rep


def clean(ax):
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(length=0, labelsize=10)
    ax.set_yticks([])
    ax.grid(axis="x", color=GRID, lw=1)
    ax.set_axisbelow(True)


# ---------- 그림 1: summary ----------
def dot_rows(ax, rows, xmax, ticks, fmt):
    """rows: [(label, color, values in %)] - 반복 1회 = 점 1개, 같은 값은 위로 쌓음."""
    for r, (label, color, vals) in enumerate(rows):
        base = (len(rows) - 1 - r) * 1.0
        ax.hlines(base - 0.12, 0, xmax, color=GRID, lw=1)
        for val, cnt in pd.Series(vals).round(6).value_counts().items():
            ax.scatter([val] * cnt, base + 0.13 * np.arange(cnt), s=95, color=color,
                       edgecolor=SURFACE, lw=1.5, zorder=3, clip_on=False)
        m = float(np.mean(vals))
        ax.vlines(m, base - 0.24, base - 0.12, color=INK, lw=2, zorder=4)
        ax.text(m, base - 0.29, f"mean {fmt(m)}", ha="center", va="top", fontsize=10, color=INK2)
        ax.text(-0.035 * xmax, base + 0.06, label, ha="right", va="center", fontsize=12,
                color=INK, fontweight="bold" if r else "normal")
    ax.set_xlim(0, xmax)
    ax.set_ylim(-0.5, len(rows) - 0.3)
    ax.set_xticks(ticks, [fmt(t) for t in ticks])
    clean(ax)


def per_rep(frame, col, n):
    return (frame.groupby("rep")[col].sum() / n * 100).to_numpy()


fig = plt.figure(figsize=(12, 5.0))
gs = fig.add_gridspec(1, 2, left=0.085, right=0.975, top=0.60, bottom=0.10, wspace=0.30)
panels = [
    ("RECALL", f"cheat videos caught, of {n_pos}", "flag_max", "flag_mean", pos, n_pos,
     100, [0, 25, 50, 75, 100], lambda x: f"{x:.0f}%"),
    ("FALSE POSITIVE RATE", f"normal videos flagged, of {n_neg}", "flag_max", "flag_mean", neg, n_neg,
     1.5, [0, 0.5, 1.0, 1.5], lambda x: f"{x:.2f}%" if x % 0.5 else f"{x:.1f}%"),
]
for i, (name, sub, c0, c1, frame, n, xmax, ticks, fmt) in enumerate(panels):
    ax = fig.add_subplot(gs[i])
    a, b = per_rep(frame, c0, n), per_rep(frame, c1, n)
    dot_rows(ax, [("Baseline", BASE, a), ("Improved", BLUE, b)], xmax, ticks, fmt)
    x0 = ax.get_position().x0
    hero = (lambda x: f"{x:.0f}%") if i == 0 else (lambda x: f"{x:.2f}%")
    tn = fig.text(x0, 0.855, name, fontsize=10.5, color=INK2, fontweight="bold")
    t = fig.text(x0, 0.70, hero(a.mean()), fontsize=30, color=MUTED)
    fig.canvas.draw()
    fig.text(x0 + tn.get_window_extent().width / fig.bbox.width + 0.012, 0.855, sub,
             fontsize=10.5, color=MUTED)
    w = t.get_window_extent().width / fig.bbox.width
    t2 = fig.text(x0 + w + 0.012, 0.70, "→", fontsize=24, color=MUTED)
    w2 = t2.get_window_extent().width / fig.bbox.width
    fig.text(x0 + w + w2 + 0.024, 0.70, hero(b.mean()), fontsize=30, color=INK, fontweight="bold")
fig.text(0.085, 0.945, "Video-level detection: baseline vs improved decision rule",
         fontsize=15, fontweight="bold")
fig.text(0.975, 0.945, f"5-fold × {n_rep} repeats · threshold set on training folds only · 1 dot = 1 repeat",
         fontsize=10, color=MUTED, ha="right", va="baseline")
fig.savefig(os.path.join(out, "summary.png"), dpi=200)
plt.close(fig)

# ---------- 그림 2: score separation (rep 0, 영상마다 out-of-fold 점수 1개) ----------
r0 = v[v.rep == 0]
rng = np.random.default_rng(0)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), sharey=True)
fig.subplots_adjust(left=0.085, right=0.975, top=0.65, bottom=0.13, wspace=0.08)
specs = [("pmax", "Baseline", "video score = highest window probability"),
         ("pmean", "Improved", "video score = mean window probability")]
for ax, (score, name, sub) in zip(axes, specs):
    h, c = r0[r0.y == 0], r0[r0.y == 1]
    ax.scatter(h[score], 1 + rng.uniform(-0.3, 0.3, len(h)), s=30, color=BLUE, alpha=0.5,
               edgecolor="none", zorder=2)
    ax.scatter(c[score], rng.uniform(-0.3, 0.3, len(c)), s=58, color=ORANGE, alpha=0.9,
               edgecolor=SURFACE, lw=1.2, zorder=3)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.6, 1.6)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("video score", fontsize=10)
    clean(ax)
    x0 = ax.get_position().x0
    hi = int((h[score] >= 0.9).sum())
    fig.text(x0, 0.815, name, fontsize=13, fontweight="bold")
    fig.text(x0, 0.76, sub, fontsize=10.5, color=INK2)
    fig.text(x0, 0.71, f"normal videos scoring 0.9 or higher: {hi} of {len(h)}",
             fontsize=10.5, color=INK2)
axes[0].text(-0.05, 1, "Normal", ha="right", va="center", fontsize=12)
axes[0].text(-0.05, 0, "Cheat", ha="right", va="center", fontsize=12)
fig.text(0.085, 0.92, "Why averaging helps: one extreme window no longer decides the video",
         fontsize=15, fontweight="bold")
fig.savefig(os.path.join(out, "score_separation.png"), dpi=200)
plt.close(fig)
print("saved:", sorted(os.listdir(out)))
