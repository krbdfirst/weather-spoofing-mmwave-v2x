#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_paper_figures.py: renders the paper figures at their embedded width, so
text is not downscaled.

  IEEEtran two-column: \\columnwidth = 3.5 in, \\textwidth = 7.16 in

Figure titles are omitted; the LaTeX captions carry them.

Outputs -> paper/figures/ (PNG only, never PDF):
  reach_range_and_prr_vs_rain.png     \\columnwidth  reliable range + long-range PRR
  mechanism_sinr_vs_distance.png      \\columnwidth  per-link SINR vs distance
  cost_tbsize_vs_rain.png             \\columnwidth  long-range transport-block size
  reach_prr_distance_rain_heatmap.png \\textwidth    PRR over distance x rain

The heatmap shows the fixed-width regime only. The defense and rain-field figures
are built by defense_sinr_consistency.py and analyze_rainfield.py respectively.

Usage:  python3 make_paper_figures.py
"""
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from matplotlib.colors import LinearSegmentedColormap

import analyze_millicar as A

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
PAPER_FIGS = os.path.join(HERE, "paper", "figures")

COL_W = 3.5     # IEEEtran \columnwidth, inches
RAINS = [0, 10, 20, 30, 37.5, 50]
RLAB = {"fixwidth": "Fixed width", "realwidth": "Real width"}

# PRR gradient from the line-chart colours: vermilion (low, force-up) -> light ->
# blue (high, force-down), so the heatmap and the line figures agree.
PRR_CMAP = LinearSegmentedColormap.from_list(
    "force_orblue", ["#D55E00", "#F2F2F2", "#0072B2"])


def set_style():
    """Sized for a 3.5 in column at 100% scale, so font sizes are the printed sizes."""
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 400, "savefig.bbox": "tight",
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif", "font.size": 8.0,
        "axes.titlesize": 7.8, "axes.labelsize": 8.0,
        "xtick.labelsize": 7.0, "ytick.labelsize": 7.0, "legend.fontsize": 6.4,
        "axes.grid": True, "grid.alpha": 0.28, "grid.linewidth": 0.45,
        "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 1.3, "lines.markersize": 3.0,
    })


def save(fig, name):
    os.makedirs(PAPER_FIGS, exist_ok=True)
    p = os.path.join(PAPER_FIGS, name)
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"  {name}")


def shared_legend(fig, ax):
    """One horizontal legend under both panels, so neither panel loses data area."""
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.10), handlelength=1.5,
               columnspacing=1.1, handletextpad=0.4)


# reach: reliable range + long-range PRR vs rain (fixed-width regime)
def fig_reach(pkt, regime="fixwidth"):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(COL_W, 1.85))
    for scen, (lab, c, mk) in A.SCEN.items():
        er_m, er_s, fp_m, fp_s = [], [], [], []
        for r in RAINS:
            Ls = pkt.get((regime, scen, r), [])
            er = [A.eff_range(L) for L in Ls]
            fp = [A.far_prr(L) for L in Ls]
            er_m.append(np.nanmean(er) if er else np.nan)
            er_s.append(np.nanstd(er) if er else 0)
            fp_m.append(np.nanmean(fp) if fp else np.nan)
            fp_s.append(np.nanstd(fp) if fp else 0)
        er_m, er_s = np.array(er_m), np.array(er_s)
        fp_m, fp_s = np.array(fp_m), np.array(fp_s)
        short = lab.split(" (")[0]
        a1.fill_between(RAINS, er_m - er_s, er_m + er_s, color=c, alpha=0.15, lw=0)
        a1.plot(RAINS, er_m, marker=mk, color=c, label=short, clip_on=False,
                markeredgecolor="white", markeredgewidth=0.35)
        a2.fill_between(RAINS, fp_m - fp_s, fp_m + fp_s, color=c, alpha=0.15, lw=0)
        a2.plot(RAINS, fp_m, marker=mk, color=c, label=short, clip_on=False,
                markeredgecolor="white", markeredgewidth=0.35)
    a1.set_ylabel("Reliable range (m)")
    a1.set_ylim(0, None)
    a1.set_title("(a) Effective range", loc="left")
    a2.set_ylabel(f"Long-range PRR (>{A.FAR_M:g} m)")
    a2.set_ylim(0, 1.04)
    a2.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    a2.set_title("(b) Long-range reliability", loc="left")
    for ax in (a1, a2):
        ax.set_xlabel("Rainfall (mm h$^{-1}$)")
    fig.tight_layout(pad=0.3)
    shared_legend(fig, a1)
    save(fig, "reach_range_and_prr_vs_rain.png")


# mechanism: per-link SINR vs distance, both regimes
def fig_sinr(phy, rain_hi=50):
    fig, axes = plt.subplots(1, 2, figsize=(COL_W, 1.85), sharey=True)
    for ax, regime in zip(axes, A.REGIMES):
        for scen, (lab, c, mk) in A.SCEN.items():
            dfs = phy.get((regime, scen, rain_hi), [])
            if not dfs:
                continue
            centers = A.binned_col(dfs[0], "sinr_db", "dist_m")[0]
            mean, sd = A.seed_stack([A.binned_col(d, "sinr_db", "dist_m")[1] for d in dfs])
            ax.fill_between(centers, mean - sd, mean + sd, color=c, alpha=0.13, lw=0)
            ax.plot(centers, mean, marker=mk, color=c, label=lab.split(" (")[0],
                    clip_on=False, markeredgecolor="white", markeredgewidth=0.35)
        ax.set_xlabel("Distance (m)")
        ax.set_xlim(0, None)
        ax.set_title(RLAB[regime], loc="left")
    axes[0].set_ylabel("Mean SINR (dB)")
    fig.tight_layout(pad=0.3)
    shared_legend(fig, axes[0])
    save(fig, "mechanism_sinr_vs_distance.png")


# cost: long-range transport-block size vs rain, both regimes
def fig_tbsize(phy):
    fig, axes = plt.subplots(1, 2, figsize=(COL_W, 1.85), sharey=True)
    for ax, regime in zip(axes, A.REGIMES):
        for scen, (lab, c, mk) in A.SCEN.items():
            ym, ys = [], []
            for r in RAINS:
                v = [A.far_mean(d, "tbsize") for d in phy.get((regime, scen, r), [])]
                ym.append(np.nanmean(v) if v else np.nan)
                ys.append(np.nanstd(v) if v else 0)
            ym, ys = np.array(ym), np.array(ys)
            ax.fill_between(RAINS, ym - ys, ym + ys, color=c, alpha=0.13, lw=0)
            ax.plot(RAINS, ym, marker=mk, color=c, label=lab.split(" (")[0],
                    clip_on=False, markeredgecolor="white", markeredgewidth=0.35)
        ax.set_xlabel("Rainfall (mm h$^{-1}$)")
        ax.set_title(RLAB[regime], loc="left")
    axes[0].set_ylabel("Transport block (B)")
    fig.tight_layout(pad=0.3)
    shared_legend(fig, axes[0])
    save(fig, "cost_tbsize_vs_rain.png")


# PRR heatmap, fixed-width regime only, full text width
def fig_heatmap(pkt, regime="fixwidth"):
    """PRR(distance, rain) heatmap, fixed-width regime, at \\textwidth.
    """
    edges = np.arange(0, 325, A.BIN_W)
    centers = 0.5 * (edges[:-1] + edges[1:])
    fig, axes = plt.subplots(1, 3, figsize=(8.80, 2.45), sharey=True)
    im = None
    for ax, scen in zip(axes, ["baseline", "attackA", "attackB"]):
        M = np.full((len(RAINS), len(centers)), np.nan)
        for j, r in enumerate(RAINS):
            Ls = pkt.get((regime, scen, r), [])
            if not Ls:
                continue
            mean, _ = A.seed_stack([A.binned_prr(L)[1] for L in Ls])
            M[j, :len(mean)] = mean
        im = ax.imshow(M, origin="lower", aspect="auto", cmap=PRR_CMAP, vmin=0.4, vmax=1.0,
                       extent=[0, edges[-1], -0.5, len(RAINS) - 0.5])
        # The same three cells are annotated with their values in every panel.
        for rv, dv in [(30, 87.5), (50, 262.5), (0, 262.5)]:
            j = int(np.argmin(np.abs(np.array(RAINS) - rv)))
            k = int(np.argmin(np.abs(centers - dv)))
            val = M[j, k]
            if np.isfinite(val):
                ax.plot(centers[k], j, "o", ms=8.5, mfc="none", mec="white", mew=1.3, zorder=4)
                ax.text(centers[k], j, f"{val*100:.0f}%", ha="center", va="center",
                        fontsize=5.6, color="black", zorder=5,
                        bbox=dict(boxstyle="round,pad=0.12", fc="white",
                                  ec="0.3", lw=0.35, alpha=0.92))
        ax.set_yticks(range(len(RAINS)))
        ax.set_yticklabels([f"{r:g}" for r in RAINS])
        ax.set_title(A.SCEN[scen][0].split(" (")[0], loc="left")
        ax.set_xlabel("TX–RX distance (m)")
        ax.grid(False)
    axes[0].set_ylabel("Rainfall (mm h$^{-1}$)")
    cb = fig.colorbar(im, ax=axes, shrink=0.92, pad=0.015)
    cb.set_label("Packet reception ratio")
    cb.ax.tick_params(labelsize=7.0)
    save(fig, "reach_prr_distance_rain_heatmap.png")


def main():
    set_style()
    print("loading results ...")
    pkt, phy = A.load(RESULTS)
    fig_heatmap(pkt)
    print(f"rebuilding paper figures at \\columnwidth = {COL_W} in -> paper/figures/")
    fig_reach(pkt)
    fig_sinr(phy)
    fig_tbsize(phy)


if __name__ == "__main__":
    main()
