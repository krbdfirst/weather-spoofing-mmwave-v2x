#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_millicar.py: reach and capacity analysis for the MilliCar
weather-spoofing study, across two bandwidth regimes (fixwidth | realwidth).

Inputs (per run tag s<seed>_<regime>_<scen>_<rain>):
  pkts-<tag>.csv  txrx,time,src,dst,seq,bytes,latency_ms,dist_m,x,y
  phy-<tag>.csv   time,rxnode,txnode,dist_m,sinr_db,numsym,tbsize,mcs
  meta-<tag>.csv  regime,scenario,rain,band_ghz,bw_mhz,numveh,simtime

The platoon uses scheduled unicast (full mesh), so the natural unit is the
directed link (src,dst): PRR = rx/tx at the link distance. Results are
aggregated over seeds. Outputs (PNG):

  Per regime:
    millicar_<regime>_fig1_prr_distance.png    PRR vs distance @ heavy rain
    millicar_<regime>_fig2_reach_vs_rain.png    reliable range + long-range PRR vs rain

  Cross-regime (fixwidth | realwidth side by side):
    millicar_x_sinr_distance.png    SINR vs distance @ heavy rain
    millicar_x_sinr_vs_rain.png     long-range SINR vs rain
    millicar_x_latency_distance.png end-to-end latency vs distance @ heavy rain
    millicar_x_mcs_vs_rain.png      mean MCS vs rain
    millicar_x_capacity_vs_rain.png mean TB size vs rain
    millicar_x_regime_contrast.png  long-range PRR: fixwidth vs realwidth overlay
"""
import argparse, glob, os, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from matplotlib.lines import Line2D

CB = {"black": "#000000", "vermilion": "#D55E00", "blue": "#0072B2"}
# Legend / draw order: baseline first, then Attack Type 1, then Attack Type 2.
SCEN = {
    "baseline": ("FIS baseline", CB["black"], "o"),
    "attackA":  ("Attack Type 1 (force up, 73 GHz)", CB["vermilion"], "^"),
    "attackB":  ("Attack Type 2 (force down, 5 GHz)", CB["blue"], "D"),
}
REGIMES = ["fixwidth", "realwidth"]
RTITLE = {"fixwidth": "Fixed-width regime (100 MHz all bands)",
          "realwidth": "Real-width regime (per-band deployable BW)"}
TITLE_SZ, SUB_SZ, LABEL_SZ = 12.5, 10.5, 9.0
BIN_W = 25.0
FAR_M = 100.0
PAT = re.compile(r"s(\d+)_(fixwidth|realwidth)_(baseline|attackA|attackB)_([0-9.]+)$")


def set_style():
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 400, "savefig.bbox": "tight",
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif", "font.size": 9.5,
        "figure.titlesize": TITLE_SZ, "axes.titlesize": SUB_SZ, "axes.labelsize": LABEL_SZ,
        "xtick.labelsize": LABEL_SZ, "ytick.labelsize": LABEL_SZ, "legend.fontsize": 8.2,
        "axes.grid": True, "grid.alpha": 0.28, "grid.linewidth": 0.5,
        "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2.0,
    })


# loading
def per_link(csv):
    """Directed-link table for one pkts CSV: dist, sent, recv, lat (mean ms)."""
    df = pd.read_csv(csv)
    tx = df[df.txrx == "tx"]; rx = df[df.txrx == "rx"]
    if tx.empty:
        return pd.DataFrame(columns=["dist", "sent", "recv", "lat"])
    sent = tx.groupby(["src", "dst"]).size().rename("sent")
    recv = rx.groupby(["src", "dst"]).size().rename("recv")
    dist = tx.groupby(["src", "dst"])["dist_m"].mean().rename("dist")
    lat = rx.groupby(["src", "dst"])["latency_ms"].mean().rename("lat")
    m = pd.concat([sent, recv, dist, lat], axis=1).reset_index()
    m["recv"] = m["recv"].fillna(0.0)
    return m.dropna(subset=["dist"])[["dist", "sent", "recv", "lat"]]


def load(d):
    """pkt{(regime,scen,rain):[link_df...]}, phy{(regime,scen,rain):[phy_df...]}."""
    pkt, phy = {}, {}
    for f in sorted(glob.glob(os.path.join(d, "pkts-s*_*.csv"))):
        m = PAT.search(os.path.basename(f)[5:-4])  # strip "pkts-" and ".csv"
        if not m:
            continue
        key = (m.group(2), m.group(3), float(m.group(4)))
        pkt.setdefault(key, []).append(per_link(f))
        pf = f.replace("pkts-", "phy-", 1)
        if os.path.exists(pf):
            phy.setdefault(key, []).append(pd.read_csv(pf))
    return pkt, phy


# metrics
def binned(x, edges_max=325):
    edges = np.arange(0, edges_max, BIN_W)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, np.digitize(x, edges) - 1


def binned_prr(L):
    centers, b = binned(L["dist"].to_numpy())
    prr = np.full(len(centers), np.nan)
    for k in range(len(centers)):
        s = L.loc[b == k, "sent"].sum()
        if s > 0:
            prr[k] = L.loc[b == k, "recv"].sum() / s
    return centers, prr


def binned_col(df, valcol, distcol):
    centers, b = binned(df[distcol].to_numpy())
    out = np.full(len(centers), np.nan)
    for k in range(len(centers)):
        v = df.loc[b == k, valcol].dropna()
        if len(v):
            out[k] = v.mean()
    return centers, out


def eff_range(L, thr=0.9):
    c, prr = binned_prr(L)
    ok = np.where(prr >= thr)[0]
    return c[ok.max()] if len(ok) else 0.0


def far_prr(L):
    far = L[L["dist"] > FAR_M]
    s = far["sent"].sum()
    return far["recv"].sum() / s if s > 0 else np.nan


def far_mean(df, col):
    far = df[df["dist_m"] > FAR_M]
    return far[col].mean() if len(far) else np.nan


def seed_stack(arrs):
    M = np.vstack(arrs)
    return np.nanmean(M, axis=0), np.nanstd(M, axis=0)


# per-regime figures
def fig_prr_distance(pkt, regime, rainHi, out, nseed):
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for scen, (lab, c, mk) in SCEN.items():
        key = (regime, scen, rainHi)
        if key not in pkt:
            continue
        centers = binned_prr(pkt[key][0])[0]
        mean, sd = seed_stack([binned_prr(L)[1] for L in pkt[key]])
        ax.fill_between(centers, mean - sd, mean + sd, color=c, alpha=0.15, lw=0)
        ax.plot(centers, mean, marker=mk, color=c, label=lab, clip_on=False,
                markeredgecolor="white", markeredgewidth=0.4)
    ax.set_xlabel("TX–RX link distance  (m)"); ax.set_ylabel("Packet reception ratio")
    ax.set_ylim(0, 1.02); ax.set_xlim(0, None)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0)); ax.legend(loc="lower left")
    ax.set_title(f"Force-up collapses long-range reach (rain {rainHi:g} mm h$^{{-1}}$)",
                 loc="left", fontsize=SUB_SZ)
    fig.suptitle(f"MilliCar reach vs distance, {RTITLE[regime]} (n={nseed})",
                 fontsize=TITLE_SZ, y=1.02)
    p = os.path.join(out, f"millicar_{regime}_fig1_prr_distance.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig); print("  " + os.path.basename(p))


def fig_reach_vs_rain(pkt, regime, rains, out, nseed):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.0, 3.7))
    for scen, (lab, c, mk) in SCEN.items():
        er_m, er_s, fp_m, fp_s = [], [], [], []
        for r in rains:
            Ls = pkt.get((regime, scen, r), [])
            er = [eff_range(L) for L in Ls]; fp = [far_prr(L) for L in Ls]
            er_m.append(np.nanmean(er) if er else np.nan); er_s.append(np.nanstd(er) if er else 0)
            fp_m.append(np.nanmean(fp) if fp else np.nan); fp_s.append(np.nanstd(fp) if fp else 0)
        er_m, er_s = np.array(er_m), np.array(er_s); fp_m, fp_s = np.array(fp_m), np.array(fp_s)
        a1.fill_between(rains, er_m - er_s, er_m + er_s, color=c, alpha=0.15, lw=0)
        a1.plot(rains, er_m, marker=mk, color=c, label=lab, clip_on=False,
                markeredgecolor="white", markeredgewidth=0.4)
        a2.fill_between(rains, fp_m - fp_s, fp_m + fp_s, color=c, alpha=0.15, lw=0)
        a2.plot(rains, fp_m, marker=mk, color=c, label=lab, clip_on=False,
                markeredgecolor="white", markeredgewidth=0.4)
    a1.set_ylabel("Reliable range  (PRR ≥ 90%, m)"); a1.set_ylim(0, None)
    a1.set_title("(a) Effective communication range", loc="left", fontsize=SUB_SZ)
    a2.set_ylabel(f"Long-range PRR  (> {FAR_M:g} m)"); a2.set_ylim(0, 1.02)
    a2.yaxis.set_major_formatter(PercentFormatter(1.0))
    a2.set_title("(b) Long-range reliability", loc="left", fontsize=SUB_SZ)
    for ax in (a1, a2):
        ax.set_xlabel("True rainfall rate  (mm h$^{-1}$)")
    a1.legend(loc="best", fontsize=7.8)
    fig.suptitle(f"Reach vs rainfall, {RTITLE[regime]} (n={nseed})", fontsize=TITLE_SZ, y=1.02)
    p = os.path.join(out, f"millicar_{regime}_fig2_reach_vs_rain.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig); print("  " + os.path.basename(p))


# cross-regime figures
def _panels(regimes):
    return plt.subplots(1, len(regimes), figsize=(4.7 * len(regimes), 3.8),
                        sharey=True, squeeze=False)


def fig_dist_panels(data, regimes, scen_iter, valfn, centersfn, rainHi, ylab, title,
                    fname, out, nseed, legloc="best"):
    fig, axarr = _panels(regimes); axes = axarr[0]
    for ax, regime in zip(axes, regimes):
        for scen, (lab, c, mk) in SCEN.items():
            key = (regime, scen, rainHi)
            if key not in data:
                continue
            centers = centersfn(data[key][0])
            mean, sd = seed_stack([valfn(x) for x in data[key]])
            ax.fill_between(centers, mean - sd, mean + sd, color=c, alpha=0.13, lw=0)
            ax.plot(centers, mean, marker=mk, color=c, label=lab, clip_on=False,
                    markeredgecolor="white", markeredgewidth=0.4)
        ax.set_xlabel("TX–RX link distance  (m)"); ax.set_xlim(0, None)
        ax.set_title(RTITLE[regime], loc="left", fontsize=SUB_SZ)
    axes[0].set_ylabel(ylab); axes[0].legend(loc=legloc, fontsize=7.6)
    fig.suptitle(f"{title} @ {rainHi:g} mm h$^{{-1}}$ (n={nseed})", fontsize=TITLE_SZ, y=1.02)
    p = os.path.join(out, fname)
    fig.savefig(p, bbox_inches="tight"); plt.close(fig); print("  " + os.path.basename(p))


def fig_far_vs_rain(phy, regimes, rains, col, ylab, title, fname, out, nseed):
    fig, axarr = _panels(regimes); axes = axarr[0]
    for ax, regime in zip(axes, regimes):
        for scen, (lab, c, mk) in SCEN.items():
            ym, ys = [], []
            for r in rains:
                dfs = phy.get((regime, scen, r), [])
                v = [far_mean(df, col) for df in dfs]
                ym.append(np.nanmean(v) if v else np.nan); ys.append(np.nanstd(v) if v else 0)
            ym, ys = np.array(ym), np.array(ys)
            ax.fill_between(rains, ym - ys, ym + ys, color=c, alpha=0.13, lw=0)
            ax.plot(rains, ym, marker=mk, color=c, label=lab, clip_on=False,
                    markeredgecolor="white", markeredgewidth=0.4)
        ax.set_xlabel("True rainfall rate  (mm h$^{-1}$)")
        ax.set_title(RTITLE[regime], loc="left", fontsize=SUB_SZ)
    axes[0].set_ylabel(ylab); axes[0].legend(loc="best", fontsize=7.6)
    fig.suptitle(f"{title} (n={nseed})", fontsize=TITLE_SZ, y=1.02)
    p = os.path.join(out, fname)
    fig.savefig(p, bbox_inches="tight"); plt.close(fig); print("  " + os.path.basename(p))


def fig_regime_contrast(pkt, regimes, rains, out, nseed):
    """Long-range PRR vs rain, fixwidth (solid) vs realwidth (dashed) overlaid."""
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    styles = {"fixwidth": ("-", 1.0), "realwidth": ("--", 0.85)}
    for scen, (lab, c, mk) in SCEN.items():
        for regime in regimes:
            ls, alpha = styles[regime]
            y = [np.nanmean([far_prr(L) for L in pkt.get((regime, scen, r), [])])
                 if pkt.get((regime, scen, r)) else np.nan for r in rains]
            ax.plot(rains, y, ls=ls, marker=mk, color=c, alpha=alpha, clip_on=False,
                    markeredgecolor="white", markeredgewidth=0.4,
                    label=(lab if regime == regimes[0] else None))
    ax.set_xlabel("True rainfall rate  (mm h$^{-1}$)")
    ax.set_ylabel(f"Long-range PRR  (> {FAR_M:g} m)")
    ax.set_ylim(0, 1.02); ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    leg1 = ax.legend(loc="lower left", fontsize=7.8); ax.add_artist(leg1)
    sh = [Line2D([0], [0], color="grey", ls=styles[g][0],
                 label=("fixed width (100 MHz)" if g == "fixwidth" else "real width (per-band)"))
          for g in regimes]
    ax.legend(handles=sh, loc="upper right", fontsize=7.8)
    fig.suptitle(f"Does the regime change the story? Long-range PRR (n={nseed})",
                 fontsize=TITLE_SZ, y=1.0)
    p = os.path.join(out, "millicar_x_regime_contrast.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig); print("  " + os.path.basename(p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results")
    ap.add_argument("--out", default="results/figures")
    ap.add_argument("--rainHi", type=float, default=50.0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    set_style()
    pkt, phy = load(args.dir)
    if not pkt:
        print("no regime-tagged pkts-*.csv found in", args.dir); return
    rains = sorted({r for (_, _, r) in pkt})
    nseed = max((len(v) for v in pkt.values()), default=1)
    regimes = [g for g in REGIMES if any(k[0] == g for k in pkt)]

    for regime in regimes:
        fig_prr_distance(pkt, regime, args.rainHi, args.out, nseed)
        fig_reach_vs_rain(pkt, regime, rains, args.out, nseed)

    if phy:
        fig_dist_panels(phy, regimes, SCEN, lambda df: binned_col(df, "sinr_db", "dist_m")[1],
                        lambda df: binned_col(df, "sinr_db", "dist_m")[0], args.rainHi,
                        "Mean received SINR  (dB)", "The cause: per-link SINR vs distance",
                        "millicar_x_sinr_distance.png", args.out, nseed, legloc="upper right")
        fig_far_vs_rain(phy, regimes, rains, "sinr_db", "Long-range SINR  (> 100 m, dB)",
                        "Physical-layer SINR vs rainfall", "millicar_x_sinr_vs_rain.png",
                        args.out, nseed)
        fig_far_vs_rain(phy, regimes, rains, "mcs", "Long-range mean MCS index",
                        "Link adaptation: MCS vs rainfall", "millicar_x_mcs_vs_rain.png",
                        args.out, nseed)
        fig_far_vs_rain(phy, regimes, rains, "tbsize", "Long-range mean TB size  (bytes)",
                        "Capacity lever: transport-block size vs rainfall",
                        "millicar_x_capacity_vs_rain.png", args.out, nseed)

    # latency vs distance (from pkt links)
    fig_dist_panels(pkt, regimes, SCEN,
                    lambda L: binned_col(L.rename(columns={"dist": "dist_m"}), "lat", "dist_m")[1],
                    lambda L: binned_col(L.rename(columns={"dist": "dist_m"}), "lat", "dist_m")[0],
                    args.rainHi, "Mean end-to-end latency  (ms)",
                    "Delivered-packet latency vs distance", "millicar_x_latency_distance.png",
                    args.out, nseed, legloc="upper left")

    fig_regime_contrast(pkt, regimes, rains, args.out, nseed)

    for regime in regimes:
        print(f"\nLong-range PRR (>100 m), {regime}:")
        for scen in ("baseline", "attackA", "attackB"):
            vals = " ".join(
                f"{r:g}:{np.nanmean([far_prr(L) for L in pkt.get((regime,scen,r),[])])*100:4.0f}%"
                for r in rains)
            print(f"  {scen:<9} {vals}")


if __name__ == "__main__":
    main()
