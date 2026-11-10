#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_rainfield.py: Phase 2 analysis of the effect of a spatially and
temporally varying rain field on the attack and on the detector.

Reads results_rainfield/ (advecting EXCELL cell, tags
s<seed>_<regime>_<scen>_<rain>_c<factor>_<dir>) and uses results/ (Phase 1,
uniform rain) as the reference arm. Cell size is expressed as a multiple of the
Capsoni et al. (1987) eq. (19) mean cell radius, so 2.0 is the near-uniform limit
and 0.25 is a small cell with a strong gradient across the platoon.

Every cell run is calibrated so the platoon-experienced mean rain equals the
nominal value, so differences come from rain variability rather than from a
different amount of rain.

Two quantities are evaluated:
  1. Attack strength vs cell size. Because gamma = k*R^alpha with alpha < 1 for
     every mmWave band, gamma is concave, so Jensen predicts a variable field
     gives slightly less mean attenuation than uniform rain at the same mean.
     The force-up attack should therefore weaken slightly as cells shrink.
  2. Detector degradation vs cell size. The detector assumes the reported rain
     predicts the attenuation on the link. Under a varying field an honest
     vehicle's reported (nominal, platoon-mean) rain no longer matches the
     path-integrated rain on its long links, so the honest residual drifts away
     from zero, widening the null and costing detection rate.

Outputs (PNG only) -> results/figures/ and paper/figures/.
"""
import glob
import os
import re

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
RF = os.path.join(HERE, "results_rainfield")     # Phase 2, cell field
P1 = os.path.join(HERE, "results")               # Phase 1, uniform
FIGDIR = os.path.join(P1, "figures")
PAPER_FIGS = os.path.join(HERE, "paper", "figures")

CB = {"black": "#000000", "vermilion": "#D55E00", "blue": "#0072B2",
      "mustard": "#E69F00", "purple": "#CC79A7", "grey": "#999999"}
SCEN = {"baseline": ("FIS baseline", CB["black"], "o"),
        "attackA": ("Attack Type 1", CB["vermilion"], "^"),
        "attackB": ("Attack Type 2", CB["blue"], "D")}
FACTORS = [0.25, 0.5, 1.0, 2.0]
RAINS = [20, 37.5, 50]
BIN_W, FAR_M, MIN_DIST = 25.0, 100.0, 150.0
WINDOWS = [5, 10, 20, 50, 100]
FPR = 0.01
NBOOT = 4000
TX_FULL = 16632          # a complete 30 s run; guards against truncated files

PAT_RF = re.compile(r"pkts-s(\d+)_(\w+)_(baseline|attackA|attackB)_([0-9.]+)_c([0-9.]+)_(\w+)\.csv$")

_KR = np.linspace(0.0, 50.0, 20)
_G73 = [0.0000, 2.1748, 3.5993, 4.8329, 5.9568, 7.0057, 7.9984, 8.9467, 9.8586,
        10.7397, 11.5945, 12.4262, 13.2374, 14.0304, 14.8069, 15.5683, 16.3160,
        17.0510, 17.7743, 18.4867]


def gamma73(R):
    """ITU-R P.838 specific attenuation at 73 GHz, with the power-law tail the
    simulator uses above the table's 50 mm/h top row."""
    R = np.asarray(R, float)
    return np.where(R > 50.0, 1.07643 * np.maximum(R, 1e-9) ** 0.7268,
                    np.interp(R, _KR, _G73))


# loading, with a completeness check
def complete(pkt_path):
    """True if the run finished; rejects truncated files from an interrupted sweep."""
    try:
        return (pd.read_csv(pkt_path, usecols=["txrx"]).txrx == "tx").sum() == TX_FULL
    except Exception:
        return False


def link_table(pkt_path):
    d = pd.read_csv(pkt_path)
    tx = d[d.txrx == "tx"].groupby(["src", "dst"]).agg(n=("seq", "size"),
                                                       dist=("dist_m", "mean"))
    rx = d[d.txrx == "rx"].groupby(["src", "dst"]).size().rename("r")
    return tx.join(rx).fillna(0.0)


def far_prr(t):
    f = t[t.dist > FAR_M]
    return f.r.sum() / f.n.sum() if f.n.sum() else np.nan


def eff_range(t):
    b = t.assign(bin=(t.dist // BIN_W * BIN_W + BIN_W / 2)) \
         .groupby("bin").apply(lambda g: g.r.sum() / g.n.sum(), include_groups=False)
    ok = b[b >= 0.90]
    return float(ok.index.max()) if len(ok) else 0.0


def load_rainfield():
    """-> DataFrame(seed, scen, rain, factor, prr_far, range_m, sinr_far)."""
    rows, skipped = [], 0
    for p in sorted(glob.glob(os.path.join(RF, "pkts-*.csv"))):
        m = PAT_RF.search(os.path.basename(p))
        if not m:
            continue
        if not complete(p):
            skipped += 1
            continue
        seed, _, scen, rain, factor, _ = m.groups()
        t = link_table(p)
        phy = pd.read_csv(p.replace("pkts-", "phy-"))
        phy = phy[(phy.dist_m > 200) & np.isfinite(phy.sinr_db)]
        rows.append(dict(seed=int(seed), scen=scen, rain=float(rain),
                         factor=float(factor), prr_far=far_prr(t),
                         range_m=eff_range(t), sinr_far=phy.sinr_db.mean()))
    if skipped:
        print(f"  ! skipped {skipped} incomplete run(s)")
    return pd.DataFrame(rows)


def load_uniform():
    """Phase 1 reference arm, same metrics."""
    rows = []
    for p in sorted(glob.glob(os.path.join(P1, "pkts-s*_fixwidth_*.csv"))):
        m = re.search(r"pkts-s(\d+)_fixwidth_(baseline|attackA|attackB)_([0-9.]+)\.csv$",
                      os.path.basename(p))
        if not m:
            continue
        seed, scen, rain = m.groups()
        if float(rain) not in RAINS:
            continue
        t = link_table(p)
        phy = pd.read_csv(p.replace("pkts-", "phy-"))
        phy = phy[(phy.dist_m > 200) & np.isfinite(phy.sinr_db)]
        rows.append(dict(seed=int(seed), scen=scen, rain=float(rain),
                         prr_far=far_prr(t), range_m=eff_range(t),
                         sinr_far=phy.sinr_db.mean()))
    return pd.DataFrame(rows)


# detector under a varying field
def phy_bins(path, min_dist=MIN_DIST):
    d = pd.read_csv(path)
    d = d[(d.dist_m > min_dist) & np.isfinite(d.sinr_db)]
    return d.assign(bin=(d.dist_m // BIN_W * BIN_W).astype(int))[["bin", "dist_m", "sinr_db"]]


def clear_sky_ref(cal_seeds=range(5)):
    """Dry 73 GHz profile, taken from Phase 1 (rain=0 is not swept in Phase 2)."""
    fr = [phy_bins(os.path.join(P1, f"phy-s{s}_fixwidth_attackA_0.csv"))
          for s in cal_seeds]
    return pd.concat(fr).groupby("bin").sinr_db.mean()


def residual(path, reported_rain, ref):
    """SINR_obs - S_exp, where S_exp uses the reported (nominal) rain rate."""
    d = phy_bins(path)
    d = d[d.bin.isin(ref.index)]
    exp = d.bin.map(ref) - float(gamma73(reported_rain)) * d.dist_m / 1000.0
    return (d.sinr_db - exp).values


def detection_rate(h0, h1, w, rng):
    null = rng.choice(h0, (NBOOT, w)).mean(1)
    alt = rng.choice(h1, (NBOOT, w)).mean(1)
    return float((alt < np.percentile(null, FPR * 100)).mean())


def detector_vs_cellsize(test_seeds=range(5, 10)):
    """Detection rate at 1% FPR for each cell size, against a composite honest null."""
    ref = clear_sky_ref()
    rng = np.random.default_rng(0)
    out = []
    for factor in FACTORS:
        def paths(scen, rain):
            return [os.path.join(RF, f"pkts-s{s}_fixwidth_{scen}_{rain:g}_c{factor}_cross.csv")
                    for s in test_seeds]

        # H0 is a vehicle on 73 GHz reporting truthfully, to match the 73 GHz
        # clear-sky reference. The baseline scenario selects 28/39 GHz at these rain
        # rates, and scoring it against a 73 GHz profile would add an offset of about
        # 8 dB. Both hypotheses therefore use the attackA runs (73 GHz) and differ
        # only in the reported rain.
        acc0 = []
        for rain in RAINS:
            for p in paths("attackA", rain):
                if os.path.exists(p) and complete(p):
                    acc0.append(residual(p.replace("pkts-", "phy-"), rain, ref))
        if not acc0:
            continue
        h0 = np.concatenate(acc0)
        row = dict(factor=factor, h0_mean=h0.mean(), h0_sd=h0.std())
        for rain in RAINS:
            acc = [residual(p.replace("pkts-", "phy-"), 0.0, ref)
                   for p in paths("attackA", rain) if os.path.exists(p) and complete(p)]
            if not acc:
                continue
            h1 = np.concatenate(acc)
            row[f"resid_{rain:g}"] = h1.mean()
            for w in WINDOWS:
                row[f"det_{rain:g}_W{w}"] = detection_rate(h0, h1, w, rng)
        out.append(row)
    return pd.DataFrame(out)


def fig_rainfield(det, rf, uni):
    """(a) rain rate at the lead vehicle over the run, (b) detection rate vs cell size."""
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 400, "savefig.bbox": "tight",
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif", "font.size": 8.0,
        "axes.titlesize": 8.0, "axes.labelsize": 8.0,
        "xtick.labelsize": 7.2, "ytick.labelsize": 7.2, "legend.fontsize": 6.6,
        "axes.grid": True, "grid.alpha": 0.28, "grid.linewidth": 0.45,
        "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.4,
    })
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.16, 2.25))
    cum = np.array([0, 15, 35, 60, 95, 140, 200, 290.])
    cols = [CB["vermilion"], CB["purple"], CB["mustard"], CB["blue"]]

    # (a) rain rate at the lead vehicle through the run, per cell size
    for c, factor in zip(cols, FACTORS):
        mp = os.path.join(RF, f"meta-s0_fixwidth_attackA_50_c{factor}_cross.csv")
        if not os.path.exists(mp):
            continue
        m = pd.read_csv(mp).iloc[0]
        ts = np.linspace(0, m.simtime, 200)
        ys = cum[0] + 20.0 * ts
        R = m.peak_mmhr * np.exp(-np.hypot(0 - (m.cx0 + m.vx * ts),
                                           ys - (m.cy0 + m.vy * ts)) / (m.rho0_km * 1000))
        a1.plot(ts, R, color=c, label=rf"$\rho_0$ = {factor:g}$\bar{{\rho}}_0$")
    a1.axhline(50, color=CB["grey"], ls="--", lw=0.9, zorder=0)
    a1.text(0.4, 51.5, "uniform (Phase 1)", fontsize=6.0, color="0.35")
    a1.set_xlabel("Time (s)")
    a1.set_ylabel("Rain rate at lead vehicle (mm h$^{-1}$)")
    a1.set_title("(a) Rain rate at the lead vehicle", loc="left")
    a1.set_ylim(top=None)
    a1.margins(y=0.30)
    a1.legend(loc="lower right", framealpha=0.9, ncol=2)

    # (b) detection rate vs cell size, uniform-rain result as the reference
    if not det.empty:
        for w, mk in [(20, "o"), (50, "s"), (100, "D")]:
            col = f"det_50_W{w}"
            if col in det.columns:
                a2.plot(det.factor, det[col] * 100, mk + "-", ms=3,
                        color=CB["vermilion"] if w == 20 else
                        (CB["purple"] if w == 50 else CB["blue"]),
                        label=f"W={w}")
        for w, y in [(20, 68), (50, 98), (100, 100)]:
            a2.axhline(y, color=CB["grey"], ls=":", lw=0.8, zorder=0)
    a2.set_xscale("log")
    a2.set_xticks(FACTORS)
    a2.set_xticklabels([f"{f:g}" for f in FACTORS])
    a2.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    a2.tick_params(axis="x", which="minor", length=0)
    a2.set_xlabel(r"Cell size $\rho_0 / \bar{\rho}_0$   (smaller = more variable)")
    a2.set_ylabel("Detection rate (%)")
    a2.set_ylim(-4, 104)
    a2.set_title("(b) Detector at 50 mm h$^{-1}$ (dotted: uniform rain)", loc="left")
    a2.legend(loc="lower left", framealpha=0.9)

    fig.tight_layout()
    for d in (FIGDIR, PAPER_FIGS):
        os.makedirs(d, exist_ok=True)
        fig.savefig(os.path.join(d, "rainfield_variability_and_detector.png"))
    plt.close(fig)
    print("wrote rainfield_variability_and_detector.png")


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    print("loading rain-field runs ...")
    rf = load_rainfield()
    uni = load_uniform()
    if rf.empty:
        print("no complete rain-field runs yet")
        return
    print(f"  {len(rf)} complete cell runs, {len(uni)} uniform reference runs")
    print(f"  seeds present: {sorted(rf.seed.unique())}")

    print("\n=== Attack strength vs cell size (50 mm/h, long-range PRR) ===")
    print(f"{'scenario':>10} {'uniform':>9} " + " ".join(f"c={c:<6}" for c in FACTORS))
    for scen in ["baseline", "attackA", "attackB"]:
        u = uni[(uni.scen == scen) & (uni.rain == 50)].prr_far.mean()
        cells = [rf[(rf.scen == scen) & (rf.rain == 50) & (rf.factor == c)].prr_far.mean()
                 for c in FACTORS]
        print(f"{scen:>10} {u:>8.1%} " + " ".join(f"{v:>7.1%}" for v in cells))

    print("\n=== Mean SINR >200 m (50 mm/h) ===")
    print(f"{'scenario':>10} {'uniform':>9} " + " ".join(f"c={c:<6}" for c in FACTORS))
    for scen in ["baseline", "attackA", "attackB"]:
        u = uni[(uni.scen == scen) & (uni.rain == 50)].sinr_far.mean()
        cells = [rf[(rf.scen == scen) & (rf.rain == 50) & (rf.factor == c)].sinr_far.mean()
                 for c in FACTORS]
        print(f"{scen:>10} {u:>7.2f}dB " + " ".join(f"{v:>6.2f}dB" for v in cells))

    print("\n=== Detector vs cell size (1% FPR, held-out seeds) ===")
    det = detector_vs_cellsize()
    if not det.empty:
        pd.set_option("display.width", 200)
        cols = ["factor", "h0_mean", "h0_sd"] + [f"resid_{r:g}" for r in RAINS] + \
               [f"det_{r:g}_W{w}" for r in RAINS for w in (20, 50, 100)]
        print(det[[c for c in cols if c in det.columns]].round(3).to_string(index=False))
        det.to_csv(os.path.join(RF, "detector_vs_cellsize.csv"), index=False)
        print(f"\nwrote {os.path.join(RF, 'detector_vs_cellsize.csv')}")
    fig_rainfield(det, rf, uni)
    return rf, uni, det


if __name__ == "__main__":
    main()
