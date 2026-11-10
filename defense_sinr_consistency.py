#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
defense_sinr_consistency.py: physical-consistency detector for weather spoofing,
evaluated on the MilliCar per-transport-block SINR logs in results/.

The detector is a receiver-side test that uses only quantities available to a
vehicle: its own carrier, the reported rain rate, the per-link distance, and the
measured SINR of each received transport block.

Idea
----
Rain attenuation is linear in link distance (gamma * d, ITU-R P.838) while path loss
is logarithmic in distance, so an unmodelled rain term leaves a distance-dependent
signature in the observed SINR.

The vehicle holds a per-band clear-sky reference profile S_ref(f, d), calibrated in
known-dry operation. Under a reported rain rate R_rep it expects

    S_exp(f, d) = S_ref(f, d) - gamma(f, R_rep) * d / 1000            [dB]

and forms the residual of each received transport block

    Delta = SINR_obs - S_exp .

Honest operation gives Delta ~ 0. A force-up spoof (R_rep = 0 while it truly rains)
leaves Delta = -gamma(f, R_true) * d / 1000, i.e. a negative ramp in distance.
The test statistic is the mean residual over a window of W transport blocks received
on links beyond MIN_DIST; an alarm is raised when it falls below a threshold set for
a target false-positive rate.

The null hypothesis is composite: honest operation at every rain rate
(reported == true), not only clear sky. Honest operation in rain
carries a small negative residual of its own, because the observed excess loss
slightly exceeds the pure ITU term (AMC shifts the mix of decoded transport blocks
within a distance bin). Calibrating the threshold against clear sky alone would
understate the false-positive rate, so H0 here pools all rain rates.

Two variants are evaluated:
  * "regression" (calibration-free): regress per-link mean SINR on [1, log10 d, d]
    and read the linear coefficient as an estimate of gamma. Its estimate is too
    noisy for detection (see the printed table).
  * "reference": the residual test described above, used in the paper.

Outputs (PNG only, sized for a single IEEEtran column at ~100% scale):
  results/figures/defense_sinr_consistency.png
and the numerical results printed to stdout.
"""
import glob
import os
import re

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
FIGDIR = os.path.join(RESULTS, "figures")
PAPER_FIGS = os.path.join(HERE, "paper", "figures")

CB = {"black": "#000000", "vermilion": "#D55E00", "blue": "#0072B2",
      "mustard": "#E69F00", "purple": "#CC79A7", "grey": "#999999"}

RAINS = [0, 10, 20, 30, 37.5, 50]
CAL_SEEDS = range(0, 5)     # seeds used to calibrate the clear-sky reference
TEST_SEEDS = range(5, 10)   # held-out seeds used for every reported number
MIN_DIST = 150.0            # detector watches long links, where the signature lives
BIN_W = 25.0
WINDOWS = [5, 10, 20, 50, 100]
FPR = 0.01                  # target false-positive rate
NBOOT = 4000

# ITU-R P.838 specific attenuation, Dataset-1 table (dB/km)
# Same table the simulator injects, so the detector is model-matched: the ROC below
# is an upper bound on what a deployment with model mismatch would achieve. The
# noise it must overcome (blockage, LOS/NLOSv transitions, fading) is not matched.
_KR = np.linspace(0.0, 50.0, 20)
_GAMMA = {
    5.0: [0.0000, 0.0011, 0.0036, 0.0072, 0.0118, 0.0172, 0.0234, 0.0304, 0.0381,
          0.0465, 0.0556, 0.0653, 0.0757, 0.0867, 0.0984, 0.1106, 0.1234, 0.1367,
          0.1507, 0.1651],
    73.0: [0.0000, 2.1748, 3.5993, 4.8329, 5.9568, 7.0057, 7.9984, 8.9467, 9.8586,
           10.7397, 11.5945, 12.4262, 13.2374, 14.0304, 14.8069, 15.5683, 16.3160,
           17.0510, 17.7743, 18.4867],
}


def gamma_itu(band_ghz, rain_mm_hr):
    """Specific attenuation gamma(f, R) in dB/km."""
    return float(np.interp(rain_mm_hr, _KR, _GAMMA[band_ghz]))


def rain_from_gamma(band_ghz, gamma_db_km):
    """Invert the ITU-R curve: implied rain rate (mm/h) for a measured gamma."""
    return float(np.interp(gamma_db_km, _GAMMA[band_ghz], _KR))


# data loading
def load_phy(scen, rain, seeds, regime="fixwidth", min_dist=MIN_DIST):
    """Received transport blocks beyond min_dist, tagged with their distance bin."""
    frames = []
    for s in seeds:
        path = os.path.join(RESULTS, f"phy-s{s}_{regime}_{scen}_{rain:g}.csv")
        if not os.path.exists(path):
            continue
        d = pd.read_csv(path)
        d = d[(d.dist_m > min_dist) & np.isfinite(d.sinr_db)]
        d = d.assign(bin=(d.dist_m // BIN_W * BIN_W).astype(int))
        frames.append(d[["bin", "dist_m", "sinr_db"]])
    if not frames:
        raise FileNotFoundError(f"no phy CSVs for {scen} @ {rain} mm/h")
    return pd.concat(frames, ignore_index=True)


# variant 1: calibration-free regression
def gamma_hat_regression(scen, rain, seeds, regime="fixwidth"):
    """Estimate gamma by regressing per-link mean SINR on [1, log10 d, d].

    The coefficient on the linear-d term is the excess specific attenuation in
    dB/m; x1000 gives dB/km. Returns one estimate per seed.
    """
    out = []
    for s in seeds:
        path = os.path.join(RESULTS, f"phy-s{s}_{regime}_{scen}_{rain:g}.csv")
        if not os.path.exists(path):
            continue
        d = pd.read_csv(path)
        d = d[(d.dist_m > 0) & np.isfinite(d.sinr_db)]
        g = d.groupby("dist_m").sinr_db.mean()
        x, y = g.index.values, g.values
        A = np.column_stack([np.ones_like(x), np.log10(x), x])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        out.append(-coef[2] * 1000.0)   # SINR slope is negative for a loss
    return np.array(out)


# variant 2: reference-profile residual detector
def clear_sky_reference(band_ghz=73.0, regime="fixwidth"):
    """Per-distance-bin clear-sky mean SINR, calibrated on CAL_SEEDS.

    attackA at rain 0 is honest operation (carrier 73 GHz, reported rain 0, true
    rain 0), so it is the legitimate calibration set for the 73 GHz profile.
    """
    scen = {73.0: "attackA", 5.0: "attackB"}[band_ghz]
    return load_phy(scen, 0, CAL_SEEDS, regime).groupby("bin").sinr_db.mean()


def residuals(scen, true_rain, reported_rain, ref, band_ghz=73.0, regime="fixwidth",
              seeds=TEST_SEEDS):
    """Per-transport-block residual SINR_obs - S_exp under the reported rain."""
    d = load_phy(scen, true_rain, seeds, regime)
    d = d[d.bin.isin(ref.index)]
    expected = d.bin.map(ref) - gamma_itu(band_ghz, reported_rain) * d.dist_m / 1000.0
    return (d.sinr_db - expected).values


def honest_null(ref, band_ghz=73.0, regime="fixwidth", seeds=TEST_SEEDS):
    """Composite H0: honest operation at every rain rate (reported == true)."""
    return np.concatenate([
        residuals("attackA", r, r, ref, band_ghz, regime, seeds) for r in RAINS
    ])


def tb_rate_per_vehicle(scen="attackA", rain=50, regime="fixwidth", seeds=TEST_SEEDS,
                        min_dist=MIN_DIST, sim_time=30.0):
    """Transport blocks per second per receiving vehicle on links beyond min_dist.

    Converts an observation window in transport blocks into seconds. Computed per
    receiver, because each vehicle runs its own detector.
    """
    rates = []
    for s in seeds:
        path = os.path.join(RESULTS, f"phy-s{s}_{regime}_{scen}_{rain:g}.csv")
        if not os.path.exists(path):
            continue
        d = pd.read_csv(path)
        d = d[(d.dist_m > min_dist) & np.isfinite(d.sinr_db)]
        rates.append(d.groupby("rxnode").size().mean() / sim_time)
    return float(np.mean(rates))


def detection_rate(h0, h1, window, fpr=FPR, rng=None):
    """P(detect) at the given FPR for a window of `window` transport blocks."""
    rng = rng or np.random.default_rng(0)
    null = rng.choice(h0, (NBOOT, window)).mean(axis=1)
    alt = rng.choice(h1, (NBOOT, window)).mean(axis=1)
    return float((alt < np.percentile(null, fpr * 100)).mean())


# mitigation: what re-selection recovers once the spoof is flagged
def reach_metrics(scen, rain, regime="fixwidth", seeds=range(10), far=100.0):
    """Long-range PRR (>far) and reliable range (largest bin with PRR >= 90%)."""
    prr_far, ranges = [], []
    for s in seeds:
        path = os.path.join(RESULTS, f"pkts-s{s}_{regime}_{scen}_{rain:g}.csv")
        if not os.path.exists(path):
            continue
        d = pd.read_csv(path)
        tx = d[d.txrx == "tx"].groupby(["src", "dst"]).agg(
            n=("seq", "size"), dist=("dist_m", "mean"))
        rx = d[d.txrx == "rx"].groupby(["src", "dst"]).size().rename("r")
        t = tx.join(rx).fillna(0.0)
        far_links = t[t.dist > far]
        if far_links.n.sum() > 0:
            prr_far.append(far_links.r.sum() / far_links.n.sum())
        t = t.assign(bin=(t.dist // BIN_W * BIN_W + BIN_W / 2))
        b = t.groupby("bin").apply(lambda g: g.r.sum() / g.n.sum(), include_groups=False)
        ok = b[b >= 0.90]
        ranges.append(float(ok.index.max()) if len(ok) else 0.0)
    return float(np.mean(prr_far)), float(np.mean(ranges))


# figure
def set_style():
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 400, "savefig.bbox": "tight",
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif", "font.size": 8.0,
        "axes.titlesize": 8.0, "axes.labelsize": 8.0,
        "xtick.labelsize": 7.2, "ytick.labelsize": 7.2, "legend.fontsize": 6.6,
        "axes.grid": True, "grid.alpha": 0.28, "grid.linewidth": 0.45,
        "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.4,
    })


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    set_style()
    rng = np.random.default_rng(0)

    ref73 = clear_sky_reference(73.0)
    ref5 = clear_sky_reference(5.0)

    # 1. calibration-free regression
    print("=" * 74)
    print("Variant 1 (calibration-free regression)")
    print("=" * 74)
    print(f"{'rain':>6} {'true gamma':>11} {'gamma_hat':>11} {'sd':>7}")
    for r in RAINS:
        est = gamma_hat_regression("attackA", r, range(10))
        print(f"{r:>6g} {gamma_itu(73.0, r):>11.3f} {est.mean():>11.3f} {est.std():>7.3f}")
    print("  -> across-seed sd exceeds the whole signal range; unusable.\n")

    # 2. reference-profile detector
    # H0 is composite: honest operation at 73 GHz across every rain rate.
    # H1: force-up spoof (reported rain 0, true rain R).
    print("=" * 74)
    print("Variant 2 (reference-profile residual)")
    print("=" * 74)
    print("Honest-operation residual by true rain (reported == true):")
    for r in RAINS:
        hr = residuals("attackA", r, r, ref73)
        print(f"  rain {r:>4g} mm/h: mean {hr.mean():+.2f} dB")
    print("  -> honest operation in rain carries its own small negative offset,")
    print("     so the null must pool all rain rates, not just clear sky.")

    h0 = honest_null(ref73)
    rate = tb_rate_per_vehicle()
    print(f"\nComposite H0: n={len(h0)}, mean {h0.mean():+.2f} dB, sd {h0.std():.2f} dB")
    print(f"\nLong-range TB rate: {rate:.1f} per second per receiving vehicle, so")
    print("  " + ", ".join(f"W={w} = {w / rate:.1f} s" for w in WINDOWS))
    print(f"\nDetection rate at {FPR:.0%} FPR, held-out seeds {list(TEST_SEEDS)}:")
    print(f"{'rain':>6} {'mean resid':>11} {'vs H0':>8} | " +
          " ".join(f"W={w:<4}" for w in WINDOWS))

    det = {}
    for r in RAINS[1:]:
        h1 = residuals("attackA", r, 0, ref73)
        det[r] = [detection_rate(h0, h1, w, rng=rng) for w in WINDOWS]
        row = f"{r:>6g} {h1.mean():>+10.2f} dB {h1.mean() - h0.mean():>+7.2f} | "
        row += " ".join(f"{p * 100:>4.0f}%" for p in det[r])
        print(row)

    # 3. force-down at 5 GHz
    print("\nForce-down (5 GHz) residual:")
    for r in [0, 50]:
        hb = residuals("attackB", r, 50, ref5, band_ghz=5.0)
        print(f"  true rain {r:>4g} mm/h, reported 50: mean residual {hb.mean():+.3f} dB")
    print(f"  gamma(5 GHz, 50 mm/h) = {gamma_itu(5.0, 50):.3f} dB/km "
          f"-> {gamma_itu(5.0, 50) * 0.29:.2f} dB over a 290 m link. Undetectable.\n")

    # 4. mitigation: band re-selection
    # Once the report is flagged, the vehicle discards it and selects the band
    # from its own link budget, which is what the honest controller already does.
    # The post-mitigation state is therefore the baseline scenario.
    print("=" * 74)
    print("Mitigation: band re-selection after the spoof is flagged (n=10 seeds)")
    print("=" * 74)
    print(f"{'rain':>6} | {'attacked PRR':>12} {'restored':>9} {'gain':>7} | "
          f"{'attacked range':>14} {'restored':>9}")
    for r in RAINS:
        pa, ra = reach_metrics("attackA", r)
        pb, rb = reach_metrics("baseline", r)
        print(f"{r:>6g} | {pa:>11.1%} {pb:>9.1%} {pb - pa:>+6.1%} | "
              f"{ra:>13.0f} m {rb:>7.0f} m")
    print()

    # figure: residual vs distance (single column)
    fig, a1 = plt.subplots(figsize=(3.5, 2.15))

    ref73_all = load_phy("attackA", 0, CAL_SEEDS, min_dist=0.0).groupby("bin").sinr_db.mean()
    ref5_all = load_phy("attackB", 0, CAL_SEEDS, min_dist=0.0).groupby("bin").sinr_db.mean()

    def profile(scen, rain, reported, ref_all, band):
        d = load_phy(scen, rain, TEST_SEEDS, min_dist=0.0)
        d = d[d.bin.isin(ref_all.index)]
        exp = d.bin.map(ref_all) - gamma_itu(band, reported) * d.dist_m / 1000.0
        return (d.sinr_db - exp).groupby(d.bin.values).mean()

    for c, r in zip([CB["mustard"], CB["purple"], CB["vermilion"]], [20, 30, 50]):
        pr = profile("attackA", r, 0, ref73_all, 73.0)
        a1.plot(pr.index + BIN_W / 2, pr.values, "o-", color=c, ms=2.6,
                label=f"Attack Type 1, {r:g} mm h$^{{-1}}$")
        xs = np.array([0, 300])
        a1.plot(xs, -gamma_itu(73.0, r) * xs / 1000.0, ":", color=c, lw=0.9, zorder=0)
    pr = profile("attackB", 50, 50, ref5_all, 5.0)
    a1.plot(pr.index + BIN_W / 2, pr.values, "D-", color=CB["blue"], ms=2.6,
            label="Attack Type 2, 50 mm h$^{-1}$")
    a1.axhline(0, color=CB["grey"], lw=0.9, ls="--", zorder=0)
    a1.axvspan(MIN_DIST, 300, color=CB["grey"], alpha=0.13, lw=0, zorder=0)
    a1.text(MIN_DIST + 5, 0.7, "detector window", fontsize=5.8, color="0.35")
    a1.set_xlim(0, 300)
    a1.set_xlabel("TX–RX distance (m)")
    a1.set_ylabel("SINR residual (dB)")
    a1.legend(loc="lower left", framealpha=0.9, handlelength=1.3, borderpad=0.3)

    fig.tight_layout(pad=0.3)
    name = "defense_sinr_consistency.png"
    for d in (FIGDIR, PAPER_FIGS):
        os.makedirs(d, exist_ok=True)
        fig.savefig(os.path.join(d, name))
        print(f"wrote {os.path.join(d, name)}")
    plt.close(fig)


if __name__ == "__main__":
    main()
