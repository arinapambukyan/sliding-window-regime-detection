"""Experiment I (two regimes, Appendix), +-1 observations (model (a)).

Exact stationary error of the uniform-window nearest-mean decoder for every
window length (forward recursion over regime and window sum), checked against
simulation; the half-window law A(m) = P_single(m) + (1-p)(m-1)/2 and its
minimiser; the sharper bounds for the symmetric chain; the two-regime bound
g_2 of the baseline bound (Proposition 3.1); exponentially weighted windows and the Bayes
forward filter (simulated)."""
from __future__ import annotations

import csv
import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.stats import binom

from common import (COLORS, DATA_DIR, FULL_WIDTH, ACCENT, INK, apply_style, legend_below, panel_grid, ewma_best, ewma_errors, fmt,
                    forward_filter_errors, mean_se, nearest_mean_stats, save_figure, simulate_sym_chain,
                    store_values, symmetric_bounds, write_tex)

SWEEP_P = [0.85, 0.90, 0.95, 0.97, 0.98, 0.99]
SWEEP_D = [0.8, 1.0, 1.2, 1.4, 1.6]
DETAILED = [(0.97, 0.8), (0.97, 1.2), (0.99, 0.8), (0.99, 1.2)]
TITLES = {(0.97, 0.8): "Lower persistence, harder separation",
          (0.97, 1.2): "Lower persistence, easier separation",
          (0.99, 0.8): "Higher persistence, harder separation",
          (0.99, 1.2): "Higher persistence, easier separation"}
TIE_DELTA = 0.8
T = 30000
R = 120
R_TUNE = 30          # independent trajectories used only to tune the exponential weights
M_MAX = 100
X_FACTOR = 3        # each panel shows windows up to X_FACTOR times the optimum (at least 15)
FILTER_BURN = M_MAX - 1
SEED = 7031


def q_wrong(delta: float) -> float:
    """Probability that a single +-1 observation points to the wrong regime."""
    return (1.0 - delta / 2.0) / 2.0


def g2(m, p, delta, sigma=1.0):
    return np.exp(-m * delta**2 / (8 * sigma**2)) + 1.0 - p ** (m - 1)


def exact_errors(p: float, delta: float, m_max: int) -> np.ndarray:
    """Exact stationary error of the nearest-mean decoder (random tie-breaking)
    for m = 1..m_max, by forward recursion over (regime, window sum)."""
    qp = np.array([q_wrong(delta), 1 - q_wrong(delta)])  # P(Y=+1 | regime 0/1)
    off = m_max
    s = np.arange(-m_max, m_max + 1)
    P = np.zeros((2, 2 * m_max + 1))
    for e in (0, 1):
        P[e, off + 1] += 0.5 * qp[e]
        P[e, off - 1] += 0.5 * (1 - qp[e])
    Q = np.array([[p, 1 - p], [1 - p, p]])
    out = np.zeros(m_max)
    for j in range(1, m_max + 1):
        if j > 1:
            mix = Q.T @ P
            Pn = np.zeros_like(P)
            for e in (0, 1):
                Pn[e, 1:] += mix[e, :-1] * qp[e]
                Pn[e, :-1] += mix[e, 1:] * (1 - qp[e])
            P = Pn
        out[j - 1] = P[0, s > 0].sum() + P[1, s < 0].sum() + 0.5 * P[:, s == 0].sum()
    return out


def exact_single_regime(delta: float, m_max: int) -> np.ndarray:
    q = q_wrong(delta)
    m = np.arange(1, m_max + 1)
    tail = binom.sf(np.floor(m / 2), m, q)          # more than m/2 wrong signs
    tie = np.where(m % 2 == 0, binom.pmf(m // 2, m, q), 0.0)
    return tail + 0.5 * tie


def run() -> dict:
    rng = np.random.default_rng(SEED)
    m = np.arange(1, M_MAX + 1)
    rows, curves, det = [], [], {}
    for p in SWEEP_P:
        for d in SWEEP_D:
            ex = exact_errors(p, d, M_MAX)
            P1 = exact_single_regime(d, M_MAX)
            A = P1 + (1 - p) * (m - 1) / 2
            assert np.all(ex <= A + 1e-12) and np.all(ex >= p ** (m - 1) * A - 1e-12), "half-window law"
            Lb, Ub = symmetric_bounds(p, P1, m)
            assert np.all(ex <= Ub + 1e-12) and np.all(ex >= Lb - 1e-12), "symmetric bounds"
            E = simulate_sym_chain(rng, p, R, T)
            qp = np.where(E == 1, 1 - q_wrong(d), q_wrong(d))
            Y = np.where(rng.random((R, T)) < qp, 1, -1).astype(np.int8)
            U = rng.random((R, T))
            err_per, _ = nearest_mean_stats(E, Y, M_MAX, U)
            err, err_se = mean_se(err_per)
            i, iA, ig = int(np.argmin(ex)), int(np.argmin(A)), int(np.argmin(g2(m, p, d)))
            row = dict(p=p, delta=d, mstar=i + 1, Pstar=ex[i], mstar_sim=int(np.argmin(err)) + 1,
                       mA=iA + 1, PA=ex[iA], mg2=ig + 1, Pg2=ex[ig],
                       maxdev=float(np.max(np.abs(err - ex))), maxz=float(np.max(np.abs(err - ex) / err_se)),
                       se_max=float(err_se.max()))
            if (p, d) in DETAILED:
                # tune the exponential weights on independent trajectories, then evaluate
                n_grid = np.geomspace(1.5, 4 * (i + 1) + 4, 26)
                Et = simulate_sym_chain(rng, p, R_TUNE, T)
                qt = np.where(Et == 1, 1 - q_wrong(d), q_wrong(d))
                Yt = np.where(rng.random((R_TUNE, T)) < qt, 1, -1).astype(np.int8)
                n_ew, _, _ = ewma_best(Et, Yt, n_grid, FILTER_BURN, rng.random((R_TUNE, T)))
                assert n_grid[0] < n_ew < n_grid[-1], "tuning grid too narrow"
                e_ew, e_ew_se = mean_se(ewma_errors(E, Y, n_ew, FILTER_BURN, U))
                llr = math.log((1 - q_wrong(d)) / q_wrong(d)) * Y
                filt, filt_se = mean_se(forward_filter_errors(E, llr, p, FILTER_BURN, U))
                row.update(ewma=float(e_ew), ewma_se=float(e_ew_se), n_ewma=n_ew, filt=float(filt),
                           filt_se=float(filt_se))
                det[(p, d)] = dict(ex=ex, sim=err, sim_se=err_se, A=A, L=Lb, U=Ub, g2=g2(m, p, d),
                                   ewma=e_ew, filt=filt, mstar=i + 1)
            rows.append(row)
            for j in range(M_MAX):
                curves.append(dict(p=p, delta=d, m=j + 1, err_sim=err[j], err_exact=ex[j], A=A[j]))
    for path, rr in ((DATA_DIR / "expI_pm1_summary.csv", rows), (DATA_DIR / "expI_pm1_curves.csv", curves)):
        keys = []
        for r in rr:
            for k in r:
                if k not in keys:
                    keys.append(k)
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rr)
    make_figure(det)
    return summarize(rows)


def make_figure(det) -> None:
    apply_style()
    K, G = INK["k"], INK["gr"]
    m = np.arange(1, M_MAX + 1)
    fig, axes = plt.subplots(2, 2, figsize=(FULL_WIDTH, 4.1))
    for ax, (p, d) in zip(axes.ravel(), DETAILED):
        r = det[(p, d)]
        ms = r["mstar"]
        xmax = max(15, X_FACTOR * ms)
        k = m <= xmax
        ax.fill_between(m[k], r["L"][k], r["U"][k], color="#EDEDED", lw=0, label="bounds for the symmetric chain")
        ax.plot(m[k], r["A"][k], color=K, lw=0.9, ls="--", label=r"$A(m)$")
        ax.plot(m[k], r["ex"][k], color=ACCENT, lw=1.2, label="uniform window, exact")
        ax.plot(m[k][::2], r["sim"][k][::2], "o", color=ACCENT, ms=2.5, mfc="white", mew=0.6,
                label="uniform window, simulated")
        ax.axhline(r["ewma"], color=K, lw=0.9, ls="-.", label="exponential weights")
        ax.axhline(r["filt"], color=G, lw=1.0, ls=":", label="Bayes filter")
        ax.set_title(f"$p={p:.2f}$, $\\Delta={d:.1f}$, $m^\\star={ms}$")
        ax.set_xlim(1, xmax)
        top = max(float(r["U"][k].max()), float(r["A"][k].max()), float(r["ex"][k].max()))
        ax.set_ylim(0, min(1.0, 1.08 * top))
    for ax in axes[1]:
        ax.set_xlabel(r"Window length $m$")
    for ax in axes[:, 0]:
        ax.set_ylabel("Error probability")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.09, 1, 1), w_pad=2.0)
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0), columnspacing=1.6,
               handletextpad=0.5, labelspacing=0.3)
    save_figure(fig, "expI_window_tradeoff.pdf")


def summarize(rows) -> dict:
    for p in SWEEP_P:
        r_ = [next(r for r in rows if r["p"] == p and r["delta"] == d)["mstar"] for d in SWEEP_D]
        assert all(a >= b for a, b in zip(r_, r_[1:])), "m* not monotone in Delta"
    for d in SWEEP_D:
        c_ = [next(r for r in rows if r["p"] == p and r["delta"] == d)["mstar"] for p in SWEEP_P]
        assert all(a <= b for a, b in zip(c_, c_[1:])), "m* not monotone in p"
    assert all(r["mstar"] % 2 == 1 for r in rows), "an even m* appeared"
    det = [r for r in rows if "ewma" in r]
    exA = [r["PA"] / r["Pstar"] - 1 for r in rows]
    exg = [r["Pg2"] / r["Pstar"] - 1 for r in rows]
    lines = [f"{r['p']:.2f} & {r['delta']:.2f} & {r['mstar']} & {r['mA']} & {r['Pstar']:.4f} & "
             f"{r['PA']:.4f} & {r['ewma']:.4f}\\,({r['ewma_se']:.4f}) & {r['filt']:.4f}\\,({r['filt_se']:.4f}) \\\\"
             for r in det]
    write_tex("table_expI_pm1.tex",
              "\\begin{tabular}{cccccccc}\n\\toprule\n"
              "\\(p\\) & \\(\\Delta\\) & \\(m^\\star\\) & \\(m_A\\) & error at \\(m^\\star\\) & "
              "error at \\(m_A\\) & exponential & Bayes filter \\\\\n\\midrule\n"
              + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
    q = q_wrong(TIE_DELTA)
    v = {
        "T": f"{T}", "R": f"{R}", "nconfig": f"{len(rows)}",
        "maxdev": fmt(max(r["maxdev"] for r in rows), 3),
        "maxz": fmt(max(r["maxz"] for r in rows), 1),
        "se-max": fmt(max(r["se_max"] for r in rows), 4),
        "Rtune": f"{R_TUNE}",
        "nagree-sim": f"{sum(r['mstar'] == r['mstar_sim'] for r in rows)}",
        "nA-exact": f"{sum(r['mA'] == r['mstar'] for r in rows)}",
        "A-excess-max": fmt(100 * max(exA), 1),
        "A-excess-med": fmt(100 * float(np.median(exA)), 1),
        "g2-excess-lo": fmt(100 * min(exg), 0), "g2-excess-hi": fmt(100 * max(exg), 0),
        "u-filt-lo": fmt(min(r["Pstar"] / r["filt"] for r in det), 2),
        "u-filt-hi": fmt(max(r["Pstar"] / r["filt"] for r in det), 2),
        "e-filt-lo": fmt(min(r["ewma"] / r["filt"] for r in det), 3),
        "e-filt-hi": fmt(max(r["ewma"] / r["filt"] for r in det), 3),
        "mstar-lo": f"{min(r['mstar'] for r in rows)}", "mstar-hi": f"{max(r['mstar'] for r in rows)}",
        "tie-prob": fmt(2 * q * (1 - q), 2), "tie-delta": f"{TIE_DELTA:.2f}",
        "nones": f"{sum(r['mstar'] == 1 for r in rows)}",
    }
    store_values("pm1", v)
    return v


if __name__ == "__main__":
    print(run())
