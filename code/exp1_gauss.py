"""Experiment I (two regimes), Gaussian residuals with sigma = 1.

(a) Exact optimal error and window of the uniform-window decoder over a grid of
    kappa = (1-p) sigma^2 / Delta^2 and Delta / sigma; the half-window law
    A(m) = P_single(m) + (1-p)(m-1)/2 and its minimiser; the minimiser of the
    two-regime bound g_2; the exact error of the Bayes forward filter.
(b) Simulated comparison of decoders at Delta = sigma = 1: exponentially weighted
    windows (causal), centred windows and the forward-backward smoother (offline).
(c) Error conditional on the age of the current regime (the half-window effect).
(d) pgfplots data for the illustration in Section 3.
(e) The half-window law beyond the symmetric chain: asymmetric switching and
    unequal noise levels (exact), and non-Markov regimes with non-geometric
    sojourns (simulated), compared with a hidden Markov filter."""
from __future__ import annotations

import csv
import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.optimize import brentq
from scipy.stats import norm

from common import (COLORS, DATA_DIR, legend_below, panel_row, FILTER_GRID, FULL_WIDTH, ACCENT, INK, MARKERS, SERIES, apply_style,
                    centred_best, centred_errors, ewma_best, ewma_errors, exact_gauss_errors,
                    exact_two_state_gauss, filter_error_grid, fmt, forward_filter_errors, mean_se,
                    ratio_se, run_lengths, save_figure, simulate_alternating, simulate_sym_chain,
                    smoother_error, store_values, symmetric_bounds, write_tex)

KAPPAS = [0.1, 0.03, 0.01, 3e-3, 1e-3, 3e-4, 1e-4]
DELTAS = [0.25, 0.5, 1.0, 2.0]
SIM_KAPPAS = [0.03, 0.01, 3e-3, 1e-3]          # Delta = sigma = 1
R_SIM, T_SIM = 40, 250_000
R_TUNE = 10                                      # independent trajectories used only for tuning
AGE_P, AGE_M = 0.999, [8, 16, 32]                # switch-age illustration, Delta = sigma = 1
ILL_P, ILL_MMAX = 0.99, 60                      # Section 3 illustration, Delta = sigma = 1
ASYM_RATIO = 3.0                                 # Q_21 / Q_12 for the asymmetric chains (pi_1 = 3/4)
NM_MEAN, NM_MMAX, NM_R, NM_T, NM_BURN = 100, 60, 40, 250_000, 20_000   # non-Markov regimes
SEED = 99173


def half_window_A(p, delta, m):
    return norm.sf(np.sqrt(m) * delta / 2) + (1 - p) * (m - 1) / 2


def g2(p, delta, m):
    return np.exp(-m * delta**2 / 8) + 1 - p ** (m - 1)


def rule_window(kappa, delta):
    """Continuous minimiser of Phi(-sqrt(m) Delta/2) + (1-p) m/2: m = 4 z^2/Delta^2
    with phi(z)/z = 4 kappa (sigma = 1)."""
    z = brentq(lambda z: norm.pdf(z) / z - 4 * kappa, 1e-3, 40)
    return 4 * z * z / delta**2


def part_a():
    rows = []
    for d in DELTAS:
        for kap in KAPPAS:
            eps = kap * d * d
            p = 1 - eps
            m_max = max(int(3 * 8 / d**2 * math.log(1 / kap)) + 40, 40)
            m = np.arange(1, m_max + 1)
            ex = exact_gauss_errors(p, d, m_max)
            A = half_window_A(p, d, m)
            lo = p ** (m - 1) * A
            assert np.all(ex <= A + 1e-12) and np.all(ex >= lo - 1e-12), "half-window law violated"
            i = int(np.argmin(ex))
            iA = int(np.argmin(A))
            ig = int(np.argmin(g2(p, d, m)))
            assert i < m_max - 5
            # rule window computed from a misestimated kappa (factor 1/2 or 2)
            rob = {c: ex[min(max(int(round(rule_window(c * kap, d))), 1), m_max) - 1] / ex[i] - 1
                   for c in (0.5, 2.0)}
            # asymmetric switching with the same switch rate: Q_12 = q, Q_21 = ASYM_RATIO q
            q = eps * (1 + ASYM_RATIO) / (2 * ASYM_RATIO)
            Qa = np.array([[1 - q, q], [ASYM_RATIO * q, 1 - ASYM_RATIO * q]])
            exa, _ = exact_two_state_gauss(Qa, (-d / 2, d / 2), (1.0, 1.0), m_max)
            pmin = 1 - ASYM_RATIO * q
            assert np.all(exa <= A + 1e-12) and np.all(exa >= pmin ** (m - 1) * A - 1e-12), \
                "half-window law violated (asymmetric switching)"
            ia = int(np.argmin(exa))
            rows.append(dict(delta=d, kappa=kap, p=p, mstar=i + 1, Pstar=ex[i], mA=iA + 1,
                             PA=ex[iA], mg2=ig + 1, Pg2=ex[ig], rule=rule_window(kap, d),
                             filt=filter_error_grid(p, d),
                             filt_fine=filter_error_grid(p, d, N=2 * FILTER_GRID), gap=1 - p ** i,
                             asym=4 * kap * math.log(1 / kap), rob_half=rob[0.5], rob_double=rob[2.0],
                             mstar_as=ia + 1, Pstar_as=exa[ia], PA_as=exa[iA]))
    # unequal noise levels, symmetric and asymmetric switching
    for Q, sd in ((np.array([[0.99, 0.01], [0.01, 0.99]]), (1.0, 2.0)),
                  (np.array([[0.998, 0.002], [0.006, 0.994]]), (0.7, 1.4))):
        m_max = 400
        m = np.arange(1, m_max + 1)
        exu, pi = exact_two_state_gauss(Q, (-0.5, 0.5), sd, m_max)
        Pbar = pi[0] * norm.sf(np.sqrt(m) * 0.5 / sd[0]) + pi[1] * norm.sf(np.sqrt(m) * 0.5 / sd[1])
        sbar = pi[0] * Q[0, 1] + pi[1] * Q[1, 0]
        A = Pbar + sbar * (m - 1) / 2
        lo = min(Q[0, 0], Q[1, 1]) ** (m - 1) * A
        assert np.all(exu <= A + 1e-12) and np.all(exu >= lo - 1e-12), "half-window law violated (unequal noise)"
    return rows


def part_b(rng, rows_a):
    """Exponential weights, centred windows and the smoother at Delta = sigma = 1. The
    tuning parameters (effective length of the exponential weights, half-width of the
    centred window) are chosen on R_TUNE independent trajectories by a grid search, and
    the errors are then estimated on R_SIM fresh trajectories."""
    out = []
    for kap in SIM_KAPPAS:
        p = 1 - kap
        ra = next(r for r in rows_a if r["delta"] == 1.0 and r["kappa"] == kap)
        ms = ra["mstar"]
        burn, pad = 6 * ms + 50, 3 * ms + 60
        n_grid = np.geomspace(1.5, 3 * ms, 26)
        h_grid = sorted(set(np.unique(np.round(np.geomspace(1, 3 * ms, 30)).astype(int))))
        Et = simulate_sym_chain(rng, p, R_TUNE, T_SIM)
        Yt = np.where(Et == 1, 0.5, -0.5) + rng.standard_normal((R_TUNE, T_SIM))
        n_ew, _, _ = ewma_best(Et, Yt, n_grid, burn)
        h_c, _ = centred_best(Et, Yt, h_grid, pad)
        assert n_grid[0] < n_ew < n_grid[-1] and h_grid[0] <= h_c < h_grid[-1], "tuning grid too narrow"
        E = simulate_sym_chain(rng, p, R_SIM, T_SIM)
        Y = np.where(E == 1, 0.5, -0.5) + rng.standard_normal((R_SIM, T_SIM))
        ew = ewma_errors(E, Y, n_ew, burn)
        ce = centred_errors(E, Y, h_c, pad)
        sm = smoother_error(E, 1.0 * Y, p, pad)
        e_ew, se_ew = mean_se(ew)
        e_c, se_c = mean_se(ce)
        e_sm, se_sm = mean_se(sm)
        r_ew, se_r_ew = ratio_se(ew, ra["filt"])
        off, se_off = ra["Pstar"] / e_c, ra["Pstar"] * se_c / e_c**2
        out.append(dict(kappa=kap, uniform=ra["Pstar"], mstar=ms, ewma=e_ew, ewma_se=se_ew, n_ewma=n_ew,
                        ewma_ratio=r_ew, ewma_ratio_se=se_r_ew, filt=ra["filt"],
                        centred=e_c, centred_se=se_c, m_centred=2 * h_c + 1,
                        smoother=e_sm, smoother_se=se_sm, offline=off, offline_se=se_off,
                        smoother_gain=e_c / e_sm))
    return out


def part_c(rng):
    """Error given the age of the current regime; per-trajectory conditional error rates."""
    E = simulate_sym_chain(rng, AGE_P, R_SIM, T_SIM)
    Y = np.where(E == 1, 0.5, -0.5) + rng.standard_normal((R_SIM, T_SIM))
    L = run_lengths(E)
    C = np.zeros((R_SIM, T_SIM + 1))
    C[:, 1:] = np.cumsum(Y, axis=1)
    res = {}
    for m in AGE_M:
        S = C[:, m:] - C[:, :T_SIM - m + 1]
        wrong = (S > 0) != (E[:, m - 1:] == 1)
        age = np.minimum(L[:, m - 1:], m)
        per = np.zeros((m, R_SIM))
        for r in range(R_SIM):
            cnt = np.bincount(age[r], minlength=m + 1)[1:]
            err = np.bincount(age[r], weights=wrong[r], minlength=m + 1)[1:]
            assert cnt.min() > 0, "empty age class"
            per[:, r] = err / cnt
        res[m] = per
    return res


def sojourn_sampler(rng, kind):
    if kind == "geometric":
        return lambda n: rng.geometric(1 / NM_MEAN, n)
    if kind == "regular":                       # sum of four geometric sojourns
        return lambda n: rng.geometric(4 / NM_MEAN, (n, 4)).sum(axis=1)
    if kind == "bursty":                        # half short, half long sojourns
        def f(n):
            short = rng.random(n) < 0.5
            return np.where(short, rng.geometric(1 / 10, n), rng.geometric(1 / 190, n))
        return f
    raise ValueError(kind)


NM_LAWS = ["geometric", "regular", "bursty"]


def part_e(rng):
    sbar = 1 / NM_MEAN
    p = 1 - sbar
    m = np.arange(1, NM_MMAX + 1)
    A = half_window_A(p, 1.0, m)
    mA = int(np.argmin(A)) + 1
    out = {}
    # exponential weights are tuned once, on independent trajectories with Markov regimes
    Et = simulate_alternating(rng, sojourn_sampler(rng, "geometric"), R_TUNE, NM_T, NM_BURN)
    Yt = np.where(Et == 1, 0.5, -0.5) + rng.standard_normal((R_TUNE, NM_T))
    n_grid = np.geomspace(3, 60, 24)
    n_ew, _, _ = ewma_best(Et, Yt, n_grid, 500)
    assert n_grid[0] < n_ew < n_grid[-1], "tuning grid too narrow"
    for kind in NM_LAWS:
        E = simulate_alternating(rng, sojourn_sampler(rng, kind), NM_R, NM_T, NM_BURN)
        Y = np.where(E == 1, 0.5, -0.5) + rng.standard_normal((NM_R, NM_T))
        C = np.zeros((NM_R, NM_T + 1))
        C[:, 1:] = np.cumsum(Y, axis=1)
        per = np.zeros((NM_MMAX, NM_R))
        for mm in m:
            S = C[:, mm:] - C[:, :NM_T - mm + 1]
            per[mm - 1] = ((S > 0) != (E[:, mm - 1:] == 1)).mean(axis=1)
        err = per.mean(axis=1)
        se = per.std(axis=1, ddof=1) / math.sqrt(NM_R)
        U = rng.random((NM_R, NM_T))
        filt_per = forward_filter_errors(E, Y.astype(float), p, 500, U)
        ew_per = ewma_errors(E, Y, n_ew, 500)
        filt, filt_se = mean_se(filt_per)
        e_ew, e_ew_se = mean_se(ew_per)
        ratio, ratio_s = ratio_se(per[mA - 1], filt_per)
        sw = float(np.mean(E[:, 1:] != E[:, :-1]))
        out[kind] = dict(err=err, se=se, filt=filt, filt_se=filt_se, ewma=e_ew, ewma_se=e_ew_se,
                         ratio=ratio, ratio_se=ratio_s, sbar_hat=sw, n_ewma=n_ew)
    return out, A, mA


def illustration(rows_a):
    p, d = ILL_P, 1.0
    m = np.arange(1, ILL_MMAX + 1)
    ex = exact_gauss_errors(p, d, ILL_MMAX)
    A = half_window_A(p, d, m)
    lo = p ** (m - 1) * A
    gg = g2(p, d, m)
    iA = int(np.argmin(A))
    Lb, Ub = symmetric_bounds(p, norm.sf(np.sqrt(m) * d / 2), m)
    assert np.all(ex <= Ub + 1e-12) and np.all(ex >= Lb - 1e-12), "symmetric bounds violated"
    pts = lambda y, step=1: " ".join(f"({mm},{yy:.5f})" for mm, yy in zip(m[::step], y[::step]))
    tex = (
        "% Generated by code/exp1_gauss.py -- do not edit by hand.\n"
        f"\\addplot[name path=upper, draw=black!30, very thin, forget plot] coordinates {{{pts(Ub)}}};\n"
        f"\\addplot[name path=lower, draw=black!30, very thin, forget plot] coordinates {{{pts(Lb)}}};\n"
        "\\addplot[black!8] fill between[of=upper and lower];\n"
        "\\addlegendentry{bounds of Corollary~\\ref{cor:symmetric-bounds}}\n"
        f"\\addplot[black!50, semithick, dashed] coordinates {{{pts(np.minimum(gg, 1.05))}}};\n"
        "\\addlegendentry{\\(g_2(m)\\), Proposition~\\ref{thm:recoverability}}\n"
        f"\\addplot[black, semithick] coordinates {{{pts(A)}}};\n"
        "\\addlegendentry{\\(A(m)\\), Theorem~\\ref{thm:half-window}}\n"
        f"\\addplot[only marks, mark=o, mark size=1.1pt, accent, mark options={{line width=0.5pt}}] coordinates {{{pts(ex, 2)}}};\n"
        "\\addlegendentry{exact error}\n"
        f"\\addplot[only marks, mark=*, mark size=1.8pt, accent] coordinates {{({iA + 1},{A[iA]:.5f})}};\n"
    )
    write_tex("fig_halfwindow.tex", tex)
    return dict(p=p, mA=iA + 1, A=A[iA], mstar=int(np.argmin(ex)) + 1, Pstar=float(ex.min()),
                g2min=float(gg.min()), g2arg=int(np.argmin(gg)) + 1)


def run() -> dict:
    rng = np.random.default_rng(SEED)
    rows_a = part_a()
    rows_b = part_b(rng, rows_a)
    age = part_c(rng)
    ill = illustration(rows_a)
    nm, A_nm, mA_nm = part_e(rng)
    for path, rows in ((DATA_DIR / "expI_gauss_exact.csv", rows_a),
                       (DATA_DIR / "expI_gauss_decoders.csv", rows_b)):
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    make_figures(rows_a, rows_b, age)
    make_nonmarkov_figure(nm, A_nm, mA_nm)
    return summarize(rows_a, rows_b, age, ill, nm, A_nm, mA_nm)


def make_figures(rows_a, rows_b, age) -> None:
    from matplotlib.lines import Line2D
    apply_style()
    K, G = INK["k"], INK["gr"]
    # ---- Figure: optimal error and optimal window against kappa -------------------------
    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 2.65))
    kk = np.logspace(-4.1, -0.95, 80)
    shapes = ["o", "s", "^", "D"]
    for d, mk in zip(DELTAS, shapes):
        rr = [r for r in rows_a if r["delta"] == d]
        x = [r["kappa"] for r in rr]
        axes[0].plot(x, [r["filt"] for r in rr], mk, color=G, ms=3.5, mfc="white", mew=0.8, zorder=3)
        axes[0].plot(x, [r["Pstar"] for r in rr], mk, color=K, ms=3.5, zorder=4)
        axes[1].plot(x, [r["mg2"] * d * d for r in rr], mk, color=G, ms=3.5, mfc="white", mew=0.8, zorder=3)
        axes[1].plot(x, [r["mstar"] * d * d for r in rr], mk, color=K, ms=3.5, zorder=4)
    uu = np.linspace(0.05, 400, 40000)
    law = [np.min(norm.sf(np.sqrt(uu) / 2) + k * uu / 2) for k in kk]
    axes[0].plot(kk, law, color=ACCENT, lw=1.1, zorder=5)
    axes[0].plot(kk, 4 * kk * np.log(1 / kk), color=G, lw=0.9, ls="--", zorder=1)
    axes[0].plot(kk, 2 * kk * np.log(1 / kk), color=G, lw=0.9, ls=":", zorder=1)
    axes[1].plot(kk, [rule_window(k, 1.0) for k in kk], color=ACCENT, lw=1.1, zorder=5)
    axes[1].plot(kk, 8 * np.log(1 / kk), color=G, lw=0.9, ls="--", zorder=1)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel(r"$\kappa=\bar s\,\sigma^2/\Delta^2$")
    axes[0].set_yscale("log")
    axes[0].set_title("Optimal error")
    axes[0].set_ylabel("Error probability")
    axes[1].set_title("Optimal window")
    axes[1].set_ylabel(r"$m\,\Delta^2/\sigma^2$")
    hd = [Line2D([], [], color=K, ls="none", marker=mk, ms=3.5, mfc="white", mew=0.8,
                 label=f"$\\Delta/\\sigma={d:g}$") for d, mk in zip(DELTAS, shapes)]
    axes[0].add_artist(axes[0].legend(handles=hd, loc="upper left", handletextpad=0.2, labelspacing=0.25,
                                      borderaxespad=0.4))
    h0 = [Line2D([], [], color=K, ls="none", marker="o", ms=3.5, label="uniform window, optimal"),
          Line2D([], [], color=G, ls="none", marker="o", ms=3.5, mfc="white", mew=0.8, label="Bayes filter"),
          Line2D([], [], color=ACCENT, lw=1.1, label=r"$\min_u\{\bar\Phi(\sqrt{u}/2)+\kappa u/2\}$"),
          Line2D([], [], color=G, lw=0.9, ls="--", label=r"$4\kappa\log(1/\kappa)$"),
          Line2D([], [], color=G, lw=0.9, ls=":", label=r"$2\kappa\log(1/\kappa)$")]
    h1 = [Line2D([], [], color=K, ls="none", marker="o", ms=3.5, label=r"optimal window $m^\star$"),
          Line2D([], [], color=G, ls="none", marker="o", ms=3.5, mfc="white", mew=0.8, label=r"minimiser of $g_2$"),
          Line2D([], [], color=ACCENT, lw=1.1, label=r"rule $4z_\kappa^2$"),
          Line2D([], [], color=G, lw=0.9, ls="--", label=r"$8\log(1/\kappa)$")]
    fig.tight_layout(w_pad=2.0)
    legend_below(axes[0], h0, ncol=2, columnspacing=1.0)
    legend_below(axes[1], h1, ncol=2, columnspacing=1.0)
    save_figure(fig, "expI_collapse.pdf")

    # ---- Figure: window shapes and offline decoding -------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 2.7))
    x = np.array([r["kappa"] for r in rows_b])
    ax = axes[0]
    ax.axhline(1.0, color=G, lw=1.0, ls="--")
    for lev, lab in ((2.0, r"$2$"), (2 * math.log(2), r"$2\log 2$")):
        ax.axhline(lev, color=G, lw=0.7, ls=":")
        ax.text(x[0], lev + 0.02, lab, va="bottom", ha="left", fontsize=8, color=INK["dk"])
    ax.plot(x, [r["uniform"] / r["filt"] for r in rows_b], color=ACCENT, marker="o")
    ax.plot(x, [r["ewma_ratio"] for r in rows_b], color=K, ls="--", marker="o", mfc="white", mew=0.8)
    ax.set_ylim(0.9, 2.15)
    ax.set_title("Causal decoders")
    ax.set_ylabel("Ratio to Bayes filter")
    ax = axes[1]
    ax.plot(x, [r["uniform"] for r in rows_b], color=ACCENT, marker="o")
    ax.plot(x, [r["filt"] for r in rows_b], color=G, lw=1.0, ls="--")
    ax.plot(x, [r["centred"] for r in rows_b], color=K, ls="-.", marker="s", ms=3)
    ax.plot(x, [r["smoother"] for r in rows_b], color=G, ls=":", marker="s", ms=3, mfc="white", mew=0.8)
    ax.set_yscale("log")
    ax.set_title("Offline decoding")
    ax.set_ylabel("Error probability")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel(r"$\kappa$  ($\Delta=\sigma=1$)")
        ax.invert_xaxis()
    h = [Line2D([], [], color=ACCENT, marker="o", label="trailing uniform window"),
         Line2D([], [], color=K, ls="--", marker="o", mfc="white", mew=0.8, label="exponential weights"),
         Line2D([], [], color=G, lw=1.0, ls="--", label="Bayes filter"),
         Line2D([], [], color=K, ls="-.", marker="s", ms=3, label="centred uniform window"),
         Line2D([], [], color=G, ls=":", marker="s", ms=3, mfc="white", mew=0.8, label="Bayes smoother")]
    fig.tight_layout(rect=(0, 0.14, 1, 1), w_pad=2.0)
    fig.legend(handles=h, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0), columnspacing=1.6,
               handletextpad=0.5, labelspacing=0.3)
    save_figure(fig, "expI_decoders.pdf")

    # ---- Figure (appendix): error given the age of the current regime -------------------
    fig, ax = plt.subplots(1, 1, figsize=(0.62 * FULL_WIDTH, 2.6))
    for m, col in zip(AGE_M, SERIES[1:]):
        kx = np.arange(1, m + 1)
        ax.plot(kx / m, age[m].mean(axis=1), color=col, marker="o", ms=2.5, label=f"$m={m}$")
        ax.plot(kx / m, norm.sf((2 * kx - m) / (2 * np.sqrt(m))), color=col, lw=0.8, ls="--")
    ax.axvline(0.5, color=G, lw=0.7, ls=":")
    ax.plot([], [], color=K, lw=0.8, ls="--", label=r"$\bar\Phi((2k-m)/(2\sqrt{m}))$")
    ax.set_xlabel(r"Age of the current regime / window length, $\min(\tau_t,m)/m$")
    ax.set_ylabel("Error given the age")
    ax.legend(loc="upper right")
    fig.tight_layout()
    save_figure(fig, "expI_switch_age.pdf")


def make_nonmarkov_figure(nm, A, mA) -> None:
    from matplotlib.lines import Line2D
    apply_style()
    K, G = INK["k"], INK["gr"]
    m = np.arange(1, NM_MMAX + 1)
    titles = {"geometric": "Geometric sojourns (Markov)", "regular": "Regular sojourns",
              "bursty": "Bursty sojourns"}
    fig, axes = plt.subplots(1, 3, figsize=(FULL_WIDTH, 2.45), sharey=True)
    for ax, kind in zip(axes, NM_LAWS):
        r = nm[kind]
        ax.plot(m, A, color=K, lw=0.9, ls="--")
        ax.plot(m, r["err"], color=ACCENT, lw=1.2)
        ax.axhline(r["filt"], color=G, lw=1.0, ls=":")
        ax.axhline(r["ewma"], color=K, lw=0.9, ls="-.")
        ax.plot([mA], [r["err"][mA - 1]], "o", color=ACCENT, ms=4, zorder=5)
        ax.set_title(titles[kind])
        ax.set_xlabel(r"Window length $m$")
        ax.set_ylim(0, 0.3)
        ax.set_xlim(1, NM_MMAX)
    axes[0].set_ylabel("Error probability")
    h = [Line2D([], [], color=ACCENT, lw=1.2, label="uniform window (simulated)"),
         Line2D([], [], color=K, lw=0.9, ls="--", label=r"$A(m)$"),
         Line2D([], [], color=ACCENT, ls="none", marker="o", ms=4, label=r"rule $m_A$"),
         Line2D([], [], color=G, lw=1.0, ls=":", label="hidden Markov filter"),
         Line2D([], [], color=K, lw=0.9, ls="-.", label="exponential weights")]
    fig.tight_layout(rect=(0, 0.1, 1, 1), w_pad=0.6)
    fig.legend(handles=h, loc="lower center", ncol=5, bbox_to_anchor=(0.5, 0.0), columnspacing=1.4,
               handletextpad=0.5)
    save_figure(fig, "expI_nonmarkov.pdf")


def summarize(rows_a, rows_b, age, ill, nm, A_nm, mA_nm) -> dict:
    small = [r for r in rows_a if r["kappa"] <= 0.03]
    tiny = [r for r in rows_a if r["kappa"] <= 1e-3]
    excess_A = [r["PA"] / r["Pstar"] - 1 for r in small]
    excess_g2 = [r["Pg2"] / r["Pstar"] - 1 for r in small]
    at3 = [r for r in rows_a if r["kappa"] == 1e-3 and r["delta"] <= 1.0]
    at3_all = [r for r in rows_a if r["kappa"] == 1e-3]
    k4 = [r for r in rows_a if r["kappa"] == 1e-4]
    ratio_u = [r["uniform"] / r["filt"] for r in rows_b]
    ratio_e = [r["ewma_ratio"] for r in rows_b]
    off = [r["offline"] for r in rows_b]
    rb3 = next(r for r in rows_b if r["kappa"] == 1e-3)
    # half-window effect: errors for ages below / above m/2 at the largest window
    mbig = AGE_M[-1]
    a = age[mbig]
    before_m, before_se = mean_se(a[: mbig // 2 - 2].mean(axis=0))
    after_m, after_se = mean_se(a[mbig // 2 + 2: -1].mean(axis=0))
    v = {
        "kappa-lo": "10^{-4}", "kappa-hi": "10^{-1}",
        "nconfig": f"{len(rows_a)}",
        "collapse-k": "10^{-3}",
        "collapse-lo": fmt(min(r["Pstar"] for r in at3), 4),
        "collapse-hi": fmt(max(r["Pstar"] for r in at3), 4),
        "collapse-d2": fmt(next(r["Pstar"] for r in at3_all if r["delta"] == 2.0), 4),
        "collapse-m-lo": fmt(min(r["mstar"] * r["delta"]**2 for r in at3_all), 0),
        "collapse-m-hi": fmt(max(r["mstar"] * r["delta"]**2 for r in at3_all), 0),
        "rule-excess-max": fmt(100 * max(excess_A), 1),
        "rule-excess-tiny": fmt(100 * max(r["PA"] / r["Pstar"] - 1 for r in rows_a if r["kappa"] <= 0.01), 1),
        "g2-excess-lo": fmt(100 * min(excess_g2), 0), "g2-excess-hi": fmt(100 * max(excess_g2), 0),
        "robust-half": fmt(100 * max(r["rob_half"] for r in small), 0),
        "robust-double": fmt(100 * max(r["rob_double"] for r in small), 0),
        "sandwich-gap-max": fmt(100 * max(r["gap"] for r in tiny), 1),
        "asym-ratio-lo": fmt(min(r["Pstar"] / r["asym"] for r in k4), 2),
        "asym-ratio-hi": fmt(max(r["Pstar"] / r["asym"] for r in k4), 2),
        "filt-ratio-k4": fmt(next(r["filt"] / (2 * 1e-4 * math.log(1e4)) for r in k4 if r["delta"] == 1.0), 2),
        "unif-filt-lo": fmt(min(r["Pstar"] / r["filt"] for r in rows_a if r["kappa"] <= 0.03), 2),
        "unif-filt-hi": fmt(max(r["Pstar"] / r["filt"] for r in rows_a), 2),
        "sim-ratio-u-lo": fmt(min(ratio_u), 2), "sim-ratio-u-hi": fmt(max(ratio_u), 2),
        "sim-ratio-e-lo": fmt(min(ratio_e), 2), "sim-ratio-e-hi": fmt(max(ratio_e), 2),
        "offline-lo": fmt(min(off), 1), "offline-hi": fmt(max(off), 1),
        "k3-uniform": fmt(rb3["uniform"], 4), "k3-ewma": fmt(rb3["ewma"], 4),
        "k3-filter": fmt(rb3["filt"], 4), "k3-centred": fmt(rb3["centred"], 4),
        "k3-smoother": fmt(rb3["smoother"], 4),
        "age-p": f"{AGE_P}", "age-m": f"{mbig}",
        "age-before": fmt(float(before_m), 2), "age-before-se": fmt(float(before_se), 3),
        "age-after": fmt(float(after_m), 3), "age-after-se": fmt(float(after_se), 3),
        "k3-ewma-se": fmt(rb3["ewma_se"], 4), "k3-centred-se": fmt(rb3["centred_se"], 4),
        "k3-smoother-se": fmt(rb3["smoother_se"], 4),
        "sim-ratio-e-se-max": fmt(max(r["ewma_ratio_se"] for r in rows_b), 2),
        "sim-se-max": fmt(max(max(r["ewma_se"], r["centred_se"], r["smoother_se"]) for r in rows_b), 4),
        "smoother-gain-lo": fmt(min(r["smoother_gain"] for r in rows_b), 1),
        "smoother-gain-hi": fmt(max(r["smoother_gain"] for r in rows_b), 1),
        "Rtune": f"{R_TUNE}", "grid": f"{FILTER_GRID}", "grid-fine": f"{2 * FILTER_GRID}",
        "grid-dev": fmt(100 * max(abs(r["filt"] / r["filt_fine"] - 1) for r in rows_a), 3),
        "ill-p": f"{ill['p']}", "ill-mA": f"{ill['mA']}", "ill-mstar": f"{ill['mstar']}",
        "ill-Pstar": fmt(ill["Pstar"], 4), "ill-A": fmt(ill["A"], 4),
        "ill-g2min": fmt(ill["g2min"], 3), "ill-g2arg": f"{ill['g2arg']}",
        "Tsim": f"{T_SIM}", "Rsim": f"{R_SIM}",
    }
    assert max(excess_A) < 0.05, "half-window rule should be within 5% for kappa <= 0.03"
    for kind in NM_LAWS:
        r = nm[kind]
        assert np.all(r["err"] <= A_nm + 3 * r["se"]), f"half-window bound violated ({kind})"
    asym_dev = [abs(r["Pstar_as"] / r["Pstar"] - 1) for r in small]
    asym_mdiff = max(abs(r["mstar_as"] - r["mstar"]) / r["mstar"] for r in small)
    ratio_nm = [nm[k]["ratio"] for k in NM_LAWS]
    v.update({
        "skew-ratio": f"{ASYM_RATIO:.0f}",
        "skew-pstar-dev": fmt(100 * max(asym_dev), 1),
        "skew-mstar-dev": fmt(100 * asym_mdiff, 0),
        "nm-mean": f"{NM_MEAN}", "nm-mA": f"{mA_nm}", "nm-A": fmt(A_nm[mA_nm - 1], 3),
        "nm-geo": fmt(nm["geometric"]["err"][mA_nm - 1], 3),
        "nm-reg": fmt(nm["regular"]["err"][mA_nm - 1], 3),
        "nm-bur": fmt(nm["bursty"]["err"][mA_nm - 1], 3),
        "nm-filt-geo": fmt(nm["geometric"]["filt"], 3),
        "nm-filt-reg": fmt(nm["regular"]["filt"], 3),
        "nm-filt-bur": fmt(nm["bursty"]["filt"], 3),
        "nm-ratio-lo": fmt(min(ratio_nm), 2), "nm-ratio-hi": fmt(max(ratio_nm), 2),
        "nm-margin": fmt(min(float(np.min(A_nm - nm[k]["err"])) for k in NM_LAWS), 3),
        "nm-best-bur": f"{int(np.argmin(nm['bursty']['err'])) + 1}",
        "nm-best-reg": f"{int(np.argmin(nm['regular']['err'])) + 1}",
        "nm-T": f"{NM_T}", "nm-R": f"{NM_R}",
        "nm-ratio-se-max": fmt(max(nm[k]["ratio_se"] for k in NM_LAWS), 3),
        "nm-se-max": fmt(max(max(float(nm[k]["se"].max()), nm[k]["filt_se"], nm[k]["ewma_se"])
                             for k in NM_LAWS), 4),
    })
    store_values("gauss", v)
    return v


if __name__ == "__main__":
    print(run())
