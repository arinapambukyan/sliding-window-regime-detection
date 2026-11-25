"""Experiment III: the reflected MMRW itself. Holding walls (a step towards a
wall leaves the walk in place): decoder error on raw increments, exact
boundary-encounter probabilities P(H_{t,m}) from the joint chain, the bound for
raw increments, and the position-aware decoder, which recovers the attempted
step at the walls. Bouncing walls (the walk always moves away from a wall): the
wall steps carry no information, and the position-aware decoder ignores them,
so the information clock runs only in the interior (Section 5)."""
from __future__ import annotations

import csv
import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from common import (COLORS, DATA_DIR, FULL_WIDTH, ACCENT, INK, MARKERS, SERIES, apply_style, fmt, legend_below,
                    mean_se, panel_row,
                    nearest_mean_stats, ordinal_list, save_figure, store_values, sym_Q)
from exp1_pm1 import exact_errors

P = 0.97
DELTA = 1.2
M_LIST = [20, 40, 80, 160]
T = 30000
R = 60
W_MAX = 100
SEED = 4242


def step_up_prob(delta: float) -> np.ndarray:
    """P(step = +1 | regime), regimes coded 0 (drift -delta/2) and 1 (+delta/2)."""
    return np.array([(1 - delta / 2) / 2, (1 + delta / 2) / 2])


def joint_kernel(p: float, delta: float, M: int, wall: str = "hold") -> np.ndarray:
    """Kernel of W_t = (X_t, E_t), state index 2x + e: E_t ~ Q(E_{t-1}, .), then
    X_t = clip(X_{t-1} +- 1, 0, M) with the step law of regime E_t (holding walls),
    or X_t = 1 from 0 and M - 1 from M (bouncing walls)."""
    up = step_up_prob(delta)
    Q = sym_Q(p)
    K = np.zeros((2 * (M + 1), 2 * (M + 1)))
    for x in range(M + 1):
        for e in range(2):
            for e2 in range(2):
                if wall == "bounce" and x in (0, M):
                    K[2 * x + e, 2 * (1 if x == 0 else M - 1) + e2] += Q[e, e2]
                    continue
                K[2 * x + e, 2 * min(x + 1, M) + e2] += Q[e, e2] * up[e2]
                K[2 * x + e, 2 * max(x - 1, 0) + e2] += Q[e, e2] * (1 - up[e2])
    return K


def stationary(K: np.ndarray) -> np.ndarray:
    w, v = np.linalg.eig(K.T)
    pi = np.real(v[:, np.argmin(np.abs(w - 1))])
    return pi / pi.sum()


def exact_boundary_prob(K, Pi, M, w_max):
    """P(H_{t,m}) for m = 1..w_max: some step in the window starts at 0 or M."""
    interior = np.array([2 * x + e for x in range(1, M) for e in range(2)])
    v = Pi[interior].copy()
    KII = K[np.ix_(interior, interior)]
    no_hit = [v.sum()]
    for _ in range(2, w_max + 1):
        v = v @ KII
        no_hit.append(v.sum())
    return 1 - np.array(no_hit)


def simulate(rng, M, Pi, wall="hold"):
    up = step_up_prob(DELTA)
    s0 = rng.choice(len(Pi), size=R, p=Pi)
    x = s0 // 2
    e_prev = s0 % 2
    flips = rng.random((R, T)) < (1 - P)
    E = ((e_prev[:, None] + np.cumsum(flips, axis=1)) % 2).astype(np.int8)
    Z = np.where(rng.random((R, T)) < up[E], 1, -1)
    Y = np.zeros((R, T), dtype=np.int8)
    B = np.zeros((R, T), dtype=bool)          # step t starts at a boundary site
    Xp = np.zeros((R, T), dtype=np.int16)     # position before step t
    for t in range(T):
        Xp[:, t] = x
        B[:, t] = (x == 0) | (x == M)
        xn = np.clip(x + Z[:, t], 0, M)
        if wall == "bounce":
            xn = np.where(x == 0, 1, np.where(x == M, M - 1, xn))
        Y[:, t] = xn - x
        x = xn
    return E, Y, B, Xp


def single_regime_pm1(delta: float, m_max: int) -> np.ndarray:
    """Error of the sign decoder on a single-regime window of +-1 steps (fair ties)."""
    from scipy.stats import binom
    q = (1 - delta / 2) / 2
    mm = np.arange(1, m_max + 1)
    return binom.sf(np.floor(mm / 2), mm, q) + 0.5 * np.where(mm % 2 == 0, binom.pmf(mm // 2, mm, q), 0.0)


def attempted_steps(Y, Xp, M):
    """Position-aware reconstruction of the attempted step: a zero increment at
    0 (at M) is a blocked step towards the wall, i.e. -1 (+1)."""
    Z = Y.astype(np.int8).copy()
    Z[(Xp == 0) & (Y == 0)] = -1
    Z[(Xp == M) & (Y == 0)] = 1
    return Z


def run() -> dict:
    rng = np.random.default_rng(SEED)
    m = np.arange(1, W_MAX + 1)
    proxy = exact_errors(P, DELTA, W_MAX)
    tail = np.exp(-m * DELTA**2 / 8)
    switch = 1 - P ** (m - 1)
    res, rows = {}, []
    for M in M_LIST:
        K = joint_kernel(P, DELTA, M)
        Pi = stationary(K)
        ph = exact_boundary_prob(K, Pi, M, W_MAX)
        E, Y, B, Xp = simulate(rng, M, Pi)
        U = rng.random((R, T))
        err, err_se = mean_se(nearest_mean_stats(E, Y, W_MAX, U)[0])
        err_pa, err_pa_se = mean_se(nearest_mean_stats(E, attempted_steps(Y, Xp, M), W_MAX, U)[0])
        Cb = np.zeros((R, T + 1))
        Cb[:, 1:] = np.cumsum(B, axis=1)
        ph_sim = np.array([((Cb[:, w:] - Cb[:, :T - w + 1]) > 0).mean() for w in m])
        bound = tail + switch + ph
        # bouncing walls: raw increments, and the position-aware decoder that drops wall steps
        Kb = joint_kernel(P, DELTA, M, "bounce")
        Pib = stationary(Kb)
        Eb, Yb, Bb, Xpb = simulate(rng, M, Pib, "bounce")
        err_b, err_b_se = mean_se(nearest_mean_stats(Eb, Yb, W_MAX, U)[0])
        err_bpa, err_bpa_se = mean_se(nearest_mean_stats(Eb, np.where(Bb, 0, Yb).astype(np.int8), W_MAX, U)[0])
        wall_b = float(Pib.reshape(M + 1, 2)[[0, M]].sum())
        # information-clock prediction: the noise term runs on the interior steps only
        P1 = single_regime_pm1(DELTA, W_MAX)
        m_eff = np.maximum(1, np.round(m * (1 - wall_b))).astype(int)
        A_eff = P1[m_eff - 1] + (1 - P) * (m - 1) / 2
        res[M] = dict(err=err, err_pa=err_pa, ph=ph, ph_sim=ph_sim, bound=bound,
                      wall=Pi.reshape(M + 1, 2)[[0, M]].sum(),
                      err_b=err_b, err_bpa=err_bpa, wall_b=wall_b, A_eff=A_eff,
                      se_max=float(max(err_se.max(), err_pa_se.max(), err_b_se.max(), err_bpa_se.max())),
                      err_bpa_se=err_bpa_se, err_b_se=err_b_se, err_se=err_se)
        for i in range(W_MAX):
            rows.append(dict(M=M, m=i + 1, err=err[i], err_position_aware=err_pa[i],
                             p_boundary_exact=ph[i], p_boundary_sim=ph_sim[i], bound=bound[i],
                             proxy_exact=proxy[i]))
    with open(DATA_DIR / "expIII_reflected.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    make_figure(res, proxy)
    return summarize(res, proxy)


def make_figure(res, proxy) -> None:
    from matplotlib.lines import Line2D
    apply_style()
    K = INK["k"]
    m = np.arange(1, W_MAX + 1)
    fig, axes = plt.subplots(1, 3, figsize=(FULL_WIDTH, 2.6))
    axes[1].sharey(axes[0])
    axes[1].tick_params(labelleft=False)
    for ax in axes[:2]:
        ax.plot(m, proxy, color=INK["band"], lw=4.5, solid_capstyle="butt", zorder=1)
    for M, col in zip(M_LIST, SERIES):
        axes[0].plot(m, res[M]["err"], color=col, lw=1.1, zorder=3)
        axes[0].plot(m, res[M]["err_pa"], color=col, lw=0.8, ls="--", zorder=4)
        axes[1].plot(m, res[M]["err_b"], color=col, lw=1.1, zorder=3)
        axes[1].plot(m, res[M]["err_bpa"], color=col, lw=0.8, ls="--", zorder=4)
        axes[2].plot(m, res[M]["ph"], color=col, lw=1.1)
        axes[2].plot(m[4::8], res[M]["ph_sim"][4::8], "o", color=col, ms=2.5, mfc="white", mew=0.6)
    axes[0].set_title("Holding walls")
    axes[1].set_title("Bouncing walls")
    axes[0].set_ylabel("Error probability")
    for ax in axes[:2]:
        ax.set_ylim(0, 0.55)
    axes[2].set_title("Wall-encounter probability")
    axes[2].set_ylabel(r"$\mathbb{P}(H_{t,m})$")
    axes[2].set_ylim(0, 1.02)
    for ax in axes:
        ax.set_xlabel(r"Window length $m$")
        ax.set_xlim(1, W_MAX)
        ax.set_xticks([20, 40, 60, 80, 100])
    h = [Line2D([], [], color=col, lw=1.1, label=f"$M={M}$") for M, col in zip(M_LIST, SERIES)]
    h += [Line2D([], [], color=K, lw=1.1, label="raw increments"),
          Line2D([], [], color=K, lw=0.8, ls="--", label="position-aware decoder"),
          Line2D([], [], color=INK["band"], lw=4.5, solid_capstyle="butt", label="free walk"),
          Line2D([], [], color=K, ls="none", marker="o", ms=2.5, mfc="white", mew=0.6, label="simulated")]
    fig.tight_layout(rect=(0, 0.12, 1, 1), w_pad=0.8)
    fig.legend(handles=h, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.0), columnspacing=1.6,
               handletextpad=0.5, labelspacing=0.3)
    save_figure(fig, "expIII_reflected.pdf")


def summarize(res, proxy) -> dict:
    m5 = 5
    for M in M_LIST:
        assert np.max(np.abs(res[M]["ph"] - res[M]["ph_sim"])) < 0.02, "P(H) simulation disagrees"
    mstar = {M: int(np.argmin(res[M]["err"]) + 1) for M in M_LIST}
    assert res[M_LIST[0]]["bound"].min() > 1, "bound expected to be vacuous for the smallest lattice"
    assert res[M_LIST[0]]["err"][-1] < proxy[-1], "walls expected to help at long windows"
    assert all(res[M]["err"][m5 - 1] > proxy[m5 - 1] for M in M_LIST), "walls expected to hurt at m=5"
    assert all(mstar[M] >= int(np.argmin(proxy) + 1) for M in M_LIST), "optimal window should lengthen"
    pa_dev = max(float(np.max(np.abs(res[M]["err_pa"] - proxy))) for M in M_LIST)
    assert pa_dev < 0.01, "position-aware decoder should match the free walk"
    best_b = {M: float(res[M]["err_bpa"].min()) for M in M_LIST}
    pred_b = {M: float(res[M]["A_eff"].min()) for M in M_LIST}
    for M in M_LIST:
        assert res[M]["err_bpa"].min() >= proxy.min() - 0.003, "bouncing walls cannot beat the free walk"
    v = {
        "pa-maxdev": fmt(pa_dev, 3),
        "b-wall-lo": f"{round(100 * res[M_LIST[-1]]['wall_b'])}",
        "b-wall-hi": f"{round(100 * res[M_LIST[0]]['wall_b'])}",
        "b-best-lo": fmt(best_b[M_LIST[-1]], 3), "b-best-hi": fmt(best_b[M_LIST[0]], 3),
        "b-pred-lo": fmt(pred_b[M_LIST[-1]], 3), "b-pred-hi": fmt(pred_b[M_LIST[0]], 3),
        "b-raw-hi": fmt(float(res[M_LIST[0]]["err_b"].min()), 3),
        "proxy-best": fmt(float(proxy.min()), 3),
        "p": f"{P}", "delta": f"{DELTA}", "T": f"{T}", "R": f"{R}",
        "M-list": ordinal_list([str(M) for M in M_LIST]),
        "M-lo": f"{M_LIST[0]}", "M-hi": f"{M_LIST[-1]}",
        "proxy-mstar": f"{int(np.argmin(proxy) + 1)}", "m5": f"{m5}",
        "wall-lo": f"{round(100 * res[M_LIST[-1]]['wall'])}",
        "wall-hi": f"{round(100 * res[M_LIST[0]]['wall'])}",
        "ph5-lo": fmt(res[M_LIST[-1]]["ph"][m5 - 1], 2),
        "ph5-hi": fmt(res[M_LIST[0]]["ph"][m5 - 1], 2),
        "err5-proxy": fmt(proxy[m5 - 1], 3),
        "err5-lo": fmt(res[M_LIST[-1]]["err"][m5 - 1], 3),
        "err5-hi": fmt(res[M_LIST[0]]["err"][m5 - 1], 3),
        "errmax-lo": fmt(res[M_LIST[-1]]["err"][-1], 2),
        "errmax-hi": fmt(res[M_LIST[0]]["err"][-1], 2),
        "mstar-list": ordinal_list([str(mstar[M]) for M in M_LIST]),
        "minbound-lo": fmt(min(res[M]["bound"].min() for M in M_LIST), 2),
        "minbound-hi": fmt(res[M_LIST[0]]["bound"].min(), 2),
        "ph5-M20-err5": fmt(res[M_LIST[0]]["err"][m5 - 1], 3),
        "maxdev-ph": fmt(max(np.max(np.abs(res[M]["ph"] - res[M]["ph_sim"])) for M in M_LIST), 3),
        "se-max": fmt(max(res[M]["se_max"] for M in M_LIST), 4),
        "b-best-hi-se": fmt(float(res[M_LIST[0]]["err_bpa_se"][int(np.argmin(res[M_LIST[0]]["err_bpa"]))]), 4),
        "b-best-lo-se": fmt(float(res[M_LIST[-1]]["err_bpa_se"][int(np.argmin(res[M_LIST[-1]]["err_bpa"]))]), 4),
    }
    store_values("refl", v)
    return v


if __name__ == "__main__":
    print(run())
