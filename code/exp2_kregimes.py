"""Experiment II: more than two regimes.

Three regimes with Gaussian residuals (sigma = 1). Three chains have the same
switch rate: a birth-death chain (switches between neighbouring means only), a
chain that switches uniformly to the other regimes (a third of the switches jump
over the middle regime) and the non-reversible cycle 1 -> 2 -> 3 -> 1. For each we
compute

(a) the exact stationary error of the uniform nearest-mean window for every window
    length, by a backward recursion over the current regime, the earliest regime in
    the window and the composition of the window;
(b) the switch-cost law A_K(m), its two-sided bounds, the pairing identity and the
    geometric switch costs c_ji = 1 - h_ij / |mu_i - mu_j|;
(c) the optimal error against its asymptotic form 8 cbar kappa log(1/kappa) and the
    window rules;
(d) the Bayes forward filter (simulated) and the ratio of the window error to it;
(e) a time-reversal example: the two directions of a cycle on unequally spaced means."""
from __future__ import annotations

import csv
import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.optimize import brentq
from scipy.special import ndtr
from scipy.stats import norm

from common import (COLORS, DATA_DIR, FULL_WIDTH, ACCENT, INK, MARKERS, apply_style, fmt, legend_below, mean_se,
                    panel_row, save_figure,
                    simulate_markov, store_values)

MU = np.array([-1.0, 0.0, 1.0])              # equally spaced means, Delta = 1
MU_UNEQ = np.array([-1.0, 0.0, 2.0])         # unequally spaced means (time reversal)
SIGMA = 1.0
KAPPAS = [0.03, 0.01, 3e-3, 1e-3, 3e-4, 1e-4]
FIG_KAPPA = 0.01
SIM_KAPPAS = [0.03, 0.01, 3e-3, 1e-3]
R_SIM, T_SIM, BURN = 40, 250_000, 500
M_MAX = 160
CHAINS = ["bd", "uni", "cyc"]
NAMES = {"bd": "neighbouring switches (birth-death)", "uni": "uniform switching",
         "cyc": r"cycle $1\to2\to3\to1$"}
SEED = 52411


# ----------------------------------------------------------------------------
# Chains, decision regions and exact errors
# ----------------------------------------------------------------------------
def chain(kind: str, sbar: float) -> np.ndarray:
    """Three-state chains with stationary law (1/3, 1/3, 1/3) and switch rate sbar."""
    if kind == "bd":
        q = 0.75 * sbar
        return np.array([[1 - q, q, 0.0], [q, 1 - 2 * q, q], [0.0, q, 1 - q]])
    if kind == "uni":
        e = sbar
        return np.array([[1 - e, e / 2, e / 2], [e / 2, 1 - e, e / 2], [e / 2, e / 2, 1 - e]])
    if kind == "cyc":
        e = sbar
        return np.array([[1 - e, e, 0.0], [0.0, 1 - e, e], [e, 0.0, 1 - e]])
    if kind == "rev":
        return chain("cyc", sbar).T.copy()
    raise ValueError(kind)


def stationary(Q: np.ndarray) -> np.ndarray:
    w, v = np.linalg.eig(Q.T)
    pi = np.real(v[:, np.argmin(np.abs(w - 1.0))])
    return pi / pi.sum()


def regions(mu: np.ndarray):
    th = 0.5 * (mu[:-1] + mu[1:])
    return np.concatenate([[-np.inf], th]), np.concatenate([th, [np.inf]])


def exact_errors(Q: np.ndarray, mu: np.ndarray, sigma: float, m_max: int) -> np.ndarray:
    """Exact stationary error of the uniform nearest-mean window, m = 1..m_max, for a
    stationary three-state chain. The window is built backwards from the current time
    with the reversed chain; the state is (current regime, earliest regime in the
    window, number of observations from regimes 1 and 2)."""
    K = 3
    pi = stationary(Q)
    Qr = (Q.T * pi[None, :]) / pi[:, None]          # Qr[a, b] = P(E_{s-1} = b | E_s = a)
    lo, hi = regions(mu)
    P = np.zeros((K, K, m_max + 2, m_max + 2))
    for e in range(K):
        n = [0, 0]
        if e < 2:
            n[e] = 1
        P[e, e, n[0], n[1]] = pi[e]
    n0 = np.arange(m_max + 2)[:, None]
    n1 = np.arange(m_max + 2)[None, :]
    out = np.zeros(m_max)
    for r in range(1, m_max + 1):
        n2 = r - n0 - n1
        valid = n2 >= 0
        mean = (n0 * mu[0] + n1 * mu[1] + np.where(valid, n2, 0) * mu[2]) / r
        sd = sigma / math.sqrt(r)
        Pe = P.sum(axis=1)
        err = 0.0
        for e in range(K):
            inside = ndtr((hi[e] - mean) / sd) - ndtr((lo[e] - mean) / sd)
            err += float(np.sum(np.where(valid, Pe[e] * (1.0 - inside), 0.0)))
        out[r - 1] = err
        mix = np.einsum("eaxy,ab->ebxy", P, Qr)
        P = np.zeros_like(P)
        P[:, 0, 1:, :] = mix[:, 0, :-1, :]
        P[:, 1, :, 1:] = mix[:, 1, :, :-1]
        P[:, 2] = mix[:, 2]
    return out


def single_errors(mu: np.ndarray, sigma: float, m: int) -> np.ndarray:
    lo, hi = regions(mu)
    sd = sigma / math.sqrt(m)
    return np.array([1.0 - (ndtr((hi[i] - mu[i]) / sd) - ndtr((lo[i] - mu[i]) / sd))
                     for i in range(len(mu))])


def switch_profiles(mu: np.ndarray, sigma: float, m: int):
    """e[j, i, k-1] = error when i is current and the k most recent of m observations come
    from i, the others from j; chi[i, j, k-1] = probability of returning a third regime."""
    K = len(mu)
    lo, hi = regions(mu)
    k = np.arange(1, m)
    sd = sigma / math.sqrt(m)
    e = np.zeros((K, K, m - 1))
    chi = np.zeros((K, K, m - 1))
    for i in range(K):
        for j in range(K):
            if i == j:
                continue
            nu = (k * mu[i] + (m - k) * mu[j]) / m
            pin_i = ndtr((hi[i] - nu) / sd) - ndtr((lo[i] - nu) / sd)
            pin_j = ndtr((hi[j] - nu) / sd) - ndtr((lo[j] - nu) / sd)
            e[j, i] = 1.0 - pin_i
            chi[i, j] = 1.0 - pin_i - pin_j
    return e, chi


def limit_costs(mu: np.ndarray) -> np.ndarray:
    """c[j, i] = 1 - h_ij / |mu_i - mu_j|, h_ij = distance from mu_i to the end point of its
    decision region facing mu_j."""
    K = len(mu)
    lo, hi = regions(mu)
    c = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            if i != j:
                h = (mu[i] - lo[i]) if mu[j] < mu[i] else (hi[i] - mu[i])
                c[j, i] = 1.0 - h / abs(mu[i] - mu[j])
    return c


def law(Q: np.ndarray, mu: np.ndarray, sigma: float, m_max: int):
    """A_K(m), the two-regime law Pbar + sbar (m-1)/2, the reversible form of A_K, and the
    geometric law Pbar + sum_ji sbar_ji (m c_ji - 1/2), for m = 1..m_max."""
    K = len(mu)
    pi = stationary(Q)
    flows = pi[:, None] * Q
    np.fill_diagonal(flows, 0.0)
    sbar = flows.sum()
    cl = limit_costs(mu)
    A = np.zeros(m_max)
    A2 = np.zeros(m_max)
    Arev = np.zeros(m_max)
    Ageo = np.zeros(m_max)
    pair_dev = 0.0
    for m in range(1, m_max + 1):
        Pbar = float(pi @ single_errors(mu, sigma, m))
        A2[m - 1] = Pbar + sbar * (m - 1) / 2
        Ageo[m - 1] = Pbar + sum(flows[j, i] * (m * cl[j, i] - 0.5)
                                 for i in range(K) for j in range(K) if i != j)
        if m == 1:
            A[0] = Arev[0] = Pbar
            continue
        e, chi = switch_profiles(mu, sigma, m)
        S = e.sum(axis=2)
        A[m - 1] = Pbar + float(np.sum(flows * S))
        Arev[m - 1] = A2[m - 1] + sum(flows[i, j] * chi[i, j].sum()
                                      for i in range(K) for j in range(K) if i < j)
        for i in range(K):
            for j in range(K):
                if i != j:
                    pair_dev = max(pair_dev, abs(S[j, i] + S[i, j] - (m - 1) - chi[i, j].sum()))
    return dict(A=A, A2=A2, Arev=Arev, Ageo=Ageo, pi=pi, flows=flows, sbar=sbar,
                pmin=float(Q.diagonal().min()), pair_dev=pair_dev, cost=cl)


def expansion_error(mu: np.ndarray, sigma: float, m: int) -> float:
    """max over pairs of |sum_k e_ji(k) - (m c_ji - 1/2)|, the Gaussian expansion of the
    proposition on the geometry of a switch."""
    cl = limit_costs(mu)
    e, _ = switch_profiles(mu, sigma, m)
    S = e.sum(axis=2)
    K = len(mu)
    return max(abs(S[j, i] - (m * cl[j, i] - 0.5)) for i in range(K) for j in range(K) if i != j)


def sci(x: float) -> str:
    """x in LaTeX scientific notation with two significant digits, e.g. 1.1\\cdot10^{-3}."""
    if x <= 0:
        return "0"
    e = int(math.floor(math.log10(x)))
    d = round(x / 10**e, 1)
    if d >= 10:
        d, e = round(d / 10, 1), e + 1
    return f"{d:.1f}\\cdot10^{{{e}}}"


def kz_prediction(Q: np.ndarray, mu: np.ndarray, sigma: float) -> float:
    """Khasminskii-Zeitouni asymptotic error of the Bayes filter for Gaussian emissions:
    sum_{i != j} sbar_ji log(1/sbar) / KL(F_i || F_j), KL = (mu_i - mu_j)^2 / (2 sigma^2)."""
    pi = stationary(Q)
    flows = pi[:, None] * Q
    np.fill_diagonal(flows, 0.0)
    sbar = flows.sum()
    K = len(mu)
    tot = sum(flows[j, i] * 2 * sigma**2 / (mu[i] - mu[j]) ** 2
              for i in range(K) for j in range(K) if i != j)
    return float(tot * math.log(1 / sbar))


def mean_cost(flows: np.ndarray, cl: np.ndarray) -> float:
    return float(np.sum(flows * cl) / flows.sum())


def rule_window(cbar: float, kappa: float, nbar: float, delta: float = 1.0) -> float:
    """Continuous minimiser of nbar Phi(-sqrt(m) Delta/2) + m cbar sbar (sigma = 1):
    m = 4 z^2 / Delta^2 with phi(z)/z = 8 cbar kappa / nbar."""
    z = brentq(lambda z: norm.pdf(z) / z - 8 * cbar * kappa / nbar, 1e-3, 40)
    return 4 * z * z / delta**2


# ----------------------------------------------------------------------------
# Simulation: Bayes forward filter and the window, for cross-checks
# ----------------------------------------------------------------------------
def filter_and_window(rng, Q, mu, sigma, m_check):
    E = simulate_markov(rng, Q, R_SIM, T_SIM)
    Y = mu[E] + sigma * rng.standard_normal(E.shape)
    K = len(mu)
    alpha = np.full((R_SIM, K), 1.0 / K)
    wrong = np.zeros(R_SIM)
    for t in range(T_SIM):
        alpha = (alpha @ Q) * np.exp(-0.5 * ((Y[:, t, None] - mu[None, :]) / sigma) ** 2)
        alpha /= alpha.sum(axis=1, keepdims=True)
        if t >= BURN:
            wrong += np.argmax(alpha, axis=1) != E[:, t]
    filt, filt_se = mean_se(wrong / (T_SIM - BURN))
    th = 0.5 * (mu[:-1] + mu[1:])
    C = np.zeros((R_SIM, T_SIM + 1))
    C[:, 1:] = np.cumsum(Y, axis=1)
    avg = (C[:, m_check:] - C[:, :T_SIM - m_check + 1]) / m_check
    dec = np.searchsorted(th, avg)
    per = (dec != E[:, m_check - 1:]).mean(axis=1)
    return float(filt), float(filt_se), float(per.mean()), float(per.std(ddof=1) / math.sqrt(R_SIM))


# ----------------------------------------------------------------------------
def run() -> dict:
    rng = np.random.default_rng(SEED)
    cl = limit_costs(MU)
    nbar = float(np.mean([1, 2, 1]))                 # mean number of end points, pi uniform
    m = np.arange(1, M_MAX + 1)
    rows, curves = [], {}
    for kind in CHAINS:
        for kap in KAPPAS:
            sbar = kap * SIGMA**2                    # Delta = 1
            Q = chain(kind, sbar)
            ex = exact_errors(Q, MU, SIGMA, M_MAX)
            L = law(Q, MU, SIGMA, M_MAX)
            A, pmin = L["A"], L["pmin"]
            lower = pmin ** (m - 1) * A
            upper = A + (m - 1) * (m - 2) / 2 * (1 - pmin) * L["sbar"]
            assert np.all(ex >= lower - 1e-12) and np.all(ex <= upper + 1e-12), f"switch-cost law ({kind}, {kap})"
            assert L["pair_dev"] < 1e-9, "pairing identity"
            if kind in ("bd", "uni"):
                assert np.max(np.abs(L["Arev"] - A)) < 1e-10, "reversible form of A_K"
            i_star = int(np.argmin(ex))
            assert i_star < M_MAX - 5, "optimum too close to the largest window"
            cbar = mean_cost(L["flows"], cl)
            iA = int(np.argmin(A))
            i2 = int(np.argmin(L["A2"]))
            mrule = int(round(rule_window(cbar, kap, nbar)))
            e_star, _ = switch_profiles(MU, SIGMA, i_star + 1)
            cm = e_star.mean(axis=2)
            rows.append(dict(kind=kind, kappa=kap, sbar=L["sbar"], pmin=pmin, cbar=cbar,
                             mstar=i_star + 1, Pstar=float(ex[i_star]), A_at=float(A[i_star]),
                             mA=iA + 1, PA=float(ex[iA]), m2=i2 + 1, P2=float(ex[i2]),
                             mrule=mrule, Prule=float(ex[min(mrule, M_MAX) - 1]),
                             asym=8 * cbar * kap * math.log(1 / kap),
                             c_adj=float(cm[0, 1]), c_jump=float(cm[0, 2]),
                             gap=1 - pmin ** i_star))
            if kap == FIG_KAPPA:
                curves[kind] = dict(ex=ex, A=A, lower=lower, upper=upper, A2=L["A2"])
    # Bayes filter (simulated) and the ratio of the window error to it
    for r in rows:
        if r["kappa"] in SIM_KAPPAS:
            filt, filt_se, win, se = filter_and_window(rng, chain(r["kind"], r["sbar"]), MU, SIGMA, r["mstar"])
            assert abs(win - r["Pstar"]) < max(4 * se, 0.003), "exact window error disagrees with simulation"
            kz = kz_prediction(chain(r["kind"], r["sbar"]), MU, SIGMA)
            r.update(filt=filt, filt_se=filt_se, win_sim=win, win_se=se, winratio=r["Pstar"] / filt,
                     winratio_se=r["Pstar"] * filt_se / filt**2, kz=kz, kz_ratio=filt / kz, kz_ratio_se=filt_se / kz)
    # time reversal on unequally spaced means
    rev = {}
    for kind in ("cyc", "rev"):
        Q = chain(kind, FIG_KAPPA)
        ex = exact_errors(Q, MU_UNEQ, SIGMA, M_MAX)
        L = law(Q, MU_UNEQ, SIGMA, M_MAX)
        lower = L["pmin"] ** (m - 1) * L["A"]
        upper = L["A"] + (m - 1) * (m - 2) / 2 * (1 - L["pmin"]) * L["sbar"]
        assert np.all(ex >= lower - 1e-12) and np.all(ex <= upper + 1e-12), "switch-cost law (unequal means)"
        rev[kind] = dict(Pstar=float(ex.min()), mstar=int(np.argmin(ex)) + 1,
                         cbar=mean_cost(L["flows"], limit_costs(MU_UNEQ)))
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(DATA_DIR / "expII_kregimes.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(rows)
    make_figure(curves, rows)
    em = {mm: expansion_error(MU, SIGMA, mm) for mm in (20, 40, 80)}
    assert em[20] > em[40] > em[80] and em[40] < 2e-3, "Gaussian expansion of the switch cost"
    em.update({("uneq", mm): expansion_error(MU_UNEQ, SIGMA, mm) for mm in (40, 80)})
    assert em[("uneq", 80)] < 1e-3, "Gaussian expansion of the switch cost (unequal means)"
    return summarize(rows, rev, em, cl)


def make_figure(curves, rows) -> None:
    from matplotlib.lines import Line2D
    apply_style()
    K, G = INK["k"], INK["gr"]
    fig, axes = panel_row(3, gap=0.72)
    m = np.arange(1, M_MAX + 1)
    show = 50
    ax = axes[0]
    for kind, col, ls in (("bd", K, "-"), ("uni", G, "-.")):
        c = curves[kind]
        ax.plot(m[:show], c["A"][:show], color=ACCENT, lw=1.1, ls="--", zorder=4)
        ax.plot(m[:show], c["ex"][:show], color=col, lw=1.1, ls=ls, zorder=3)
    ax.set_title(r"Error, $\kappa=0.01$")
    ax.set_xlabel(r"Window length $m$")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True, steps=[1, 2, 5, 10]))
    ax.set_ylabel("Error probability")
    ax.set_ylim(0.05, 0.3)
    ax.set_xlim(1, show)
    h0 = [Line2D([], [], color=K, lw=1.1, label="neighbours"),
          Line2D([], [], color=G, lw=1.1, ls="-.", label="uniform switching"),
          Line2D([], [], color=ACCENT, lw=1.1, ls="--", label=r"$A_K(m)$")]
    ax = axes[1]
    mm = np.arange(2, 81)
    prof = [switch_profiles(MU, SIGMA, int(k))[0].mean(axis=2) for k in mm]
    cl = limit_costs(MU)
    for (j, i), col, ls, lw, z in (((0, 1), INK["band"], "-", 4.0, 1), ((1, 0), K, "-", 1.1, 3), ((0, 2), K, "--", 1.1, 3)):
        ax.plot(mm, [pr[j, i] for pr in prof], color=col, lw=lw, ls=ls, solid_capstyle="butt", zorder=z)
        ax.plot(mm, (mm * cl[j, i] - 0.5) / (mm - 1), color=ACCENT, lw=1.5, ls=":", zorder=4)
    ax.set_title("Cost of a switch")
    ax.set_xlabel(r"Window length $m$")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True, steps=[1, 2, 5, 10]))
    ax.set_ylabel(r"$c_{ji}(m)$")
    ax.set_ylim(0.46, 0.80)
    ax.set_xlim(2, 80)
    ax.set_yticks([0.5, 0.6, 0.7, 0.75])
    h1 = [Line2D([], [], color=K, lw=1.1, label="into an end regime"),
          Line2D([], [], color=INK["band"], lw=4.0, solid_capstyle="butt", label="into the middle"),
          Line2D([], [], color=K, lw=1.1, ls="--", label="over the middle"),
          Line2D([], [], color=ACCENT, lw=1.5, ls=":", label="expansion")]
    ax = axes[2]
    for kind, col, ls, mk, mfc in (("bd", K, "-", "o", K), ("uni", G, "-.", "o", "white"), ("cyc", K, ":", "^", K)):
        rr = [r for r in rows if r["kind"] == kind and "winratio" in r]
        ax.plot([r["kappa"] for r in rr], [r["winratio"] for r in rr], color=col, ls=ls, marker=mk, mfc=mfc, mew=0.8)
    for lev, lab in ((2.0, r"$2$"), (28 / 9, r"$28/9$")):
        ax.axhline(lev, color=ACCENT, lw=1.5, ls=":")
        ax.text(0.028, lev + 0.05, lab, color=ACCENT, fontsize=8, va="bottom", ha="left")
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_ylim(1.0, 4.3)
    ax.set_xlabel(r"$\kappa=\bar s\,\sigma^2/\Delta^2$")
    ax.set_ylabel("Ratio to Bayes filter")
    ax.set_title("Window and Bayes filter")
    h2 = [Line2D([], [], color=K, marker="o", label="neighbours"),
          Line2D([], [], color=G, ls="-.", marker="o", mfc="white", mew=0.8, label="uniform switching"),
          Line2D([], [], color=K, ls=":", marker="^", label="cycle"),
          Line2D([], [], color=ACCENT, lw=1.5, ls=":", label="limits")]
    for ax, h in zip(axes, (h0, h1, h2)):
        legend_below(ax, h)
    save_figure(fig, "expII_kregimes.pdf")


def summarize(rows, rev, em, cl) -> dict:
    at = {r["kind"]: r for r in rows if r["kappa"] == FIG_KAPPA}
    small = [r for r in rows if r["kappa"] <= 0.03]
    k4 = [r for r in rows if r["kappa"] == 1e-4]
    k3 = {r["kind"]: r for r in rows if r["kappa"] == 1e-3}
    two_law_excess = [r["P2"] / r["Pstar"] - 1 for r in rows]
    v = {
        "kappa": "0.01", "kappa-lo": "10^{-4}", "kappa-hi": "0.03",
        "nconfig": f"{len(rows)}", "Mmax": f"{M_MAX}",
        "bd-pstar": fmt(at["bd"]["Pstar"], 3), "uni-pstar": fmt(at["uni"]["Pstar"], 3),
        "cyc-pstar": fmt(at["cyc"]["Pstar"], 3),
        "bd-mstar": f"{at['bd']['mstar']}", "uni-mstar": f"{at['uni']['mstar']}",
        "jump-excess": fmt(100 * (at["uni"]["Pstar"] / at["bd"]["Pstar"] - 1), 0),
        "cbar-uni": "7/12",
        "cost-adj-max": fmt(max(r["c_adj"] for r in small), 3),
        "cost-jump-lo": fmt(min(r["c_jump"] for r in small), 3),
        "cost-jump-hi": fmt(max(r["c_jump"] for r in small), 3),
        "rule-A-excess": fmt(100 * max(r["PA"] / r["Pstar"] - 1 for r in rows), 1),
        "rule-geo-excess": fmt(100 * max(r["Prule"] / r["Pstar"] - 1 for r in rows), 1),
        "two-law-excess-hi": fmt(100 * max(two_law_excess), 1),
        "asym-lo": fmt(min(r["Pstar"] / r["asym"] for r in k4), 2),
        "asym-hi": fmt(max(r["Pstar"] / r["asym"] for r in k4), 2),
        "gap-max": fmt(100 * max(r["gap"] for r in rows if r["kappa"] <= 1e-3), 1),
        "winratio-bd-k3": fmt(k3["bd"]["winratio"], 2), "winratio-uni-k3": fmt(k3["uni"]["winratio"], 2),
        "winratio-cyc-k3": fmt(k3["cyc"]["winratio"], 2),
        "filt-bd-k3": fmt(k3["bd"]["filt"], 4), "filt-uni-k3": fmt(k3["uni"]["filt"], 4),
        "win-bd-k3": fmt(k3["bd"]["Pstar"], 4), "win-uni-k3": fmt(k3["uni"]["Pstar"], 4),
        "rev-fwd": fmt(rev["cyc"]["Pstar"], 4), "rev-rev": fmt(rev["rev"]["Pstar"], 4),
        "rev-ratio": fmt(rev["cyc"]["Pstar"] / rev["rev"]["Pstar"], 3),
        "rev-limit": fmt(rev["cyc"]["cbar"] / rev["rev"]["cbar"], 2),
        "Rsim": f"{R_SIM}", "Tsim": f"{T_SIM}",
        "em-20": sci(em[20]), "em-40": sci(em[40]), "em-80": sci(em[80]),
    }
    winratios_bd = [r["winratio"] for r in rows if r["kind"] == "bd" and "winratio" in r]
    winratios_j = [r["winratio"] for r in rows if r["kind"] != "bd" and "winratio" in r]
    kz3 = [r["kz_ratio"] for r in rows if r["kappa"] == 1e-3 and "kz_ratio" in r]
    v.update({"kz-lo": fmt(min(kz3), 2), "kz-hi": fmt(max(kz3), 2),
              "winratio-se-max": fmt(max(r["winratio_se"] for r in rows if "winratio_se" in r), 2),
              "kz-se-max": fmt(max(r["kz_ratio_se"] for r in rows if "kz_ratio_se" in r), 2),
              "filt-relse-max": fmt(100 * max(r["filt_se"] / r["filt"] for r in rows if "filt_se" in r), 1)})
    v.update({"winratio-bd-lo": fmt(min(winratios_bd), 2), "winratio-bd-hi": fmt(max(winratios_bd), 2),
              "winratio-j-lo": fmt(min(winratios_j), 2), "winratio-j-hi": fmt(max(winratios_j), 2)})
    for kap in sorted({r["kappa"] for r in rows if "winratio" in r}):
        b = next(r for r in rows if r["kind"] == "bd" and r["kappa"] == kap)
        u = next(r for r in rows if r["kind"] == "uni" and r["kappa"] == kap)
        assert u["winratio"] > b["winratio"], "jumps should raise the ratio of the window error to the filter error"
    store_values("kreg", v)
    return v


if __name__ == "__main__":
    import json
    print(json.dumps(run(), indent=1))
