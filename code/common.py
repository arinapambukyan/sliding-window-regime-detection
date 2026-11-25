"""Shared utilities: plotting style, Markov-chain helpers, decoders, and the
store of generated values that the LaTeX source reads through \\G{key}."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import logging  # noqa: E402
logging.getLogger("fontTools").setLevel(logging.ERROR)
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
DATA_DIR = ROOT / "data"
GEN_DIR = ROOT / "generated"
for _d in (FIG_DIR, DATA_DIR, GEN_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# Plotting style
# ----------------------------------------------------------------------------
# Grayscale. Data are black or gray, and ordered parameters (noise level, lattice
# size, window length) use a gray scale from light to black. The dark red ACCENT is
# reserved for the theoretical prediction a figure tests, and is used for nothing else.
INK = {"k": "#000000", "dk": "#404040", "gr": "#808080", "lt": "#A6A6A6", "band": "#D4D4D4"}
ACCENT = "#8C1C1C"
SEQ = [INK["lt"], INK["gr"], INK["dk"], INK["k"]]
COLORS = {"black": INK["k"], "gray": INK["gr"], "navy": INK["k"], "accent": ACCENT}
SERIES = SEQ
MARKERS = ["o", "o", "o", "o"]

# Figures are drawn at their printed size (the 6.5in text width) and included at
# that width, so 8-9pt here is 8-9pt on the page, next to 10pt body text.
STYLE = {
    "font.family": "serif",
    "font.serif": ["cmr10"],
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": True,
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.titleweight": "normal",
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.facecolor": "white",
    "axes.edgecolor": "#262626",
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "axes.grid.which": "major",
    "axes.axisbelow": True,
    "grid.color": "#EBEBEB",
    "grid.linewidth": 0.5,
    "grid.linestyle": "-",
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.minor.size": 1.5,
    "ytick.minor.size": 1.5,
    "xtick.color": "#262626",
    "ytick.color": "#262626",
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "lines.linewidth": 1.1,
    "lines.markersize": 3,
    "legend.frameon": False,
    "legend.fontsize": 8,
    "legend.handlelength": 2.2,
    "legend.borderaxespad": 0.3,
    "figure.facecolor": "white",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
}
FULL_WIDTH = 6.5  # inches; all figures are included at \textwidth = 6.5in


def apply_style() -> None:
    plt.rcParams.update(STYLE)


def panel_row(n: int, *, gap: float = 0.75, left: float = 0.55, right: float = 0.08,
              top: float = 0.3, bottom: float = 0.5, width: float = FULL_WIDTH, sharey: bool = False):
    """n square panels in a row, spanning the given width."""
    side = (width - left - right - (n - 1) * gap) / n
    height = top + side + bottom
    fig = plt.figure(figsize=(width, height))
    axes = []
    for i in range(n):
        rect = [(left + i * (side + gap)) / width, bottom / height, side / width, side / height]
        axes.append(fig.add_axes(rect, sharey=axes[0] if (sharey and axes) else None))
    if sharey:
        for ax in axes[1:]:
            ax.tick_params(labelleft=False)
    return fig, axes


def panel_grid(rows: int, cols: int, *, side: float, hgap: float = 0.75, vgap: float = 0.6,
               left: float = 0.55, top: float = 0.3, bottom: float = 0.5):
    """rows x cols square panels of the given side (inches)."""
    width = left + cols * side + (cols - 1) * hgap + 0.08
    height = top + rows * side + (rows - 1) * vgap + bottom
    fig = plt.figure(figsize=(width, height))
    axes = np.empty((rows, cols), dtype=object)
    for r in range(rows):
        for c in range(cols):
            y0 = bottom + (rows - 1 - r) * (side + vgap)
            axes[r, c] = fig.add_axes([(left + c * (side + hgap)) / width, y0 / height,
                                       side / width, side / height])
    return fig, axes


def legend_below(axes, handles, ncol: int = 1, drop: float = 30, **kw):
    """Legend centred under one axis or a row of axes, a fixed distance (points) below
    the axes, so that it never touches the tick labels or the axis label."""
    from matplotlib.transforms import offset_copy
    axes = list(np.atleast_1d(axes))
    ax0, ax1 = axes[0], axes[-1]
    x0, w0 = ax0.get_position().x0, ax0.get_position().width
    xc = ((ax0.get_position().x0 + ax1.get_position().x1) / 2 - x0) / w0
    tr = offset_copy(ax0.transAxes, fig=ax0.figure, y=-drop, units="points")
    opts = dict(loc="upper center", bbox_to_anchor=(xc, 0.0), bbox_transform=tr, ncol=ncol,
                handletextpad=0.5, labelspacing=0.3, columnspacing=1.6)
    opts.update(kw)
    return ax0.legend(handles=handles, **opts)


def save_figure(fig, name: str) -> Path:
    path = FIG_DIR / name
    fig.savefig(path)
    plt.close(fig)
    return path


# ----------------------------------------------------------------------------
# Monte Carlo summaries. Each simulated error is the average of the error rates
# of R independent trajectories; its standard error is the sample standard
# deviation of those R rates divided by sqrt(R), which accounts for the
# dependence between time points within a trajectory.
# ----------------------------------------------------------------------------
def mean_se(per) -> tuple[float, float]:
    """Mean and standard error of per-trajectory error rates (last axis = trajectories)."""
    per = np.asarray(per, float)
    R = per.shape[-1]
    return per.mean(axis=-1), per.std(axis=-1, ddof=1) / math.sqrt(R)


def ratio_se(a, b) -> tuple[float, float]:
    """Ratio of means abar/bbar of paired per-trajectory rates and its delta-method
    standard error. b may be a scalar computed without simulation error."""
    a = np.asarray(a, float)
    R = a.shape[-1]
    if np.ndim(b) == 0:
        r = a.mean() / b
        return float(r), float(a.std(ddof=1) / math.sqrt(R) / b)
    b = np.asarray(b, float)
    r = a.mean() / b.mean()
    return float(r), float((a - r * b).std(ddof=1) / math.sqrt(R) / b.mean())


# ----------------------------------------------------------------------------
# Markov-chain helpers
# ----------------------------------------------------------------------------
def sym_Q(p: float) -> np.ndarray:
    return np.array([[p, 1.0 - p], [1.0 - p, p]])


def simulate_sym_chain(rng: np.random.Generator, p: float, R: int, T: int) -> np.ndarray:
    """R stationary trajectories (E_1..E_T) of the symmetric two-state chain.
    States are coded 0 (paper: regime 1, mean -Delta/2) and 1 (regime 2)."""
    flips = rng.random((R, T)) < (1.0 - p)
    flips[:, 0] = False
    start = rng.integers(0, 2, size=R)
    return ((start[:, None] + np.cumsum(flips, axis=1)) % 2).astype(np.int8)


def simulate_markov(rng: np.random.Generator, Q: np.ndarray, R: int, T: int) -> np.ndarray:
    """R stationary trajectories of a finite Markov chain with transition matrix Q."""
    Q = np.asarray(Q, float)
    w, v = np.linalg.eig(Q.T)
    pi = np.real(v[:, np.argmin(np.abs(w - 1.0))])
    pi = pi / pi.sum()
    cum = np.cumsum(Q, axis=1)
    E = np.empty((R, T), dtype=np.int8)
    E[:, 0] = np.searchsorted(np.cumsum(pi), rng.random(R))
    U = rng.random((R, T))
    for t in range(1, T):
        E[:, t] = (U[:, t, None] > cum[E[:, t - 1]]).sum(axis=1)
    return E


def simulate_alternating(rng: np.random.Generator, sampler, R: int, T: int, burn: int) -> np.ndarray:
    """R trajectories of length T of a two-regime process whose sojourns are drawn
    independently from `sampler(n)` (alternating regimes 0/1). Each trajectory is
    cut from a long path after `burn` steps, so that it is close to stationary."""
    out = np.empty((R, T), dtype=np.int8)
    for r in range(R):
        L = np.empty(0, dtype=np.int64)
        while L.sum() < burn + T + 1:
            L = np.concatenate([L, sampler(4096)])
        lab = (np.arange(len(L)) + rng.integers(0, 2)) % 2
        path = np.repeat(lab.astype(np.int8), L)
        out[r] = path[burn:burn + T]
    return out


def run_lengths(E: np.ndarray) -> np.ndarray:
    """L[t] = number of consecutive equal regimes ending at t, so that
    NoSwitch_{t,m} = {L[t] >= m}."""
    R, T = E.shape
    idx = np.broadcast_to(np.arange(T), (R, T))
    change = np.ones((R, T), dtype=bool)
    change[:, 1:] = E[:, 1:] != E[:, :-1]
    last = np.maximum.accumulate(np.where(change, idx, 0), axis=1)
    return idx - last + 1


# ----------------------------------------------------------------------------
# Decoders (regimes coded 0/1; regime 1 has the larger mean)
# ----------------------------------------------------------------------------
def nearest_mean_stats(E, Y, m_max, U, collect_split=False):
    """Nearest-mean window decoder for two regimes with means -Delta/2, +Delta/2
    (equivalently the sign of the window sum; ties broken by U < 1/2).
    Returns per-trajectory error rates for m = 1..m_max and, optionally, the
    split into single-regime (NoSwitch) and switched windows."""
    R, T = E.shape
    C = np.zeros((R, T + 1))
    C[:, 1:] = np.cumsum(Y, axis=1)
    L = run_lengths(E) if collect_split else None
    err = np.zeros((m_max, R))
    split = {k: np.zeros(m_max) for k in ("n", "n_pure", "e_pure", "e_sw")}
    for m in range(1, m_max + 1):
        S = C[:, m:] - C[:, : T - m + 1]
        Et = E[:, m - 1:]
        tie = U[:, m - 1:] < 0.5
        dec = np.where(S > 0, 1, np.where(S < 0, 0, tie.astype(np.int8)))
        wrong = dec != Et
        err[m - 1] = wrong.mean(axis=1)
        if collect_split:
            pure = L[:, m - 1:] >= m
            split["n"][m - 1] = wrong.size
            split["n_pure"][m - 1] = pure.sum()
            split["e_pure"][m - 1] = (wrong & pure).sum()
            split["e_sw"][m - 1] = (wrong & ~pure).sum()
    return err, split


def _predict(ell, p):
    lp, lq = math.log(p), math.log(1.0 - p)
    return np.logaddexp(lp + ell, lq) - np.logaddexp(lq + ell, lp)


def forward_filter_errors(E, llr, p, burn, U):
    """Bayes classifier of E_t given the whole past Y_1..Y_t (forward filter)."""
    R, T = E.shape
    ell = np.zeros(R)
    wrong = np.zeros(R)
    for t in range(T):
        if t > 0:
            ell = _predict(ell, p)
        ell = ell + llr[:, t]
        if t >= burn:
            dec = np.where(ell > 0, 1, np.where(ell < 0, 0, (U[:, t] < 0.5).astype(np.int8)))
            wrong += dec != E[:, t]
    return wrong / (T - burn)


def exact_gauss_errors(p: float, delta: float, m_max: int, sigma: float = 1.0) -> np.ndarray:
    """Exact stationary error of the uniform-window nearest-mean decoder with
    N(-+delta/2, sigma^2) observations, m = 1..m_max, by forward recursion over
    (current regime, number of regime-2 steps in the window)."""
    Q = sym_Q(p)
    P = np.zeros((2, m_max + 1))
    P[0, 0] = 0.5
    P[1, 1] = 0.5
    out = np.zeros(m_max)
    n = np.arange(m_max + 1)
    from scipy.special import ndtr
    for j in range(1, m_max + 1):
        if j > 1:
            mix = Q.T @ P
            Pn = np.zeros_like(P)
            Pn[0] = mix[0]
            Pn[1, 1:] = mix[1, :-1]
            P = Pn
        z = 0.5 * delta * (2 * n - j) / (sigma * math.sqrt(j))
        out[j - 1] = np.sum(P[0] * ndtr(z)) + np.sum(P[1] * ndtr(-z))
    return out


def exact_two_state_gauss(Q, mu, sd, m_max):
    """Exact stationary error of the uniform-window decoder (threshold at the
    midpoint of the means) for a stationary two-state chain Q and observations
    N(mu_i, sd_i^2), m = 1..m_max, by forward recursion over (current regime,
    number of regime-1 positions in the window). Returns (errors, pi)."""
    from scipy.special import ndtr
    Q = np.asarray(Q, float)
    pi1 = Q[1, 0] / (Q[0, 1] + Q[1, 0])
    pi = np.array([pi1, 1.0 - pi1])
    th = 0.5 * (mu[0] + mu[1])
    d1 = np.zeros(m_max + 2)
    d2 = np.zeros(m_max + 2)
    d1[1], d2[0] = pi[0], pi[1]
    out = np.zeros(m_max)
    for m in range(1, m_max + 1):
        c = np.arange(m + 1)
        mean = c * mu[0] + (m - c) * mu[1]
        sdv = np.sqrt(c * sd[0] ** 2 + (m - c) * sd[1] ** 2)
        z = (m * th - mean) / sdv
        out[m - 1] = np.sum(d1[:m + 1] * ndtr(-z)) + np.sum(d2[:m + 1] * ndtr(z))
        n1 = np.zeros_like(d1)
        n1[1:] = d1[:-1] * Q[0, 0] + d2[:-1] * Q[1, 0]
        n2 = d1 * Q[0, 1] + d2 * Q[1, 1]
        d1, d2 = n1, n2
    return out, pi


def symmetric_bounds(p: float, Pbar: np.ndarray, m: np.ndarray):
    """Lower and upper bounds of the corollary for the symmetric chain:
    L = Pbar + (m-1)(1-p)p^(m-2)(1/2 - Pbar),
    U = p^(m-1) Pbar + 1 - p^r + [m even](1-p)p^(m/2-1)/2, r = ceil(m/2) - 1."""
    m = np.asarray(m, float)
    r = np.ceil(m / 2) - 1
    U = p ** (m - 1) * Pbar + 1 - p ** r + np.where(m % 2 == 0, 0.5 * (1 - p) * p ** (m / 2 - 1), 0.0)
    L = Pbar + (m - 1) * (1 - p) * p ** np.maximum(m - 2, 0) * (0.5 - Pbar)
    return L, U


FILTER_GRID = 3000         # cells of the log-odds grid; checked against 2 * FILTER_GRID


def filter_error_grid(p: float, delta: float, sigma: float = 1.0, N: int = FILTER_GRID) -> float:
    """Numerical approximation of the stationary error of the Bayes forward filter
    for the symmetric chain with N(-+delta/2, sigma^2) observations. The predicted
    log-odds in favour of the current regime is a Markov chain on [-c, c],
    c = log(p/(1-p)); its kernel is discretised on N cells (midpoint rule) and the
    stationary law of the discretised chain solved for directly. The error of the
    discretisation is second order in the cell width."""
    from scipy.special import ndtr
    q = 1.0 - p
    c = math.log(p / q)
    b = np.linspace(-c, c, N + 1)
    lam = 0.5 * (b[:-1] + b[1:])
    eb = np.exp(b[1:-1])
    finv = np.empty(N + 1)
    finv[0], finv[-1] = -np.inf, np.inf
    finv[1:-1] = np.log((p * eb - q) / (p - q * eb))
    s = delta / sigma                    # log-likelihood ratio ~ N(+-s^2/2, s^2)
    mu = s * s / 2
    G = np.diff(ndtr((finv[None, :] - lam[:, None] - mu) / s), axis=1)
    K = p * G + q * G[:, ::-1]
    A = K.T - np.eye(N)
    A[-1, :] = 1.0
    rhs = np.zeros(N)
    rhs[-1] = 1.0
    pi = np.linalg.solve(A, rhs)
    return float(np.sum(pi * ndtr((-lam - mu) / s)))


def ewma_errors(E, Y, n_eff, burn, U=None) -> np.ndarray:
    """Per-trajectory error of the exponentially weighted decoder sign(S_t),
    S_t = (1-beta) sum_j beta^j Y_{t-j}, with effective length n = 2/(1-beta)."""
    from scipy.signal import lfilter
    beta = 1 - 2 / n_eff
    S = lfilter([1 - beta], [1, -beta], Y.astype(float), axis=1)
    if U is None:
        dec = (S > 0)
    else:
        dec = np.where(S > 0, 1, np.where(S < 0, 0, (U < 0.5).astype(np.int8)))
    return (dec[:, burn:] != (E[:, burn:] == 1)).mean(axis=1)


def ewma_best(E, Y, n_grid, burn, U=None):
    """Tune the effective length n over a grid on the given (tuning) trajectories.
    Returns (best n, its error, errors over the grid)."""
    errs = np.array([float(ewma_errors(E, Y, n, burn, U).mean()) for n in n_grid])
    i = int(np.argmin(errs))
    return float(n_grid[i]), float(errs[i]), errs


def centred_errors(E, Y, h, pad) -> np.ndarray:
    """Per-trajectory error of the centred uniform window Y_{t-h..t+h} (offline)."""
    R, T = E.shape
    C = np.zeros((R, T + 1))
    C[:, 1:] = np.cumsum(Y, axis=1)
    t = np.arange(pad, T - pad)
    S = C[:, t + h + 1] - C[:, t - h]
    return ((S > 0) != (E[:, t] == 1)).mean(axis=1)


def centred_best(E, Y, h_grid, pad):
    """Tune the half-width h over a grid on the given (tuning) trajectories.
    Returns (best h, its error)."""
    errs = [float(centred_errors(E, Y, h, pad).mean()) for h in h_grid]
    i = int(np.argmin(errs))
    return int(h_grid[i]), errs[i]


def smoother_error(E, llr, p, pad):
    """Per-trajectory error of the Bayes forward-backward smoother (offline benchmark)."""
    R, T = E.shape
    F = np.zeros((R, T))
    ell = np.zeros(R)
    for s in range(T):
        if s > 0:
            ell = _predict(ell, p)
        ell = ell + llr[:, s]
        F[:, s] = ell
    B = np.zeros(R)
    Sm = np.zeros((R, T))
    for s in range(T - 1, -1, -1):
        Sm[:, s] = F[:, s] + B
        B = _predict(B + llr[:, s], p)
    t = np.arange(pad, T - pad)
    return ((Sm[:, t] > 0) != (E[:, t] == 1)).mean(axis=1)


# ----------------------------------------------------------------------------
# Generated values and tables
# ----------------------------------------------------------------------------
VALUES_FILE = DATA_DIR / "generated_values.json"


def store_values(prefix: str, values: dict) -> None:
    """Merge values (already formatted as strings) into the shared store."""
    store = json.loads(VALUES_FILE.read_text()) if VALUES_FILE.exists() else {}
    for k, v in values.items():
        store[f"{prefix}-{k}"] = v
    VALUES_FILE.write_text(json.dumps(store, indent=1, sort_keys=True))


def write_tex(name: str, content: str) -> None:
    (GEN_DIR / name).write_text(content)


def fmt(x: float, nd: int) -> str:
    return f"{x:.{nd}f}"


def ordinal_list(items):
    items = list(items)
    return ", ".join(items[:-1]) + " and " + items[-1] if len(items) > 1 else items[0]
