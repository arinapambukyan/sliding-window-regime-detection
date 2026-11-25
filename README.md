# Sliding-window detection of hidden regimes: code

Code and results for

A. Pambukyan and H. Nersisyan, *Sliding-Window Detection of Hidden Regimes under Stochastic Switching*, 2026.

Version 1.0.0. It reproduces every figure, table and number of the numerical results
in the paper (Figure 1, Section 6 and Appendix B).

## Running

From this folder:

```
python code/run_all.py              # everything, about six minutes
python code/run_all.py gauss refl   # only the named modules; other values are kept
```

Python 3.10 or later with NumPy, SciPy and Matplotlib (see `requirements.txt`; the
results in this archive were produced with Python 3.13, NumPy 2.5, SciPy 1.18 and
Matplotlib 3.11). Random seeds are fixed, so a re-run gives the same numbers.

The script writes

- `data/`: raw results as CSV files,
- `figures/`: the figures of the paper (PDF),
- `generated/`: `numbers.tex`, which defines every number quoted in the text, a LaTeX
  table, and the pgfplots data for Figure 1.

The LaTeX source of the paper reads each number through `\G{key}`, so re-running the
script and recompiling updates the paper. The outputs of version 1.0.0 are included.

## Modules

| Module | `run_all.py` key | Paper |
| --- | --- | --- |
| `exp1_gauss.py` | `gauss` | Experiment I with Gaussian residuals: exact error of uniform windows and of the Bayes filter, the half-window law and the window rule, exponential and centred windows, the Bayes smoother, error given the age of the regime, non-geometric sojourns, data for Figure 1 |
| `exp1_pm1.py` | `pm1` | Experiment I with ±1 observations (Appendix A): exact recursion checked by simulation, the half-window law, the bounds for the symmetric chain, exponential weights and the Bayes filter |
| `exp2_kregimes.py` | `kreg` | Experiment II, three regimes: exact error by a recursion over the composition of the window, the switch-cost law and its bounds, switch costs and their limits, the simulated Bayes filter, a time-reversal example |
| `exp3_reflected.py` | `refl` | Experiment III, the reflected walk with holding and bouncing walls: raw-increment and position-aware decoders, exact boundary probabilities |
| `common.py` | | plotting style, chain and sojourn simulation, exact error recursions, the grid Bayes filter, decoders, Monte Carlo summaries |

Each module checks the statements the text relies on, for example that the exact error
lies between the two sides of the half-window law in every configuration, and stops
with an error if a re-run breaks one of them.

## Methods

Exact errors of the uniform window are computed by recursions over the regime and the
composition of the window. The stationary error of the Bayes filter for two Gaussian
regimes is approximated numerically on a grid of 3000 cells (`filter_error_grid`), and
each run recomputes it on 6000 cells and reports the largest relative change.

A simulated error is the average of the error rates of independent stationary
trajectories. Its standard error is the standard deviation of those rates divided by
the square root of their number (`mean_se`), and ratios carry delta-method standard
errors (`ratio_se`). Tuning parameters, namely the effective length of exponential
weights and the half-width of centred windows, are chosen by grid search on separate
trajectories, and the errors are then estimated on new ones.

## Licence and citation

The code is released under the MIT licence (`LICENSE`). If you use it, please cite the
paper and this archive (`CITATION.cff`).
