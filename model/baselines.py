"""Benchmark forecasters the neural net has to beat.

- seasonal_naive: trailing mean of the last 4 same-weekday sales (the same
  brain as the status-quo par sheet, minus the pad).
- ridge: global linear regression on the exact covariates the net sees, plus
  lag features, with item one-hots. Closed-form solve.

Both emit quantiles by assuming gaussian residuals (per item, train-period)
in log1p space -- the standard shortcut, and precisely what a pinball-trained
quantile head improves on.
"""
import numpy as np
import pandas as pd

from . import features


def _norm_ppf(p):
    """Inverse standard normal CDF (Acklam's approximation)."""
    p = np.asarray(p, dtype=float)
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    x = np.empty_like(p)
    lo = p < plow
    hi = p > phigh
    mid = ~(lo | hi)
    if lo.any():
        q = np.sqrt(-2 * np.log(p[lo]))
        x[lo] = (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
                ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    if hi.any():
        q = np.sqrt(-2 * np.log(1 - p[hi]))
        x[hi] = -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
                 ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    if mid.any():
        q = p[mid] - 0.5
        r = q * q
        x[mid] = (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5])*q / \
                 (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1)
    return x


def _lag_features(df, stats):
    """log1p-z lags per item row: lag1, lag7, lag14, trailing 4-same-dow mean."""
    out = np.zeros((len(df), 4), dtype=np.float32)
    for item, grp in df.groupby("item"):
        st = stats["items"][item]
        z = (np.log1p(grp.sold.values) - st["mean"]) / st["std"]
        idx = grp.index.values
        n = len(z)
        for j, lag in enumerate((1, 7, 14)):
            v = np.zeros(n, dtype=np.float32)
            v[lag:] = z[:-lag]
            out[idx, j] = v
        tr = np.zeros(n, dtype=np.float32)
        for t in range(n):
            past = [z[t - 7 * k] for k in range(1, 5) if t - 7 * k >= 0]
            tr[t] = np.mean(past) if past else 0.0
        out[idx, 3] = tr
    return out


def naive_forecast(df, b):
    """Trailing 4 same-weekday sales mean, in units, aligned with b's rows."""
    key = pd.MultiIndex.from_arrays([b["item"], b["date"]])
    trail = {}
    for item, grp in df.groupby("item"):
        z = grp.sold.values
        dates = grp.date.values
        for t in range(len(grp)):
            past = [z[t - 7 * k] for k in range(1, 5) if t - 7 * k >= 0]
            trail[(item, dates[t])] = float(np.mean(past)) if past else float(z[max(t - 1, 0)])
    return np.array([trail[k] for k in key])


def fit_predict_ridge(b, lam=3.0):
    """Global ridge on [cov, lags, item one-hots]; returns z-space predictions."""
    n_items = len(b["items"])
    onehot = np.zeros((len(b["iidx"]), n_items), dtype=np.float32)
    onehot[np.arange(len(b["iidx"])), b["iidx"]] = 1.0
    X = np.concatenate([b["cov"], b["lags"], onehot], axis=1)
    X = np.concatenate([X, np.ones((len(X), 1), dtype=np.float32)], axis=1)
    tr = b["split"] == "train"
    keep = tr & (b["cens"] == 0)          # simple censoring handling: drop sellout rows
    A = X[keep]
    y = b["y"][keep]
    w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ y)
    return X @ w


def quantiles_from_point(pred_z, b, taus, which_resid_split="train"):
    """Gaussian quantiles around a z-space point forecast, per-item residual std."""
    mask = (b["split"] == which_resid_split) & (b["cens"] == 0)
    sig = {}
    for i, item in enumerate(b["items"]):
        m = mask & (b["iidx"] == i)
        r = b["y"][m] - pred_z[m]
        sig[item] = float(max(r.std(), 1e-3))
    zq = _norm_ppf(taus)
    sigma = np.array([sig[it] for it in b["item"]])[:, None]
    return pred_z[:, None] + sigma * zq[None, :]
