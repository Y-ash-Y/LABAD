# models/early_warning.py
"""
Critical Slowing Down (CSD) — early-warning signals for an APPROACHING
behavioral bifurcation.

THE PHYSICS (this is the differentiator, made rigorous):
As a dynamical system approaches a bifurcation / tipping point, it
recovers more and more slowly from small perturbations. That loss of
resilience leaves two statistical fingerprints in the system's output,
visible BEFORE the transition actually happens:

  1. Rising lag-1 autocorrelation — today looks more like yesterday
     (the system is "sticky", slow to return to baseline).
  2. Rising variance — fluctuations grow as the restoring force weakens.

This is the canonical early-warning theory from tipping-point science
(Scheffer et al., Nature 2009), used for ecosystem collapse, climate
transitions, and financial crashes.

Applied to LABAD: where BOCPD/CUSUM tell you the regime ALREADY changed,
CSD tries to flag that a user is APPROACHING a behavioral bifurcation —
potentially days BEFORE the labeled onset, as their behavior destabilizes.

WHY detrend inside each window?
A rising mean (trend) would by itself inflate variance and
autocorrelation, faking the signal. We remove a linear trend within each
sliding window so the metrics reflect genuine loss of resilience, not the
drift itself.
"""

import numpy as np
from scipy.stats import kendalltau


def _lag1_autocorr(x):
    """Lag-1 autocorrelation of a 1D array (0 if degenerate)."""
    x = x - x.mean()
    denom = np.sum(x * x)
    if denom <= 0:
        return 0.0
    return float(np.sum(x[:-1] * x[1:]) / denom)


def critical_slowing_down(data, window=30):
    """
    Rolling autocorrelation and variance over a sliding window.

    Returns two arrays (NaN until the first full window):
        ar1:  lag-1 autocorrelation in the trailing `window` days
        var:  variance in the trailing `window` days (detrended)
    """
    data = np.asarray(data, dtype=float)
    n = len(data)
    ar1 = np.full(n, np.nan)
    var = np.full(n, np.nan)

    for t in range(window, n + 1):
        w = data[t - window:t]
        # linear-detrend the window so trend doesn't fake the signal
        idx = np.arange(window)
        slope, intercept = np.polyfit(idx, w, 1)
        resid = w - (slope * idx + intercept)
        ar1[t - 1] = _lag1_autocorr(resid)
        var[t - 1] = float(np.var(resid))

    return ar1, var


def detect_early_warning(data, window=30, trend_window=20, tau_thresh=0.5):
    """
    Fire an early warning when BOTH autocorrelation and variance are
    significantly TRENDING UP together — the joint CSD signature.

    We measure "trending up" with Kendall's tau (a rank correlation with
    time) over a trailing `trend_window`. Requiring BOTH signals to rise
    sharply (tau > tau_thresh each) suppresses false alarms from one noisy
    metric alone.

    Args:
        data:          1D array — a user's daily anomaly scores
        window:        window for the rolling AR1 / variance
        trend_window:  how many recent days to assess the trend over
        tau_thresh:    minimum Kendall tau (0..1) for "rising"

    Returns:
        ar1, var:      the rolling signals (for plotting)
        warn_idx:      index of the first early warning, or None
    """
    ar1, var = critical_slowing_down(data, window=window)
    n = len(data)
    warn_idx = None

    start = window - 1 + trend_window
    for t in range(start, n):
        a = ar1[t - trend_window + 1:t + 1]
        v = var[t - trend_window + 1:t + 1]
        if np.any(np.isnan(a)) or np.any(np.isnan(v)):
            continue
        time = np.arange(trend_window)
        tau_a, _ = kendalltau(time, a)
        tau_v, _ = kendalltau(time, v)
        if (tau_a is not None and tau_v is not None
                and tau_a > tau_thresh and tau_v > tau_thresh):
            warn_idx = t
            break

    return ar1, var, warn_idx
