# models/cusum.py
"""
CUSUM (Cumulative Sum) change detection — the gradual-drift complement
to BOCPD.

WHY add CUSUM when we already have BOCPD?
BOCPD detects SHARP regime changes — its run-length posterior collapses
when a single point is wildly inconsistent with the current regime. That
is exactly why it nails CERT scenario 1 (abrupt after-hours/USB/exfil)
and misses scenario 2 (a slow ramp of job-site browsing): a gradual
drift never produces a sharp collapse.

CUSUM is the opposite tool. It ACCUMULATES small, persistent deviations
from the in-control mean. A drift that is individually invisible day to
day eventually pushes the cumulative sum over the threshold. So BOCPD and
CUSUM together cover both failure modes — sharp vs gradual.

Reference: Page (1954), "Continuous Inspection Schemes".
"""

import numpy as np


def cusum(data, burn_in=30, k_sigma=0.5, h_sigma=5.0):
    """
    One-sided (upper) CUSUM for an UPWARD shift in mean.

    We only care about upward shifts: anomaly scores RISING is the
    suspicious direction. A drop back toward normal is not a threat.

    The recursion:
        S_t = max(0, S_{t-1} + (x_t - mu0 - k))
    fires an alarm when S_t > h, then resets.

    Args:
        data:     1D array — a user's daily anomaly scores
        burn_in:  days used to estimate the in-control baseline
                  (mu0, sigma). Assumed benign — the model is trained
                  on normal behavior, so a user's first weeks are their
                  baseline. No alarm is raised inside the burn-in.
        k_sigma:  slack / allowance as a multiple of sigma. The classic
                  choice 0.5*sigma makes CUSUM optimal for detecting a
                  ~1-sigma shift: it ignores noise below k and only
                  accumulates genuine excess.
        h_sigma:  decision threshold as a multiple of sigma. Larger =
                  fewer false alarms, slower detection. 4-5 is standard.

    Returns:
        alarms:       bool array — True on days the statistic crossed h
        S:            the CUSUM statistic over time
        first_alarm:  index of the first alarm after burn_in, or None
    """
    data = np.asarray(data, dtype=float)
    n = len(data)

    if n <= burn_in:
        return np.zeros(n, dtype=bool), np.zeros(n), None

    # In-control baseline from the burn-in window
    base  = data[:burn_in]
    mu0   = base.mean()
    sigma = base.std(ddof=1)
    if sigma <= 0:
        sigma = 1e-9  # degenerate flat baseline — avoid divide-by-zero

    k = k_sigma * sigma   # allowance (raw units)
    h = h_sigma * sigma   # threshold  (raw units)

    S = np.zeros(n)
    alarms = np.zeros(n, dtype=bool)
    s = 0.0
    first_alarm = None

    for t in range(n):
        # accumulate excess above (mu0 + k); floor at 0 so quiet periods
        # don't bank "credit" that would delay a later detection
        s = max(0.0, s + (data[t] - mu0 - k))
        S[t] = s
        if s > h:
            alarms[t] = True
            if first_alarm is None and t >= burn_in:
                first_alarm = t
            s = 0.0  # reset after signalling, then keep watching

    return alarms, S, first_alarm
