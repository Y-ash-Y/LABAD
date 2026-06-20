# models/bocpd.py
"""
Bayesian Online Changepoint Detection for behavioral bifurcation.

Detects the moment a user's behavioral regime changes — the
"bifurcation point" where benign behavior transitions to malicious.

Reference: Adams & MacKay (2007), "Bayesian Online Changepoint Detection"

WHY from scratch instead of a library?
1. The bayesian_changepoint_detection PyPI package is unmaintained
   and has subtle bugs in the online variant.
2. You must be able to explain every line in an interview.
   "I used a library" is a weak answer. "I implemented the
   run-length recursion with a Student-t predictive" is a strong one.
"""

import numpy as np
from scipy.stats import t as student_t


class GaussianUnknownMeanVar:
    """
    Predictive model for BOCPD: Gaussian likelihood with unknown
    mean AND variance, using a Normal-Inverse-Gamma conjugate prior.

    WHY unknown variance (not just unknown mean)?
    A user's behavioral noise level isn't fixed. Some users are
    naturally erratic, some are routine. Modeling variance as
    unknown lets BOCPD adapt to each user's baseline volatility —
    so it doesn't false-alarm on a naturally noisy user.

    The four NIG parameters per run length:
      mu0    — prior belief about the mean
      kappa0 — confidence in that mean (pseudo-observations)
      alpha0 — shape of the variance prior
      beta0  — scale of the variance prior
    """

    def __init__(self, mu0=0.0, kappa0=1.0, alpha0=1.0, beta0=1.0):
        # Store priors as the first (run length = 0) entry.
        # These arrays GROW by one element each timestep — one slot
        # per possible run length.
        self.mu0    = mu0
        self.kappa0 = kappa0
        self.alpha0 = alpha0
        self.beta0  = beta0

        self.mu    = np.array([mu0])
        self.kappa = np.array([kappa0])
        self.alpha = np.array([alpha0])
        self.beta  = np.array([beta0])

    def predictive_prob(self, x):
        """
        P(x | each run length) — the Student-t predictive.

        For a Normal-Inverse-Gamma posterior, the predictive
        distribution of a new point is Student-t with:
          df    = 2 * alpha
          loc   = mu
          scale = sqrt( beta * (kappa + 1) / (alpha * kappa) )

        Returns an array — one probability per current run length.
        """
        df    = 2 * self.alpha
        scale = np.sqrt(
            self.beta * (self.kappa + 1) / (self.alpha * self.kappa)
        )
        return student_t.pdf(x, df=df, loc=self.mu, scale=scale)

    def update(self, x):
        """
        Fold observation x into the sufficient statistics.

        WHY prepend the prior each step?
        Run length 0 (a fresh changepoint) always starts from the
        prior — it has seen zero data. So every update, we prepend
        the original prior, then update all existing runs with x.
        """
        # Standard NIG update for all existing runs
        # (applied to the slots that already held data)
        updated_mu    = (self.kappa * self.mu + x) / (self.kappa + 1)
        updated_kappa = self.kappa + 1
        updated_alpha = self.alpha + 0.5
        updated_beta  = self.beta + (
            self.kappa * (x - self.mu) ** 2
        ) / (2 * (self.kappa + 1))

        # Assemble: slot 0 is the fresh prior, slots 1..n are updated
        self.mu    = np.concatenate([[self.mu0],    updated_mu])
        self.kappa = np.concatenate([[self.kappa0], updated_kappa])
        self.alpha = np.concatenate([[self.alpha0], updated_alpha])
        self.beta  = np.concatenate([[self.beta0],  updated_beta])


def bocpd(data, hazard_lambda=50, short_window=5,
          mu0=0.0, kappa0=1.0, alpha0=1.0, beta0=1.0):
    """
    Run Bayesian Online Changepoint Detection over a 1D sequence.

    Args:
        data:           1D array — e.g. a user's daily anomaly scores
        hazard_lambda:  expected run length. Hazard = 1/lambda.
                        Larger = fewer expected changepoints.
                        WHY 50? Insiders go rogue rarely — once in
                        a ~50-day window is a reasonable prior.
        short_window:   how many of the smallest run lengths count as
                        "the regime just reset". cp_prob is the posterior
                        mass on run lengths <= short_window.

    Returns:
        R:              run-length posterior matrix (T+1, T+1)
        cp_prob:        per-timestep changepoint signal — the posterior
                        probability that the run length is SHORT, i.e.
                        P(r_t <= short_window). This is the "mass collapsed
                        toward r=0" quantity from the concept note.

    WHY not cp_prob = R[0, t]?
    R[0, t] is provably identical to the hazard at every step:
        R[0,t] = (H * S) / S = H,  where S = sum_r R[r,t-1] * pred_r
    so it carries no information about WHERE changepoints are. The
    regime change shows up instead as the whole run-length posterior
    collapsing onto small run lengths — which is exactly what
    P(r_t <= short_window) measures: it sits near 0 in a stable regime
    and jumps toward 1 the moment the run resets.
    """
    T = len(data)
    hazard = 1.0 / hazard_lambda

    # R[r, t] = P(run length = r at time t)
    R = np.zeros((T + 1, T + 1))
    R[0, 0] = 1.0  # at t=0, run length is certainly 0

    model = GaussianUnknownMeanVar(mu0, kappa0, alpha0, beta0)
    cp_prob = np.zeros(T)

    for t in range(1, T + 1):
        x = data[t - 1]

        # Step 1: predictive probability of x under each run length
        pred_probs = model.predictive_prob(x)

        # Step 2: growth — run continues (no changepoint)
        # shift mass UP one run length, weighted by (1 - hazard)
        R[1:t + 1, t] = R[0:t, t - 1] * pred_probs * (1 - hazard)

        # Step 3: changepoint — run resets to 0
        # sum all mass flowing into run length 0
        R[0, t] = np.sum(R[0:t, t - 1] * pred_probs * hazard)

        # Step 4: normalize to a valid distribution
        evidence = np.sum(R[:, t])
        if evidence > 0:
            R[:, t] /= evidence

        # Changepoint signal: posterior mass on SHORT run lengths.
        # In a stable regime the mass sits at large r, so this ~0;
        # when the regime resets the mass collapses onto small r, so
        # this jumps toward 1. (R[0, t] alone is just the hazard.)
        cp_prob[t - 1] = R[0:short_window + 1, t].sum()

        # Update sufficient statistics with x
        model.update(x)

    return R, cp_prob
