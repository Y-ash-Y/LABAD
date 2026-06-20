# eval/run_ensemble.py
"""
Onset-detection ENSEMBLE: BOCPD + CUSUM + Critical Slowing Down.

Each detector is matched to a different failure mode:
  - BOCPD  -> sharp regime shifts        (CERT scenario 1)
  - CUSUM  -> gradual drift              (CERT scenario 2)
  - CSD    -> EARLY WARNING before onset (the physics differentiator)

Honest, CAUSAL evaluation:
  - Every detector reports its FIRST credible alarm (true online behavior),
    not the crossing cherry-picked closest to the labeled onset.
  - A detection counts as a HIT only if it lands within +/-14 days of the
    true onset; anything further is a miss, not a flattering "match".
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from models.bocpd import bocpd
from models.cusum import cusum
from models.early_warning import detect_early_warning
# reuse the exact per-user scoring + collapse threshold from run_bocpd
from run_bocpd import get_user_daily_scores, CP_THRESHOLD

BURN_IN  = 30   # days of baseline before any detector may alarm
HIT_DAYS = 14   # |latency| must be within this to count as a real detection


def _signed_latency(idx, dates, true_start):
    """detected_date - true_onset, in days (negative = before onset)."""
    if idx is None:
        return None
    return (pd.to_datetime(dates[idx]) - true_start).days


def bocpd_first_alarm(scores):
    """First UPWARD collapse crossing after burn-in (causal)."""
    _, cp = bocpd(scores, hazard_lambda=50)
    cross = np.where((cp[1:] > CP_THRESHOLD) &
                     (cp[:-1] <= CP_THRESHOLD))[0] + 1
    cross = cross[cross >= BURN_IN]
    return int(cross[0]) if len(cross) else None


def main():
    import torch
    from models.encoder import LSTMAutoencoder
    import pickle

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'

    features_df = pd.read_csv("data/processed/daily_features.csv",
                              parse_dates=['date'])
    with open("data/processed/scaler.pkl", 'rb') as f:
        scaler = pickle.load(f)
    insiders = pd.read_csv("data/raw/answers/insiders.csv")
    release = pd.to_numeric(insiders['dataset'], errors='coerce')
    insiders = insiders[np.isclose(release, 4.2)].copy()

    ckpt = torch.load("checkpoints/best_model.pt", map_location=device)
    model = LSTMAutoencoder(
        n_features=ckpt['n_features'], hidden_dim=128,
        latent_dim=64, n_layers=2, dropout=0.0
    ).to(device)
    model.load_state_dict(ckpt['model_state'])

    feature_cols = [c for c in features_df.columns
                    if c not in ['user', 'date', 'is_malicious']]

    print("=" * 78)
    print("Onset-Detection Ensemble — BOCPD + CUSUM + Critical Slowing Down")
    print("=" * 78)
    print(f"\n{'User':<9}{'Scn':<4}{'Onset':<12}"
          f"{'BOCPD':<8}{'CUSUM':<8}{'CSD lead':<9}{'Ensemble':<9}")
    print("-" * 78)

    rows = []
    for _, r in insiders.iterrows():
        user_id = r['user']
        true_start = pd.to_datetime(r['start'])
        scn = r['scenario']

        scores, dates = get_user_daily_scores(
            model, features_df, scaler, user_id, feature_cols, device=device
        )
        if scores is None or len(scores) < BURN_IN + 5:
            continue

        b_idx = bocpd_first_alarm(scores)
        _, _, c_idx = cusum(scores, burn_in=BURN_IN)
        _, _, w_idx = detect_early_warning(scores)

        b_lat = _signed_latency(b_idx, dates, true_start)
        c_lat = _signed_latency(c_idx, dates, true_start)
        w_lat = _signed_latency(w_idx, dates, true_start)
        # CSD "lead" = days BEFORE onset it warned (positive = early warning)
        csd_lead = (-w_lat) if w_lat is not None else None

        # Ensemble alert = earliest credible BOCPD/CUSUM alarm
        cand = [i for i in (b_idx, c_idx) if i is not None]
        e_idx = min(cand) if cand else None
        e_lat = _signed_latency(e_idx, dates, true_start)

        rows.append(dict(user=user_id, scn=scn, b=b_lat, c=c_lat,
                         csd=csd_lead, e=e_lat))

        def f(v, plus=True):
            if v is None:
                return "—"
            return f"{v:+d}" if plus else f"{v}"
        print(f"{user_id:<9}{str(scn):<4}{str(true_start.date()):<12}"
              f"{f(b_lat):<8}{f(c_lat):<8}{f(csd_lead):<9}{f(e_lat):<9}")

    df = pd.DataFrame(rows)
    tot = len(df)

    def summarize(col, label):
        v = df[col].dropna()
        hits = v[v.abs() <= HIT_DAYS]
        within7 = v[v.abs() <= 7]
        print(f"\n{label}")
        print(f"  detections within ±{HIT_DAYS}d: {len(hits)}/{tot} "
              f"({len(hits)/tot*100:.0f}%)   within ±7d: {len(within7)}/{tot} "
              f"({len(within7)/tot*100:.0f}%)")
        if len(hits):
            print(f"  median |latency|: {hits.abs().median():.1f}d   "
                  f"mean signed: {hits.mean():+.1f}d")

    print("\n" + "=" * 78)
    print("SUMMARY (causal, honest: |latency|>14d = miss)")
    print("=" * 78)
    summarize('b', "BOCPD (sharp shifts):")
    summarize('c', "CUSUM (gradual drift):")
    summarize('e', "ENSEMBLE (earliest of BOCPD/CUSUM):")

    # CSD: report EARLY warnings (lead > 0 = warned before onset)
    csd = df['csd'].dropna()
    early = csd[(csd > 0) & (csd <= 60)]
    print("\nCritical Slowing Down (early warning):")
    print(f"  warned BEFORE onset (1–60d lead): {len(early)}/{tot} "
          f"({len(early)/tot*100:.0f}%)")
    if len(early):
        print(f"  median lead time: {early.median():.1f}d   "
              f"max lead: {early.max():.0f}d")

    # Per-scenario ensemble detection (the headline story)
    print("\nEnsemble detection within ±7d, by CERT scenario:")
    for s, g in df.groupby('scn'):
        e = g['e'].dropna()
        hit = (e.abs() <= 7).sum()
        print(f"  scenario {s}: {hit}/{len(g)}")

    print("\nEnsemble complete.")


if __name__ == '__main__':
    main()
