# eval/run_bocpd.py
"""
Apply BOCPD to per-user anomaly score sequences.
Validate detected changepoints against CERT ground-truth onset dates.
"""
import sys
from pathlib import Path

# Put the repo root on sys.path so `models` imports resolve regardless of
# the directory this script is launched from (matches run_week2.py).
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from models.bocpd import bocpd
from models.encoder import LSTMAutoencoder
from models.dataset import UserBehaviorDataset
import pickle

# Collapse threshold: fraction of run-length posterior mass on short runs
# that counts as a regime reset. See bocpd() cp_prob semantics.
CP_THRESHOLD = 0.5


def get_user_daily_scores(model, features_df, scaler, user_id,
                          feature_cols, window_size=30, device='mps'):
    """
    Produce a per-day anomaly score sequence for one user.

    We slide the 30-day window across the user's full timeline and
    score each window. The score at window ending on day t is the
    anomaly score for day t. This gives a continuous daily signal
    that BOCPD can analyze for regime changes.
    """
    user_data = features_df[
        features_df['user'] == user_id
    ].sort_values('date').reset_index(drop=True)

    feats = scaler.transform(user_data[feature_cols].values).astype(np.float32)

    if len(feats) < window_size:
        return None, None

    model.eval()
    scores, dates = [], []
    with torch.no_grad():
        for start in range(len(feats) - window_size + 1):
            window = feats[start:start + window_size]
            x = torch.FloatTensor(window).unsqueeze(0).to(device)
            score = model.anomaly_score(x).item()
            scores.append(score)
            # date of the LAST day in the window
            dates.append(user_data.iloc[start + window_size - 1]['date'])

    return np.array(scores), dates


def main():
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'

    # ── Load everything ──────────────────────────────────────────
    features_df = pd.read_csv("data/processed/daily_features.csv",
                              parse_dates=['date'])
    with open("data/processed/scaler.pkl", 'rb') as f:
        scaler = pickle.load(f)
    insiders = pd.read_csv("data/raw/answers/insiders.csv")

    # insiders.csv spans every CERT release (datasets 2, 3.1, ... 4.2).
    # Filter to r4.2 only — matches CERTFeatureEngineer's release filter,
    # so the denominator is the 70 r4.2 incidents and we never score a
    # cross-release user-ID collision against the wrong onset date.
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

    # ── Run BOCPD on each malicious user ─────────────────────────
    print("="*60)
    print("BOCPD Bifurcation Detection — Validation")
    print("="*60)

    detection_latencies = []
    results = []

    for _, row in insiders.iterrows():
        user_id = row['user']
        true_start = pd.to_datetime(row['start'])

        scores, dates = get_user_daily_scores(
            model, features_df, scaler, user_id,
            feature_cols, device=device
        )
        if scores is None or len(scores) < 35:
            continue

        # Run BOCPD on the daily score sequence
        _, cp_prob = bocpd(scores, hazard_lambda=50)

        # A bifurcation is where the run-length posterior COLLAPSES onto
        # short run lengths — cp_prob rises through the threshold. We look
        # for upward crossings (low -> high), which marks the day the
        # regime reset and ignores the warm-up (where run length is
        # naturally small and cp_prob starts high).
        # WHY 0.5? More than half the posterior mass landing on short
        # run lengths is a decisive collapse, not noise.
        detected_idx = np.where((cp_prob[1:] > CP_THRESHOLD) &
                                (cp_prob[:-1] <= CP_THRESHOLD))[0] + 1

        if len(detected_idx) == 0:
            results.append((user_id, true_start.date(), None, None))
            continue

        # Use the detected changepoint closest to (but not long after)
        # the true onset — the bifurcation we care about
        detected_dates = [dates[i] for i in detected_idx]
        # latency = detected date - true start (signed, in days)
        latencies = [(pd.to_datetime(d) - true_start).days
                     for d in detected_dates]
        # pick the detection with smallest absolute latency
        best = np.argmin(np.abs(latencies))
        detected_date = detected_dates[best]
        latency = latencies[best]

        detection_latencies.append(latency)
        results.append(
            (user_id, true_start.date(),
             pd.to_datetime(detected_date).date(), latency)
        )

    # ── Report ───────────────────────────────────────────────────
    print(f"\n{'User':<10} {'True onset':<13} "
          f"{'Detected':<13} {'Latency (days)':<14}")
    print("-"*55)
    for user, true_s, det, lat in results:
        det_str = str(det) if det else "NOT DETECTED"
        lat_str = f"{lat:+d}" if lat is not None else "—"
        print(f"{user:<10} {str(true_s):<13} {det_str:<13} {lat_str:<14}")

    lats = np.array(detection_latencies)
    detected_count = len(lats)
    total = len(results)

    print("\n" + "="*55)
    print(f"Bifurcations detected:  {detected_count}/{total} "
          f"({detected_count/total*100:.0f}%)")
    if detected_count > 0:
        within_7 = np.sum(np.abs(lats) <= 7)
        print(f"Detected within ±7 days: {within_7}/{detected_count} "
              f"({within_7/detected_count*100:.0f}%)")
        print(f"Median absolute latency: "
              f"{np.median(np.abs(lats)):.1f} days")
        print(f"Mean signed latency:     {np.mean(lats):+.1f} days")
        print("  (negative = detected BEFORE labeled onset,")
        print("   which can happen as behavior ramps up pre-onset)")

    # ── Plot one example ─────────────────────────────────────────
    if results:
        example = next((r for r in results if r[3] is not None), None)
        if example:
            user_id = example[0]
            scores, dates = get_user_daily_scores(
                model, features_df, scaler, user_id,
                feature_cols, device=device
            )
            _, cp_prob = bocpd(scores, hazard_lambda=50)

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7),
                                           sharex=True)
            ax1.plot(scores, color='steelblue')
            ax1.axvline(
                [i for i, d in enumerate(dates)
                 if pd.to_datetime(d) >= pd.to_datetime(example[1])][0],
                color='red', linestyle='--', label='True onset'
            )
            ax1.set_ylabel('Anomaly score')
            ax1.set_title(f'{user_id} — anomaly scores')
            ax1.legend()

            ax2.plot(cp_prob, color='crimson')
            ax2.axhline(CP_THRESHOLD, color='gray', linestyle=':',
                        label='Detection threshold')
            ax2.set_ylabel('P(run length ≤ w)')
            ax2.set_xlabel('Day index')
            ax2.set_title('BOCPD bifurcation probability')
            ax2.legend()

            plt.tight_layout()
            plt.savefig("data/processed/bocpd_example.png", dpi=150)
            print(f"\nExample plot: data/processed/bocpd_example.png")

    print("\nBOCPD complete.")


if __name__ == '__main__':
    main()
