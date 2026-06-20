# eval/run_week2.py
"""
Week 2 evaluation script.
Run this after train.py completes.
"""
import sys
from pathlib import Path

# Ensure project root is on sys.path so top-level packages like
# `models` can be imported when running this script from the repo root
# or from the `eval/` directory. Compute the repo root relative to
# this file and insert it at the front of sys.path.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import pickle
import os
from anomaly_scorer import AnomalyScorer
from models.dataset import CERTFeatureEngineer, UserBehaviorDataset

def main():
    print("="*50)
    print("LABAD — Week 2: Anomaly Scoring + NP Calibration")
    print("="*50)
    
    # ── 1. Reload processed features ────────────────────────────
    # We saved these in Week 1 — no need to reprocess
    print("\nLoading processed features...")
    features_df = pd.read_csv("data/processed/daily_features.csv")
    
    with open("data/processed/scaler.pkl", 'rb') as f:
        scaler = pickle.load(f)
    
    # Recreate the same train/test split as Week 1
    # WHY same split? Consistency — the NP threshold must be
    # calibrated on the same held-out normal users the model
    # never saw during training.
    all_users       = features_df['user'].unique()
    malicious_users = features_df[
        features_df['is_malicious'] == 1
    ]['user'].unique()
    benign_users    = [u for u in all_users 
                       if u not in malicious_users]
    
    n_train       = int(0.8 * len(benign_users))
    train_users   = benign_users[:n_train]
    test_benign   = benign_users[n_train:]
    test_malicious = list(malicious_users)
    test_users    = test_benign + test_malicious
    
    # Normalize using saved scaler
    feature_cols = [c for c in features_df.columns 
                    if c not in ['user', 'date', 'is_malicious']]
    
    test_df = features_df[
        features_df['user'].isin(test_users)
    ].copy()
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])
    
    # Separate benign-only test set for NP calibration
    calib_df = test_df[test_df['user'].isin(test_benign)]
    
    # ── 2. Build datasets ────────────────────────────────────────
    # Full test set (benign + malicious) for evaluation
    test_dataset = UserBehaviorDataset(
        test_df, window_size=30, mode='test'
    )
    
    # Benign-only for calibration
    calib_dataset = UserBehaviorDataset(
        calib_df, window_size=30, mode='test'
    )
    
    # ── 3. Load model and compute scores ────────────────────────
    scorer = AnomalyScorer("checkpoints/best_model.pt")
    
    print("\nComputing anomaly scores on test set...")
    test_result = scorer.compute_scores(test_dataset)
    
    print("\nComputing anomaly scores on calibration set...")
    calib_result = scorer.compute_scores(calib_dataset)
    
    # ── 4. NP Calibration ───────────────────────────────────────
    # Calibrate on BENIGN-ONLY scores
    # These are users the model has never seen — held-out normal
    print("\n" + "="*40)
    print("Neyman-Pearson Threshold Calibration")
    print("="*40)
    
    normal_scores = calib_result['scores']
    
    # Try multiple FPR targets
    print("\nCalibrating at multiple FPR targets:")
    for target_fpr in [0.01, 0.05, 0.10]:
        tau = scorer.calibrate_threshold_np(normal_scores, target_fpr)
    
    # Lock in 1% FPR as our operating point
    scorer.calibrate_threshold_np(normal_scores, target_fpr=0.01)
    
    # ── 5. Full evaluation ───────────────────────────────────────
    print("\n" + "="*40)
    print("Evaluation Results")
    print("="*40)
    
    # Window-level evaluation
    print("\nWindow-level metrics:")
    window_results = scorer.evaluate(
        pd.DataFrame({
            'score':        test_result['scores'],
            'is_malicious': test_result['labels'],
            'user':         test_result['users'],
        }).rename(columns={'score': 'max_score'}),
        score_col='max_score'
    )
    
    # User-level evaluation (more meaningful)
    print("\nUser-level metrics (max score aggregation):")
    user_df = scorer.aggregate_user_scores(test_result, 'max')
    user_results = scorer.evaluate(user_df, score_col='max_score')
    
    print("\nUser-level metrics (top-3 mean aggregation):")
    user_results_top3 = scorer.evaluate(user_df, score_col='top3_score')
    
    # ── 6. Save results ──────────────────────────────────────────
    os.makedirs("data/processed", exist_ok=True)
    
    user_df.to_csv("data/processed/user_scores.csv", index=False)
    np.save("data/processed/threshold.npy", 
            np.array([scorer.threshold]))
    
    print(f"\nUser scores saved to data/processed/user_scores.csv")
    
    # ── 7. Plots ─────────────────────────────────────────────────
    scorer.plot_score_distributions(
        test_result,
        save_path="data/processed/anomaly_distributions.png"
    )
    
    # ── 8. Sanity check — inspect individual malicious users ────
    print("\n" + "="*40)
    print("Malicious User Inspection")
    print("="*40)
    
    malicious_in_test = user_df[user_df['is_malicious'] == 1]
    benign_in_test    = user_df[user_df['is_malicious'] == 0]
    
    print(f"\nMalicious users in test set: {len(malicious_in_test)}")
    print(f"Benign users in test set:    {len(benign_in_test)}")
    print(f"\nMalicious user scores:")
    print(malicious_in_test[
        ['user','max_score','mean_score','top3_score']
    ].sort_values('max_score', ascending=False).to_string())
    
    print(f"\nTop 5 benign users by score (false positive risk):")
    print(benign_in_test.nlargest(5, 'max_score')[
        ['user','max_score','mean_score','top3_score']
    ].to_string())
    
    print(f"\nScore gap:")
    print(f"  Avg malicious score: "
          f"{malicious_in_test['max_score'].mean():.4f}")
    print(f"  Avg benign score:    "
          f"{benign_in_test['max_score'].mean():.4f}")
    print(f"  Separation ratio:    "
          f"{malicious_in_test['max_score'].mean() / benign_in_test['max_score'].mean():.2f}x")

if __name__ == '__main__':
    main()