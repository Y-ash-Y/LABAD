# eval/anomaly_scorer.py
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score, roc_curve, 
    precision_recall_curve, average_precision_score,
    confusion_matrix
)
import seaborn as sns
from models.encoder import LSTMAutoencoder
from models.dataset import CERTFeatureEngineer, UserBehaviorDataset
import pickle
import os

class AnomalyScorer:
    """
    Converts LSTM reconstruction errors into calibrated anomaly scores.
    
    The core insight: an autoencoder trained only on normal behavior
    learns the manifold of normal sequences. Anomalous sequences
    lie OFF this manifold, so reconstruction error is high.
    
    Week 2 has three jobs:
    1. Compute per-window and per-user anomaly scores
    2. Understand the score distributions (normal vs malicious)
    3. Calibrate threshold via Neyman-Pearson to guarantee FPR ≤ α
    """
    
    def __init__(self, model_path: str, device: str = None):
        self.device = device or (
            'mps' if torch.backends.mps.is_available() else 'cpu'
        )
        checkpoint = torch.load(model_path, map_location=self.device)
        
        self.model = LSTMAutoencoder(
            n_features=checkpoint['n_features'],
            hidden_dim=128,
            latent_dim=64,
            n_layers=2,
            dropout=0.0,   # dropout OFF at inference — deterministic scores
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state'])
        self.model.eval()
        
        self.threshold = None       # set by NP calibration
        self.normal_scores = None   # stored for calibration
        
        print(f"Model loaded from {model_path}")
        print(f"Device: {self.device}")
    
    def compute_scores(
        self, 
        dataset: UserBehaviorDataset,
        batch_size: int = 64
    ) -> dict:
        """
        Run the full dataset through the model and collect:
        - Per-window reconstruction error (anomaly score)
        - Ground truth labels
        - User IDs for per-user aggregation
        
        WHY no gradient computation here?
        torch.no_grad() reduces memory by ~50% and speeds up
        inference ~30% by skipping gradient bookkeeping.
        We're not updating weights, so gradients are useless.
        """
        loader = DataLoader(
            dataset, batch_size=batch_size, 
            shuffle=False   # CRITICAL: never shuffle eval
        )
        
        all_scores = []
        all_labels = []
        
        with torch.no_grad():
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                scores  = self.model.anomaly_score(batch_x)
                
                all_scores.extend(scores.cpu().numpy())
                all_labels.extend(batch_y.squeeze().numpy())
        
        scores = np.array(all_scores)
        labels = np.array(all_labels)
        users  = np.array(dataset.users)
        
        return {
            'scores': scores,   # per-window reconstruction error
            'labels': labels,   # 0 = benign window, 1 = malicious window
            'users':  users,
        }
    
    def calibrate_threshold_np(
        self, 
        normal_scores: np.ndarray,
        target_fpr: float = 0.01
    ) -> float:
        """
        Neyman-Pearson threshold calibration.
        
        The math is simple but the reasoning is important:
        
        We want P(score > τ | benign) ≤ α
        
        Empirically, this means: find the (1-α) quantile of the
        normal score distribution. Any score above this is flagged.
        
        Example: if α = 0.01 (1% FPR target):
        - Sort all normal scores
        - Take the 99th percentile value
        - That value is τ
        - By construction, only 1% of normal scores exceed τ
        
        WHY is this a "guarantee"?
        On the training distribution, exactly α fraction of normal
        samples will be falsely flagged. It's an empirical guarantee,
        not a probabilistic bound — which is why we need held-out
        normal data that the model hasn't seen during training.
        
        WHY not just use ROC curve to pick threshold?
        ROC picks the threshold that maximizes some combination of
        TPR and FPR. NP lets YOU specify the FPR constraint and
        finds the best TPR achievable under that constraint.
        This matches how a SOC actually operates: the analyst
        capacity is fixed, so the FPR budget is fixed.
        """
        self.normal_scores = normal_scores
        tau = np.quantile(normal_scores, 1 - target_fpr)
        self.threshold = tau
        
        empirical_fpr = (normal_scores > tau).mean()
        print(f"\nNP Calibration (target FPR = {target_fpr*100:.1f}%):")
        print(f"  Threshold τ = {tau:.4f}")
        print(f"  Empirical FPR on normal = {empirical_fpr*100:.2f}%")
        
        return tau
    
    def aggregate_user_scores(
        self, 
        result: dict,
        aggregation: str = 'max'
    ) -> pd.DataFrame:
        """
        Convert per-window scores into per-user scores.
        
        WHY aggregate to user level?
        An insider doesn't just have one anomalous window —
        they have a sustained period of unusual behavior.
        Window-level detection is noisy. User-level aggregation
        smooths that noise and gives you a cleaner signal.
        
        WHY 'max' aggregation by default?
        Taking the maximum window score for each user captures
        the "worst day" — the peak of their anomalous activity.
        
        Alternatives:
        - mean: more robust to noise, but dilutes sharp spikes
        - top-k mean: average of k highest scores — best of both
        - percentile (95th): similar to max but outlier-robust
        
        In practice, top-3 mean tends to work best on CERT.
        We'll implement all and compare.
        """
        df = pd.DataFrame({
            'user':  result['users'],
            'score': result['scores'],
            'label': result['labels'],
        })
        
        # Ground truth: a user is malicious if ANY of their 
        # windows is malicious
        user_df = df.groupby('user').agg(
            max_score    = ('score', 'max'),
            mean_score   = ('score', 'mean'),
            top3_score   = ('score', lambda x: 
                           x.nlargest(min(3, len(x))).mean()),
            p95_score    = ('score', lambda x: 
                           np.percentile(x, 95)),
            n_windows    = ('score', 'count'),
            is_malicious = ('label', 'max'),
        ).reset_index()
        
        return user_df
    def smooth_user_scores(self,result: dict, window: int = 7) -> dict:
        """
        Apply rolling average over each user's daily anomaly scores.
        
        WHY smooth? An insider's activity ramps up over days/weeks —
        they don't just have one anomalous day. A rolling average
        catches the gradual escalation that single-window scoring misses.
        
        This is the same intuition as EMA in financial time series:
        smooth out noise, preserve trend.
        """
        df = pd.DataFrame({
            'user':  result['users'],
            'score': result['scores'],
            'label': result['labels'],
        })
        
        # Sort by user then by natural sequence order
        df['seq_idx'] = df.groupby('user').cumcount()
        df = df.sort_values(['user', 'seq_idx'])
        
        # Rolling mean per user
        df['smoothed_score'] = df.groupby('user')['score'].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )
        
        return df
    def evaluate(
        self, 
        user_df: pd.DataFrame,
        score_col: str = 'max_score'
    ) -> dict:
        """
        Full evaluation suite.
        
        We report multiple metrics because each tells a different story:
        
        AUC-ROC: overall discrimination ability. 
                 How well does the model separate normal from malicious?
                 Threshold-independent — tells you about the score 
                 distribution, not a specific operating point.
        
        TPR@1%FPR: at the NP operating point, how many malicious
                   users do we catch? This is the number that maps
                   directly to SOC workload and detection rate.
        
        AP (Average Precision): area under precision-recall curve.
                   More informative than AUC when classes are imbalanced
                   (which they always are in insider threat detection —
                   maybe 5% of users are malicious).
        
        We also compute these at multiple FPR targets because different
        deployments have different analyst capacities.
        """
        scores = user_df[score_col].values
        labels = user_df['is_malicious'].values
        
        # ── Core metrics ─────────────────────────────────────────
        auc   = roc_auc_score(labels, scores)
        ap    = average_precision_score(labels, scores)
        
        # ROC curve
        fpr_arr, tpr_arr, thresholds = roc_curve(labels, scores)
        
        # TPR at specific FPR targets (NP operating points)
        results = {
            'auc':    auc,
            'ap':     ap,
            'fpr':    fpr_arr,
            'tpr':    tpr_arr,
            'thresholds': thresholds,
        }
        
        for target_fpr in [0.01, 0.05, 0.10]:
            # Find closest point on ROC curve to target FPR
            idx = np.argmin(np.abs(fpr_arr - target_fpr))
            tpr_at_fpr = tpr_arr[idx]
            actual_fpr = fpr_arr[idx]
            results[f'tpr@{int(target_fpr*100)}fpr'] = tpr_at_fpr
            print(f"  TPR@{int(target_fpr*100)}%FPR = "
                  f"{tpr_at_fpr*100:.1f}% "
                  f"(actual FPR: {actual_fpr*100:.2f}%)")
        
        print(f"\n  AUC-ROC:           {auc:.4f}")
        print(f"  Average Precision: {ap:.4f}")
        
        # ── Confusion matrix at NP threshold ─────────────────────
        if self.threshold is not None:
            preds = (scores > self.threshold).astype(int)
            cm    = confusion_matrix(labels, preds)
            tn, fp, fn, tp = cm.ravel()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            print(f"\n  At NP threshold (τ={self.threshold:.4f}):")
            print(f"  TP={tp}  FP={fp}  TN={tn}  FN={fn}")
            print(f"  Precision: {precision*100:.1f}%")
            print(f"  Recall:    {recall*100:.1f}%")
            
            results.update({
                'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
                'precision': precision, 'recall': recall,
            })
        
        return results
    
    def plot_score_distributions(
        self, 
        result: dict,
        save_path: str = None
    ):
        """
        The most important diagnostic plot.
        
        If your model is working, normal and malicious score
        distributions should be clearly separated. Overlap = 
        the model can't distinguish them at that operating point.
        
        What to look for:
        - Good: two distinct peaks, malicious peak shifted right
        - Bad: complete overlap — model learned nothing useful
        - Acceptable: partial overlap — some detection possible
        """
        normal_scores   = result['scores'][result['labels'] == 0]
        malicious_scores = result['scores'][result['labels'] == 1]
        
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        
        # ── Plot 1: Score distributions ──────────────────────────
        axes[0].hist(normal_scores, bins=50, alpha=0.6, 
                     color='steelblue', label='Benign', density=True)
        axes[0].hist(malicious_scores, bins=50, alpha=0.6, 
                     color='crimson', label='Malicious', density=True)
        if self.threshold:
            axes[0].axvline(self.threshold, color='orange', 
                           linestyle='--', linewidth=2,
                           label=f'NP threshold (τ={self.threshold:.3f})')
        axes[0].set_xlabel('Reconstruction Error (Anomaly Score)')
        axes[0].set_ylabel('Density')
        axes[0].set_title('Score Distributions')
        axes[0].legend()
        
        # ── Plot 2: ROC Curve ────────────────────────────────────
        # Compute from window-level scores
        auc = roc_auc_score(result['labels'], result['scores'])
        fpr, tpr, _ = roc_curve(result['labels'], result['scores'])
        
        axes[1].plot(fpr, tpr, color='steelblue', linewidth=2,
                    label=f'AUC = {auc:.3f}')
        axes[1].plot([0,1],[0,1], 'k--', alpha=0.3, label='Random')
        # Mark the NP operating points
        for target, color in [(0.01,'red'),(0.05,'orange'),(0.10,'green')]:
            idx = np.argmin(np.abs(fpr - target))
            axes[1].scatter(fpr[idx], tpr[idx], color=color, 
                          zorder=5, s=80,
                          label=f'FPR={target*100:.0f}%→TPR={tpr[idx]*100:.0f}%')
        axes[1].set_xlabel('False Positive Rate')
        axes[1].set_ylabel('True Positive Rate')
        axes[1].set_title('ROC Curve (window-level)')
        axes[1].legend(fontsize=8)
        
        # ── Plot 3: Per-user score heatmap ───────────────────────
        # Show top 20 highest-scored users
        user_df = pd.DataFrame({
            'user':  result['users'],
            'score': result['scores'],
            'label': result['labels'],
        }).groupby('user').agg(
            max_score=('score','max'),
            is_malicious=('label','max')
        ).sort_values('max_score', ascending=False).head(20)
        
        colors = ['crimson' if m else 'steelblue' 
                  for m in user_df['is_malicious']]
        axes[2].barh(range(len(user_df)), user_df['max_score'], 
                    color=colors)
        if self.threshold:
            axes[2].axvline(self.threshold, color='orange',
                           linestyle='--', label='NP threshold')
        axes[2].set_yticks(range(len(user_df)))
        axes[2].set_yticklabels(
            [f"{'[M]' if m else '[B]'} {u[:8]}"
            for u, m in zip(user_df.index, user_df['is_malicious'])],
            fontsize=8
        )
        axes[2].set_xlabel('Max Anomaly Score')
        axes[2].set_title('Top 20 Users by Anomaly Score\n(red=malicious, blue=benign)')
        axes[2].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
        return fig