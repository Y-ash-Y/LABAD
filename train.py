import argparse
import os
from pathlib import Path

import matplotlib
import mlflow
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd

from models.dataset import CERTFeatureEngineer, UserBehaviorDataset
from models.encoder import LSTMAutoencoder
import config
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

matplotlib.use('Agg')
import matplotlib.pyplot as plt

def train_epoch(model, loader, optimizer, scheduler, device, epoch):
    model.train()
    total_loss = 0
    for batch_idx, (batch_x, _) in enumerate(loader):  # _ = labels, unused
        batch_x = batch_x.to(device)
        
        optimizer.zero_grad()
        x_reconstructed, _ = model(batch_x)
        
        # MSE reconstruction loss
        # WHY MSE and not BCE or cross-entropy?
        # Our features are continuous values (counts, hours), not
        # probabilities or classes. MSE penalizes large deviations
        # proportionally — exactly what we want.
        loss = nn.MSELoss()(x_reconstructed, batch_x)
        
        loss.backward()
        
        # Gradient clipping: prevents exploding gradients in LSTMs
        # WHY 1.0? Standard value for LSTMs — large enough to not
        # interfere with normal training, small enough to prevent spikes.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step(epoch + batch_idx / len(loader))

        total_loss += loss.item()
    
    return total_loss / len(loader)

def evaluate(model, loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch_x, _ in loader:
            batch_x = batch_x.to(device)
            x_reconstructed, _ = model(batch_x)
            loss = nn.MSELoss()(x_reconstructed, batch_x)
            total_loss += loss.item()
    return total_loss / len(loader)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs (default: 50).',
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        help='Training and validation batch size (default: 64).',
    )
    parser.add_argument(
        '--step-size',
        type=int,
        default=1,
        help='Days to move each 30-day sequence window (default: 1).',
    )
    parser.add_argument(
        '--setup-only',
        action='store_true',
        help='Build datasets and model, then exit before training.',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.epochs < 1 and not args.setup_only:
        raise ValueError('--epochs must be at least 1')
    if args.batch_size < 1:
        raise ValueError('--batch-size must be at least 1')
    if args.step_size < 1:
        raise ValueError('--step-size must be at least 1')

    device = torch.device('mps' if torch.backends.mps.is_available() 
                          else 'cpu')
    # MPS = Apple Metal — your M4 chip. Will be ~5-10x faster than CPU.
    print(f"Using device: {device}")
    
    # ── 1. Feature engineering ───────────────────────────────────
    engineer = CERTFeatureEngineer(data_dir="data/raw")
    engineer.load_raw_data()

    features_path = Path(config.PROCESSED_DATA_PATH) / "daily_features.csv"
    if features_path.is_file():
        print(f"Loading cached daily features from {features_path}...")
        features_df = pd.read_csv(features_path, parse_dates=['date'])
    else:
        features_df = engineer.build_daily_features()
    
    # Split users into train/test BEFORE normalization
    # WHY user-level split (not row-level)?
    # If we split rows, the same user appears in both train and test.
    # The model would learn that user's baseline from training rows,
    # making test evaluation unrealistically easy. User-level split
    # is the correct evaluation for a UEBA system.
    all_users      = features_df['user'].unique()
    malicious_users = engineer.insiders['user'].unique()
    benign_users   = [u for u in all_users if u not in malicious_users]
    
    # 80% benign users for training, 20% + all malicious for testing
    n_train = int(0.8 * len(benign_users))
    train_users = benign_users[:n_train]
    test_users  = list(benign_users[n_train:]) + list(malicious_users)
    
    # Fit scaler on TRAIN data only, then transform both
    train_df = features_df[features_df['user'].isin(train_users)]
    test_df  = features_df[features_df['user'].isin(test_users)]
    
    train_df_norm = engineer.normalize_features(train_df, fit=True)
    test_df_norm  = engineer.normalize_features(test_df,  fit=False)
    
    engineer.save(features_df, config.PROCESSED_DATA_PATH)
    
    # ── 2. Build datasets ────────────────────────────────────────
    train_dataset = UserBehaviorDataset(
        train_df_norm,
        window_size=30,
        step_size=args.step_size,
        mode='train',
        benign_users=train_users,
    )
    
    test_dataset = UserBehaviorDataset(
        test_df_norm,
        window_size=30,
        step_size=args.step_size,
        mode='test',
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=True,   # shuffle within training — order doesn't matter
        num_workers=0,  # 0 for MPS compatibility
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size,
        shuffle=False,  # never shuffle test — evaluation must be deterministic
    )
    
    # ── 3. Model + optimizer ─────────────────────────────────────
    model = LSTMAutoencoder(
        n_features=train_dataset.n_features,
        hidden_dim=128,
        latent_dim=64,
        n_layers=2,
        dropout=0.2,
    ).to(device)
    
    print(f"\nModel parameters: "
          f"{sum(p.numel() for p in model.parameters()):,}")
    
    # WHY Adam? Adaptive learning rates handle the varying scales of
    # our features well. SGD would require careful LR tuning per feature.
    # WHY lr=1e-3? Standard starting point for Adam on sequence models.
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    

    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=30,        # restart every 30 epochs
        T_mult=2,      # double the period after each restart
        eta_min=1e-5,  # minimum LR floor
    )

    if args.setup_only:
        print("Setup complete; skipping training.")
        return
    
    # ── 4. Training loop ─────────────────────────────────────────
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    os.makedirs("checkpoints", exist_ok=True)

    # ── Experiment tracking ──────────────────────────────────────
    # Log the ACTUAL config (CosineAnnealingWarmRestarts, not the older
    # OneCycleLR) so the MLflow run is a faithful record of this schedule.
    mlflow.set_experiment("labad-lstm-autoencoder")
    mlflow.start_run()
    mlflow.log_params({
        "epochs":         args.epochs,
        "batch_size":     args.batch_size,
        "step_size":      args.step_size,
        "n_features":     train_dataset.n_features,
        "hidden_dim":     128,
        "latent_dim":     64,
        "n_layers":       2,
        "dropout":        0.2,
        "window_size":    30,
        "loss":           "MSE",
        "optimizer":      "Adam",
        "lr":             1e-3,
        "scheduler":      "CosineAnnealingWarmRestarts",
        "scheduler_T_0":  30,
        "scheduler_T_mult": 2,
        "scheduler_eta_min": 1e-5,
        "device":         str(device),
    })

    print("\nStarting training...")
    for epoch in range(1, args.epochs + 1):
        
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, epoch)
        val_loss   = evaluate(model, test_loader, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'val_loss': val_loss,
                'n_features': train_dataset.n_features,
            }, "checkpoints/best_model.pt")
        
        mlflow.log_metrics(
            {"train_loss": train_loss, "val_loss": val_loss},
            step=epoch,
        )

        print(f"Epoch {epoch:3d} | "
              f"Train loss: {train_loss:.4f} | "
              f"Val loss: {val_loss:.4f} | "
              f"Best: {best_val_loss:.4f}")

    # ── 5. Plot training curve ───────────────────────────────────
    plt.figure(figsize=(10, 4))
    plt.plot(train_losses, label='Train loss')
    plt.plot(val_losses,   label='Val loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('LSTM Autoencoder — Reconstruction Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig("data/processed/training_curve.png")
    plt.close()

    # ── Log final outcome + artifacts, then close the run ────────
    mlflow.log_metric("best_val_loss", best_val_loss)
    mlflow.log_artifact("checkpoints/best_model.pt")
    mlflow.log_artifact("data/processed/training_curve.png")
    mlflow.end_run()

    print(f"\nWeek 1 complete.")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to checkpoints/best_model.pt")
    print(f"Next: Week 2 — anomaly scoring + NP threshold calibration")

if __name__ == '__main__':
    main()
