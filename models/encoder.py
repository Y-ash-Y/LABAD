import torch
import torch.nn as nn

class LSTMAutoencoder(nn.Module):
    """
    Encoder: compresses a 30-day behavioral sequence into a 
             fixed-size latent vector (the "behavioral fingerprint")
    Decoder: reconstructs the original sequence from that vector
    
    Training objective: minimize reconstruction error on NORMAL sequences
    
    At inference time:
    - Normal sequence  → low reconstruction error  → benign
    - Anomalous sequence → high reconstruction error → FLAG
    
    WHY bidirectional encoder? A user's behavior on day 15 is
    influenced by both past context (days 1-14) and what follows
    (days 16-30 — e.g., a gradually escalating pattern). BiLSTM
    captures both directions for a richer encoding.
    
    WHY unidirectional decoder? At inference time, you're predicting
    what "should" happen next based on past context only — the future
    isn't available. Unidirectional matches real-world deployment.
    """
    
    def __init__(
        self,
        n_features: int,     # number of behavioral features per day
        hidden_dim: int = 128,
        latent_dim: int = 64,
        n_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.n_layers   = n_layers
        
        # ── Encoder ──────────────────────────────────────────────
        # Input: (batch, seq_len, n_features)
        # Output: (batch, seq_len, hidden_dim*2) — *2 for bidirectional
        self.encoder = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0,
        )
        
        # Project bidirectional hidden state → latent vector
        # hidden_dim*2 because bidirectional concatenates both directions
        self.encoder_fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, latent_dim),
            nn.Tanh(),   # Tanh keeps latent values in [-1, 1] — stable
        )
        
        # ── Decoder ──────────────────────────────────────────────
        # Expands latent back to sequence
        self.decoder_fc = nn.Linear(latent_dim, hidden_dim)
        
        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if n_layers > 1 else 0,
        )
        
        # Project hidden state back to feature space
        self.output_layer = nn.Linear(hidden_dim, n_features)
    
    def encode(self, x):
        """
        x: (batch, seq_len, n_features)
        returns: (batch, latent_dim) — the behavioral fingerprint
        """
        _, (hidden, _) = self.encoder(x)
        # hidden: (n_layers*2, batch, hidden_dim) for bidirectional
        # Take the last layer's forward + backward hidden states
        # Forward: hidden[-2], Backward: hidden[-1]
        last_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        # last_hidden: (batch, hidden_dim*2)
        return self.encoder_fc(last_hidden)
    
    def decode(self, z, seq_len):
        """
        z: (batch, latent_dim) — behavioral fingerprint
        seq_len: length of sequence to reconstruct
        returns: (batch, seq_len, n_features)
        """
        # Expand latent to initial hidden state
        h0 = self.decoder_fc(z)           # (batch, hidden_dim)
        h0 = h0.unsqueeze(0).repeat(       # (n_layers, batch, hidden_dim)
            self.n_layers, 1, 1
        )
        c0 = torch.zeros_like(h0)          # cell state initialized to zero
        
        # Repeat latent as input at every timestep
        # The decoder "unfolds" the latent by attending to it repeatedly
        decoder_input = self.decoder_fc(z)                    # (batch, hidden_dim)
        decoder_input = decoder_input.unsqueeze(1).repeat(    # (batch, seq_len, hidden_dim)
            1, seq_len, 1
        )
        
        output, _ = self.decoder(decoder_input, (h0, c0))
        # output: (batch, seq_len, hidden_dim)
        
        return self.output_layer(output)  # (batch, seq_len, n_features)
    
    def forward(self, x):
        """Full encode → decode pass."""
        z = self.encode(x)
        x_reconstructed = self.decode(z, seq_len=x.shape[1])
        return x_reconstructed, z
    
    def anomaly_score(self, x):
        """
        Compute reconstruction error per sample.
        
        WHY mean over both seq_len and features?
        We want a single scalar score per window that captures
        how "surprised" the model was by this sequence overall.
        Mean is more robust than sum (independent of sequence length).
        """
        x_reconstructed, _ = self.forward(x)
        # MSE per sample: average over time steps and features
        error = torch.mean((x - x_reconstructed) ** 2, dim=(1, 2))
        return error  # shape: (batch,)

    def anomaly_score_per_feature(self, x):
        """
        Per-FEATURE reconstruction error: average over time only.

        WHY keep the feature axis?
        The scalar anomaly_score() averages over all 18 features, so a
        sharp anomaly in ONE feature (e.g. job-site browsing) gets diluted
        by 17 stable features and vanishes. That is exactly why gradual
        insiders (CERT scenario 2) are invisible to the scalar score —
        their score barely moves at onset.

        Returning the per-feature error preserves WHICH feature went
        anomalous and WHEN, so a changepoint detector can catch a shift
        confined to a single behavioral dimension.
        """
        x_reconstructed, _ = self.forward(x)
        # MSE per (sample, feature): average over time steps only
        error = torch.mean((x - x_reconstructed) ** 2, dim=1)
        return error  # shape: (batch, n_features)