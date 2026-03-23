"""
act_model.py
Faithful implementation of the ACT (Action Chunking with Transformers) policy.

Architecture (Zhao et al., 2023 — arXiv:2304.13705)
─────────────────────────────────────────────────────
Training path
  obs (state + image) ──┐
                         ├─► Transformer encoder ─► CVAE decoder ─► action chunk
  action_seq ──► CVAE   ┘
                encoder ─► z ~ N(mu, sigma)

Inference path
  obs ─► Transformer encoder ─► CVAE decoder (z=0) ─► action chunk

Loss
  L = L1(pred, gt) + β · KL(N(mu,σ) ‖ N(0,I))
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Image Encoder ────────────────────────────────────────────────────────────

class ImageEncoder(nn.Module):
    """Small CNN: (B, 3, H, W) → (B, hidden_dim)."""

    def __init__(self, img_size: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3,  16, 4, stride=2, padding=1), nn.ReLU(),   # H/2
            nn.Conv2d(16, 32, 4, stride=2, padding=1), nn.ReLU(),   # H/4
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.ReLU(),   # H/8
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 3, img_size, img_size)
            flat_dim = self.net(dummy).shape[1]
        self.proj = nn.Linear(flat_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.net(x))


# ─── Positional Encoding ──────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, :x.size(1)])


# ─── CVAE Encoder ─────────────────────────────────────────────────────────────

class CVAEEncoder(nn.Module):
    """
    Encodes (observation summary, action sequence) → (mu, log_var).
    Used only during training to produce the style latent z.
    """

    def __init__(self, obs_dim: int, action_dim: int, chunk_size: int,
                 hidden_dim: int, latent_dim: int, nhead: int, num_layers: int):
        super().__init__()
        self.action_proj = nn.Linear(action_dim, hidden_dim)
        self.cls_token    = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.pos_enc      = PositionalEncoding(hidden_dim, max_len=chunk_size + 2)
        enc_layer = nn.TransformerEncoderLayer(
            hidden_dim, nhead, dim_feedforward=hidden_dim * 4,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers)
        self.fc_mu      = nn.Linear(hidden_dim, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim, latent_dim)

    def forward(self, action_seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        action_seq: (B, chunk_size, action_dim)
        returns: mu (B, latent_dim), log_var (B, latent_dim)
        """
        B = action_seq.size(0)
        act_emb  = self.action_proj(action_seq)                       # (B, C, H)
        cls      = self.cls_token.expand(B, -1, -1)                   # (B, 1, H)
        x        = torch.cat([cls, act_emb], dim=1)                   # (B, C+1, H)
        x        = self.pos_enc(x)
        out      = self.transformer(x)                                 # (B, C+1, H)
        cls_out  = out[:, 0]                                           # (B, H)
        return self.fc_mu(cls_out), self.fc_log_var(cls_out)


# ─── ACT Policy (full model) ──────────────────────────────────────────────────

class ACTPolicy(nn.Module):
    """
    Full ACT policy.

    Parameters
    ----------
    obs_dim    : dimensionality of proprioceptive observation
    action_dim : dimensionality of one action step
    img_size   : side length of square input images
    chunk_size : number of future action steps to predict per forward pass
    hidden_dim : transformer hidden dimension
    latent_dim : CVAE latent dimension
    nhead      : transformer attention heads
    num_enc_layers : encoder transformer layers (obs + CVAE)
    num_dec_layers : decoder transformer layers
    beta       : KL weight in training loss
    """

    def __init__(
        self,
        obs_dim: int        = 13,
        action_dim: int     = 7,
        img_size: int       = 32,
        chunk_size: int     = 10,
        hidden_dim: int     = 256,
        latent_dim: int     = 32,
        nhead: int          = 8,
        num_enc_layers: int = 4,
        num_dec_layers: int = 4,
        beta: float         = 10.0,
    ):
        super().__init__()
        self.chunk_size = chunk_size
        self.latent_dim = latent_dim
        self.beta       = beta

        # Observation encoders
        self.state_proj   = nn.Linear(obs_dim, hidden_dim)
        self.image_enc    = ImageEncoder(img_size, hidden_dim)
        # Learnable type tokens to distinguish state vs image tokens
        self.obs_type_emb = nn.Embedding(2, hidden_dim)

        # CVAE encoder (training only)
        self.cvae_enc = CVAEEncoder(
            obs_dim, action_dim, chunk_size,
            hidden_dim, latent_dim, nhead, num_enc_layers
        )

        # Latent → hidden projection
        self.latent_proj = nn.Linear(latent_dim, hidden_dim)

        # Transformer encoder for observations + latent token
        enc_layer = nn.TransformerEncoderLayer(
            hidden_dim, nhead, dim_feedforward=hidden_dim * 4,
            dropout=0.1, batch_first=True
        )
        self.obs_transformer = nn.TransformerEncoder(enc_layer, num_enc_layers)

        # Learnable query embeddings for action chunk
        self.query_emb = nn.Embedding(chunk_size, hidden_dim)
        self.pos_enc   = PositionalEncoding(hidden_dim, max_len=chunk_size)

        # Transformer decoder
        dec_layer = nn.TransformerDecoderLayer(
            hidden_dim, nhead, dim_feedforward=hidden_dim * 4,
            dropout=0.1, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_dec_layers)

        # Action prediction head
        self.action_head = nn.Linear(hidden_dim, action_dim)

        self._init_weights()

    # ── Weight initialisation ─────────────────────────────────────────────────

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    # ── Reparameterisation trick ──────────────────────────────────────────────

    @staticmethod
    def reparameterise(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        return mu + std * torch.randn_like(std)

    # ── Build observation memory ──────────────────────────────────────────────

    def _encode_obs(self, state: torch.Tensor, image: torch.Tensor,
                    z: torch.Tensor) -> torch.Tensor:
        """
        state : (B, obs_dim)
        image : (B, 3, H, W)  – float32, pixels in [0, 1]
        z     : (B, hidden_dim)  – projected latent token
        returns: memory (B, N, H) for transformer decoder
        """
        B = state.size(0)
        state_tok = self.state_proj(state).unsqueeze(1)          # (B, 1, H)
        image_tok = self.image_enc(image).unsqueeze(1)           # (B, 1, H)
        # Add type embeddings
        state_tok = state_tok + self.obs_type_emb(torch.zeros(B, 1, dtype=torch.long, device=state.device))
        image_tok = image_tok + self.obs_type_emb(torch.ones(B, 1,  dtype=torch.long, device=state.device))
        z_tok     = z.unsqueeze(1)                               # (B, 1, H)
        memory    = torch.cat([z_tok, state_tok, image_tok], 1) # (B, 3, H)
        return self.obs_transformer(memory)

    # ── Forward pass ─────────────────────────────────────────────────────────

    def forward(
        self,
        state: torch.Tensor,
        image: torch.Tensor,
        action_seq: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        state      : (B, obs_dim)
        image      : (B, 3, H, W)
        action_seq : (B, chunk_size, action_dim)  – None at inference

        returns
        -------
        pred_actions : (B, chunk_size, action_dim)
        mu           : (B, latent_dim)
        log_var      : (B, latent_dim)
        """
        B = state.size(0)
        device = state.device

        if action_seq is not None:
            mu, log_var = self.cvae_enc(action_seq)
            z = self.reparameterise(mu, log_var)
        else:
            mu = log_var = z = torch.zeros(B, self.latent_dim, device=device)

        z_proj = self.latent_proj(z)
        memory = self._encode_obs(state, image, z_proj)          # (B, 3, H)

        queries = self.pos_enc(
            self.query_emb.weight.unsqueeze(0).expand(B, -1, -1)
        )                                                         # (B, chunk_size, H)
        out         = self.decoder(queries, memory)               # (B, chunk_size, H)
        pred_actions = self.action_head(out)                      # (B, chunk_size, action_dim)

        return pred_actions, mu, log_var

    # ── Loss ─────────────────────────────────────────────────────────────────

    def compute_loss(
        self,
        pred_actions: torch.Tensor,
        gt_actions:   torch.Tensor,
        mu:           torch.Tensor,
        log_var:      torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        l1_loss = F.l1_loss(pred_actions, gt_actions)
        kl_loss = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).sum(dim=-1).mean()
        total   = l1_loss + self.beta * kl_loss
        return {"total": total, "l1": l1_loss, "kl": kl_loss}

    # ── Parameter count ──────────────────────────────────────────────────────

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─── Quick sanity check ───────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = dict(obs_dim=13, action_dim=7, img_size=32, chunk_size=10,
               hidden_dim=256, latent_dim=32, nhead=8,
               num_enc_layers=4, num_dec_layers=4, beta=10.0)
    model = ACTPolicy(**cfg)
    print(f"ACTPolicy  |  parameters: {model.count_parameters():,}")

    B = 4
    state  = torch.randn(B, 13)
    image  = torch.randn(B, 3, 32, 32)
    acts   = torch.randn(B, 10, 7)

    pred, mu, lv = model(state, image, acts)
    losses = model.compute_loss(pred, acts, mu, lv)
    print(f"Output shape : {pred.shape}")
    print(f"Losses       : { {k: f'{v.item():.4f}' for k, v in losses.items()} }")
