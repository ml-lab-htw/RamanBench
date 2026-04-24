import math

import torch
import torch.nn as nn
from autogluon.common import space

from raman_bench.models.custom.base import BaseCustomModel


class _PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        return x + self.pe[:, :x.size(1)]


class _RamanFormerNetwork(nn.Module):
    """Transformer encoder for spectral data (Koyun et al., 2024).

    1. Patchify: non-overlapping patches -> Linear projection + ReLU + positional encoding
    2. Transformer encoder layers
    3. Two Conv1d layers with BN + GELU
    4. Global average pooling -> MLP head -> output
    """

    def __init__(
        self, n_features, n_outputs,
        patch_size=128, d_model=256, nhead=8,
        dim_feedforward=1024, n_layers=3, dropout=0.1,
        post_processing_dim=512,
    ):
        super().__init__()
        self.patch_size = patch_size

        # Number of patches (pad input if needed)
        self.n_patches = math.ceil(n_features / patch_size)
        self.padded_len = self.n_patches * patch_size

        # Patch embedding
        self.patch_proj = nn.Linear(patch_size, d_model)
        self.pos_enc = _PositionalEncoding(d_model, max_len=self.n_patches)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            activation="gelu",
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Conv1d post-processing (operates on sequence dim as "length")
        self.conv_layers = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Conv1d(d_model, post_processing_dim, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm1d(post_processing_dim),
            nn.GELU(),
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Classification / regression head
        self.head = nn.Sequential(
            nn.Linear(post_processing_dim, int(post_processing_dim / 2)),
            nn.ReLU(inplace=True),
            nn.Linear(int(post_processing_dim / 2), n_outputs),
        )

    def forward(self, x):
        # x: (batch, n_features)
        batch_size = x.size(0)

        # Pad to multiple of patch_size
        if x.size(1) < self.padded_len:
            x = nn.functional.pad(x, (0, self.padded_len - x.size(1)))

        # Patchify: (batch, n_patches, patch_size)
        x = x.view(batch_size, self.n_patches, self.patch_size)

        # Patch projection + positional encoding
        x = self.patch_proj(x)  # (batch, n_patches, d_model)
        x = self.pos_enc(x)

        # Transformer
        x = self.transformer(x)  # (batch, n_patches, d_model)

        # Conv1d expects (batch, channels, length)
        x = x.permute(0, 2, 1)
        x = self.conv_layers(x)
        x = self.global_pool(x).squeeze(-1)  # (batch, 512)

        return self.head(x)


class RamanFormerModel(BaseCustomModel):
    """AutoGluon-compatible RamanFormer model for spectral data.

    Transformer encoder architecture with patch embedding and Conv1d
    post-processing. Supports both classification and regression.

    Reference:
        Koyun, O. C., Keser, R. K., Sahin, S. O., Bulut, D., Yorulmaz, M.,
        Yucesoy, V., & Toreyin, B. U. (2024). RamanFormer: A
        transformer-based quantification approach for Raman mixture
        components. ACS Omega, 9(22), 23241-23251.
        https://doi.org/10.1021/acsomega.3c09247
    """

    def _set_default_params(self):
        default_params = {
            # Architecture
            "patch_size": 128,
            "d_model": 256,
            "nhead": 8,
            "dim_feedforward": 1024,
            "n_layers": 3,
            "post_processing_dim": 512,
            # Training
            "n_epochs": 100,
            "lr": 1e-4,
            "batch_size": 128,
            "patience": 10,
            "val_fraction": 0.1,
            "warmup_epochs": 10,
            # Regularization
            "dropout": 0.1,
            "weight_decay": 1e-4,
            # Augmentation
            "aug_noise_sigma": 0.01,
            "aug_mixup_alpha": 1e-12,
            "per_epoch_augmentation": False,
            "aug_max_train_samples": 2000,
            "aug_n_per_epoch": 20,
            "grad_clip_norm": 1.0,
        }
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    def _get_default_searchspace(self):
        return self._get_base_hp_searchspace()

    @staticmethod
    def _get_base_hp_searchspace():
        return {
            # Architecture
            "patch_size": space.Categorical(32, 64, 128, 256),
            "d_model": space.Categorical(64, 128, 256),
            "dim_feedforward": space.Categorical(256, 512, 1024),
            "n_layers": space.Int(lower=1, upper=6),
            "post_processing_dim": space.Categorical(256, 512, 1024),
            # Training
            "lr": space.Real(lower=1e-5, upper=1e-3, log=True),
            # Regularization
            "dropout": space.Categorical(0.1, 0.2, 0.3),
            "weight_decay": space.Real(1e-6, 1e-1, log=True),
            # Augmentation
            "aug_noise_sigma": space.Real(1e-3, 0.1, log=True),
            "aug_mixup_alpha": space.Real(1e-3, 1e3, log=True),
            "per_epoch_augmentation": space.Categorical(True, False),
        }

    def _fit(self, X, y, time_limit=None, **kwargs):
        params = self._get_model_params()
        self._log_params(params)

        # Architecture
        patch_size = params["patch_size"]
        d_model = params["d_model"]
        nhead = params["nhead"]
        dim_feedforward = params["dim_feedforward"]
        n_layers = params["n_layers"]
        post_processing_dim = params["post_processing_dim"]

        # Training
        n_epochs = params["n_epochs"]
        lr = params["lr"]
        batch_size = params["batch_size"]
        patience = params["patience"]
        val_fraction = params["val_fraction"]
        warmup_epochs = params["warmup_epochs"]

        # Regularization
        dropout = params["dropout"]
        weight_decay = params["weight_decay"]

        # Augmentation
        aug_noise_sigma = params["aug_noise_sigma"]
        aug_mixup_alpha = params["aug_mixup_alpha"]
        per_epoch_augmentation = params["per_epoch_augmentation"]
        aug_max_train_samples = params.get("aug_max_train_samples", None)
        aug_n_per_epoch = params.get("aug_n_per_epoch", 1)
        grad_clip_norm = params.get("grad_clip_norm", None)

        self._setup_device()
        X_np, y_np, n_outputs, criterion = self._prepare_labels(X, y)
        X_train_t, X_val_t, y_train_t, y_val_t = self._train_val_split(X_np, y_np, val_fraction)

        self.model = _RamanFormerNetwork(
            n_features=X_np.shape[1],
            n_outputs=n_outputs,
            patch_size=patch_size,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            n_layers=n_layers,
            dropout=dropout,
            post_processing_dim=post_processing_dim,
        ).to(self._device)

        self._run_training_loop(
            X_train_t=X_train_t,
            y_train_t=y_train_t,
            X_val_t=X_val_t,
            y_val_t=y_val_t,
            n_epochs=n_epochs,
            patience=patience,
            time_limit=time_limit,
            criterion=criterion,
            per_epoch_augmentation=per_epoch_augmentation,
            batch_size=batch_size,
            aug_noise_sigma=aug_noise_sigma,
            aug_mixup_alpha=aug_mixup_alpha,
            lr=lr,
            weight_decay=weight_decay,
            warmup_epochs=warmup_epochs,
            aug_max_train_samples=aug_max_train_samples,
            aug_n_per_epoch=aug_n_per_epoch,
            grad_clip_norm=grad_clip_norm,
        )
